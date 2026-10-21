# Modified for the V100 (sm70) port, 2026-10 - see README.md / NOTES_V100.md.
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import math
import inspect
import queue
import random
import time
import traceback
from contextlib import nullcontext
from datetime import timedelta
from functools import partial

import cv2
import numpy as np
from einops import rearrange
from loguru import logger
from decord import VideoReader
from easydict import EasyDict
from tqdm import tqdm

try:
    import moviepy.editor as mpy
except:
    import moviepy as mpy

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard, FSDPModule

from core import build_object_from_dict, BaseInferencePipeline

from wanxiang import ops
from wanxiang.models.clip import AttentionBlock
from wanxiang.models.vae import VideoVAE
from wanxiang.utils.utils import TensorList, get_sharding_strategy, get_dtype
from wanxiang.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from wanxiang.eval_i2v import T5Encoder, CLIP, get_i2v_mask
from wanxiang.distributed.fsdp import shard_model, shard_transformer
from wanxiang.models import attention as wan_attention
from wanxiang.utils.precision import resolve_precision, parse_dtype
from wanxiang.utils.fp16_guard import check_fp16_castable_cached, force_fp32, OverflowGuard

from pipelines.utils.multiclip_utils import zigzag_padding, get_padding_len
from pipelines.utils.human_video_omni_utils import resize_by_area, get_frame_indices
from pipelines.utils.utils import ffmpeg_extract_audio_by_fps, merge_video_audio, get_video_duration
from safetensors import safe_open


def load_safetensors(path):
    tensors = {}
    with safe_open(path,  framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def merge_configs(main_cfg, backup_cfg):
    merged_cfg = EasyDict(main_cfg.copy())
    changes = []
    
    def _merge_dict(d1, d2, path=""):
        for key, value in d2.items():
            current_path = f"{path}.{key}" if path else key
            
            if key in d1:
                if isinstance(value, dict) and isinstance(d1[key], dict):
                    _merge_dict(d1[key], value, current_path)
                elif d1[key] != value:
                    changes.append(f"{current_path}: {d1[key]} -> {value}")
                    d1[key] = value
            else:
                d1[key] = value
                changes.append(f"新增 {current_path}: {value}")
    
    _merge_dict(merged_cfg, backup_cfg)
    
    if changes:
        logger.info("配置覆盖信息:")
        for change in changes:
            logger.info(f"  - {change}")
    else:
        logger.info("没有配置项被覆盖")
    
    return merged_cfg

def get_sampling_sigmas(sampling_steps, shift):
    sigma = np.linspace(1, 0, sampling_steps + 1)[:sampling_steps]
    sigma = shift * sigma / (1 + (shift - 1) * sigma)
    return sigma


def retrieve_timesteps(
    scheduler,
    num_inference_steps=None,
    device=None,
    timesteps=None,
    sigmas=None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def vae_encode(vae, videos, prec):
    """V100 port: run the VAE in prec.vae (fp32 on V100). Upstream path: unchanged call."""
    if prec.upstream:
        return vae.encode(videos)
    try:
        with torch.autocast(device_type="cuda", dtype=prec.vae):
            return vae.encode([u.to(prec.vae) for u in videos])
    finally:  # the causal conv caches stay alive until the next call otherwise (~GBs at 720P)
        vae.model.clear_cache()


def vae_decode(vae, zs, prec):
    """Upstream path: VideoVAE.decode (clamped to [-1, 1]). V100 path: same maths but returned
    UNclamped, so the caller can check for inf/NaN first (clamp would turn inf into +-1)."""
    if prec.upstream:
        return vae.decode(zs)
    try:
        with torch.autocast(device_type="cuda", dtype=prec.vae):
            return [vae.model.decode(u.to(prec.vae).unsqueeze(0), vae.scale).float().squeeze(0) for u in zs]
    finally:
        vae.model.clear_cache()


def t5_autocast(prec):
    """V100 port: T5 computes in prec.t5 (fp32) inside the fp16 DiT autocast. Upstream path: none."""
    return nullcontext() if prec.upstream else torch.autocast(device_type="cuda", dtype=prec.t5)


def debug_dump(name, obj, idex, gpu):
    """Opt-in debugging (env WAN_DUMP_DIR): save first-clip stage tensors as fp32 CPU files
    (rank-0 view) so two runs can be compared with scripts/compare_dumps.py. No-op otherwise."""
    out = os.environ.get("WAN_DUMP_DIR")
    if not out or idex != 1 or gpu != 0 or obj is None:
        return
    os.makedirs(out, exist_ok=True)
    debug_dump.n = getattr(debug_dump, "n", 0) + 1
    ts = obj.tensors if isinstance(obj, TensorList) else (list(obj) if isinstance(obj, (list, tuple)) else [obj])
    ts = [t.detach().float().cpu() for t in ts]
    t = torch.stack(ts) if len({tuple(x.shape) for x in ts}) == 1 else ts[0]
    torch.save(t, os.path.join(out, f"{debug_dump.n:03d}_{name}.pt"))


def gpu_memory_checkpoint(label, device, enabled=False):
    """Log measured CUDA peaks per phase; the offline estimator is not a GPU measurement."""
    if not enabled:
        return
    torch.cuda.synchronize(device)
    gib = 1024 ** 3
    logger.info(f"[VRAM {device}] {label}: "
                f"allocated={torch.cuda.memory_allocated(device) / gib:.2f} GiB "
                f"peak_allocated={torch.cuda.max_memory_allocated(device) / gib:.2f} GiB "
                f"reserved={torch.cuda.memory_reserved(device) / gib:.2f} GiB "
                f"peak_reserved={torch.cuda.max_memory_reserved(device) / gib:.2f} GiB")
    torch.cuda.reset_peak_memory_stats(device)


def local_weight_allocation(module, device):
    """Count *local* parameter-shard bytes, never all-gather distributed tensors."""
    target = torch.device(device)
    by_dtype = {}
    for param in module.parameters():
        shard = param.to_local() if hasattr(param, "to_local") else param
        if shard.device != target:
            continue
        label = str(shard.dtype).replace("torch.", "")
        by_dtype[label] = by_dtype.get(label, 0) + shard.numel() * shard.element_size()
    return by_dtype


def log_weight_allocation(label, module, device, enabled=False):
    if enabled:
        sizes = local_weight_allocation(module, device)
        gib = 1024 ** 3
        details = ", ".join(f"{dtype}={n / gib:.3f}GiB" for dtype, n in sorted(sizes.items()))
        logger.info(f"[WEIGHTS {device}] {label}: local={sum(sizes.values()) / gib:.3f} GiB ({details})")


T5_HINT = "T5 runs in fp32 on V100 (test_cfg.t5_dtype); a non-finite text embedding means bad inputs/weights."
CLIP_HINT = "set test_cfg.clip_dtype=fp32 (or --cfg-override test_cfg.clip_dtype=fp32)."
VAE_HINT = "the VAE runs in test_cfg.vae_dtype (fp32 on V100); check the input video / weights."


class ForwardStepWrapper(nn.Module):
    def __init__(self, module):
        super(ForwardStepWrapper, self).__init__()
        self.wrapped_module = module

    def forward(self, *args, step=None, counter=None, **kwargs):
        return self.wrapped_module(*args, **kwargs)


def inference_core(item, gpu, cfg, clip, vae, model, t5, use_t5, context, context_null, device,
                   prec=None, guard=None):
    prec = prec or resolve_precision(cfg.test_cfg, torch.device(device))
    guard = guard or OverflowGuard(model, mode="off")
    hints = {}
    height = item.height
    width = item.width
    CLIP_LEN = item.clip_len
    low_vram = bool(cfg.test_cfg.get("low_vram", False))
    log_gpu_memory = bool(cfg.test_cfg.get("log_gpu_memory", False))
    origin_len = CLIP_LEN
    origin_area = [width, height]

    # real_frame_len = item.real_frame_len
    
    first_num = item.first_num
    # first_img = item.first_img
    fps = item.fps
    seed = item.seed
    output_path = item.output_path
    # cond_images = item.cond_images

    prompt = item.prompt
    prompt_ref = item.prompt_ref
    neg_prompt = item.neg_prompt
    sample_guide_scale = item.sample_guide_scale
    step = item.step

    tpl_video_path = item.tpl_video_path
    refer_img_path = item.refer_img_path

    cfg.test_cfg.sample_guide_scale = sample_guide_scale

    setup_seed(seed)
    seed_g = torch.Generator(device=gpu)
    seed_g.manual_seed(seed)
    
    start = 0
    end = CLIP_LEN
    all_out_frames = []
    previous_tail_cpu = None  # Opt-in: retain only overlap frames between chunks on CPU

    logger.info(f"Processing reference image: {refer_img_path}")
    first_img = cv2.imread(refer_img_path)[..., ::-1]
    first_img, pading_info = resize_by_area(first_img, width * height, divisor=16, return_padding_info=True)


    # 读取视频
    video_reader = VideoReader(tpl_video_path)
    frame_num = len(video_reader)
    logger.info('frame_num: {}'.format(frame_num))

    # fps采样，短的首尾循环
    video_fps = video_reader.get_avg_fps()
    logger.info(f"[GPU {gpu}] video fps {video_fps}")

    audio_save_path = os.path.join(output_path, 'tpl_audio.wav')
    ffmpeg_extract_audio_by_fps(tpl_video_path, audio_save_path, fps)
    
    # TODO: Maybe we can switch to PyAV later, which can get accurate frame num
    duration = float(video_reader.get_frame_timestamp(-1)[-1])
    if not (np.isfinite(duration) and 0 < duration < frame_num / video_fps * 10):
        logger.warning(f"[GPU {gpu}] invalid duration {duration} from decord, fallback to ffprobe")
        duration = get_video_duration(tpl_video_path)
        logger.info(f"[GPU {gpu}] duration (ffprobe): {duration}")
    expected_frame_num = int(duration * video_fps + 0.5)
    ratio = abs((frame_num - expected_frame_num) / frame_num)
    if ratio > 0.1:
        logger.warning(f"[GPU {gpu}] Warning: The difference between the actual number of frames and the expected number of frames is too large")
        frame_num = expected_frame_num

    target_num = int(frame_num / video_fps * fps)
    idxs = get_frame_indices(frame_num, video_fps, target_num, fps)

    frames = video_reader.get_batch(idxs).asnumpy()
    cond_images = [resize_by_area(frame, width * height, divisor=16) for frame in frames]
    real_frame_len = len(frames)

    logger.info('real_frame_len: {}'.format(real_frame_len))
    target_len = get_padding_len(real_frame_len, CLIP_LEN, first_num)

    logger.info('target_len: {}'.format(target_len))
    cond_images = zigzag_padding(cond_images, target_len)

    torch.cuda.empty_cache()
    idex = 0
    while True:
        if start + first_num >= len(cond_images):
            break
        
        if start == 0:
            mask_reft_len = 0
        else:
            mask_reft_len = first_num

        if len(cond_images) - start < CLIP_LEN:
            CLIP_LEN = len(cond_images) - start
            logger.info(f"[GPU {gpu}] Start: {start} End: {end} CLIP_LEN: {CLIP_LEN}")
            
        # process the input
        batch = {
            "conditioning_pixel_values": torch.zeros(1, 3, CLIP_LEN, height, width),
            "refer_pixel_values": torch.zeros(1, 3, height, width),
            "refer_t_pixel_values": torch.zeros(first_num, 3, height, width),
        }
       
        batch["conditioning_pixel_values"] = rearrange(
            torch.tensor(np.stack(cond_images[start:end]) / 127.5 - 1),
            "t h w c -> 1 c t h w",
        )

        batch["refer_pixel_values"] = rearrange(
            torch.tensor(first_img / 127.5 - 1), "h w c -> 1 c h w"
        )

        if start > 0:
            tail = previous_tail_cpu if low_vram else out_frames[0, :, -first_num:]
            batch["refer_t_pixel_values"] = rearrange(
                tail.clone().detach(), "c t h w -> t c h w",
            )

        batch['pixel_values'] = rearrange(batch["refer_t_pixel_values"],
                                          "t c h w -> 1 c t h w",
                                          )
        idex += 1
        dist.barrier(device_ids=[gpu])
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device=device, dtype=prec.vae)

        ref_pixel_values = batch["refer_pixel_values"]
        conditioning_pixel_values = batch["conditioning_pixel_values"]

        B, C, H, W = ref_pixel_values.shape

        T = CLIP_LEN + 1
        lat_h = H // 8
        lat_w = W // 8
        lat_t = T // 4 + 1 + 1
        target_shape = [lat_t, lat_h, lat_w]
        grid_sizes = torch.stack([torch.tensor([lat_t, lat_h // 2, lat_w // 2], dtype=torch.long)])

        noise = [
            torch.randn(
                16,
                target_shape[0],
                target_shape[1],
                target_shape[2],
                dtype=torch.float32,
                device=device,
                generator=seed_g,
            )
        ]

        torch.cuda.empty_cache()

        with (
            torch.autocast(device_type=str(device), dtype=prec.dit, enabled=True),
            torch.no_grad()
        ):  
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=cfg.test_cfg.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sampling_sigmas = get_sampling_sigmas(step, cfg.test_cfg.sample_shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler, device=device, sigmas=sampling_sigmas
            )
            latents = noise

            ref_pixel_values = rearrange(ref_pixel_values, "t c h w -> 1 c t h w")
            ref_latents = vae_encode(vae, ref_pixel_values.to(prec.vae), prec)
            ref_latents = torch.stack(ref_latents)
            mask_ref = get_i2v_mask(1, lat_h, lat_w, 1, device=device)
            y_ref = torch.concat([mask_ref, ref_latents[0]]).to(dtype=prec.dit, device=device)

            img = ref_pixel_values[0, :, 0]
            guard.begin_component()
            clip_context = clip.visual([img[:, None, :, :]]).to(dtype=prec.dit, device=device)
            debug_dump("clip_context", clip_context, idex, gpu)
            clip_attn_ref = guard.attention_flag(device)

            msk_reft = get_i2v_mask(lat_t - 1, lat_h, lat_w, mask_reft_len, device=device)

            if mask_reft_len > 0:
                y_reft = vae_encode(vae,
                    [
                        torch.concat(
                            [
                                torch.nn.functional.interpolate(batch["pixel_values"][0, :, :mask_reft_len].cpu(),
                                                                size=(H, W), mode="bicubic"),
                                torch.zeros(3, T - mask_reft_len - 1, H, W),
                            ],
                            dim=1,
                        ).to(device)
                    ], prec
                )[0]
            else:
                y_reft = vae_encode(vae,
                    [
                        torch.concat(
                            [
                                torch.zeros(3, T - mask_reft_len - 1, H, W),
                            ],
                            dim=1,
                        ).to(device)
                    ], prec
                )[0]

            y_reft = torch.concat([msk_reft, y_reft]).to(dtype=prec.dit, device=device)
            y = torch.concat([y_ref, y_reft], dim=1)
            debug_dump("y", y, idex, gpu)

            condition_latents = vae_encode(vae, conditioning_pixel_values.to(prec.vae), prec)
            condition_latents = torch.stack(condition_latents)
            condition_latents = torch.cat([condition_latents], dim=2)
            debug_dump("condition_latents", condition_latents, idex, gpu)

            B, C, T, H, W = conditioning_pixel_values.shape
            B, C, lat_t, lat_h, lat_w = condition_latents.shape
            grid_sizes_ref = torch.stack([torch.tensor([lat_t, lat_h // 2, lat_w // 2], dtype=torch.long)])

            condition_img =  batch["conditioning_pixel_values"][0, :, 0]
            guard.begin_component()
            condition_clip_context = clip.visual([condition_img[:, None, :, :]]).to(dtype=prec.dit, device=device)
            debug_dump("condition_clip_context", condition_clip_context, idex, gpu)
            clip_attn_cond = guard.attention_flag(device)
            
            mask_len = T
            condition_y = vae_encode(vae,
                [
                    torch.concat(
                        [
                            torch.nn.functional.interpolate(batch["conditioning_pixel_values"][0, :, :mask_len].cpu(),
                                                            size=(H, W), mode="bicubic"),
                            torch.zeros(3, T - mask_len, H, W),
                        ],
                        dim=1,
                    ).to(device)
                ], prec
            )[0]

            condition_msk_y = get_i2v_mask(lat_t, lat_h, lat_w, mask_len, device=device)
            condition_y = torch.concat([condition_msk_y, condition_y]).to(dtype=prec.dit, device=device)
            debug_dump("condition_y", condition_y, idex, gpu)

            # V100 port: one collective inf/NaN check of every conditioning tensor (CLIP checked
            # including its attention kernels); all ranks raise the same error
            guard.assert_finite("conditioning", {
                "VAE latents (reference image)": y_ref, "VAE latents (continuation frames)": y_reft,
                "VAE latents (driving video)": condition_latents, "VAE latents (driving mask)": condition_y,
                "CLIP visual (reference image)": [clip_context, clip_attn_ref],
                "CLIP visual (first driving frame)": [condition_clip_context, clip_attn_cond],
            }, hints={"VAE latents (reference image)": VAE_HINT, "VAE latents (continuation frames)": VAE_HINT,
                      "VAE latents (driving video)": VAE_HINT, "VAE latents (driving mask)": VAE_HINT,
                      "CLIP visual (reference image)": CLIP_HINT, "CLIP visual (first driving frame)": CLIP_HINT})
            if not prec.upstream:  # V100 port: pixel tensors are not needed again in this clip
                del batch, ref_pixel_values, conditioning_pixel_values, img, condition_img

            if use_t5:
                logger.info('prompt:{}'.format(prompt))
                with t5_autocast(prec):  # fp32 on V100; upstream path: no extra context
                    context = t5(prompt)[0]
                    context_ref = t5(prompt_ref)[0]
                debug_dump("t5_context", context, idex, gpu)
                debug_dump("t5_context_ref", context_ref, idex, gpu)
                guard.assert_finite("T5", {"T5 context (prompt)": context, "T5 context (prompt_ref)": context_ref},
                                    hints={"T5 context (prompt)": T5_HINT, "T5 context (prompt_ref)": T5_HINT})

            max_seq_len = int(math.ceil(np.prod(target_shape) // 4 / cfg.test_cfg.sp_size)) * cfg.test_cfg.sp_size
            target_ref_shape = condition_latents.shape[2:]
            max_seq_len_ref = int(math.ceil(np.prod(target_ref_shape) // 4 / cfg.test_cfg.sp_size)) * cfg.test_cfg.sp_size

            arg_c = {
                    "context":  [context],
                    "seq_len": max_seq_len,
                    "clip_fea": clip_context,
                    "y": [y],
                    "origin_len": origin_len, 
                    "origin_area": origin_area,
                }
            
            arg_ref_c = {
                    "context_ref": [context_ref],
                    "seq_len_ref": max_seq_len_ref,
                    "clip_fea_ref": condition_clip_context,
                    "y_ref": [condition_y]
                }

            if cfg.test_cfg.sample_guide_scale > 1:
                logger.info('sample_guide_scale: {}'.format(cfg.test_cfg.sample_guide_scale))

                if use_t5:
                    with t5_autocast(prec):
                        context_null = t5(neg_prompt)[0]
                    guard.assert_finite("T5", {"T5 context (negative prompt)": context_null},
                                        hints={"T5 context (negative prompt)": T5_HINT})
                
                arg_null = {
                        "context": [context_null],
                        "seq_len": max_seq_len,
                        "clip_fea": clip_context,
                        "y": [y],
                        "origin_len": origin_len, 
                        "origin_area": origin_area,
                        "is_uncondtion": True
                    }

            if max_seq_len % cfg.test_cfg.sp_size != 0:
                raise ValueError(f"max_seq_len {max_seq_len} is not divisible by sp_size {cfg.test_cfg.sp_size}")

            gpu_memory_checkpoint(f"chunk {idex} conditioning", device, log_gpu_memory)
            # cache ref kv
            k_cache = {}
            v_cache = {}
            t = timesteps[0]
            timestep = [t]
            timestep = torch.stack(timestep)
            guard.run(lambda: model(condition_latents, grid_sizes=grid_sizes, k_cache=k_cache, v_cache=v_cache, t=timestep,  method='forward_ref', **arg_ref_c),
                      step="ref", branch="ref")
            for _layer in (0, 20, 39):
                if _layer in k_cache:
                    debug_dump(f"ref_k_layer{_layer:02d}", k_cache[_layer], idex, gpu)
                    debug_dump(f"ref_v_layer{_layer:02d}", v_cache[_layer], idex, gpu)
            gpu_memory_checkpoint(f"chunk {idex} reference", device, log_gpu_memory)
            for i, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                noise_pred_cond = TensorList(guard.run(
                    lambda: model(TensorList(latent_model_input), k_cache=k_cache, v_cache=v_cache, t=timestep, grid_sizes_ref=grid_sizes_ref, method='forward_gen', **arg_c),
                    step=i, branch="cond")
                )

                if cfg.test_cfg.sample_guide_scale > 1:
                    noise_pred_uncond = TensorList(guard.run(
                        lambda: model(
                            TensorList(latent_model_input), k_cache=k_cache, v_cache=v_cache, t=timestep, grid_sizes_ref=grid_sizes_ref, method='forward_gen', **arg_null
                        ), step=i, branch="uncond")
                    )
                    noise_pred = noise_pred_uncond + cfg.test_cfg.sample_guide_scale * (
                        noise_pred_cond - noise_pred_uncond
                    )
                else:
                    noise_pred = noise_pred_cond
                debug_dump(f"step{i}_noise_pred_cond", noise_pred_cond, idex, gpu)
                debug_dump(f"step{i}_noise_pred", noise_pred, idex, gpu)

                temp_x0 = sample_scheduler.step(
                    noise_pred[0].unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g,
                )[0]
                latents[0] = temp_x0.squeeze(0)
                debug_dump(f"step{i}_latents", latents, idex, gpu)

            gpu_memory_checkpoint(f"chunk {idex} denoising", device, log_gpu_memory)
            x0 = latents
            # 将latents转换为float32进行解码
            x0 = [x.to(dtype=torch.float32) for x in x0]
            debug_dump("x0", x0, idex, gpu)
            if not prec.upstream:  # V100 port: free the reference K/V before VAE decode
                k_cache.clear(); v_cache.clear()
            torch.cuda.empty_cache()
            out_frames = torch.stack(vae_decode(vae, [x0[0][:, 1:]], prec))
            if not prec.upstream:
                guard.assert_finite("VAE decode", {"VAE decoded frames (before clamp)": out_frames},
                                    hints={"VAE decoded frames (before clamp)": VAE_HINT})
                out_frames = out_frames.clamp_(-1, 1)

            debug_dump("decoded_frames", out_frames, idex, gpu)
            out = (
                rearrange(((out_frames + 1) * 127.5), "1 c t h w -> t h w c")
                .detach()
                .cpu()
                .float()
                .numpy()
                .astype(np.uint8)
            )
            gpu_memory_checkpoint(f"chunk {idex} decoded", device, log_gpu_memory)
            if low_vram:
                previous_tail_cpu = out_frames[0, :, -first_num:].detach().cpu().clone()
                del out_frames

            if start != 0:
                out = out[first_num:]

            all_out_frames.extend([*out])
            start += CLIP_LEN - first_num
            end += CLIP_LEN - first_num
            torch.cuda.empty_cache()
    
    if gpu == 0:
        os.makedirs(os.path.split(output_path)[0], exist_ok=True)
        all_out_frames_array = np.stack(all_out_frames, axis=0)
        all_out_frames_array = all_out_frames_array[:real_frame_len]

        padding_type, padding, side_long = pading_info['padding_type'], pading_info['padding'], pading_info['side_long']
        if padding_type == 'width':
            all_out_frames_array = all_out_frames_array[:, :, padding:padding+side_long, :]
        else:
            all_out_frames_array = all_out_frames_array[:, padding:padding+side_long, :, :]

        video_path = os.path.join(output_path, 'results.mp4')
        mpy.ImageSequenceClip([*all_out_frames_array], fps=fps).write_videofile(
            video_path
        )
        if os.path.exists(audio_save_path):
            merge_video_audio(video_path, audio_save_path)

        res = {'result': video_path, 'status': 'success'}
        return res
    
    return None
        

def prepare_clip(clip, prec, log=None):
    """V100 port: CLIP was strictly loaded in fp32 (checkpoint values unrounded). Check that the
    visual tower survives the cast to prec.clip, then cast it (and the unused text tower)."""
    if prec.clip == torch.float16:
        check_fp16_castable_cached(clip.model.visual.named_parameters, [getattr(clip, "checkpoint_path", None)],
                                   underflow_frac=prec.underflow_frac, log=log, what="CLIP visual",
                                   hint=CLIP_HINT)
    clip.model.visual = clip.model.visual.to(prec.clip)
    clip.model.textual = clip.model.textual.to(prec.clip)
    clip.dtype = prec.clip  # CLIP.visual() autocasts to clip.dtype
    return clip


def build_dit(cfg, device, prec, gpu):
    """Build, FSDP2-shard and load the DiT exactly as upstream's worker did (strict, broadcast from
    rank 0). V100 port additions: compute/output dtypes from ``prec``, optional fp32 allow-list
    (test_cfg.fp32_modules) and a load-time fp16 castability check. Returns (model, fp32_modules)."""
    device_type = torch.device(device).type
    fp32_modules = [str(m) for m in (cfg.test_cfg.get("fp32_modules", None) or [])]
    # FSDP2 stores its original parameter dtype in persistent shards. Keep the 32GB
    # default (fp32) unchanged; opt in to checkpoint-exact bf16 storage on 16GB cards.
    # Compute remains prec.dit (guarded fp16 on V100), not bf16 emulation.
    shard_dtype = parse_dtype(cfg.test_cfg.get("dit_shard_dtype", "auto")) or torch.float32
    if shard_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("test_cfg.dit_shard_dtype must be auto, fp32 or bf16 (never fp16)")
    try:
        with torch.device('meta'):
            model = build_object_from_dict(cfg.model.transformer)
            if shard_dtype == torch.bfloat16:
                model = model.to(dtype=torch.bfloat16)
                # Explicit fp32 fallback blocks must retain fp32 parameters.
                for name in fp32_modules:
                    model.get_submodule(name).to(dtype=torch.float32)
            torch.set_default_dtype(prec.dit)
    except Exception as e:
        logger.error(f"[GPU {gpu}] Failed to initialize transformer: {str(e)}")
        logger.error(f"[GPU {gpu}] Error traceback: {traceback.format_exc()}")
        raise
    
    cfg.test_cfg.sharding_size = min(cfg.test_cfg.sharding_size, cfg.test_cfg.world_size)
    # prepare fsdp args
    fsdp_mesh = init_device_mesh(
            device_type=device_type, 
            mesh_shape=(dist.get_world_size() // cfg.test_cfg.sharding_size, cfg.test_cfg.sharding_size),
            mesh_dim_names=('replicate', 'shard')
        )

    def mp_policy(dtype):
        return MixedPrecisionPolicy(
            param_dtype=dtype,
            reduce_dtype=dtype,
            output_dtype=prec.dit_output,
            cast_forward_inputs=False
        )

    fsdp_kwargs = dict(
            mesh=fsdp_mesh,
            reshard_after_forward=True,
            shard_placement_fn=None,
            mp_policy=mp_policy(prec.dit),
            offload_policy=None,
            ignored_params=None
        )

    for name in fp32_modules:  # explicit allow-list: own FSDP unit, fp32 params, fp32 compute
        fully_shard(model.get_submodule(name), **dict(fsdp_kwargs, mp_policy=mp_policy(torch.float32)))
    for block in model.blocks:
            if not isinstance(block, FSDPModule):
                fully_shard(block, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)
    
    model = model.to_empty(device=device)
    model = model.eval().requires_grad_(False)

    # [model] load checkpoint
    if gpu == 0:
        model_path = cfg.get("model_path", False)
        if model_path:
            sd = load_safetensors(model_path)
            logger.info('gpu: {} model_path: {}'.format(gpu, model_path))
    else:
        sd = {}

    dist.barrier(device_ids=[gpu]) if device_type == "cuda" else dist.barrier()
    set_model_state_dict(
        model=model,
        model_state_dict=sd,
        options=StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
            broadcast_from_rank0=True,
            strict=True
        )
    )
    del sd

    # dtype conversion happens after loading (FSDP2 casts the loaded fp32 shards per forward);
    # verify that cast is lossless enough before running anything in fp16.
    if prec.dit == torch.float16:
        check_fp16_castable_cached(model.named_parameters, [cfg.get("model_path", None)], fp32_modules=fp32_modules,
                                   underflow_frac=prec.underflow_frac,
                                   group=fsdp_mesh.get_group("shard"), log=logger, what="DiT")
    for name in fp32_modules:
        force_fp32(model.get_submodule(name), device_type=device_type)
    return model, fp32_modules


def worker(gpu, cfg, in_q_list, out_q, initialized_events, status_q=None):
    """status_q (V100 port): ('init_ok' | 'init_fail' | 'fail', gpu, message) and ('done', gpu,
    request_id) after every completed request, so that the parent never waits forever on a failed
    worker nor returns a result before every worker finished (see WorkerMonitor)."""
    event = initialized_events[gpu]
    in_q = in_q_list[gpu]
    out_q = out_q
    initialized = failed = False

    def report(kind, msg=None):
        if status_q is not None:
            try:
                status_q.put((kind, gpu, msg))
            except Exception as put_e:  # the parent may already be gone
                logger.warning(f"[GPU {gpu}] could not report {kind}: {put_e}")

    cfg.gpu = gpu
    cfg.rank = cfg.pmi_rank * cfg.test_cfg.gpu_infer_per_machine + gpu

    init_device = "cpu"
    device = f'cuda:{gpu}'
    vae = None
    model = None
    context = None
    context_null = None

    try:
        torch.cuda.set_device(gpu)
        logger.info(f"[GPU {gpu}] Initializing process group with rank {cfg.rank}, world_size {cfg.world_size}")
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=cfg.rank,
            world_size=cfg.world_size,
            timeout=timedelta(seconds=int(os.getenv("NCCL_INIT_TIMEOUT_SEC", "3600"))),
        )

        if cfg.test_cfg.sp_size > cfg.test_cfg.world_size:
            logger.warning(f"[GPU {gpu}] Reset sp_size to world_size ({cfg.test_cfg.world_size})")
            cfg.test_cfg.sp_size = min(cfg.test_cfg.sp_size, cfg.test_cfg.world_size)

        if cfg.test_cfg.sp_size > 1:
            cfg.model.transformer.use_context_parallel = True
            ops.init_model_parallel_groups(cfg.test_cfg.sp_size)
            assert cfg.test_cfg.sp_size == ops.get_tensor_parallel_world_size()
            logger.info(f"[GPU {gpu}] Tensor parallel world size verified: {ops.get_tensor_parallel_world_size()}")
    
        # V100 port: capability switch (sm>=80 + 'auto' -> upstream bf16/flash recipe)
        prec = resolve_precision(cfg.test_cfg, torch.device(device))
        wan_attention.set_attention_backend(
            cfg.test_cfg.get("attn_backend", None) or os.environ.get("WAN_ATTN_BACKEND", "auto"))
        wan_attention.set_compute_dtype(prec.dit)
        if gpu == 0:
            logger.info(f"GPU capability sm{''.join(map(str, torch.cuda.get_device_capability(gpu)))}; "
                        f"precision: {prec.describe()}; attention backend: "
                        f"{wan_attention.attention_backend(torch.device(device))}")

        if gpu == 0:
            logger.info(f"Sharding strategy: {cfg.test_cfg.sharding_strategy}")
            logger.info(f"Param dtype: {cfg.test_cfg.param_dtype}")
            logger.info(f"Reduce dtype: {cfg.test_cfg.reduce_dtype}")
            logger.info(f"Buffer dtype: {cfg.test_cfg.buffer_dtype}")

        # [model] sharding function
        sharding_strategy = get_sharding_strategy(cfg.test_cfg.sharding_strategy.lower())
        
        # Convert dtype strings to torch.dtype
        param_dtype = get_dtype(cfg.test_cfg.param_dtype)
        reduce_dtype = get_dtype(cfg.test_cfg.reduce_dtype)
        buffer_dtype = get_dtype(cfg.test_cfg.buffer_dtype)
        
        shard_fn = partial(
            shard_model,
            device_id=gpu,
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype,
            sharding_strategy=sharding_strategy
        )
        
        t5 = None
        if cfg.model.use_t5:
            with torch.device(init_device):
                t5 = T5Encoder(
                    name=cfg.model.t5_model,
                    text_len=cfg.model.text_len,
                    dtype=torch.bfloat16,
                    device=init_device,
                    checkpoint_path=cfg.model.t5_checkpoint,
                    tokenizer_path=cfg.model.t5_tokenizer,
                )
            # V100 port: compute in prec.t5 (fp32) with the bf16 weights as stored; the token embedding
            # and final norm get their own FSDP units so nothing stays unsharded after encoding
            # (FSDP1 keeps the root unit's unsharded params: 4.2 GB in fp32)
            t5.model = shard_fn(t5.model, sync_module_states=False, param_dtype=prec.t5,
                                extra_wrap=() if prec.upstream else (t5.model.token_embedding, t5.model.norm))
            log_weight_allocation("T5 FSDP1", t5.model, device,
                                  bool(cfg.test_cfg.get("log_gpu_memory", False)))
            gpu_memory_checkpoint("T5 initialization", device,
                                  bool(cfg.test_cfg.get("log_gpu_memory", False)))

        try:
            with torch.device(init_device):
                clip = CLIP(
                    name=cfg.model.clip_model,
                    # V100 port: load the checkpoint values unrounded, check, then cast (prepare_clip)
                    dtype=prec.clip if prec.upstream else torch.float32,
                    device=init_device,
                    checkpoint_path=cfg.model.clip_checkpoint,
                    tokenizer_path=cfg.model.clip_tokenizer,
                )
            logger.info(f" --> CLIP: {torch.cuda.memory_allocated()/1e9:.2f} GB, {torch.cuda.memory_reserved()/1e9:.2f} GB")

            
            if not prec.upstream:
                prepare_clip(clip, prec, log=logger)
            clip.model.visual = shard_transformer(
                clip.model.visual, 
                {AttentionBlock}, 
                device_id=gpu,
                param_dtype=prec.clip,
                reduce_dtype=prec.clip,
                buffer_dtype=prec.clip,
                sharding_strategy=sharding_strategy,
            )
            if prec.clip_text_on_gpu:  # unused by inference; 'auto' keeps it on CPU for an fp16 DiT
                clip.model.textual = clip.model.textual.to(device)
            
            logger.info(f" --> CLIP FSDP: {torch.cuda.memory_allocated()/1e9:.2f} GB, {torch.cuda.memory_reserved()/1e9:.2f} GB")
            log_weight_allocation("CLIP visual FSDP1", clip.model.visual, device,
                                  bool(cfg.test_cfg.get("log_gpu_memory", False)))
            gpu_memory_checkpoint("CLIP initialization", device,
                                  bool(cfg.test_cfg.get("log_gpu_memory", False)))
        except Exception as e:
            logger.error(f"[GPU {gpu}] Failed to initialize VAE: {str(e)}")
            logger.error(f"[GPU {gpu}] Error traceback: {traceback.format_exc()}")
            raise

        try:
            with torch.device(init_device):
                vae = VideoVAE(
                    vae_pth=cfg.model.vae_checkpoint,
                    device=init_device,
                    dtype=torch.bfloat16
                )
            if cfg.test_cfg.get("low_vram", False) and not prec.upstream:
                # 16GB path: finish the BF16 -> FP32 VAE cast on CPU BEFORE GPU
                # transfer. Avoid temporarily holding both GPU copies on every rank.
                # The source checkpoint values and parameter names stay unchanged.
                vae.model = vae.model.to(dtype=prec.vae)
                vae.model = vae.model.to(device)
                vae.scale = [x.to(device=device, dtype=prec.vae) for x in vae.scale]
            else:
                vae.model = vae.model.to(device)
                vae.scale = [vae.scale[0].to(device), vae.scale[1].to(device)]
                if not prec.upstream:  # original 32GB recipe unchanged
                    vae.model = vae.model.to(dtype=prec.vae)
                    vae.scale = [x.to(dtype=prec.vae) for x in vae.scale]
            logger.info(f" --> VAE: {torch.cuda.memory_allocated()/1e9:.2f} GB, {torch.cuda.memory_reserved()/1e9:.2f} GB")
            log_weight_allocation("VAE replicated", vae.model, device,
                                  bool(cfg.test_cfg.get("log_gpu_memory", False)))
            gpu_memory_checkpoint("VAE initialization", device,
                                  bool(cfg.test_cfg.get("log_gpu_memory", False)))
        except Exception as e:
            logger.error(f"[GPU {gpu}] Failed to initialize VAE: {str(e)}")
            logger.error(f"[GPU {gpu}] Error traceback: {traceback.format_exc()}")
            raise

        # [model] transformer (V100 port: moved into build_dit, statements unchanged)
        model, fp32_modules = build_dit(cfg, device, prec, gpu)
        log_weight_allocation("DiT FSDP2", model, device,
                              bool(cfg.test_cfg.get("log_gpu_memory", False)))
        gpu_memory_checkpoint("DiT initialization", device,
                              bool(cfg.test_cfg.get("log_gpu_memory", False)))
        guard = OverflowGuard(
            model,
            mode=prec.guard,
            action=cfg.test_cfg.get("overflow_action", "fp32"),
            warn_at=float(cfg.test_cfg.get("overflow_warn_at", 2.0 ** 14)),
            fp32_modules=fp32_modules,
            log=logger,
        )

        torch.cuda.empty_cache()
        gpu_memory_checkpoint("worker initialization", device,
                              bool(cfg.test_cfg.get("log_gpu_memory", False)))

        # init done
        logger.info(f'[GPU {gpu}] Worker initialization completed')
        initialized = True
        report("init_ok")
        event.set()

        # start inference
        logger.info(f"[GPU {gpu}] Starting inference loop")
        while True:
            try:
                # receive data from queue
                item = in_q.get()
                if item.get("overwrite_cfg", None):
                    cfg = merge_configs(cfg, item["overwrite_cfg"])
                res = inference_core(item, gpu, cfg, clip, vae, model, t5, cfg.model.use_t5, context, context_null, device,
                                     prec=prec, guard=guard)
                
                if res is not None:
                    out_q.put(res)
                report("done", item.get("request_id"))  # V100 port: per-request completion ack
                
            except Exception as e:
                logger.error(f"[GPU {gpu}] Error during inference: {str(e)}")
                logger.error(f"[GPU {gpu}] Error traceback: {traceback.format_exc()}")
                failed = True
                report("fail", f"{type(e).__name__}: {e}")
                if gpu == 0:
                    res = {'result': str(e), 'status': 'fail'}
                    out_q.put(res)
                break

    except Exception as e:
        logger.error(f"[GPU {gpu}] Fatal error: {str(e)}")
        logger.error(f"[GPU {gpu}] Error traceback: {traceback.format_exc()}")
        failed = True
        report("fail" if initialized else "init_fail", f"{type(e).__name__}: {e}")
        if gpu == 0:
            res = {'result': str(e), 'status': 'fail'}
            out_q.put(res)

    finally:
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as sync_e:
            logger.warning(f"[GPU {gpu}] cuda.synchronize failed: {sync_e}")

        if dist.is_available() and dist.is_initialized():
            # V100 port: after a failure, peers may be stuck elsewhere (another phase or collective);
            # do not wait for them - the parent terminates the remaining workers.
            if not failed:
                try:
                    logger.info(f"[GPU {gpu}] Waiting at barrier before cleanup")
                    dist.barrier()
                except Exception as barrier_e:
                    logger.warning(f"[GPU {gpu}] barrier failed: {barrier_e}")

            try:
                logger.info(f"[GPU {gpu}] Destroying process group")
                dist.destroy_process_group()
            except Exception as destroy_e:
                logger.warning(f"[GPU {gpu}] destroy_process_group failed: {destroy_e}")
        else:
            logger.info(f"[GPU {gpu}] Process group not initialized, skip barrier/destroy")
        try:
            torch.cuda.empty_cache()
        except Exception as cache_e:
            logger.warning(f"[GPU {gpu}] empty_cache failed: {cache_e}")

        logger.info(f"[GPU {gpu}] Worker cleanup completed")

def main(cfg):
    if int(os.environ.get("LOCAL_WORLD_SIZE", "1")) > 1:  # V100 port: guard against torchrun --nproc_per_node>1
        raise RuntimeError("This pipeline spawns one worker per visible GPU itself; launch ONE process per "
                           "node (python ... or torchrun --nproc_per_node=1), not one per GPU.")
    cfg.pmi_rank = int(os.environ["RANK"])
    cfg.pmi_world_size = int(os.environ["WORLD_SIZE"])
    cfg.test_cfg.gpu_infer_per_machine = torch.cuda.device_count()
    cfg.world_size = cfg.pmi_world_size * cfg.test_cfg.gpu_infer_per_machine
    cfg.test_cfg.world_size = cfg.world_size
    
    in_q_list = [
        torch.multiprocessing.Manager().Queue()
        for _ in range(cfg.test_cfg.gpu_infer_per_machine)
    ]
    out_q = torch.multiprocessing.Manager().Queue()
    initialized_events = [
        torch.multiprocessing.Manager().Event()
        for _ in range(cfg.test_cfg.gpu_infer_per_machine)
    ]

    status_q = torch.multiprocessing.Manager().Queue()

    context = mp.spawn(
        worker,
        nprocs=cfg.test_cfg.gpu_infer_per_machine,
        args=(cfg, in_q_list, out_q, initialized_events, status_q),
        join=False,
    )

    # V100 port: instead of polling only the init events (a failed worker never sets its event and
    # upstream waited forever), watch explicit init reports AND the worker processes.
    monitor = WorkerMonitor(context, status_q, cfg.test_cfg.gpu_infer_per_machine)
    monitor.wait_init(timeout=float(os.getenv("WAN_INIT_TIMEOUT_SEC", "0")) or None)
    logger.info("Inference model is initialized", flush=True)
    inference_pids = context.pids()

    return in_q_list, out_q, inference_pids, context, monitor


def _drain(status_q):
    msgs = []
    while True:
        try:
            msgs.append(status_q.get_nowait())
        except (queue.Empty, EOFError, OSError):
            return msgs


def _shutdown_workers(context, grace=10.0):
    for p in context.processes:
        if p.is_alive():
            p.terminate()
    deadline = time.time() + grace
    for p in context.processes:
        p.join(timeout=max(0.0, deadline - time.time()))
        if p.is_alive():
            p.kill()
            p.join(timeout=5)


class WorkerMonitor:
    """V100 port: parent-side view of the spawned workers.

    Every status report is LATCHED: once any worker reported a failure (or died), no request can
    succeed any more. A request succeeds only when rank 0's result has arrived AND every worker has
    acknowledged completing that request ('done', request id) AND all workers are still alive."""

    def __init__(self, context, status_q, n):
        self.context, self.status_q, self.n = context, status_q, n
        self.failures = []   # latched (kind, gpu, message)
        self.ready = set()   # gpus that finished initialisation
        self.done = {}       # request id -> gpus that completed it

    def poll(self, timeout=0.0):
        msgs = []
        if timeout:
            try:
                msgs.append(self.status_q.get(timeout=timeout))
            except queue.Empty:
                pass
        msgs += _drain(self.status_q)
        for kind, gpu, msg in msgs:
            if kind == "init_ok":
                self.ready.add(gpu)
            elif kind == "done":
                self.done.setdefault(msg, set()).add(gpu)
            elif kind in ("init_fail", "fail"):
                self.failures.append((kind, gpu, msg))

    def dead(self):
        return [(i, p.exitcode) for i, p in enumerate(self.context.processes) if not p.is_alive()]

    def check(self, what, grace=0.5):
        """Raise (after stopping every worker) if a failure was ever reported or a worker died."""
        if not self.failures and self.dead():
            self.poll(grace)  # a dying worker's report may still be in flight
        if self.failures or self.dead():
            raise self.error(what)

    def error(self, what):
        errors = [f"worker {gpu}: {msg}" for _, gpu, msg in self.failures]
        dead = self.dead()
        _shutdown_workers(self.context)
        if dead:
            errors.append(("worker(s) exited without a report" if not errors else "dead workers")
                          + f" (index, exit code): {dead}")
        return RuntimeError(f"Wan-Animate-2 {what} failed: " + "; ".join(errors))

    def wait_init(self, timeout=None, poll=0.5):
        start = time.time()
        while len(self.ready) < self.n:
            self.poll(poll)
            self.check("initialisation")
            if timeout and time.time() - start > timeout:
                _shutdown_workers(self.context)
                raise TimeoutError(f"Wan-Animate-2 initialisation did not finish within {timeout:.0f}s "
                                   f"({len(self.ready)}/{self.n} workers ready)")

    def wait_request(self, out_q, request_id, timeout=None, poll=0.5):
        """Return rank 0's result once every worker acked ``request_id``; raise on any failure."""
        result, start = None, time.time()
        while True:
            self.poll(poll if result is not None else 0.0)
            self.check("inference")
            if result is None:
                try:
                    result = out_q.get(timeout=poll)
                except queue.Empty:
                    result = None
                if result is not None and result.get("status") == "fail":
                    self.failures.append(("fail", 0, result.get("result")))
                    raise self.error("inference")
            elif self.done.get(request_id, set()) >= set(range(self.n)):
                self.poll(0.0)
                self.check("inference")  # latched failures + liveness, right before accepting
                return result
            if timeout and time.time() - start > timeout:
                _shutdown_workers(self.context)
                raise TimeoutError(f"Wan-Animate-2 request {request_id} did not finish within {timeout:.0f}s "
                                   f"(result={'yes' if result is not None else 'no'}, acks="
                                   f"{sorted(self.done.get(request_id, set()))})")


class WanAnimate2MPIPipeline(BaseInferencePipeline):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.setup()

    def setup(self):
        if os.getenv("MASTER_ADDR") is None:
            os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDRESS", "localhost")
            os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12345")
            os.environ["RANK"] = os.getenv("rank_id", "0")
            os.environ["WORLD_SIZE"] = os.getenv("WORLD_SIZE", "1")

        self.in_q_list, self.out_q, self.inference_pids, self._context, self._monitor = main(self.hparams)
        self._request_id = 0
    
    def __call__(
        self, refer_img_path, tpl_video_path, output_path, width=720, height=1280, fps=24,
        seed=-1, clip_len=81, sample_guide_scale=1.0, step=10,
        prompt=['视频中的人在做动作， 背景静止'],
        prompt_ref=['视频中的人在做动作， 背景静止'],
        neg_prompt=['过曝，静态，细节模糊不清，字幕，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，三条腿']
    ):
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        logger.info(f"seed: {seed}")

        first_num = 1
        item = EasyDict(
            {
                "height": height,
                "width": width,
                "clip_len": clip_len,
                "refer_len": clip_len,
                "first_num": first_num,
                "refer_img_path": refer_img_path,
                "tpl_video_path": tpl_video_path,
                "fps": fps,
                "seed": seed,
                "output_path": output_path,
                "sample_guide_scale": sample_guide_scale,
                "step": step,
                "prompt": prompt,
                "prompt_ref": prompt_ref,
                "neg_prompt": neg_prompt
            }
        )

        logger.info(f"Processing model inference")

        # V100 port: per-request id + completion acks from ALL workers; any failure, ever, or a dead
        # worker raises here (workers stopped) instead of hanging or returning a partial success.
        self._monitor.check("inference")
        self._request_id += 1
        item["request_id"] = self._request_id
        for in_q in self.in_q_list:
            in_q.put((item))

        result = self._monitor.wait_request(
            self.out_q, self._request_id, timeout=float(os.getenv("WAN_REQUEST_TIMEOUT_SEC", "0")) or None)
        return result['result']

    def close(self):
        """V100 port: stop the (non-daemon) workers; without this the interpreter waits for them
        forever at exit."""
        context = getattr(self, "_context", None)
        if context is not None:
            _shutdown_workers(context)
            self._context = None