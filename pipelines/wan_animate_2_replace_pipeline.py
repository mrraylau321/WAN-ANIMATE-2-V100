"""EXPERIMENTAL, UNOFFICIAL replace mode for Wan-Animate-2 (a hypothesis, not a released feature; no
fine-tuning): put the reference character into the driving video's own scene. See README, "Experimental:
replace mode". The original pipeline file is not modified; this module reuses its worker and copies its
inference_core (pipelines/wan_animate_2_pipeline.py) with three marked changes (search "REPLACE MODE"),
so keep the copy in step if that function changes:

  1. one person mask per driving frame (scripts/make_person_mask.py), dilated by test_cfg.replace_dilate px
     (or its per-frame bounding box with test_cfg.replace_mask_mode=bbox);
  2. the continuation conditioning y carries the driving frames with the person region filled with the
     surrounding background colours (test_cfg.replace_fill=blur|grey|black), and its mask channels become
     per-pixel: 1 = keep (background, previous-clip frames), 0 = generate (person region, grown by
     test_cfg.replace_grow latent cells).
     The public weights were trained with whole-frame masks only, so this input is out of distribution;
  3. the generated frames are pasted back onto the original background with a feathered mask
     (results.mp4); the unblended model output is saved as results_raw.mp4.

The driving branch (motion source), the reference image slot, CLIP and the prompts are unchanged.
Optional retarget (scripts/retarget_driving.py): the driving video fed to the model is a copy with the
person rescaled; test_cfg.replace_background_video then names the ORIGINAL video, which supplies the
background in (2) and the paste-back in (3).
Select it with:  --cfg-override type=pipelines.wan_animate_2_replace_pipeline.WanAnimate2ReplacePipeline
                 test_cfg.replace_mask=<masks.npz or mask video>   (see scripts/run_v100_replace.sh)
"""
import numpy as np
import torch
import torch.nn.functional as F

import pipelines.wan_animate_2_pipeline as base
from pipelines.wan_animate_2_pipeline import *  # noqa: F401,F403  (names used by the copied inference core)
from wanxiang.eval_i2v import get_i2vmask_mask


def load_person_masks(cfg, idxs, area, target_len):
    """Masks for the sampled driving frames (idxs), resized/padded like the frames, then dilated.
    Returns target_len uint8 arrays (1 = person / region to generate), zigzag-padded like the frames."""
    path = cfg.test_cfg.get("replace_mask", None)
    if not path:
        raise ValueError("replace mode needs test_cfg.replace_mask (run scripts/make_person_mask.py)")
    if str(path).endswith(".npz"):
        masks = np.load(path)["masks"]
    else:  # a mask video, white = person
        reader = VideoReader(path)
        masks = reader.get_batch(list(range(len(reader)))).asnumpy()[..., 0]
    if max(idxs) >= len(masks):
        raise ValueError(f"{path} has {len(masks)} masks but frame {max(idxs)} of the driving video is used")
    dilate = int(cfg.test_cfg.get("replace_dilate", 15))
    mode = str(cfg.test_cfg.get("replace_mask_mode", "dilate"))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1)) if dilate > 0 else None
    out = []
    for m in masks[np.asarray(idxs)]:
        m3 = np.repeat(((m > 127) * 255).astype(np.uint8)[..., None], 3, axis=2)
        m = resize_by_area(m3, area, divisor=16)[..., 0] > 127  # same geometry as the driving frames
        if mode == "bbox" and m.any():
            ys, xs = np.nonzero(m)
            m = np.zeros_like(m)
            m[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
        if kernel is not None:
            m = cv2.dilate(m.astype(np.uint8), kernel) > 0
        out.append(m.astype(np.uint8))
    return zigzag_padding(out, target_len)


def load_background_frames(cfg, idxs, area, target_len, n_frames):
    """Frames of test_cfg.replace_background_video (the original video when the driving video was
    retargeted), sampled / resized / padded exactly like the driving frames. None when not set."""
    path = cfg.test_cfg.get("replace_background_video", None)
    if not path:
        return None
    reader = VideoReader(path)
    if len(reader) != n_frames:
        raise ValueError(f"{path} has {len(reader)} frames but the driving video has {n_frames}")
    frames = reader.get_batch(list(idxs)).asnumpy()
    return zigzag_padding([resize_by_area(f, area, divisor=16) for f in frames], target_len)


def fill_hole(cond, m, fill="blur", scale=8):
    """cond 3,n,H,W in [-1, 1]; m n,H,W (1 = hole). Returns cond with the hole filled (grey | black | blur).
    blur: normalized convolution of the known pixels at 1/scale resolution, widened until every hole pixel
    has known neighbours (the hole gets the surrounding background colours, no foreign colour)."""
    if fill == "grey":
        return cond * (1 - m)[None]
    if fill == "black":
        return cond * (1 - m)[None] - m[None]
    if fill != "blur":
        raise ValueError(f"test_cfg.replace_fill must be grey, black or blur, not {fill!r}")
    out = cond.clone()
    H, W = m.shape[1:]
    sw, sh = max(1, W // scale), max(1, H // scale)
    for t in range(m.shape[0]):
        hole = m[t].numpy() > 0.5
        if not hole.any() or hole.all():
            out[:, t] = cond[:, t] * (1 - m[t])[None]      # nothing to fill / nothing to fill from: grey
            continue
        img = cond[:, t].permute(1, 2, 0).numpy()
        keep = cv2.resize((~hole).astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
        f = cv2.resize(img * (~hole)[..., None], (sw, sh), interpolation=cv2.INTER_AREA)  # known pixels only
        est = np.zeros_like(f); done = np.zeros((sh, sw), bool); k = 3
        while not done.all():
            num, den = cv2.blur(f, (k, k)), cv2.blur(keep, (k, k))
            new = (den > 1e-3) & ~done
            est[new] = num[new] / den[new][:, None]
            done |= new; k = 2 * k + 1
        est = cv2.resize(est, (W, H), interpolation=cv2.INTER_LINEAR)
        img = img.copy(); img[hole] = est[hole]
        out[:, t] = torch.from_numpy(img).permute(2, 0, 1)
    return out


def replace_conditioning(vae, batch, clip_bg, clip_masks, mask_reft_len, H, W, lat_t, lat_h, lat_w, prec, device,
                         fill="blur", grow=1):
    """(y_reft, msk_reft) for one clip: VAE latents of the background frames with the person region filled
    and the matching per-pixel mask channels (same shapes as the move-mode tensors they replace).

    fill: what goes into the hole before VAE encoding. "grey" (0, like move mode's unknown frames) leaks
    into the neighbouring KEPT latent cells through the VAE's receptive field and shows up as a light rim
    on dark scenes; "blur" (default) uses the surrounding background colours instead; "black" = v1.
    grow: extra latent cells (8 px each) around the hole that are generated instead of kept, so the
    cells that saw the fill are redrawn by the model."""
    n = batch["conditioning_pixel_values"].shape[2]
    cond = torch.from_numpy(np.stack(clip_bg[:n]).astype(np.float32) / 127.5 - 1).permute(3, 0, 1, 2)
    cond = F.interpolate(cond, size=(H, W), mode="bicubic")                # 3, n, H, W in [-1, 1]
    m = torch.from_numpy(np.stack(clip_masks[:n])).float()[None]           # 1, n, h, w (1 = person)
    m = F.interpolate(m, size=(H, W), mode="nearest")[0]                   # n, H, W
    frames = fill_hole(cond, m, fill)
    if mask_reft_len > 0:  # previous-clip frames stay fully known, exactly as in move mode
        frames[:, :mask_reft_len] = F.interpolate(batch["pixel_values"][0, :, :mask_reft_len].float().cpu(),
                                                  size=(H, W), mode="bicubic")
    dtype = batch["conditioning_pixel_values"].dtype
    y_reft = vae_encode(vae, [frames.to(device=device, dtype=dtype)], prec)[0]
    gen = F.max_pool2d(m[:, None], kernel_size=8)                          # n, 1, lat_h, lat_w: any person pixel -> generate
    if grow > 0:
        gen = F.max_pool2d(gen, kernel_size=2 * grow + 1, stride=1, padding=grow)
    known = 1 - gen[:, 0]
    msk_reft = get_i2vmask_mask(known[None].to(device), lat_t - 1, lat_h, lat_w, mask_reft_len, device=device)
    return y_reft, msk_reft


def composite_clip(out, bg_images, person_masks, start, cfg):
    """Paste the generated frames (uint8 t,h,w,3) onto the original frames inside the person mask."""
    if not cfg.test_cfg.get("replace_composite", True):
        return out
    res = []
    for k, frame in enumerate(out):
        bg = np.clip(np.asarray(bg_images[start + k]), 0, 255).astype(np.uint8)
        m = person_masks[start + k] * 255
        if bg.shape[:2] != frame.shape[:2]:
            bg = cv2.resize(bg, (frame.shape[1], frame.shape[0]))
            m = cv2.resize(m, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        res.append(feather_blend(frame, bg, m))
    return np.stack(res)


def feather_blend(fg, bg, mask, border=15):
    """fg inside mask (0/255), bg outside, with an eroded + blurred edge. Same recipe as the unused
    upstream blend_layer_with_mask (pipelines/utils/multiclip_utils.py), which lacks its cv2 import."""
    m = mask.astype(np.uint8).copy()
    border = min(border, m.shape[0] // 10, m.shape[1] // 10)
    if border < 1:
        w = (m > 0).astype(np.float32)[..., None]
    else:
        m[:, :1] = 0; m[:1, :] = 0; m[-1:, :] = 0; m[:, -1:] = 0  # keep erode effective at the borders
        m = cv2.erode(m, np.ones((border, border), np.uint8))
        w = cv2.blur(m, (border, border)).astype(np.float32)[..., None] / 255
    return (fg * w + bg * (1 - w)).astype(np.uint8)


def inference_core_replace(item, gpu, cfg, clip, vae, model, t5, use_t5, context, context_null, device,
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
    # REPLACE MODE (1/3): one person mask per driving frame, same sampling/resize/padding as the frames
    person_masks = load_person_masks(cfg, idxs, width * height, target_len)
    bg_images = load_background_frames(cfg, idxs, width * height, target_len, len(video_reader)) or cond_images
    all_raw_frames = []

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

            # REPLACE MODE (2/3): known = driving frames with the person blanked out, per-pixel mask
            # (1 = keep background / continuation frames, 0 = generate the new character there)
            y_reft, msk_reft = replace_conditioning(vae, batch, bg_images[start:start + CLIP_LEN],
                                                    person_masks[start:start + CLIP_LEN],
                                                    mask_reft_len, H, W, lat_t, lat_h, lat_w, prec, device,
                                                    fill=str(cfg.test_cfg.get("replace_fill", "blur")),
                                                    grow=int(cfg.test_cfg.get("replace_grow", 1)))
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

            # REPLACE MODE (3/3): composite onto the original background (results.mp4); raw -> results_raw.mp4
            raw = out
            out = composite_clip(out, bg_images, person_masks, start, cfg)
            if start != 0:
                out = out[first_num:]
                raw = raw[first_num:]

            all_out_frames.extend([*out])
            all_raw_frames.extend([*raw])
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

        raw_array = np.stack(all_raw_frames, axis=0)[:real_frame_len]
        if padding_type == 'width':
            raw_array = raw_array[:, :, padding:padding+side_long, :]
        else:
            raw_array = raw_array[:, padding:padding+side_long, :, :]
        mpy.ImageSequenceClip([*raw_array], fps=fps).write_videofile(os.path.join(output_path, 'results_raw.mp4'))

        res = {'result': video_path, 'status': 'success'}
        return res
    
    return None


_base_worker = base.worker


def worker_replace(gpu, cfg, *args, **kwargs):
    """Spawned once per GPU: the original worker, running the replace-mode inference core."""
    base.inference_core = inference_core_replace
    return _base_worker(gpu, cfg, *args, **kwargs)


class WanAnimate2ReplacePipeline(base.WanAnimate2MPIPipeline):
    """Experimental replace mode; same call signature as WanAnimate2MPIPipeline."""

    def setup(self):
        base.worker = worker_replace  # base.main() spawns base.worker
        super().setup()
