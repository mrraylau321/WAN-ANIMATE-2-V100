"""EXPERIMENTAL, UNOFFICIAL: our training-free replace recipe on Wan2.2-TI2V-5B (public diffusers weights).

Wan-Animate-2 follows the driving video through its trained in-context branch; TI2V-5B has no such input (only a
first frame and text). So the motion here comes from SDEdit: the driving clip's own latents are noised to
--strength and denoised. The rest is the recipe that works on Wan-Animate-2:
  g0        first frame = the new character on the clip's clean plate (grounded_sam2.py scene-ref), put in
            TI2V's image slot (first latent frame kept clean, its tokens at timestep 0)
  blending  every step, outside the dilated person mask the velocity is set so that x0 = the driving latent
            (flow matching: x0 = x_t - sigma * v): the background is the real clip
  person    generated from the noised driving latents (pose and layout from --strength), look from g0 + prompt
Expect: a higher --strength frees the character (less of the old person) but follows the pose less.

Weights (34 GB): huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers --local-dir <dir>; needs diffusers >= 0.35.
Other TI2V-5B-architecture weights (same VAE, text encoder and first-frame scheme): --transformer-ckpt <file> loads a
single .pth/.safetensors (original Wan or diffusers key names) into the --model config; --expert <file>@<lo>-<hi>
(repeatable) uses several such checkpoints by timestep range (t = 1000 * sigma, lo <= t < hi), one at a time on the GPU.
  python scripts/ti2v5b_replace.py --model <dir> --drive clip.mp4 --g0 ref_scene.png --masks clip_gsam2_person_mask.npz \\
      --prompt "..." --out outputs/ti2v5b
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    import imageio.v2 as imageio
except ImportError:  # older imageio
    import imageio

NEGATIVE = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，"
            "残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，"
            "三条腿，背景人很多，倒着走")
VAE_STRIDE, T_STRIDE = 16, 4  # Wan2.2 VAE: 16x16 pixels and 4 frames per latent cell


def cover(img, w, h, interp=cv2.INTER_AREA):
    """Resize + centre-crop to exactly w x h (keeps the aspect ratio)."""
    s = max(w / img.shape[1], h / img.shape[0])
    r = cv2.resize(img, (max(w, round(img.shape[1] * s)), max(h, round(img.shape[0] * s))), interpolation=interp)
    y, x = (r.shape[0] - h) // 2, (r.shape[1] - w) // 2
    return r[y:y + h, x:x + w]


def sample_indices(n_src, src_fps, fps, n):
    """Source frame of each output frame at `fps` (first n, clamped to the clip)."""
    idx = [min(int(round(k * src_fps / fps)), n_src - 1) for k in range(n)]
    return idx


def read_clip(path, fps, n, w, h):
    reader = imageio.get_reader(path)
    src_fps = float(reader.get_meta_data().get("fps", fps))
    frames = [f for f in reader]
    usable = min(n, int((len(frames) - 1) * fps / src_fps) + 1)
    usable = (usable - 1) // T_STRIDE * T_STRIDE + 1  # 4k + 1 frames
    idx = sample_indices(len(frames), src_fps, fps, usable)
    return np.stack([cover(frames[i], w, h) for i in idx]), idx


def load_masks(path, idx, w, h, dilate):
    masks = np.load(path)["masks"]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1,) * 2) if dilate > 0 else None
    out = []
    for i in idx:
        m = cover(((masks[min(i, len(masks) - 1)] > 127) * 255).astype(np.uint8), w, h, cv2.INTER_NEAREST) > 127
        out.append(cv2.dilate(m.astype(np.uint8), k) > 0 if k is not None else m)
    return np.stack(out)


def latent_keep(masks, grow):
    """1 where the driving latent is kept (background), 0 where the person is generated; shape 1,1,T,h,w.
    Latent frame 0 is pixel frame 0, latent frame k >= 1 covers pixel frames 4k-3 .. 4k; a cell is 16x16 px."""
    m = torch.from_numpy(masks.astype(np.float32))[:, None]                 # n,1,H,W
    gen = F.max_pool2d(m, VAE_STRIDE)                                       # n,1,h,w: any person pixel
    if grow > 0:
        gen = F.max_pool2d(gen, 2 * grow + 1, stride=1, padding=grow)
    gen = gen[:, 0]
    n_lat = (len(masks) - 1) // T_STRIDE + 1
    groups = [gen[:1]] + [gen[1 + T_STRIDE * (k - 1):1 + T_STRIDE * k] for k in range(1, n_lat)]
    gen = torch.stack([g.amax(0) for g in groups])
    return (1 - gen)[None, None]


def vae_norm(vae, device):
    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(device)
    std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(device)
    return mean, std


@torch.no_grad()
def vae_encode(vae, frames, device):
    """uint8 n,H,W,3 -> normalized latents 1,48,T,h,w (float32)."""
    x = torch.from_numpy(frames).float().div(127.5).sub(1).permute(3, 0, 1, 2)[None].to(device, vae.dtype)
    z = vae.encode(x).latent_dist.mode().float()
    mean, std = vae_norm(vae, device)
    return (z - mean) / std


@torch.no_grad()
def vae_decode(vae, z):
    mean, std = vae_norm(vae, z.device)
    x = vae.decode((z * std + mean).to(vae.dtype), return_dict=False)[0]
    return ((x[0].float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 3, 0).cpu().numpy()


def read_state_dict(path):
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    for k in ("state_dict", "model", "module", "ema"):  # common training-checkpoint wrappers
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict) and len(sd) <= 4:
            sd = sd[k]
    return sd


def load_transformer_ckpt(path, config_dir, dtype, log=print):
    """A WanTransformer3DModel with the config of config_dir and the weights of one checkpoint file
    (original Wan names like blocks.0.self_attn.q.weight are converted; module./model. prefixes stripped)."""
    from diffusers import WanTransformer3DModel
    from diffusers.loaders.single_file_utils import convert_wan_transformer_to_diffusers
    sd = read_state_dict(path)
    for pre in ("module.", "model.", "_orig_mod.", "dit.", "transformer."):
        if sd and all(k.startswith(pre) for k in sd):
            sd = {k[len(pre):]: v for k, v in sd.items()}
    if not any(".attn1.to_q." in k for k in sd):
        sd = convert_wan_transformer_to_diffusers(dict(sd))
    model = WanTransformer3DModel.from_config(WanTransformer3DModel.load_config(config_dir))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"{path}: {len(missing)} missing / {len(unexpected)} unexpected keys after conversion "
                         f"(first: {missing[:3]} {unexpected[:3]}) -> not the TI2V-5B layout")
    log(f"loaded {path} ({sum(v.numel() for v in sd.values()) / 1e9:.2f} B params)")
    return model.to(dtype).eval()


class Experts:
    """Which transformer to use at timestep t: [(lo, hi, model)], lo <= t < hi. Only the active one sits on the
    device (the rest on the CPU), so several 5B experts fit one GPU."""

    def __init__(self, ranges, device):
        self.ranges, self.device, self.active = sorted(ranges, key=lambda r: -r[1]), device, None

    def __call__(self, t):
        t = float(t)
        for lo, hi, m in self.ranges:
            if lo <= t < hi or (hi >= 1000 and t >= lo):
                if m is not self.active:
                    if self.active is not None and len(self.ranges) > 1:
                        self.active.to("cpu")
                        torch.cuda.empty_cache()
                    self.active = m.to(self.device)
                return m
        raise ValueError(f"no expert covers timestep {t:.1f}: {[(lo, hi) for lo, hi, _ in self.ranges]}")


@torch.no_grad()
def denoise(transformer, scheduler, z_drive, cond0, keep, ctx, ctx_null, steps, strength, guide, free_steps,
            generator, dtype, log=print):
    """SDEdit from the driving latents + TI2V first-frame condition + background pinning.
    z_drive, cond0: 1,C,T,h,w and 1,C,1,h,w (normalized); keep: 1,1,T,h,w or None. Returns 1,C,T,h,w."""
    device = z_drive.device
    scheduler.set_timesteps(steps, device=device)
    timesteps = scheduler.timesteps
    k0 = min(max(int(round(steps * (1 - strength))), 0), steps - 1)
    sig0 = float(scheduler.sigmas[k0])
    noise = torch.randn(z_drive.shape, generator=generator, device="cpu").to(device)
    lat = (1 - sig0) * z_drive + sig0 * noise
    ffm = torch.ones(1, 1, *z_drive.shape[2:], device=device)              # 0 = the clean first frame (g0)
    ffm[:, :, 0] = 0
    if hasattr(scheduler, "set_begin_index"):
        scheduler.set_begin_index(k0)
    run = timesteps[k0:]
    pick = transformer if isinstance(transformer, Experts) else (lambda _t: transformer)
    log(f"SDEdit: {len(run)} of {steps} steps from sigma {sig0:.3f} (strength {strength}); "
        f"pinned background {'off' if keep is None else f'{float(keep[:, :, 1:].mean()):.0%}'}")
    for i, t in enumerate(run):
        x_in = ((1 - ffm) * cond0 + ffm * lat).to(dtype)
        tt = (ffm[0, 0][:, ::2, ::2] * t).flatten()[None]                    # per-token timestep, first frame 0
        model = pick(t)
        v = model(hidden_states=x_in, timestep=tt, encoder_hidden_states=ctx, return_dict=False)[0]
        if guide > 1:
            vu = model(hidden_states=x_in, timestep=tt, encoder_hidden_states=ctx_null, return_dict=False)[0]
            v = vu + guide * (v - vu)
        v = v.float()
        if not torch.isfinite(v).all():
            raise RuntimeError(f"non-finite transformer output at step {i} ({dtype}): rerun with DTYPE=fp32")
        if keep is not None and i < len(run) - free_steps:
            sigma = float(scheduler.sigmas[k0 + i])
            v = v * (1 - keep) + keep * (lat - z_drive) / sigma            # x0 prediction = driving latent
        lat = scheduler.step(v, t, lat, return_dict=False)[0]
    return (1 - ffm) * cond0 + ffm * lat


@torch.no_grad()
def encode_prompts(model_dir, prompts, device, max_len=512):
    """UMT5 embeddings like the diffusers Wan pipelines (trimmed to each prompt, zero-padded to max_len)."""
    from transformers import AutoTokenizer, UMT5EncoderModel
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    enc = UMT5EncoderModel.from_pretrained(os.path.join(model_dir, "text_encoder"), torch_dtype=torch.float32).to(device)
    t = tok(prompts, padding="max_length", max_length=max_len, truncation=True, add_special_tokens=True,
            return_attention_mask=True, return_tensors="pt")
    out = enc(t.input_ids.to(device), t.attention_mask.to(device)).last_hidden_state.float().cpu()
    lens = t.attention_mask.gt(0).sum(1)
    embs = [torch.cat([u[:n], u.new_zeros(max_len - int(n), u.shape[1])]) for u, n in zip(out, lens)]
    del enc
    torch.cuda.empty_cache()
    return [e[None] for e in embs]


@torch.no_grad()
def encode_prompts_wan(t5_ckpt, tokenizer_dir, prompts, device, max_len=512):
    """The same UMT5-XXL encoder from a Wan checkpoint (models_t5_umt5-xxl-enc-bf16.pth, as used by Wan-Animate-2),
    so the 11 GB diffusers text_encoder need not be downloaded. Output like encode_prompts."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from wanxiang.eval_i2v import T5Encoder
    t5 = T5Encoder(name="umt5_xxl", text_len=max_len, dtype=torch.float32, device=torch.device(device),
                   checkpoint_path=t5_ckpt, tokenizer_path=tokenizer_dir)
    outs = [u.float().cpu() for u in t5(prompts)]
    del t5
    torch.cuda.empty_cache()
    return [torch.cat([u, u.new_zeros(max_len - u.shape[0], u.shape[1])])[None] for u in outs]


def write_video(path, frames, fps):
    writer = imageio.get_writer(path, fps=fps, codec="libx264", macro_block_size=1, pixelformat="yuv420p",
                                ffmpeg_params=["-crf", "16"])
    for f in frames:
        writer.append_data(f[: f.shape[0] // 2 * 2, : f.shape[1] // 2 * 2])
    writer.close()


def run(a, transformer, vae, scheduler, ctx, ctx_null, device, dtype, log=print):
    frames, idx = read_clip(a.drive, a.fps, a.frames, a.width, a.height)
    masks = load_masks(a.masks, idx, a.width, a.height, a.dilate) if a.masks else None
    g0 = cover(cv2.cvtColor(cv2.imread(a.g0), cv2.COLOR_BGR2RGB), a.width, a.height)
    log(f"{len(frames)} frames {a.width}x{a.height} @ {a.fps} fps from {a.drive}")
    z_drive = vae_encode(vae, frames, device)
    cond0 = vae_encode(vae, g0[None], device)
    keep = latent_keep(masks, a.grow).to(device) if masks is not None and not a.no_blend else None
    if keep is not None and keep.shape[2:] != z_drive.shape[2:]:
        raise ValueError(f"mask latents {tuple(keep.shape[2:])} != video latents {tuple(z_drive.shape[2:])}")
    g = torch.Generator().manual_seed(a.seed)
    lat = denoise(transformer, scheduler, z_drive, cond0, keep, ctx.to(device, dtype), ctx_null.to(device, dtype),
                  a.steps, a.strength, a.guide, a.free_steps, g, dtype, log)
    out = vae_decode(vae, lat)
    os.makedirs(a.out, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(a.out, f"ti2v5b_s{a.strength}_{stamp}.mp4")
    write_video(path, out, a.fps)
    write_video(os.path.join(a.out, f"drive_{stamp}.mp4"), frames, a.fps)
    imageio.imwrite(os.path.join(a.out, f"g0_{stamp}.png"), g0)
    with open(os.path.join(a.out, f"args_{stamp}.json"), "w") as fh:
        json.dump(vars(a), fh, ensure_ascii=False, indent=1)
    log(f"-> {path}")
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Wan2.2-TI2V-5B-Diffusers directory")
    p.add_argument("--drive", required=True)
    p.add_argument("--g0", required=True, help="first frame: the new character on the clip's clean plate")
    p.add_argument("--masks", help="person masks of the driving video (.npz, grounded_sam2.py video)")
    p.add_argument("--prompt", default="")
    p.add_argument("--negative", default=NEGATIVE)
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=832)
    p.add_argument("--frames", type=int, default=121, help="at most this many (4k+1); 121 = 5 s at 24 fps")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--strength", type=float, default=0.6,
                   help="SDEdit noise level, 1 = pure noise (flow shift 5: 0.6 -> sigma 0.88, 0.8 -> 0.95)")
    p.add_argument("--guide", type=float, default=5.0)
    p.add_argument("--dilate", type=int, default=15, help="px around the person mask that is generated too")
    p.add_argument("--grow", type=int, default=1, help="extra latent cells (16 px) generated around it")
    p.add_argument("--free-steps", type=int, default=2, help="last steps without background pinning")
    p.add_argument("--no-blend", action="store_true", help="plain SDEdit, nothing pinned")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", choices=["fp16", "fp32", "bf16"], default="fp16")
    p.add_argument("--t5-device", default="auto", help="auto: the 2nd GPU if there is one, else cpu")
    p.add_argument("--t5-wan", help="Wan-format UMT5 checkpoint (models_t5_umt5-xxl-enc-bf16.pth) instead of "
                                    "<model>/text_encoder; needs --t5-tokenizer")
    p.add_argument("--t5-tokenizer", help="umt5-xxl tokenizer dir for --t5-wan")
    p.add_argument("--transformer-ckpt", help="one checkpoint file instead of <model>/transformer")
    p.add_argument("--expert", action="append", default=[], metavar="FILE@LO-HI",
                   help="checkpoint used for timesteps LO <= t < HI (0..1000); repeat to cover 0-1000")
    p.add_argument("--out", default="outputs/ti2v5b")
    a = p.parse_args()
    if a.width % 32 or a.height % 32:
        raise SystemExit("--width/--height must be multiples of 32 (16 px VAE cell x 2x2 patch)")

    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanTransformer3DModel
    dtype = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[a.dtype]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t5_dev = a.t5_device if a.t5_device != "auto" else ("cuda:1" if torch.cuda.device_count() > 1 else "cpu")
    print(f"text encoder on {t5_dev} (fp32) ...")
    if a.t5_wan:
        ctx, ctx_null = encode_prompts_wan(a.t5_wan, a.t5_tokenizer, [a.prompt or " ", a.negative], t5_dev)
    else:
        ctx, ctx_null = encode_prompts(a.model, [a.prompt or " ", a.negative], t5_dev)
    vae = AutoencoderKLWan.from_pretrained(os.path.join(a.model, "vae"), torch_dtype=torch.float32).to(device)
    vae.enable_tiling()
    cfg_dir = os.path.join(a.model, "transformer")
    if a.expert:
        ranges = []
        for spec in a.expert:
            path, _, rng = spec.rpartition("@")
            lo, hi = (float(x) for x in rng.split("-"))
            ranges.append((lo, hi, load_transformer_ckpt(path, cfg_dir, dtype)))
        transformer = Experts(ranges, device)
    elif a.transformer_ckpt:
        transformer = load_transformer_ckpt(a.transformer_ckpt, cfg_dir, dtype).to(device)
    else:
        transformer = WanTransformer3DModel.from_pretrained(cfg_dir, torch_dtype=dtype).to(device).eval()
    scheduler = UniPCMultistepScheduler.from_pretrained(os.path.join(a.model, "scheduler"))
    run(a, transformer, vae, scheduler, ctx, ctx_null, device, dtype)


if __name__ == "__main__":
    main()
