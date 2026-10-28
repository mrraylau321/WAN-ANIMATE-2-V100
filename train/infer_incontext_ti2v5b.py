"""Try the TI2V-5B in-context pilot: a reference image moves like the driving video.

The driving clip gets the same treatment as in training (grey + blur + random contrast, then its latents noised to
--drive-noise), the reference image is latent frame 0 (clean), the rest is sampled with Euler steps on the shifted
flow schedule. Without --lora it is the untrained base with the same in-context input: the baseline to compare with.
  python train/infer_incontext_ti2v5b.py --model <TI2V-5B diffusers dir> --lora runs/pilot/lora_002000.safetensors \\
      --text <cache>/text.pt --ref ref.png --drive drive.mp4 --out outputs/pilot
Writes <out>/<name>.mp4 (the generated video) and <name>_side.mp4 (driving | generated).
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
from prep_incontext_ti2v5b import destroy_appearance  # noqa: E402
from train_incontext_ti2v5b import incontext_forward, load_lora, token_timesteps  # noqa: E402


def flow_sigmas(steps, shift=5.0):
    s = torch.linspace(1, 0, steps + 1, dtype=torch.float64)
    return shift * s / (1 + (shift - 1) * s)


@torch.no_grad()
def sample(model, ref, drive, text, neg=None, steps=40, shift=5.0, drive_level=0.4, guide=1.0, generator=None,
           amp=False, use_drive=True, drive_rope="same"):
    """ref [1, C, 1, h, w], drive [1, C, F, h, w] (normalized latents) -> generated latents [1, C, F, h, w]."""
    B, C, F_, h, w = drive.shape
    dev = drive.device
    x = torch.randn(drive.shape, generator=generator).to(dev)
    x[:, :, :1] = ref
    d = (1 - drive_level) * drive + drive_level * torch.randn(drive.shape, generator=generator).to(dev)
    t_drive = token_timesteps(torch.full((B, F_), drive_level * 1000, device=dev), h, w)
    sig = flow_sigmas(steps, shift)
    for i in range(steps):
        t_frames = torch.full((B, F_), float(sig[i]) * 1000, device=dev)
        t_frames[:, 0] = 0
        t_x = token_timesteps(t_frames, h, w)
        kw = dict(drive=d, t_drive=t_drive, drive_rope=drive_rope) if use_drive else {}
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            v = incontext_forward(model, x, t_x, text, **kw).float()
            if guide > 1 and neg is not None:
                v = incontext_forward(model, x, t_x, neg, **kw).float() * (1 - guide) + v * guide
        if not torch.isfinite(v).all():
            raise RuntimeError(f"non-finite output at step {i}: inspect overflowing projections; "
                               "fp32 weight storage still uses fp16 autocast")
        x = x + float(sig[i + 1] - sig[i]) * v
        x[:, :, :1] = ref
    return x


def side_by_side(a, b):
    h = min(a.shape[1], b.shape[1])
    fit = lambda f: cv2.resize(f, (round(f.shape[1] * h / f.shape[0]), h), interpolation=cv2.INTER_AREA)
    return np.stack([np.concatenate([fit(x), fit(y)], 1) for x, y in zip(a, b)])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Wan2.2-TI2V-5B-Diffusers folder")
    p.add_argument("--lora", help="trained LoRA (.safetensors); none = the untrained baseline")
    p.add_argument("--text", required=True, help="text.pt written by prep_incontext_ti2v5b.py (same prompt as training)")
    p.add_argument("--ref", required=True, help="reference image")
    p.add_argument("--drive", required=True, help="driving video")
    p.add_argument("--out", required=True)
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=832)
    p.add_argument("--frames", type=int, default=33, help="4k + 1")
    p.add_argument("--fps", type=float, default=24)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--drive-noise", type=float, default=0.4, help="driving latent noise (training range 0.2-0.7)")
    p.add_argument("--guide", type=float, default=1.0, help="> 1: CFG against the negative prompt (2x time)")
    p.add_argument("--no-drive", action="store_true", help="plain TI2V image-to-video (no driving tokens)")
    p.add_argument("--drive-rope", choices=["same", "offset"], default="same", help="must match the LoRA's training")
    p.add_argument("--weights", choices=["fp32", "fp16"], default="fp32")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--name", help="output file name (default: from the inputs and settings)")
    a = p.parse_args()
    from diffusers import AutoencoderKLWan, WanTransformer3DModel
    from ti2v5b_replace import cover, read_clip, vae_decode, vae_encode, write_video
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if a.weights == "fp32" else torch.float16
    model = WanTransformer3DModel.from_pretrained(os.path.join(a.model, "transformer"), torch_dtype=dtype)
    if a.lora:
        load_lora(model, a.lora)
    model.requires_grad_(False).eval()
    vae = AutoencoderKLWan.from_pretrained(os.path.join(a.model, "vae"), torch_dtype=torch.float32).to(device)
    vae.requires_grad_(False).eval()
    if hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
    texts = torch.load(a.text, map_location="cpu", weights_only=True)

    frames, _ = read_clip(a.drive, a.fps, a.frames, a.width, a.height)
    ref = cover(cv2.cvtColor(cv2.imread(a.ref), cv2.COLOR_BGR2RGB), a.width, a.height)
    g = torch.Generator().manual_seed(a.seed)
    drive_aug = destroy_appearance(frames, g)
    z_ref = vae_encode(vae, ref[None], device)
    z_drive = vae_encode(vae, drive_aug, device)
    vae.to("cpu")
    if device != "cpu":
        torch.cuda.empty_cache()
    model.to(device)
    t0 = time.time()
    z = sample(model, z_ref, z_drive, texts["text"].to(device).float(), texts["neg"].to(device).float(), a.steps,
               a.shift, a.drive_noise, a.guide, g, amp=device != "cpu", use_drive=not a.no_drive,
               drive_rope=a.drive_rope)
    model.to("cpu")
    if device != "cpu":
        torch.cuda.empty_cache()
    vae.to(device)
    out = vae_decode(vae, z)
    os.makedirs(a.out, exist_ok=True)
    tag = (os.path.splitext(os.path.basename(a.lora))[0] if a.lora else "base") + ("_nodrive" if a.no_drive else "")
    name = a.name or (f"{os.path.splitext(os.path.basename(a.ref))[0]}_{tag}_n{a.drive_noise}_g{a.guide:g}_"
                      f"{time.strftime('%H%M%S')}")
    write_video(os.path.join(a.out, f"{name}.mp4"), out, a.fps)
    write_video(os.path.join(a.out, f"{name}_side.mp4"), side_by_side(frames[:len(out)], out), a.fps)
    write_video(os.path.join(a.out, f"{name}_drive_input.mp4"), drive_aug, a.fps)
    print(f"{len(out)} frames in {time.time() - t0:.0f} s -> {os.path.join(a.out, name)}.mp4", flush=True)


if __name__ == "__main__":
    main()
