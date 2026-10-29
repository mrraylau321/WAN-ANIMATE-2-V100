"""Conditional Euler evaluation: reference frame 0, cached clean driving, and driving controls."""
import argparse
import json
import math
import os

import cv2
import torch

from infer_incontext_ti2v5b import flow_sigmas, side_by_side
from prep_wa2_ti2v5b import (HERE, VERSION, file_identity, load_mask_models, prepare_drive, read_frames, video_info)
from ti2v5b_replace import cover, vae_decode, vae_encode, write_video
from train_incontext_ti2v5b import token_timesteps
from train_wa2_ti2v5b import load_qkv_state
from wa2_ti2v5b import assert_5b, forward_drive, set_shared_qkv_trainable, to_fp16_storage, wa2_forward


@torch.no_grad()
def sample(model, ref, drive, text, steps=40, shift=5.0, generator=None, amp=False, use_drive=True):
    """Build driving K/V once; pin the clean reference on every call and after every Euler step."""
    b, _, frames, h, w = drive.shape
    x = torch.randn(drive.shape, generator=generator).to(drive.device)
    sigmas = flow_sigmas(steps, shift)
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        cache = forward_drive(model, drive, text) if use_drive else None
    for i in range(steps):
        x[:, :, :1] = ref
        times = torch.full((b, frames), float(sigmas[i]) * 1000, device=x.device)
        times[:, 0] = 0
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            v = wa2_forward(model, x, token_timesteps(times, h, w), text, drive_cache=cache).float()
        if not torch.isfinite(v).all():
            raise RuntimeError(f"non-finite velocity at step {i}")
        x = x + float(sigmas[i + 1] - sigmas[i]) * v
        x[:, :, :1] = ref
    x[:, :, :1] = ref
    return x


def load_qkv(model, path, model_dir, blur_sigma):
    from safetensors import safe_open
    from safetensors.torch import load_file
    with safe_open(path, framework="pt", device="cpu") as fh:
        config = json.loads(fh.metadata()["config"])
    identity = json.loads(json.dumps(file_identity(model_dir)))
    if config["version"] != VERSION or config["model"] != identity:
        raise ValueError("QKV export belongs to a different base checkpoint or layout")
    if config["manifest"]["blur_sigma"] != blur_sigma:
        raise ValueError("blur sigma differs from training")
    load_qkv_state(model, load_file(path))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--qkv")
    p.add_argument("--text", required=True)
    p.add_argument("--ref", help="reference image; or --ref-from-drive")
    p.add_argument("--ref-from-drive", action="store_true",
                   help="reference = the --drive clip's frame 1 s after the 33-frame window (as in training)")
    p.add_argument("--drive", required=True)
    p.add_argument("--drive-clip", help="use another driving clip (wrong-drive control)")
    p.add_argument("--reverse-drive", action="store_true")
    p.add_argument("--no-drive", action="store_true")
    p.add_argument("--out", required=True)
    p.add_argument("--frames", type=int, choices=[33], default=33)
    p.add_argument("--fps", type=float, default=24)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--blur-sigma", type=float, default=2.0)
    p.add_argument("--gdino", default=os.path.join(HERE, "../ckpts/grounding-dino-base"))
    p.add_argument("--sam2", default=os.path.join(HERE, "../ckpts/sam2.1-hiera-large"))
    p.add_argument("--weights", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--name")
    a = p.parse_args()
    if bool(a.ref) == a.ref_from_drive:
        p.error("give --ref or --ref-from-drive")
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from diffusers import AutoencoderKLWan, WanTransformer3DModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # fp32 load, as in the pilot (fp16 from_pretrained needs accelerate), then fp16 storage
    model = WanTransformer3DModel.from_pretrained(os.path.join(a.model, "transformer"), torch_dtype=torch.float32,
                                                local_files_only=True)
    if a.weights == "fp16":
        to_fp16_storage(model)
    params = set_shared_qkv_trainable(model)
    assert_5b(model)
    for param in params:
        param.data = param.data.float()
    if a.qkv:
        load_qkv(model, a.qkv, a.model, a.blur_sigma)
    model.requires_grad_(False).eval()
    vae = AutoencoderKLWan.from_pretrained(os.path.join(a.model, "vae"), torch_dtype=torch.float32,
                                          local_files_only=True).to(device).requires_grad_(False).eval()
    path = a.drive_clip or a.drive
    n, fps = video_info(path)
    idx = [round(j * fps / a.fps) for j in range(33)]
    if idx[-1] >= n:
        raise ValueError("driving clip is too short for 33 frames at the requested fps")
    raw = read_frames(path, idx)
    if a.reverse_drive:
        raw = raw[::-1].copy()
    detector, segmenter = load_mask_models(a.gdino, a.sam2)
    encoded, _, _ = prepare_drive(vae, raw, detector, segmenter, device, a.blur_sigma)
    if encoded is None:
        raise ValueError("driving clip has an empty latent person-mask group")
    drive, _ = encoded
    del detector, segmenter
    if a.ref_from_drive:
        n_ref, fps_ref = video_info(a.drive)
        at = round(32 * fps_ref / a.fps) + math.ceil(fps_ref)
        if at >= n_ref:
            raise ValueError(f"{a.drive} is too short for a reference 1 s after the window")
        ref_rgb = cover(read_frames(a.drive, [at])[0], 480, 832)
    else:
        ref_rgb = cover(cv2.cvtColor(cv2.imread(a.ref), cv2.COLOR_BGR2RGB), 480, 832)
    ref = vae_encode(vae, ref_rgb[None], device)
    vae.to("cpu")
    if device != "cpu":
        torch.cuda.empty_cache()
    model.to(device)
    text = torch.load(a.text, map_location="cpu", weights_only=True)["text"].to(device).float()
    z = sample(model, ref, drive[None], text, a.steps, a.shift, torch.Generator().manual_seed(a.seed),
               amp=device != "cpu", use_drive=not a.no_drive)
    model.to("cpu")
    if device != "cpu":
        torch.cuda.empty_cache()
    vae.to(device)
    vae.enable_tiling()  # decode only: the drive and reference were encoded untiled, like the training cache
    z[:, :, :1] = ref
    output, blurred = vae_decode(vae, z), vae_decode(vae, drive[None])
    os.makedirs(a.out, exist_ok=True)
    tag = os.path.splitext(os.path.basename(a.qkv))[0] if a.qkv else "base"
    tag += "_nodrive" if a.no_drive else "_drive"
    tag += "_wrong" if a.drive_clip else ""
    tag += "_reverse" if a.reverse_drive else ""
    name = a.name or f"{tag}_s{a.seed}"
    write_video(os.path.join(a.out, name + ".mp4"), output, a.fps)
    write_video(os.path.join(a.out, name + "_side.mp4"), side_by_side(blurred, output), a.fps)
    print(f"33 frames (frame 0 is conditioning), Euler guide=1, shift={a.shift}, "
          f"sigmas={flow_sigmas(a.steps, a.shift).tolist()} -> {a.out}/{name}.mp4", flush=True)


if __name__ == "__main__":
    main()
