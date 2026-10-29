"""Joint target encodes and deterministic person-blurred driving latents for WA2 TI2V-5B."""
import argparse
import fcntl
import glob
import hashlib
import json
import math
import os
import sys
import time
import zlib

import cv2
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
from prep_incontext_ti2v5b import (PROMPT, atomic_save, encode_text, pick_window, read_frames, reused_target,
                                  video_info)
from ti2v5b_replace import cover, vae_decode, vae_encode, write_video
from train_incontext_ti2v5b import CachedLatents as PilotCachedLatents

VERSION = "wa2-ti2v5b-person-blur-v1"


def person_mask_to_latent(masks):
    """Pixel frame 0, then groups of four; 16-cell pooling and one latent-cell dilation."""
    m = torch.as_tensor(masks).float()
    if m.ndim != 3 or (len(m) - 1) % 4 or m.shape[1] % 16 or m.shape[2] % 16:
        raise ValueError("masks must be [4k+1, 16h, 16w]")
    m = F.max_pool2d(m[:, None], 16)[:, 0]
    groups = [m[0]] + [m[j:j + 4].amax(0) for j in range(1, len(m), 4)]
    return F.max_pool2d(torch.stack(groups)[:, None], 3, stride=1, padding=1)[:, 0].bool()


def blur_person_latents(z, mask, sigma=2.0):
    """Separable spatial Gaussian on [C,F,H,W]; background values stay bit-identical."""
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("blur sigma must be positive and finite")
    radius = math.ceil(3 * sigma)
    u = torch.arange(-radius, radius + 1, device=z.device, dtype=torch.float32)
    kernel = torch.exp(-u.square() / (2 * sigma**2))
    kernel /= kernel.sum()
    c, frames, h, w = z.shape
    x = z.float().reshape(c * frames, 1, h, w)
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="replicate"), kernel.view(1, 1, 1, -1))
    x = F.conv2d(F.pad(x, (0, 0, radius, radius), mode="replicate"), kernel.view(1, 1, -1, 1))
    return torch.where(mask.to(z.device)[None], x.reshape_as(z).to(z.dtype), z)


def person_masks(raw, detector, segmenter, width=480, height=832):
    """Detect the first visible person at max_side 512, track, then apply the frames' cover crop."""
    from grounded_sam2 import main_object, shrink
    small = [shrink(f, 512) for f in raw]
    masks = [np.zeros(f.shape[:2], bool) for f in small]
    for i, frame in enumerate(small):
        box = detector.best(frame, "person", pick="largest")
        if box is not None:
            masks = [main_object(m) for m in segmenter.track(small, i, box)]
            break
    h, w = raw.shape[1:3]
    masks = [cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) for m in masks]
    return np.stack([cover(m, width, height, cv2.INTER_NEAREST).astype(bool) for m in masks])


def load_mask_models(gdino, sam2):
    """Local dirs or HF ids (downloaded once into the HF cache); the TI2V-5B weights are always local_files_only."""
    from grounded_sam2 import Detector, Segmenter
    return Detector(gdino, 0.35, 0.25), Segmenter(sam2)


def encode_drive(vae, clip, masks, device, sigma=2.0):
    """Shared prep/inference encoding; an empty latent group rejects the sample."""
    mask = person_mask_to_latent(masks)
    if not mask.flatten(1).any(1).all():
        return None
    z = vae_encode(vae, clip, device)[0]
    return blur_person_latents(z, mask, sigma), mask


def prepare_drive(vae, raw, detector, segmenter, device, sigma=2.0, width=480, height=832):
    masks = person_masks(raw, detector, segmenter, width, height)
    clip = np.stack([cover(f, width, height) for f in raw])
    encoded = encode_drive(vae, clip, masks, device, sigma)
    return encoded, clip, masks


def file_identity(path):
    """Local file identity for resume/cache validation, without reading multi-GB weights."""
    path = os.path.realpath(path)
    files = sorted(glob.glob(os.path.join(path, "**", "*"), recursive=True)) if os.path.isdir(path) else [path]
    entries = []
    for f in files:
        if os.path.isfile(f):
            stat = os.stat(f)
            entries.append((os.path.relpath(f, path), stat.st_size, stat.st_mtime_ns))
    return dict(path=path, files=entries)


def validate_cache_manifest(folder):
    with open(os.path.join(folder, "manifest.json")) as fh:
        manifest = json.load(fh)
    if manifest.get("version") != VERSION:
        raise ValueError("not a WA2 cache (pilot caches are incompatible)")
    return manifest


class CachedLatents(PilotCachedLatents):
    """New samples live below samples/: the pilot's root-level glob refuses this cache."""

    def __init__(self, folder):
        self.folder = os.path.realpath(folder)
        self.manifest = validate_cache_manifest(folder)
        self.files = sorted(glob.glob(os.path.join(self.folder, "samples", "*.pt")))
        if not self.files:
            raise FileNotFoundError(f"no WA2 samples in {folder}")
        self.text = torch.load(os.path.join(folder, "text.pt"), map_location="cpu", weights_only=True)
        identity = [file_identity(f) for f in self.files + [os.path.join(folder, "text.pt")]]
        self.fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    def __getitem__(self, i):
        s = super().__getitem__(i)
        if s["mask"].dtype != torch.bool or s["mask"].shape != s["drive"].shape[1:]:
            raise ValueError(f"invalid latent mask: {self.files[i]}")
        if not s["mask"].flatten(1).any(1).all():
            raise ValueError(f"empty latent mask: {self.files[i]}")
        return s


def check_manifest(a, names):
    manifest = dict(version=VERSION, width=480, height=832, frames=33, fps=a.fps, windows=a.windows,
                    ref_gap=1.0, blur_sigma=a.blur_sigma, dilation=1, prompt=a.prompt, seed=a.seed,
                    model=file_identity(a.model), gdino=file_identity(a.gdino), sam2=file_identity(a.sam2),
                    videos=os.path.realpath(a.videos) if a.videos else None, names=names,
                    t5_wan=file_identity(a.t5_wan) if a.t5_wan else None,
                    t5_tokenizer=file_identity(a.t5_tokenizer) if a.t5_tokenizer else None)
    manifest = json.loads(json.dumps(manifest))
    path = os.path.join(a.out, "manifest.json")
    with open(os.path.join(a.out, ".manifest.lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        old = None
        if os.path.exists(path):
            with open(path) as fh:
                old = json.load(fh)
        samples = glob.glob(os.path.join(a.out, "samples", "*.pt"))
        if old is None and (samples or glob.glob(os.path.join(a.out, "*.pt"))):
            raise ValueError("unversioned cache: use a new cache directory")
        if old not in (None, manifest):
            if samples:
                raise ValueError("cache manifest differs: use a new cache directory")
            # no sample yet (an earlier run stopped before its first one): start this cache over
            if os.path.exists(os.path.join(a.out, "text.pt")):
                os.remove(os.path.join(a.out, "text.pt"))
        if old != manifest:
            tmp = os.path.join(a.out, f".tmp_manifest.{os.getpid()}")
            with open(tmp, "w") as fh:
                json.dump(manifest, fh, indent=2)
            os.replace(tmp, path)


def check_reuse(folder, a):
    """Pilot targets are reused only from a cache made with the same frames, size, fps, reference gap, seed and VAE."""
    with open(os.path.join(folder, "manifest.json")) as fh:
        old = json.load(fh)
    want = dict(width=480, height=832, frames=33, fps=a.fps, ref_gap=1.0, seed=a.seed, teacher=False,
                model=os.path.realpath(a.model))
    bad = [k for k, v in want.items() if old.get(k) != v]
    if bad:
        raise ValueError(f"--reuse-target {folder} was made with other settings ({', '.join(bad)}): drop it")


def windows(path, count, fps, seed):
    """Keep the pilot's window RNG sequence, including its unused augmentation draws."""
    n, src_fps = video_info(path)
    rng = np.random.default_rng(seed)
    for window in range(count):
        picked = pick_window(n, src_fps, fps, 33, 1.0, rng)
        if picked is None:
            break
        idx, ref = picked
        rng.uniform(0.85, 1.0)
        rng.uniform(-0.05, 0.05, 2)
        yield idx, ref, dict(src=os.path.basename(path), window=window, start=int(idx[0]), ref=int(ref),
                             src_fps=float(src_fps), fps=float(fps))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--videos", required=True)
    p.add_argument("--list")
    p.add_argument("--out", required=True)
    p.add_argument("--frames", type=int, choices=[33], default=33)
    p.add_argument("--fps", type=float, default=24)
    p.add_argument("--windows", type=int, default=2)
    p.add_argument("--blur-sigma", type=float, default=2.0)
    p.add_argument("--reuse-target")
    p.add_argument("--prompt", default=PROMPT)
    p.add_argument("--t5-wan")
    p.add_argument("--t5-tokenizer")
    p.add_argument("--gdino", default=os.path.join(HERE, "../ckpts/grounding-dino-base"))
    p.add_argument("--sam2", default=os.path.join(HERE, "../ckpts/sam2.1-hiera-large"))
    p.add_argument("--preview", type=int, default=0, help="number of mask | blurred-drive previews (e.g. 4), on shard 0")
    p.add_argument("--text-only", action="store_true")
    p.add_argument("--shard", default="0/1")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if not math.isfinite(a.blur_sigma) or a.blur_sigma <= 0:
        p.error("--blur-sigma must be positive and finite")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = ([line.strip() for line in open(a.list) if line.strip()] if a.list else
             sorted(os.path.basename(f) for f in glob.glob(os.path.join(a.videos, "*.mp4"))))
    os.makedirs(a.out, exist_ok=True)
    check_manifest(a, names)
    if a.reuse_target:
        check_reuse(a.reuse_target, a)
    text_path = os.path.join(a.out, "text.pt")
    if not os.path.exists(text_path) and (a.text_only or a.shard.split("/")[0] == "0"):
        atomic_save(encode_text(a, device), text_path)
    if a.text_only:
        return
    from diffusers import AutoencoderKLWan
    detector, segmenter = load_mask_models(a.gdino, a.sam2)
    vae = AutoencoderKLWan.from_pretrained(os.path.join(a.model, "vae"), torch_dtype=torch.float32,
                                          local_files_only=True).to(device).requires_grad_(False).eval()
    shard, shards = map(int, a.shard.split("/"))
    sample_dir = os.path.join(a.out, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    done = skipped = empty = reused = previews = 0
    mine, t0 = names[shard::shards], time.time()
    for k, name in enumerate(mine, 1):
        stem = os.path.splitext(name)[0]
        seed = a.seed * 1_000_003 + zlib.crc32(stem.encode()) % 1_000_003
        count = 0
        for idx, ref, meta in windows(os.path.join(a.videos, name), a.windows, a.fps, seed):
            count += 1
            path = os.path.join(sample_dir, f"{stem}_w{meta['window']}.pt")
            if os.path.exists(path):
                continue
            raw = read_frames(os.path.join(a.videos, name), idx + [ref])
            encoded, clip, masks = prepare_drive(vae, raw[:-1], detector, segmenter, device, a.blur_sigma)
            if encoded is None:
                empty += 1
                continue
            drive, mask = encoded
            target = reused_target(a.reuse_target, os.path.basename(path), meta, (9, 52, 30))
            reused += target is not None
            if target is None:
                ref_rgb = cover(raw[-1], 480, 832)
                target = vae_encode(vae, np.concatenate([ref_rgb[None], clip[1:]]), device)[0].cpu()
            meta.update(blur_sigma=a.blur_sigma, dilation=1, mask_area=mask.float().mean((1, 2)).tolist())
            atomic_save(dict(target=target.cpu(), drive=drive.cpu(), mask=mask.cpu(), meta=meta), path)
            done += 1
            if shard == 0 and previews < a.preview:
                overlay = clip.copy()
                overlay[masks] = (0.5 * overlay[masks] + [127, 0, 0]).astype(np.uint8)
                preview = np.concatenate([overlay, vae_decode(vae, drive[None])], axis=2)
                write_video(os.path.join(a.out, f"preview_{previews}.mp4"), preview, a.fps)
                previews += 1
        skipped += a.windows - count
        if k % 10 == 0 or k == len(mine):
            rate = (time.time() - t0) / k
            print(f"shard {a.shard}: {k}/{len(mine)} clips, {done} new samples, {empty} empty-mask skips, "
                  f"{rate:.0f} s/clip, ~{rate * (len(mine) - k) / 3600:.1f} h left", flush=True)
    print(f"shard {a.shard}: {done} samples, {reused} reused targets, {empty} empty-mask skips, "
          f"{skipped} missing-window skips -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
