"""Data prep for the TI2V-5B in-context pilot: clips -> cached latents for train_incontext_ti2v5b.py.

Per clip, --windows samples (self-reconstruction, conditions made harder to copy):
  window    --frames frames at --fps from a random start (4k + 1 frames: TI2V-5B's VAE)
  target    [reference frame, window frames 1..]: latent frame 0 = the reference image, then the clip to rebuild
  reference a frame at least --ref-gap seconds away from the window (another moment); skip if none is found
  drive     the window itself, zoomed out and shifted by one random amount, then grey + blur + random contrast;
            the training loop also noises its latents. Cross-identity evaluation must establish motion transfer.
Plus text.pt: the UMT5 embedding of --prompt (one generic prompt for every clip) and of the negative prompt.

  python train/prep_incontext_ti2v5b.py --model <TI2V-5B diffusers dir> --videos <dir> --list pass.txt --out <cache> \\
      --shard 0/4        (run one shard per GPU; finished samples are skipped, so it can be restarted)
"""
import argparse
import fcntl
import glob
import json
import os
import sys
import zlib

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
from train_incontext_qkv import augment_driving  # noqa: E402
from ti2v5b_replace import NEGATIVE, cover  # noqa: E402

PROMPT = "一个人在自然地运动，动作流畅，画面清晰，光线自然。"


def read_frames(path, indices, max_side=None):
    """uint8 [len(indices), H, W, 3] RGB (decord if installed, else OpenCV), optionally shrunk to max_side."""
    try:
        from decord import VideoReader
        vr = VideoReader(path)
        frames = vr.get_batch([min(int(i), len(vr) - 1) for i in indices]).asnumpy()
    except ImportError:
        want, got, cap, i = sorted(set(indices)), {}, cv2.VideoCapture(path), 0
        while i <= want[-1]:
            ok, f = cap.read()
            if not ok:
                break
            if i in want:
                got[i] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            i += 1
        cap.release()
        last = got[max(got)]
        frames = np.stack([got.get(j, last) for j in indices])
    if max_side:
        s = min(1.0, max_side / max(frames.shape[1:3]))
        if s < 1:
            frames = np.stack([cv2.resize(f, (round(f.shape[1] * s), round(f.shape[0] * s)),
                                          interpolation=cv2.INTER_AREA) for f in frames])
    return frames


def video_info(path):
    """(frame count, fps); decord's count when installed (OpenCV's container count can be off by a few frames)."""
    try:
        from decord import VideoReader
        vr = VideoReader(path)
        return len(vr), float(vr.get_avg_fps()) or 24.0
    except ImportError:
        pass
    cap = cv2.VideoCapture(path)
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS) or 24.0
    cap.release()
    return n, fps


def pick_window(n_src, src_fps, fps, frames, ref_gap, rng):
    """(window source indices, reference source index) or None if the clip is too short."""
    step = src_fps / fps
    span = (frames - 1) * step
    if n_src - 1 < span:
        return None
    gap = ref_gap * src_fps
    for _ in range(10):
        start = rng.uniform(0, n_src - 1 - span)
        idx = [min(int(round(start + j * step)), n_src - 1) for j in range(frames)]
        far = [i for i in range(n_src) if (i < idx[0] and idx[0] - i >= gap)
               or (i > idx[-1] and i - idx[-1] >= gap)]
        if far:
            return idx, int(rng.choice(far))
    return None


def jitter(frames, scale, dx, dy):
    """The same zoom-out + shift on every frame (border replicated)."""
    h, w = frames.shape[1:3]
    m = np.float32([[scale, 0, (1 - scale) * w / 2 + dx * w], [0, scale, (1 - scale) * h / 2 + dy * h]])
    return np.stack([cv2.warpAffine(f, m, (w, h), borderMode=cv2.BORDER_REPLICATE) for f in frames])


def destroy_appearance(frames, generator=None, norm=False):
    """uint8 [n, H, W, 3] -> grey, blurred, random contrast (augment_driving), uint8. norm: every frame is also set to
    the same mean and spread of brightness, so lighting changes (strobes, a red stage turning light grey) carry no
    signal the model could copy; shapes and motion stay."""
    x = torch.from_numpy(frames).float().div(127.5).sub(1).permute(3, 0, 1, 2)
    y = augment_driving(x, generator)
    if norm:
        mean = y.mean(dim=(0, 2, 3), keepdim=True)
        std = y.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-3)
        y = ((y - mean) / std * 0.4).clamp(-1, 1)
    return ((y + 1) * 127.5).round().clamp(0, 255).byte().permute(1, 2, 3, 0).numpy()


def make_sample(path, a, rng, generator):
    """-> list of (target uint8 [F, H, W, 3], drive uint8 [F, H, W, 3], meta) for one clip."""
    n_src, src_fps = video_info(path)
    out = []
    for w in range(a.windows):
        picked = pick_window(n_src, src_fps, a.fps, a.frames, a.ref_gap, rng)
        if picked is None:
            break
        idx, ref = picked
        raw = read_frames(path, idx + [ref])
        raw = np.stack([cover(f, a.width, a.height) for f in raw])
        clip, ref_img = raw[:-1], raw[-1]
        target = np.concatenate([ref_img[None], clip[1:]])
        scale = rng.uniform(*a.scale)
        dx, dy = rng.uniform(-a.max_shift, a.max_shift, 2)
        drive = destroy_appearance(jitter(clip, scale, dx, dy), generator, getattr(a, "drive_norm", False))
        out.append((target, drive, dict(src=os.path.basename(path), window=w, start=int(idx[0]), ref=int(ref),
                                        src_fps=float(src_fps), fps=float(a.fps), scale=float(scale),
                                        dx=float(dx), dy=float(dy))))
    return out


def make_teacher_sample(job, a, rng, generator):
    """P2 pair from a teacher job (make_teacher_jobs.py + Wan-Animate-2): target = [reference B, teacher frames 1..],
    drive = A's clip with the same treatment as self-reconstruction. -> (target, drive, meta), or None if the
    teacher video is missing or too short."""
    teacher = os.path.join(job["out"], "results.mp4")
    if not os.path.exists(teacher):
        return None
    n = min(video_info(teacher)[0], video_info(job["drive"])[0], a.frames)
    n = (n - 1) // 4 * 4 + 1
    if n < 5:
        return None
    frames = lambda path: np.stack([cover(f, a.width, a.height) for f in read_frames(path, list(range(n)))])
    teacher_frames, drive_frames = frames(teacher), frames(job["drive"])
    ref = cover(cv2.cvtColor(cv2.imread(job["ref"]), cv2.COLOR_BGR2RGB), a.width, a.height)
    scale = rng.uniform(*a.scale)
    dx, dy = rng.uniform(-a.max_shift, a.max_shift, 2)
    drive = destroy_appearance(jitter(drive_frames, scale, dx, dy), generator, getattr(a, "drive_norm", False))
    meta = dict(src=f"teacher_{job['id']}", window=0, teacher=job["id"], src_a=job.get("src_a", ""),
                src_b=job.get("src_b", ""), scale=float(scale), dx=float(dx), dy=float(dy))
    return np.concatenate([ref[None], teacher_frames[1:]]), drive, meta


def reused_target(folder, name, meta, shape):
    """The target latents of the same sample in another cache (only the driving treatment changed), or None.
    Same clip, window start and reference frame (per-clip seeds make them deterministic) and the same latent shape."""
    path = os.path.join(folder, name) if folder else None
    if not path or not os.path.exists(path):
        return None
    old = torch.load(path, map_location="cpu", weights_only=True)
    same = all(old["meta"].get(k) == meta.get(k) for k in ("src", "window", "start", "ref", "teacher"))
    return old["target"] if same and tuple(old["target"].shape[1:]) == shape else None


def encode_text(a, device):
    from ti2v5b_replace import encode_prompts, encode_prompts_wan
    if a.t5_wan:
        text, neg = encode_prompts_wan(a.t5_wan, a.t5_tokenizer, [a.prompt, NEGATIVE], device)
    else:
        text, neg = encode_prompts(a.model, [a.prompt, NEGATIVE], device)
    return dict(text=text, neg=neg, prompt=a.prompt)


def atomic_save(value, path):
    """Publish a complete tensor cache file; temporary names are hidden from the sample glob."""
    tmp = os.path.join(os.path.dirname(path), f".tmp_{os.path.basename(path)}.{os.getpid()}")
    torch.save(value, tmp)
    os.replace(tmp, path)


def check_manifest(a):
    """Serialize shard startup and refuse incompatible or unversioned caches."""
    manifest = {k: getattr(a, k) for k in ("width", "height", "frames", "fps", "windows", "ref_gap",
                                          "max_shift", "prompt", "seed")}
    manifest["drive_norm"] = bool(getattr(a, "drive_norm", False))
    manifest["teacher"] = bool(getattr(a, "teacher", None))
    manifest.update(model=os.path.realpath(a.model), scale=list(a.scale),
                    t5_wan=os.path.realpath(a.t5_wan) if a.t5_wan else None,
                    t5_tokenizer=os.path.realpath(a.t5_tokenizer) if a.t5_tokenizer else None)
    path = os.path.join(a.out, "manifest.json")
    with open(os.path.join(a.out, ".manifest.lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if os.path.exists(path):
            with open(path) as fh:
                old = json.load(fh)
            if old != manifest:
                changed = [k for k in manifest if old.get(k) != manifest[k]]
                raise ValueError(f"cache manifest differs ({', '.join(changed)}): use a new cache directory")
        else:
            if glob.glob(os.path.join(a.out, "*.pt")):
                raise ValueError(f"cache {a.out} has no manifest: use a new cache directory")
            tmp = os.path.join(a.out, f".tmp_manifest.json.{os.getpid()}")
            with open(tmp, "w") as fh:
                json.dump(manifest, fh, indent=2)
                fh.write("\n")
            os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Wan2.2-TI2V-5B-Diffusers folder (vae/, text_encoder/, tokenizer/)")
    p.add_argument("--videos", help="folder of mp4 clips (self-reconstruction samples)")
    p.add_argument("--teacher", help="jobs.jsonl of make_teacher_jobs.py: build P2 pairs from the teacher videos instead")
    p.add_argument("--reuse-target", help="another cache of the same clips: copy its target latents (only the driving "
                                          "treatment changed), encode only the driving clip")
    p.add_argument("--list", help="only these clips (file names, one per line, e.g. filter_clips.py select output)")
    p.add_argument("--out", required=True)
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=832)
    p.add_argument("--frames", type=int, default=33, help="4k + 1")
    p.add_argument("--fps", type=float, default=24)
    p.add_argument("--windows", type=int, default=2, help="samples per clip")
    p.add_argument("--ref-gap", type=float, default=1.0, help="seconds between the window and the reference frame")
    p.add_argument("--scale", type=float, nargs=2, default=(0.85, 1.0), help="driving zoom-out range")
    p.add_argument("--max-shift", type=float, default=0.05, help="driving shift, fraction of width/height")
    p.add_argument("--drive-norm", action="store_true", help="per-frame brightness normalisation of the driving clip")
    p.add_argument("--prompt", default=PROMPT)
    p.add_argument("--t5-wan", help="Wan-format UMT5 checkpoint instead of the diffusers text_encoder")
    p.add_argument("--t5-tokenizer", help="umt5-xxl tokenizer dir (with --t5-wan)")
    p.add_argument("--shard", default="0/1", help="i/n: this process does every n-th clip from i")
    p.add_argument("--text-only", action="store_true", help="only write text.pt")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if (a.frames - 1) % 4:
        p.error("--frames must be 4k + 1")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if a.teacher:
        a.windows = 1                                    # one sample per teacher video
    os.makedirs(a.out, exist_ok=True)
    check_manifest(a)
    text_path = os.path.join(a.out, "text.pt")
    if not os.path.exists(text_path):
        if a.shard.split("/")[0] == "0" or a.text_only:
            atomic_save(encode_text(a, device), text_path)
            print(f"-> {text_path}", flush=True)
    if a.text_only:
        return

    i, n = (int(x) for x in a.shard.split("/"))
    if a.teacher:
        jobs = [json.loads(l) for l in open(a.teacher) if l.strip()][i::n]
        names = [f"t{j['id']}.job" for j in jobs]
    else:
        if not a.videos:
            p.error("--videos or --teacher is required")
        names = [l.strip() for l in open(a.list)] if a.list else sorted(os.path.basename(f) for f in
                                                                         glob.glob(os.path.join(a.videos, "*.mp4")))
        names = [x for x in names if x][i::n]
    from diffusers import AutoencoderKLWan
    from ti2v5b_replace import vae_encode
    vae = AutoencoderKLWan.from_pretrained(os.path.join(a.model, "vae"), torch_dtype=torch.float32).to(device)
    done = skipped = reused = 0
    for k, name in enumerate(names):
        stem = os.path.splitext(name)[0]
        if all(os.path.exists(os.path.join(a.out, f"{stem}_w{w}.pt")) for w in range(a.windows)):
            continue
        seed = a.seed * 1_000_003 + zlib.crc32(stem.encode()) % 1_000_003
        rng, g = np.random.default_rng(seed), torch.Generator().manual_seed(seed)
        try:
            if a.teacher:
                one = make_teacher_sample(jobs[k], a, rng, g)
                samples = [one] if one else []
            else:
                samples = make_sample(os.path.join(a.videos, name), a, rng, g)
        except Exception as e:  # unreadable clip: report and go on
            print(f"skip {name}: {e}", flush=True)
            skipped += 1
            continue
        if len(samples) < a.windows:
            why = "teacher video missing or too short" if a.teacher else "no separated reference/window pair"
            print(f"skip {name}: {why} for {a.windows - len(samples)} window(s)", flush=True)
            skipped += 1
        for target, drive, meta in samples:
            path = os.path.join(a.out, f"{stem}_w{meta['window']}.pt")
            if os.path.exists(path):
                continue
            shape = ((len(target) - 1) // 4 + 1, a.height // 16, a.width // 16)
            z_target = reused_target(a.reuse_target, os.path.basename(path), meta, shape)
            reused += z_target is not None
            if z_target is None:
                z_target = vae_encode(vae, target, device)[0].half().cpu()
            atomic_save(dict(target=z_target, drive=vae_encode(vae, drive, device)[0].half().cpu(), meta=meta), path)
            done += 1
        if (k + 1) % 20 == 0:
            print(f"shard {a.shard}: {k + 1}/{len(names)} clips, {done} samples, {skipped} skipped", flush=True)
    print(f"shard {a.shard} done: {done} samples ({reused} targets reused), {skipped} skipped -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
