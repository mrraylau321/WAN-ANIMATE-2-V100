"""P2 teacher-pair jobs: Wan-Animate-2 will make "person B does person A's motion" (infer/wan_animate_2_batch.py), so
the TI2V-5B LoRA sees training pairs where the driving person and the target person differ (self-reconstruction
cannot teach that: the driving clip there shows the target person).

Per job: A = a --frames window of one clip at --fps, written as its own mp4 at exactly that fps and size (so the
teacher and our prep read the same frames); B = one frame of a different clip, cover-cropped to the same size (move
mode keeps B's own scene). Writes <out>/drive/, <out>/ref/ and <out>/jobs.jsonl (teacher output dir per job).
  python train/make_teacher_jobs.py --videos <dir> --list pass.txt --out <work>/teacher --n 300
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
from prep_incontext_ti2v5b import read_frames, video_info  # noqa: E402
from ti2v5b_replace import cover, write_video  # noqa: E402


def window(n_src, src_fps, fps, frames, rng):
    """Source indices of a random --frames window at --fps, or None if the clip is too short."""
    step = src_fps / fps
    span = (frames - 1) * step
    if n_src - 1 < span:
        return None
    start = rng.uniform(0, n_src - 1 - span)
    return [min(int(round(start + j * step)), n_src - 1) for j in range(frames)]


def make_job(k, clip_a, clip_b, a, rng):
    """Write job k's inputs; -> job dict or None (a clip too short)."""
    n_a, fps_a = video_info(clip_a)
    idx = window(n_a, fps_a, a.fps, a.frames, rng)
    if idx is None:
        return None
    n_b, _ = video_info(clip_b)
    t_b = int(rng.integers(0, n_b))
    drive = np.stack([cover(f, a.width, a.height) for f in read_frames(clip_a, idx)])
    ref = cover(read_frames(clip_b, [t_b])[0], a.width, a.height)
    jid = f"{k:05d}"
    paths = {d: os.path.join(a.out, d) for d in ("drive", "ref", "teacher")}
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    drive_path, ref_path = os.path.join(paths["drive"], f"{jid}.mp4"), os.path.join(paths["ref"], f"{jid}.png")
    write_video(drive_path, drive, a.fps)
    cv2.imwrite(ref_path, cv2.cvtColor(ref, cv2.COLOR_RGB2BGR))
    return dict(id=jid, drive=drive_path, ref=ref_path, out=os.path.join(paths["teacher"], jid),
                src_a=os.path.basename(clip_a), start_a=int(idx[0]), src_b=os.path.basename(clip_b), frame_b=t_b)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos", required=True)
    p.add_argument("--list", required=True, help="clip names, one per line (e.g. filter_clips.py select output)")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=300, help="number of pairs")
    p.add_argument("--frames", type=int, default=33, help="4k + 1, the TI2V training window")
    p.add_argument("--fps", type=float, default=24)
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=832)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    names = [l.strip() for l in open(a.list) if l.strip()]
    if len(names) < 2:
        p.error("need at least 2 clips")
    os.makedirs(a.out, exist_ok=True)
    jobs_path = os.path.join(a.out, "jobs.jsonl")
    done = {json.loads(l)["id"] for l in open(jobs_path)} if os.path.exists(jobs_path) else set()
    k = made = 0
    with open(jobs_path, "a") as fh:
        while made < a.n and k < 10 * a.n:
            if f"{k:05d}" in done:                            # resume: jobs already written
                made += 1
            else:
                rng = np.random.default_rng([a.seed, k])      # per-job RNG: the same seed gives the same jobs
                i, j = rng.choice(len(names), 2, replace=False)
                job = make_job(k, os.path.join(a.videos, names[i]), os.path.join(a.videos, names[j]), a, rng)
                if job:
                    fh.write(json.dumps(job) + "\n")
                    fh.flush()
                    made += 1
            k += 1
    print(f"{made} jobs -> {jobs_path}")


if __name__ == "__main__":
    main()
