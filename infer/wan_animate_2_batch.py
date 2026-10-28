# Batch runner for the V100 port: many Wan-Animate-2 jobs with ONE loaded model (P2 teacher pairs, train/).
"""Same model/precision options as wan_animate_2_demo.py, plus --jobs: a jsonl file with one job per line
{"id", "ref", "drive", "out"}; each job writes <out>/results.mp4 (move mode: the reference does the driving motion).
Finished jobs are skipped, so rerunning the same command resumes. Launched by train/run_teacher_pairs.sh."""
import argparse
import json
import os
import time

from core import build_object_from_config_file


def pending(jobs):
    return [j for j in jobs if not os.path.exists(os.path.join(j["out"], "results.mp4"))]


def run_jobs(pipeline, jobs, a, log=print):
    todo = pending(jobs)
    log(f"{len(jobs)} jobs, {len(jobs) - len(todo)} done, {len(todo)} to run")
    t0 = time.time()
    for k, job in enumerate(todo, 1):
        os.makedirs(job["out"], exist_ok=True)
        pipeline(refer_img_path=job["ref"], tpl_video_path=job["drive"], output_path=job["out"], width=a.width,
                 height=a.height, fps=a.fps, seed=a.seed + int(job["id"]), clip_len=a.clip_len,
                 sample_guide_scale=a.sample_guide_scale, step=a.step, prompt=a.prompt, prompt_ref=a.prompt_ref)
        rate = (time.time() - t0) / k
        log(f"[{k}/{len(todo)}] {job['id']} done, {rate:.0f} s/job, ~{rate * (len(todo) - k) / 3600:.1f} h left")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jobs", required=True)
    p.add_argument("--config", default="./wan_animate_2_distillation_v100.yaml")
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=832)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--clip_len", type=int, default=33)
    p.add_argument("--sample_guide_scale", type=float, default=1.0)
    p.add_argument("--step", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt", default="视频中的人在做动作，背景静止")
    p.add_argument("--prompt_ref", default="人物动作的参考视频")
    p.add_argument("--dtype", default=None, choices=["auto", "fp16", "bf16", "fp32"])
    p.add_argument("--cfg-override", nargs="*", default=[])
    a = p.parse_args()
    jobs = [json.loads(l) for l in open(a.jobs) if l.strip()]
    overrides = list(a.cfg_override) + ([f"test_cfg.dit_dtype={a.dtype}"] if a.dtype else [])
    if not pending(jobs):
        print(f"all {len(jobs)} jobs done")
        return
    pipeline = build_object_from_config_file(a.config, overrides)
    try:
        run_jobs(pipeline, jobs, a, log=lambda m: print(m, flush=True))
    finally:  # stop the GPU workers so the process exits
        if hasattr(pipeline, "close"):
            pipeline.close()


if __name__ == "__main__":
    main()
