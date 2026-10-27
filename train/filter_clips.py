"""Pick clips for the in-context pilot: one main person, big enough in the frame, and actually moving.

score   Grounding DINO "person" boxes on --samples frames per clip; the motion is the mean grey-level change inside the
        main person's box between each sampled frame and the frame --gap seconds later (static cameras: the background
        does not change, so this is body motion)
          python train/filter_clips.py score --videos <dir> --gdino <grounding-dino-base dir> --out scores_0.csv --shard 0/4
select  thresholds on one or more score files -> the file names that pass
          python train/filter_clips.py select --scores scores_*.csv --out pass.txt
"""
import argparse
import csv
import glob
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from prep_incontext_ti2v5b import read_frames, video_info  # noqa: E402

FIELDS = ["name", "frames", "fps", "person_frac", "multi_frac", "height", "motion"]


def nms(boxes, scores, iou=0.5):
    """Indices of the boxes kept by greedy non-maximum suppression (Grounding DINO often boxes one person twice)."""
    order, keep = list(np.argsort(-scores)), []
    while order:
        i = order.pop(0)
        keep.append(i)
        rest = []
        for j in order:
            x1, y1 = max(boxes[i, 0], boxes[j, 0]), max(boxes[i, 1], boxes[j, 1])
            x2, y2 = min(boxes[i, 2], boxes[j, 2]), min(boxes[i, 3], boxes[j, 3])
            inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
            if inter / (area(boxes[i]) + area(boxes[j]) - inter + 1e-9) <= iou:
                rest.append(j)
        order = rest
    return keep


def frame_stats(boxes, scores, shape, min_area=0.05):
    """(number of people covering >= min_area of the frame, main box or None) for one frame."""
    if not len(boxes):
        return 0, None
    boxes = np.asarray(boxes, float)[nms(np.asarray(boxes, float), np.asarray(scores, float))]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) / (shape[0] * shape[1])
    return int((areas >= min_area).sum()), boxes[int(np.argmax(areas))]


def box_motion(a, b, box, pad=0.1):
    """Mean |grey(a) - grey(b)| / 255 inside the box (padded by pad of its size), uint8 RGB frames."""
    h, w = a.shape[:2]
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1 = int(max(0, x1 - px)), int(max(0, y1 - py))
    x2, y2 = int(min(w, x2 + px)), int(min(h, y2 + py))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    ga = cv2.cvtColor(a[y1:y2, x1:x2], cv2.COLOR_RGB2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b[y1:y2, x1:x2], cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(np.abs(ga - gb).mean() / 255)


def score_clip(path, detector, samples=8, gap=0.5, max_side=640, min_area=0.05):
    """detector.boxes(img, "person") -> (boxes xyxy, scores). Returns a FIELDS row."""
    n, fps = video_info(path)
    d = max(1, int(round(gap * fps)))
    if n <= d + 1:
        raise ValueError(f"{n} frames: too short")
    idx = np.linspace(0, n - 1 - d, samples).round().astype(int).tolist()
    frames = read_frames(path, idx + [i + d for i in idx], max_side)
    people, heights, motions = [], [], []
    for k in range(samples):
        f, g = frames[k], frames[samples + k]
        count, main = frame_stats(*detector.boxes(f, "person"), f.shape, min_area)
        people.append(count)
        heights.append(0.0 if main is None else (main[3] - main[1]) / f.shape[0])
        motions.append(0.0 if main is None else box_motion(f, g, main))
    people = np.array(people)
    return dict(name=os.path.basename(path), frames=n, fps=round(fps, 3),
                person_frac=round(float((people >= 1).mean()), 3), multi_frac=round(float((people >= 2).mean()), 3),
                height=round(float(np.median(heights)), 3), motion=round(float(np.median(motions)), 4))


def select(rows, min_person=0.75, max_multi=0.25, min_height=0.4, drop_stillest=0.35, log=print):
    """Names of the rows that pass; drop_stillest = the fraction of the remaining clips with the least motion."""
    a = [r for r in rows if float(r["person_frac"]) >= min_person]
    b = [r for r in a if float(r["multi_frac"]) <= max_multi]
    c = [r for r in b if float(r["height"]) >= min_height]
    cut = float(np.quantile([float(r["motion"]) for r in c], drop_stillest)) if c and drop_stillest > 0 else -1
    d = [r for r in c if float(r["motion"]) > cut] if cut >= 0 else c
    log(f"{len(rows)} clips -> person {len(a)} -> single {len(b)} -> height {len(c)} -> "
        f"motion > {cut:.4f}: {len(d)}")
    return [r["name"] for r in d]


def cmd_score(a):
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
    from grounded_sam2 import Detector
    det = Detector(a.gdino, a.box_threshold, a.text_threshold)
    names = sorted(os.path.basename(f) for f in glob.glob(os.path.join(a.videos, "*.mp4")))
    i, n = (int(x) for x in a.shard.split("/"))
    names = names[i::n]
    done = set()
    if os.path.exists(a.out):                                  # resume
        done = {r["name"] for r in csv.DictReader(open(a.out))}
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        if new:
            w.writeheader()
        for k, name in enumerate(names):
            if name in done:
                continue
            try:
                w.writerow(score_clip(os.path.join(a.videos, name), det, a.samples, a.gap))
            except Exception as e:
                print(f"skip {name}: {e}", flush=True)
            fh.flush()
            if (k + 1) % 50 == 0:
                print(f"shard {a.shard}: {k + 1}/{len(names)}", flush=True)
    print(f"shard {a.shard} done -> {a.out}", flush=True)


def cmd_select(a):
    rows = [r for f in a.scores for r in csv.DictReader(open(f))]
    names = select(rows, a.min_person, a.max_multi, a.min_height, a.drop_stillest)
    with open(a.out, "w") as fh:
        fh.write("\n".join(names) + "\n")
    print(f"{len(names)} clips -> {a.out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--videos", required=True)
    s.add_argument("--out", required=True, help="csv (appended to; finished clips are skipped)")
    s.add_argument("--gdino", default="IDEA-Research/grounding-dino-base", help="HF id or local dir")
    s.add_argument("--box-threshold", type=float, default=0.35)
    s.add_argument("--text-threshold", type=float, default=0.25)
    s.add_argument("--samples", type=int, default=8)
    s.add_argument("--gap", type=float, default=0.5, help="seconds between the two frames of a motion pair")
    s.add_argument("--shard", default="0/1")
    s.set_defaults(fn=cmd_score)
    s = sub.add_parser("select")
    s.add_argument("--scores", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--min-person", type=float, default=0.75, help="share of sampled frames with a person")
    s.add_argument("--max-multi", type=float, default=0.25, help="max share of frames with 2+ people >= 5%% of frame")
    s.add_argument("--min-height", type=float, default=0.4, help="median main-box height / frame height")
    s.add_argument("--drop-stillest", type=float, default=0.35)
    s.set_defaults(fn=cmd_select)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
