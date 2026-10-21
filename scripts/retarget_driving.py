"""Retarget the driving video for the experimental replace mode: rescale the person around the feet so
the new character is drawn smaller (or larger), and write the hole mask that covers BOTH the original
person (must be removed) and the rescaled one (must be drawn).

    python scripts/retarget_driving.py --video inputs/drive.mp4 --masks inputs/drive_person_mask.npz --scale 0.6
    -> inputs/drive_retarget_s0.6.mp4          driving video for the model (motion source only)
       inputs/drive_hole_s0.6.npz              hole masks (use as MASK)
       inputs/drive_retarget_s0.6_preview.jpg  original | retargeted | hole

A single reference image carries no real-world size, so the scale is chosen by the user
(0.5 = the character is half the height of the driving person). The original video stays the
background: pass it as test_cfg.replace_background_video (run_replace_final.sh does this).
"""
import argparse
import os

import cv2
import numpy as np
from decord import VideoReader

try:
    import imageio.v2 as imageio
except ImportError:  # older imageio
    import imageio


def smooth(values, win=9):
    """Centered moving median: keeps the feet anchor from jittering frame to frame."""
    v = np.asarray(values, dtype=np.float64)
    h = win // 2
    return np.array([np.median(v[max(0, i - h):i + h + 1]) for i in range(len(v))])


def main_person(mask):
    """Keep only the main person: parts closer than ~3% of the frame height are grouped first (a hand
    split off by motion blur stays with the body), then the largest group wins (a bystander's arm
    stays untouched)."""
    r = max(3, int(0.03 * mask.shape[0]))
    grouped = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)
    if n <= 2:
        return mask
    return mask & (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))


def fill_background(frame, mask, scale=8, k=25):
    """Cover the person with a smooth estimate of the surrounding background (normalized convolution at
    1/scale resolution). Only the driving branch sees this, so a plausible blur is enough."""
    h, w = mask.shape
    sw, sh = max(1, w // scale), max(1, h // scale)
    f = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
    keep = cv2.resize((~mask).astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
    num = cv2.blur(f * keep[..., None], (k, k))
    den = cv2.blur(keep, (k, k))[..., None]
    est = cv2.resize(num / np.maximum(den, 1e-3), (w, h), interpolation=cv2.INTER_LINEAR)
    out = frame.copy()
    out[mask] = np.clip(est[mask], 0, 255).astype(np.uint8)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--masks", required=True, help="person masks .npz from scripts/make_person_mask.py")
    p.add_argument("--scale", type=float, required=True, help="character height / driving person height")
    p.add_argument("--dilate", type=int, default=7, help="px added around the person before cutting it out")
    p.add_argument("--out-prefix", help="output path prefix (default: next to the video)")
    a = p.parse_args()
    if not 0.2 <= a.scale <= 3.0:
        raise SystemExit("--scale must be between 0.2 and 3.0")
    stem = a.out_prefix or os.path.splitext(a.video)[0]
    out_video = f"{stem}_retarget_s{a.scale:g}.mp4"
    out_hole = f"{stem}_hole_s{a.scale:g}.npz"
    preview = f"{stem}_retarget_s{a.scale:g}_preview.jpg"

    reader = VideoReader(a.video)
    n, fps = len(reader), reader.get_avg_fps()
    masks = np.load(a.masks)["masks"]
    if len(masks) != n:
        raise SystemExit(f"{a.masks} has {len(masks)} masks but {a.video} has {n} frames")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.dilate + 1,) * 2) if a.dilate > 0 else None

    # feet anchor per frame: bottom-centre of the person box, smoothed over time
    boxes, last = [], None
    masks = [main_person(m > 127) for m in masks]
    for m in masks:
        ys, xs = np.nonzero(m)
        if len(ys):
            last = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
        boxes.append(last)
    first = next((b for b in boxes if b is not None), None)
    if first is None:
        raise SystemExit("no person found in any mask")
    boxes = [b if b is not None else first for b in boxes]
    ax = smooth([(b[0] + b[2]) / 2 for b in boxes])
    ay = smooth([b[3] for b in boxes])

    writer = imageio.get_writer(out_video, fps=fps, codec="libx264", macro_block_size=1,
                                ffmpeg_params=["-crf", "12"])
    holes = []
    for i in range(n):
        frame = reader[i].asnumpy()
        h, w = frame.shape[:2]
        person = masks[i]
        cut = cv2.dilate(person.astype(np.uint8), kernel) > 0 if kernel is not None else person
        plate = fill_background(frame, cut)
        # scale the person (with a soft alpha) around the feet anchor
        x0, y0, x1, y1 = boxes[i]
        alpha = cv2.GaussianBlur(cut.astype(np.float32), (7, 7), 0)
        M = np.float32([[a.scale, 0, ax[i] * (1 - a.scale)], [0, a.scale, ay[i] * (1 - a.scale)]])
        interp = cv2.INTER_AREA if a.scale < 1 else cv2.INTER_LINEAR
        moved = cv2.warpAffine(frame, M, (w, h), flags=interp, borderMode=cv2.BORDER_CONSTANT)
        moved_a = cv2.warpAffine(alpha, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)[..., None]
        out = (plate * (1 - moved_a) + moved * moved_a).astype(np.uint8)
        writer.append_data(out)
        hole = cut | (moved_a[..., 0] > 0.5)  # remove the original person AND draw the rescaled one
        holes.append((hole * 255).astype(np.uint8))
        if i == n // 2:
            ov = out.copy()
            ov[hole] = (0.5 * ov[hole] + [127, 0, 0]).astype(np.uint8)
            cv2.imwrite(preview, cv2.cvtColor(np.concatenate([frame, out, ov], axis=1), cv2.COLOR_RGB2BGR))
    writer.close()
    np.savez_compressed(out_hole, masks=np.stack(holes))

    check = len(VideoReader(out_video))
    if check != n:
        raise SystemExit(f"re-encoded video has {check} frames, expected {n}")
    print(f"scale {a.scale:g}: {n} frames @ {fps:.2f} fps\n  driving (motion only) -> {out_video}\n"
          f"  hole masks -> {out_hole}\n  preview -> {preview}")


if __name__ == "__main__":
    main()
