"""Padding variant of the experimental replace mode: the anchor strip of real video sits OUTSIDE the output.

    pad:  python scripts/pad_replace.py pad --video drive.mp4 --frac 0.05 --out-prefix inputs/drive_pad
          -> inputs/drive_pad.mp4      the driving video with a mirrored (reflect) border on every side
             inputs/drive_pad_hole.npz hole masks: 255 = the original frame (generated), 0 = the border (kept)
             inputs/drive_pad.json     border size, to crop it off afterwards
    crop: python scripts/pad_replace.py crop --video results.mp4 --meta inputs/drive_pad.json
          -> results_crop.mp4          the border removed: the original framing, fully generated

Run the replace pipeline on the padded video with MASK=<hole.npz> MASK_MODE=dilate DILATE=0 GROW=0, so the
mirrored border stays known (scene anchor) and everything inside is generated; the seam between the two
then lies in the part that is cropped off. Build a scene reference (grounded_sam2.py scene-ref) from the
ORIGINAL video first: its clean plate needs the original person masks.
"""
import argparse
import json
import os

import cv2
import numpy as np

try:
    import imageio.v2 as imageio
except ImportError:  # older imageio
    import imageio


def write_video(path, frames, fps, crf=12):
    writer = imageio.get_writer(path, fps=fps, codec="libx264", macro_block_size=1, ffmpeg_params=["-crf", str(crf)])
    for f in frames:
        writer.append_data(f[: f.shape[0] // 2 * 2, : f.shape[1] // 2 * 2])
    writer.close()


def cmd_pad(a):
    reader = imageio.get_reader(a.video)
    fps = reader.get_meta_data()["fps"]
    frames = [f for f in reader]
    h, w = frames[0].shape[:2]
    p = max(1, int(round(a.frac * min(h, w))))
    padded = [cv2.copyMakeBorder(f, p, p, p, p, cv2.BORDER_REFLECT_101) for f in frames]
    ph, pw = padded[0].shape[:2]
    hole = np.zeros((ph // 2 * 2, pw // 2 * 2), np.uint8)
    hole[p:p + h, p:p + w] = 255
    write_video(a.out_prefix + ".mp4", padded, fps)
    np.savez_compressed(a.out_prefix + "_hole.npz", masks=np.repeat(hole[None], len(frames), axis=0))
    meta = {"pad": p, "height": h, "width": w, "padded_height": ph // 2 * 2, "padded_width": pw // 2 * 2}
    with open(a.out_prefix + ".json", "w") as fh:
        json.dump(meta, fh)
    print(f"{len(frames)} frames {w}x{h} -> {meta['padded_width']}x{meta['padded_height']} (border {p} px)\n"
          f"  -> {a.out_prefix}.mp4, {a.out_prefix}_hole.npz, {a.out_prefix}.json")


def crop_box(meta, out_h, out_w):
    """The original frame inside a (resized) padded frame, in output pixels."""
    sy, sx = out_h / meta["padded_height"], out_w / meta["padded_width"]
    y0, x0 = int(round(meta["pad"] * sy)), int(round(meta["pad"] * sx))
    return y0, x0, y0 + int(round(meta["height"] * sy)), x0 + int(round(meta["width"] * sx))


def cmd_crop(a):
    with open(a.meta) as fh:
        meta = json.load(fh)
    reader = imageio.get_reader(a.video)
    fps = reader.get_meta_data()["fps"]
    frames = [f for f in reader]
    y0, x0, y1, x1 = crop_box(meta, *frames[0].shape[:2])
    out = a.out or os.path.splitext(a.video)[0] + "_crop.mp4"
    write_video(out, [f[y0:y1, x0:x1] for f in frames], fps, crf=16)
    print(f"cropped {frames[0].shape[1]}x{frames[0].shape[0]} -> {x1 - x0}x{y1 - y0}: {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("pad")
    s.add_argument("--video", required=True)
    s.add_argument("--frac", type=float, default=0.05, help="border per side, as a fraction of the shorter side")
    s.add_argument("--out-prefix", required=True)
    s = sub.add_parser("crop")
    s.add_argument("--video", required=True)
    s.add_argument("--meta", required=True)
    s.add_argument("--out")
    a = p.parse_args()
    cmd_pad(a) if a.cmd == "pad" else cmd_crop(a)


if __name__ == "__main__":
    main()
