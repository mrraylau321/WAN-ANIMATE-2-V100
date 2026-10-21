"""Grounded-SAM2 helpers for the experimental replace mode (transformers only, no extra pip packages):
a text target -> Grounding DINO box -> SAM2 pixel masks (tracked through the video).

  video:      masks of the target in every frame of the driving video
      python scripts/grounded_sam2.py video --video inputs/drive.mp4 --target person
      -> inputs/drive_target_mask.npz (uint8 0/255 [N, H, W]) + inputs/drive_target_mask_preview.mp4

  scene-ref:  the reference character pasted onto the driving video's own background (a clean plate:
              the per-pixel median of the frames where the target is absent), so the reference image
              brings no foreign background (e.g. a white studio backdrop) into the generation
      python scripts/grounded_sam2.py scene-ref --ref inputs/ref.png --ref-target cat \\
          --video inputs/drive.mp4 --masks inputs/drive_target_mask.npz
      -> inputs/ref_scene.png (+ inputs/ref_scene_plate.png)

Models (download once, e.g. into the checkpoint folder; HF_ENDPOINT=https://hf-mirror.com if needed):
    hf download IDEA-Research/grounding-dino-base --local-dir <dir>/grounding-dino-base
    hf download facebook/sam2.1-hiera-large --local-dir <dir>/sam2.1-hiera-large
"""
import argparse
import os

import cv2
import numpy as np
import torch
from decord import VideoReader
try:
    import imageio.v2 as imageio
except ImportError:  # older imageio
    import imageio
from PIL import Image
from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2VideoModel,
                          Sam2VideoProcessor)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def write_preview(path, frames, fps):
    """H.264 mp4 (plays in VS Code / browsers; OpenCV's mp4v does not). Even size for yuv420p."""
    writer = imageio.get_writer(path, fps=fps, codec="libx264", macro_block_size=1, ffmpeg_params=["-crf", "23"])
    for f in frames:
        writer.append_data(f[: f.shape[0] // 2 * 2, : f.shape[1] // 2 * 2])
    writer.close()


def shrink(img, max_side):
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    return img if s == 1.0 else cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


class Detector:
    """Grounding DINO: the box of a text target in one image."""

    def __init__(self, path, box_threshold, text_threshold):
        self.proc = AutoProcessor.from_pretrained(path)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(path).eval().to(DEVICE)
        self.box_threshold, self.text_threshold = box_threshold, text_threshold

    @torch.no_grad()
    def boxes(self, img, target):
        text = target.strip().lower().rstrip(".") + "."
        inputs = self.proc(images=Image.fromarray(img), text=text, return_tensors="pt").to(DEVICE)
        res = self.proc.post_process_grounded_object_detection(
            self.model(**inputs), inputs.input_ids, threshold=self.box_threshold,
            text_threshold=self.text_threshold, target_sizes=[img.shape[:2]])[0]
        return res["boxes"].cpu().numpy(), res["scores"].cpu().numpy()

    def best(self, img, target, pick="largest"):
        boxes, scores = self.boxes(img, target)
        if not len(boxes):
            return None
        key = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) if pick == "largest" else scores
        return boxes[int(np.argmax(key))].tolist()


class Segmenter:
    """SAM2 (video model): pixel masks from a box, tracked forwards and backwards through the frames."""

    def __init__(self, path):
        self.proc = Sam2VideoProcessor.from_pretrained(path)
        self.model = Sam2VideoModel.from_pretrained(path).eval().to(DEVICE)

    def _mask(self, out, size):
        m = self.proc.post_process_masks([out.pred_masks], original_sizes=[list(size)], binarize=True)[0]
        while m.dim() > 2:
            m = m[0]
        return m.cpu().numpy().astype(bool)

    @torch.no_grad()
    def track(self, frames, frame_idx, box):
        """frames: list of HxWx3 uint8. Returns one bool mask per frame."""
        size = frames[0].shape[:2]
        sess = self.proc.init_video_session(video=frames, inference_device=DEVICE, video_storage_device="cpu",
                                            dtype=torch.float32)
        self.proc.add_inputs_to_inference_session(inference_session=sess, frame_idx=frame_idx, obj_ids=1,
                                                  input_boxes=[[box]])
        masks = [None] * len(frames)
        masks[frame_idx] = self._mask(self.model(inference_session=sess, frame_idx=frame_idx), size)
        for reverse in (False, True):
            if reverse and frame_idx == 0:
                break
            for out in self.model.propagate_in_video_iterator(sess, start_frame_idx=frame_idx, reverse=reverse):
                masks[out.frame_idx] = self._mask(out, size)
        empty = np.zeros(size, bool)
        return [m if m is not None else empty for m in masks]


def main_object(mask):
    """Keep the main object only: parts closer than ~3% of the frame height are grouped first (a hand
    split off by motion blur stays), then the largest group wins (a bystander's arm is dropped)."""
    r = max(3, int(0.03 * mask.shape[0]))
    grouped = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)
    if n <= 2:
        return mask
    return mask & (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))


def read_frames(path, max_side):
    reader = VideoReader(path)
    full = reader[0].asnumpy().shape[:2]
    frames = [shrink(reader[i].asnumpy(), max_side) for i in range(len(reader))]
    return reader, full, frames


def cmd_video(a):
    out = a.out or os.path.splitext(a.video)[0] + "_target_mask.npz"
    preview = os.path.splitext(out)[0] + "_preview.mp4"
    reader, (H, W), frames = read_frames(a.video, a.max_side)
    det = Detector(a.gdino, a.box_threshold, a.text_threshold)
    f0, box = None, None
    for i in range(0, len(frames), max(1, a.search_step)):  # first frame where the target is found
        box = det.best(frames[i], a.target, a.pick)
        if box is not None:
            f0 = i
            break
    if box is None:
        raise SystemExit(f"Grounding DINO found no '{a.target}' in {a.video} (try --box-threshold lower)")
    print(f"'{a.target}' found on frame {f0}: box {[round(v) for v in box]}")
    del det
    small = Segmenter(a.sam2).track(frames, f0, box)
    if not a.keep_all:
        small = [main_object(m) for m in small]
    masks = np.stack([(cv2.resize(m.astype(np.uint8) * 255, (W, H), interpolation=cv2.INTER_LINEAR) > 127)
                      .astype(np.uint8) * 255 for m in small])
    np.savez_compressed(out, masks=masks)
    def overlay(i):
        frame = reader[i].asnumpy().copy()
        sel = masks[i] > 0
        frame[sel] = (0.5 * frame[sel] + [127, 0, 0]).astype(np.uint8)
        return frame
    write_preview(preview, (overlay(i) for i in range(len(masks))), reader.get_avg_fps())
    empty = int(np.sum(masks.reshape(len(masks), -1).max(1) == 0))
    print(f"{len(masks)} masks -> {out}  (target area {np.mean(masks > 0):.1%}; empty frames: {empty})\n"
          f"preview -> {preview}")


def clean_plate(video, masks, max_side=960, samples=60, dilate=15, min_seen=0.2):
    """Per-pixel median of the frames where the (dilated) target is absent. Frames are sampled evenly
    over the whole video; a pixel seen in fewer than ``min_seen`` of them (e.g. always behind a dancer
    who barely moves) is filled from its well-seen neighbourhood instead, so no ghost of the person
    survives. Returns (plate, fraction of pixels that had to be filled)."""
    reader = VideoReader(video)
    n = len(reader)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1,) * 2)
    stack = []
    for i in np.unique(np.linspace(0, n - 1, min(n, samples)).round().astype(int)):
        f = shrink(reader[int(i)].asnumpy(), max_side).astype(np.float32)
        m = cv2.dilate((masks[int(i)] > 127).astype(np.uint8), kernel)
        m = cv2.resize(m, (f.shape[1], f.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        f[m] = np.nan
        stack.append(f)
    stack = np.stack(stack)
    seen = np.isfinite(stack[..., 0]).sum(0)
    with np.errstate(all="ignore"):
        plate = np.nanmedian(stack, axis=0)
    hole = seen < max(1, min_seen * len(stack))
    if hole.any():  # normalized-convolution fill from the well-seen neighbourhood, widening until done
        known = (~hole).astype(np.float32)
        val = np.nan_to_num(plate) * known[..., None]
        todo = hole.copy()
        k = 15
        while todo.any() and known.any():
            num = cv2.blur(val, (k, k))
            den = cv2.blur(known, (k, k))
            ok = todo & (den > 1e-3)
            plate[ok] = (num / np.maximum(den, 1e-3)[..., None])[ok]
            todo &= ~ok
            k = 2 * k + 1
    return np.clip(np.nan_to_num(plate), 0, 255).astype(np.uint8), float(hole.mean())


def cover(img, w, h):
    """Resize + centre-crop to exactly w x h (keeps the aspect ratio)."""
    s = max(w / img.shape[1], h / img.shape[0])
    r = cv2.resize(img, (max(w, round(img.shape[1] * s)), max(h, round(img.shape[0] * s))),
                   interpolation=cv2.INTER_AREA)
    y, x = (r.shape[0] - h) // 2, (r.shape[1] - w) // 2
    return r[y:y + h, x:x + w]


def cmd_scene_ref(a):
    out = a.out or os.path.splitext(a.ref)[0] + "_scene.png"
    ref = cv2.cvtColor(cv2.imread(a.ref), cv2.COLOR_BGR2RGB)
    small = shrink(ref, a.max_side)
    box = Detector(a.gdino, a.box_threshold, a.text_threshold).best(small, a.ref_target, a.pick)
    if box is None:
        raise SystemExit(f"Grounding DINO found no '{a.ref_target}' in {a.ref}")
    m = Segmenter(a.sam2).track([small], 0, box)[0]
    alpha = cv2.resize(m.astype(np.float32), (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_LINEAR)
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)[..., None]
    plate, unseen = clean_plate(a.video, np.load(a.masks)["masks"])
    bg = cover(plate, ref.shape[1], ref.shape[0]).astype(np.float32)
    comp = (ref * alpha + bg * (1 - alpha)).astype(np.uint8)
    cv2.imwrite(out, cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.splitext(out)[0] + "_plate.png", cv2.cvtColor(plate, cv2.COLOR_RGB2BGR))
    print(f"'{a.ref_target}' cut out of {a.ref} ({float(alpha.mean()):.0%} of the image) and placed on the "
          f"video's background (plate pixels never visible: {unseen:.1%}) -> {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("video", "scene-ref"):
        s = sub.add_parser(name)
        s.add_argument("--gdino", default="IDEA-Research/grounding-dino-base", help="HF id or local dir")
        s.add_argument("--sam2", default="facebook/sam2.1-hiera-large", help="HF id or local dir")
        s.add_argument("--box-threshold", type=float, default=0.35)
        s.add_argument("--text-threshold", type=float, default=0.25)
        s.add_argument("--pick", choices=["largest", "score"], default="largest",
                       help="which detection to follow when several match")
        s.add_argument("--max-side", type=int, default=1024, help="frames are processed at this size")
        s.add_argument("--video", required=True)
        s.add_argument("--out")
        if name == "video":
            s.add_argument("--target", default="person", help="what to replace, in words")
            s.add_argument("--search-step", type=int, default=4, help="frames skipped while looking for it")
            s.add_argument("--keep-all", action="store_true", help="keep every tracked part, not just the main one")
        else:
            s.add_argument("--ref", required=True)
            s.add_argument("--ref-target", required=True, help="the character in the reference image")
            s.add_argument("--masks", required=True, help="target masks of the video (from the video command)")
    a = p.parse_args()
    cmd_video(a) if a.cmd == "video" else cmd_scene_ref(a)


if __name__ == "__main__":
    main()
