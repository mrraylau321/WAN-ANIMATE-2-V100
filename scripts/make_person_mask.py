"""Person masks for the experimental replace mode: one mask per frame of the driving video, decoded with
decord exactly like the pipeline, saved as uint8 0/255 [N, H, W] in an .npz, plus a red-overlay preview.

    python scripts/make_person_mask.py --video inputs/drive_2s.mp4 [--model <HF id or local dir>]
    -> inputs/drive_2s_person_mask.npz  and  inputs/drive_2s_person_mask_preview.mp4

Segmenter: a Hugging Face SegFormer (transformers, already installed). Default: the human-parsing model
mattmdjaga/segformer_b2_clothes (every non-background label = person). A model whose labels include
"person" (e.g. an ADE20K SegFormer) also works. Download once, e.g.:
    hf download mattmdjaga/segformer_b2_clothes config.json preprocessor_config.json model.safetensors \
        --local-dir <CKPT_DIR>/segformer_b2_clothes          (HF_ENDPOINT=https://hf-mirror.com if needed)
"""
import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation


def person_ids(id2label):
    names = {int(i): str(n).lower() for i, n in id2label.items()}
    ids = [i for i, n in names.items() if "person" in n]
    return ids or [i for i, n in names.items() if n not in ("background", "bg")]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--out", help="default: <video>_person_mask.npz")
    p.add_argument("--model", default="mattmdjaga/segformer_b2_clothes", help="HF model id or local dir")
    p.add_argument("--batch", type=int, default=8)
    a = p.parse_args()
    out = a.out or os.path.splitext(a.video)[0] + "_person_mask.npz"
    preview = os.path.splitext(out)[0] + "_preview.mp4"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(a.model)
    model = AutoModelForSemanticSegmentation.from_pretrained(a.model).eval().to(device)
    ids = torch.tensor(person_ids(model.config.id2label), device=device)
    print(f"segmenter {a.model}: person labels {ids.tolist()}")

    reader = VideoReader(a.video)
    n = len(reader)
    masks = []
    with torch.no_grad():
        for s in range(0, n, a.batch):
            frames = reader.get_batch(list(range(s, min(s + a.batch, n)))).asnumpy()  # B, H, W, 3 RGB
            h, w = frames.shape[1:3]
            inputs = processor(images=list(frames), return_tensors="pt").to(device)
            logits = F.interpolate(model(**inputs).logits, size=(h, w), mode="bilinear", align_corners=False)
            person = torch.isin(logits.argmax(1), ids)
            masks.extend((person.to(torch.uint8) * 255).cpu().numpy())
    masks = np.stack(masks)
    np.savez_compressed(out, masks=masks)

    writer = cv2.VideoWriter(preview, cv2.VideoWriter_fourcc(*"mp4v"), reader.get_avg_fps(),
                             (masks.shape[2], masks.shape[1]))
    for i in range(n):
        frame = reader[i].asnumpy().copy()
        sel = masks[i] > 0
        frame[sel] = (0.5 * frame[sel] + [127, 0, 0]).astype(np.uint8)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    empty = int(np.sum(masks.reshape(n, -1).max(1) == 0))
    print(f"{n} masks -> {out}  (person area {np.mean(masks > 0):.1%}; empty frames: {empty})\npreview -> {preview}")


if __name__ == "__main__":
    main()
