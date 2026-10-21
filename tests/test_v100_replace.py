"""Experimental replace mode (pipelines/wan_animate_2_replace_pipeline.py, scripts/retarget_driving.py,
scripts/run_v100_replace.sh): CPU checks of the pieces that differ from move mode. No weights needed."""
import builtins
import os
import pickle
import runpy
import stat
import subprocess
import sys
import types

import cv2
import numpy as np
import pytest
import torch
from easydict import EasyDict

from conftest import REPO, stub_heavy_video_deps

stub_heavy_video_deps()
import pipelines.wan_animate_2_pipeline as base  # noqa: E402
import pipelines.wan_animate_2_replace_pipeline as rp  # noqa: E402
from pipelines.utils.human_video_omni_utils import resize_by_area  # noqa: E402
from wanxiang.eval_i2v import get_i2v_mask  # noqa: E402

N, H0, W0, AREA = 10, 192, 108, 48 * 80


@pytest.fixture
def masks_npz(tmp_path):
    m = np.zeros((N, H0, W0), np.uint8)
    m[:, 80:150, 40:70] = 255
    path = tmp_path / "m.npz"
    np.savez_compressed(path, masks=m)
    return str(path)


def _cfg(**kw):
    return EasyDict(test_cfg=EasyDict(**kw))


def test_copied_core_resolves_every_name():
    def names(code):
        out = set(code.co_names)
        for c in code.co_consts:
            if isinstance(c, types.CodeType):
                out |= names(c)
        return out
    missing = [n for n in names(rp.inference_core_replace.__code__)
               if hasattr(base, n) and not hasattr(rp, n) and not hasattr(builtins, n)]
    assert not missing


def test_masks_follow_frame_geometry_and_padding(masks_npz):
    idxs = list(range(N))
    shape = resize_by_area(np.zeros((H0, W0, 3), np.uint8), AREA, divisor=16).shape[:2]
    plain = rp.load_person_masks(_cfg(replace_mask=masks_npz, replace_dilate=0), idxs, AREA, 13)
    dil = rp.load_person_masks(_cfg(replace_mask=masks_npz, replace_dilate=3), idxs, AREA, 13)
    box = rp.load_person_masks(_cfg(replace_mask=masks_npz, replace_dilate=0, replace_mask_mode="bbox"),
                               idxs, AREA, 13)
    assert len(dil) == 13 and dil[0].shape == shape
    assert dil[0].sum() > plain[0].sum() > 0 and box[0].sum() >= plain[0].sum()
    with pytest.raises(ValueError):
        rp.load_person_masks(_cfg(), idxs, AREA, 13)


def test_conditioning_matches_move_mode_shapes(masks_npz, monkeypatch):
    shape = resize_by_area(np.zeros((H0, W0, 3), np.uint8), AREA, divisor=16).shape[:2]
    H, W = shape
    pm = rp.load_person_masks(_cfg(replace_mask=masks_npz, replace_dilate=3), list(range(N)), AREA, 13)
    clip_len = 9
    T = clip_len + 1
    lat_t, lat_h, lat_w = T // 4 + 2, H // 8, W // 8
    seen = {}

    def fake_encode(vae, videos, prec):
        seen["frames"] = videos[0].clone()
        return [torch.zeros(16, (videos[0].shape[1] - 1) // 4 + 1, H // 8, W // 8)]
    monkeypatch.setattr(rp, "vae_encode", fake_encode)
    cond = torch.rand(1, 3, clip_len, H, W) * 2 - 1
    batch = {"conditioning_pixel_values": cond, "pixel_values": torch.full((1, 3, 1, H, W), 0.5)}
    bg = [((cond[0, :, k].permute(1, 2, 0).numpy() + 1) * 127.5) for k in range(clip_len)]
    for mrl in (0, 1):
        y, msk = rp.replace_conditioning(None, batch, bg, pm[:clip_len], mrl, H, W, lat_t, lat_h, lat_w,
                                         None, "cpu")
        assert msk.shape == get_i2v_mask(lat_t - 1, lat_h, lat_w, mrl, device="cpu").shape
        assert y.shape[1] == lat_t - 1
        person = torch.from_numpy(pm[mrl]).bool()
        frame = seen["frames"][:, mrl]
        assert torch.all(frame[:, person] == 0)                                  # hole = grey (0)
        assert torch.allclose(frame[:, ~person], cond[0, :, mrl][:, ~person], atol=1e-4)
        assert 0 < msk.mean() < 1                                                 # per-pixel mask
        if mrl:
            assert torch.all(msk[:, 0] == 1)                                      # overlap frame known


def test_composite_keeps_background_outside_the_hole(masks_npz):
    pm = rp.load_person_masks(_cfg(replace_mask=masks_npz, replace_dilate=3), list(range(N)), AREA, 13)
    h, w = pm[0].shape
    out = np.full((9, h, w, 3), 200, np.uint8)
    bgs = [np.full((h, w, 3), 20.0) for _ in range(13)]
    comp = rp.composite_clip(out, bgs, pm, 0, _cfg())
    hole = pm[0].astype(bool)
    assert comp[0][~hole].max() <= 20 and comp[0][hole].max() == 200
    assert np.array_equal(rp.composite_clip(out, bgs, pm, 0, _cfg(replace_composite=False)), out)


def test_background_video_is_sampled_like_the_driving_frames(monkeypatch):
    class Reader:
        def __init__(self, path):
            pass

        def __len__(self):
            return N

        def get_batch(self, idx):
            return types.SimpleNamespace(asnumpy=lambda: np.full((len(idx), H0, W0, 3), 7, np.uint8))
    monkeypatch.setattr(rp, "VideoReader", Reader)
    cfg = _cfg(replace_background_video="orig.mp4")
    bgs = rp.load_background_frames(cfg, list(range(N)), AREA, 13, N)
    assert len(bgs) == 13 and abs(float(bgs[0][bgs[0].shape[0] // 2, bgs[0].shape[1] // 2, 0]) - 7) < 1
    assert rp.load_background_frames(_cfg(), list(range(N)), AREA, 13, N) is None
    with pytest.raises(ValueError):
        rp.load_background_frames(cfg, list(range(N)), AREA, 13, N + 1)


def test_spawned_workers_run_the_replace_core(monkeypatch):
    assert pickle.loads(pickle.dumps(rp.worker_replace)) is rp.worker_replace
    monkeypatch.setattr(base.WanAnimate2MPIPipeline, "setup", lambda self: None)
    monkeypatch.setattr(base, "worker", base.worker)
    pipe = object.__new__(rp.WanAnimate2ReplacePipeline)
    pipe.setup()
    assert base.worker is rp.worker_replace


class _CvReader:
    """decord.VideoReader stand-in (decord has no macOS wheel)."""

    def __init__(self, path):
        cap = cv2.VideoCapture(path)
        self.fps, self.frames = cap.get(5), []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            self.frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        return types.SimpleNamespace(asnumpy=lambda: self.frames[i])

    def get_avg_fps(self):
        return self.fps


def test_retarget_rescales_around_the_feet_and_covers_both_people(tmp_path, monkeypatch):
    imageio = pytest.importorskip("imageio")
    n, h, w = 12, 96, 64
    video = str(tmp_path / "drive.mp4")
    masks = np.zeros((n, h, w), np.uint8)
    masks[:, 30:90, 20:44] = 255                         # a 60-px tall "person" standing at y=90
    with imageio.get_writer(video, fps=24, codec="libx264", macro_block_size=1,
                            ffmpeg_params=["-crf", "0"]) as wr:
        for m in masks:
            frame = np.full((h, w, 3), 90, np.uint8)
            frame[m > 0] = 230
            wr.append_data(frame)
    np.savez_compressed(tmp_path / "m.npz", masks=masks)
    monkeypatch.setitem(sys.modules, "decord", types.SimpleNamespace(VideoReader=_CvReader))
    monkeypatch.setattr(sys, "argv", ["retarget_driving.py", "--video", video, "--masks",
                                      str(tmp_path / "m.npz"), "--scale", "0.5", "--dilate", "0"])
    runpy.run_path(os.path.join(REPO, "scripts", "retarget_driving.py"), run_name="__main__")
    out = _CvReader(str(tmp_path / "drive_retarget_s0.5.mp4"))
    hole = np.load(tmp_path / "drive_hole_s0.5.npz")["masks"]
    assert len(out) == n and hole.shape == (n, h, w)
    ys = np.nonzero(out.frames[n // 2][..., 0] > 170)[0]  # the rescaled person
    assert abs((ys.max() + 1 - ys.min()) - 30) <= 3 and abs(ys.max() + 1 - 90) <= 3   # half height, same feet
    assert hole[n // 2][masks[n // 2] > 0].all()          # the original person is inside the hole


def test_launcher_wires_retarget_and_replace(tmp_path):
    fake = tmp_path / "bin"
    fake.mkdir()
    py = fake / "python"
    py.write_text('#!/bin/bash\n'
                  'if [[ "$1" == -c ]]; then exec "%s" "$@"; fi\n'
                  'if [[ "$1" == scripts/make_person_mask.py ]]; then echo "MASKGEN $*"; touch "$5"; exit 0; fi\n'
                  'if [[ "$1" == scripts/retarget_driving.py ]]; then echo "RETARGET $*"; pre="${@: -1}"; '
                  'touch "${pre}_retarget_s0.6.mp4" "${pre}_hole_s0.6.npz"; exit 0; fi\n'
                  'if [[ "$1" == wan_animate_2_demo.py ]]; then printf "ARG %%s\\n" "$@"; exit 0; fi\n'
                  'exit 0\n' % sys.executable)
    py.chmod(py.stat().st_mode | stat.S_IEXEC)
    ck = tmp_path / "ck"
    for d in ("wan_animate_2", "videomodel/Wan-AI/umt5-xxl", "videomodel/Wan-AI/xlm-roberta-large"):
        (ck / d).mkdir(parents=True)
    for f in ("wan_animate_2/wan_animate_2_bf16_distillation.safetensors",
              "videomodel/Wan-AI/models_t5_umt5-xxl-enc-bf16.pth", "videomodel/Wan-AI/vae.pth",
              "videomodel/Wan-AI/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"):
        (ck / f).write_bytes(b"")
    drive, ref = tmp_path / "clip.mp4", tmp_path / "ref.png"
    drive.write_bytes(b"")
    ref.write_bytes(b"")
    env = dict(os.environ, PATH=f"{fake}:{os.environ['PATH']}", CKPT_DIR=str(ck), DRIVE=str(drive),
               REF_IMG=str(ref), PROMPT="a cat", SCALE="0.60", MASK_MODE="bbox", GPUS="0,1",
               MASK=str(tmp_path / "m.npz"), OUT_DIR=str(tmp_path / "out"), WORK_DIR=str(tmp_path / "work"))
    p = subprocess.run(["bash", os.path.join(REPO, "scripts", "run_v100_replace.sh")], env=env,
                       capture_output=True, text=True, timeout=60)
    out = p.stdout + p.stderr
    assert p.returncode == 0, out[-2000:]
    assert "MASKGEN" in out and "--scale 0.6 " in out                     # 0.60 normalised
    assert "type=pipelines.wan_animate_2_replace_pipeline.WanAnimate2ReplacePipeline" in out
    assert f"test_cfg.replace_background_video={drive}" in out           # original = background
    assert "clip_retarget_s0.6.mp4" in out and "clip_hole_s0.6.npz" in out and "replace_mask_mode=bbox" in out
