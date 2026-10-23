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
                                         None, "cpu", fill="grey", grow=0)
        assert msk.shape == get_i2v_mask(lat_t - 1, lat_h, lat_w, mrl, device="cpu").shape
        assert y.shape[1] == lat_t - 1
        person = torch.from_numpy(pm[mrl]).bool()
        frame = seen["frames"][:, mrl]
        assert torch.all(frame[:, person] == 0)                                  # hole = grey (0)
        assert torch.allclose(frame[:, ~person], cond[0, :, mrl][:, ~person], atol=1e-4)
        assert 0 < msk.mean() < 1                                                 # per-pixel mask
        if mrl:
            assert torch.all(msk[:, 0] == 1)                                      # overlap frame known


def test_hole_fill_uses_the_surrounding_background_not_a_foreign_colour():
    n, H, W = 2, 64, 96
    cond = torch.empty(3, n, H, W)
    cond[:, :, :, :48] = -0.6                       # dark scene: left half
    cond[:, :, :, 48:] = -0.2                       # right half
    m = torch.zeros(n, H, W)
    m[:, 16:48, 32:64] = 1                          # hole across the seam
    cond[:, :, 16:48, 32:64] = 0.9                  # the person's own (bright) colour must not leak in
    hole = m[0].bool()
    blur = rp.fill_hole(cond, m, "blur")
    assert blur[:, :, hole].min() >= -0.6 - 1e-4 and blur[:, :, hole].max() <= -0.2 + 1e-4
    assert blur[0, 0, 32, 33] < blur[0, 0, 32, 62]  # left part of the hole darker, like its side
    assert torch.equal(blur[:, :, ~hole], cond[:, :, ~hole])
    assert torch.all(rp.fill_hole(cond, m, "grey")[:, :, hole] == 0)
    assert torch.all(rp.fill_hole(cond, m, "black")[:, :, hole] == -1)
    with pytest.raises(ValueError):
        rp.fill_hole(cond, m, "pink")


def test_grow_generates_one_more_latent_cell_around_the_hole(masks_npz, monkeypatch):
    H, W = resize_by_area(np.zeros((H0, W0, 3), np.uint8), AREA, divisor=16).shape[:2]
    pm = rp.load_person_masks(_cfg(replace_mask=masks_npz, replace_dilate=0), list(range(N)), AREA, 13)
    clip_len, lat_h, lat_w = 9, H // 8, W // 8
    lat_t = (clip_len + 1) // 4 + 2
    monkeypatch.setattr(rp, "vae_encode", lambda vae, videos, prec: [torch.zeros(16, lat_t - 1, lat_h, lat_w)])
    cond = torch.rand(1, 3, clip_len, H, W) * 2 - 1
    batch = {"conditioning_pixel_values": cond, "pixel_values": torch.zeros(1, 3, 1, H, W)}
    bg = [((cond[0, :, k].permute(1, 2, 0).numpy() + 1) * 127.5) for k in range(clip_len)]
    gen = {}
    for g in (0, 1):
        _, msk = rp.replace_conditioning(None, batch, bg, pm[:clip_len], 0, H, W, lat_t, lat_h, lat_w, None, "cpu",
                                         grow=g)
        gen[g] = msk[0, 1] == 0                     # one generated frame: cells to generate
    want = torch.nn.functional.max_pool2d(gen[0][None, None].float(), 3, stride=1, padding=1)[0, 0] > 0
    assert gen[1].sum() > gen[0].sum() and torch.equal(gen[1], want)


def test_composite_keeps_background_outside_the_hole(masks_npz):
    pm = rp.load_person_masks(_cfg(replace_mask=masks_npz, replace_dilate=3), list(range(N)), AREA, 13)
    h, w = pm[0].shape
    out = np.full((9, h, w, 3), 200, np.uint8)
    bgs = [np.full((h, w, 3), 20.0) for _ in range(13)]
    comp = rp.composite_clip(out, bgs, pm, 0, _cfg())
    hole = pm[0].astype(bool)
    assert comp[0][~hole].max() <= 20 and comp[0][hole].max() == 200
    assert np.array_equal(rp.composite_clip(out, bgs, pm, 0, _cfg(replace_composite=False)), out)


def test_frame_mode_generates_everything_but_a_border(masks_npz):
    pm = rp.load_person_masks(_cfg(replace_mask=masks_npz, replace_mask_mode="frame", replace_border=8),
                              list(range(N)), AREA, 13)
    m = pm[0]
    assert m[8:-8, 8:-8].all() and not m[:8].any() and not m[-8:].any() and not m[:, :8].any()
    assert all(np.array_equal(x, m) for x in pm)          # the person's position plays no part


def test_pad_then_crop_returns_the_original_framing(tmp_path):
    import imageio.v2 as imageio
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    import pad_replace
    h, w, n = 96, 64, 5
    frames = [np.full((h, w, 3), 40 * k, np.uint8) for k in range(n)]
    for f in frames:
        f[:, : w // 2, 0] = 250                                   # left half red: the mirror must keep it
    src = str(tmp_path / "drive.mp4")
    pad_replace.write_video(src, frames, 10)
    prefix = str(tmp_path / "drive_pad")
    sys.argv = ["pad_replace.py", "pad", "--video", src, "--frac", "0.1", "--margin", "0.05", "--out-prefix", prefix]
    pad_replace.main()
    meta = __import__("json").load(open(prefix + ".json"))
    p = meta["pad"]
    assert p == 6 and (meta["padded_height"], meta["padded_width"]) == (h + 2 * p, w + 2 * p)
    hole = np.load(prefix + "_hole.npz")["masks"]
    m = meta["margin"]
    assert m == 3 and hole.shape == (n, h + 2 * p, w + 2 * p) and hole[0, p - m:p + h + m, p - m:p + w + m].min() == 255
    assert hole[0, :p - m].max() == 0 and hole[0, :, :p - m].max() == 0      # outer border stays known
    padded = [f for f in imageio.get_reader(prefix + ".mp4")]
    assert padded[0].shape[:2] == (h + 2 * p, w + 2 * p) and padded[0][h // 2, 2, 0] > 200   # mirrored red
    # a pipeline output is the padded frame at another size; crop must give back the original framing
    out = str(tmp_path / "results.mp4")
    pad_replace.write_video(out, [cv2.resize(f, (2 * (w + 2 * p), 2 * (h + 2 * p))) for f in padded], 10)
    sys.argv = ["pad_replace.py", "crop", "--video", out, "--meta", prefix + ".json"]
    pad_replace.main()
    cropped = [f for f in imageio.get_reader(str(tmp_path / "results_crop.mp4"))]
    assert cropped[0].shape[:2] == (2 * h, 2 * w) and len(cropped) == n


def test_feather_band_scales_with_the_object():
    def blend(h, w, box):
        mask = np.zeros((h, w), np.uint8)
        y0, x0, y1, x1 = box
        mask[y0:y1, x0:x1] = 255
        fg, bg = np.full((h, w, 3), 200, np.uint8), np.zeros((h, w, 3), np.uint8)
        return rp.feather_blend(fg, bg, mask)[..., 0].astype(int)
    small = blend(464, 824, (200, 300, 260, 500))          # a 60 px tall bird: 3 px band, 2 px contour skipped
    assert small[201, 400] < 100 and small[205, 400] >= 190 and small[230, 305] >= 190 and small[199, 400] == 0
    big = blend(832, 464, (100, 100, 700, 400))            # a dancer: the band is capped at 15 px
    assert big[400, 103] < 120 and big[400, 120] >= 190 and big[95, 250] == 0


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
                  'if [[ "$1" == scripts/grounded_sam2.py ]]; then echo "GSAM2 $*"; '
                  'while [[ $# -gt 0 ]]; do [[ "$1" == --out ]] && touch "$2"; shift; done; exit 0; fi\n'
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
               SCENE_REF="1", REF_TARGET="cat", TARGET="spinning top",
               MASK=str(tmp_path / "m.npz"), OUT_DIR=str(tmp_path / "out"), WORK_DIR=str(tmp_path / "work"))
    p = subprocess.run(["bash", os.path.join(REPO, "scripts", "run_v100_replace.sh")], env=env,
                       capture_output=True, text=True, timeout=60)
    out = p.stdout + p.stderr
    assert p.returncode == 0, out[-2000:]
    assert "GSAM2 scripts/grounded_sam2.py video" in out and "--target spinning top" in out
    assert "GSAM2 scripts/grounded_sam2.py scene-ref" in out and "--ref-target cat" in out
    assert "ref_scene.png" in out                                         # the scene ref is what runs
    assert "--scale 0.6 " in out                                          # 0.60 normalised
    assert "type=pipelines.wan_animate_2_replace_pipeline.WanAnimate2ReplacePipeline" in out
    assert f"test_cfg.replace_background_video={drive}" in out           # original = background
    assert "clip_retarget_s0.6.mp4" in out and "clip_hole_s0.6.npz" in out and "replace_mask_mode=bbox" in out


def _gsam2():
    stub_heavy_video_deps()
    return runpy.run_path(os.path.join(REPO, "scripts", "grounded_sam2.py"))


def test_main_object_keeps_split_parts_and_drops_a_bystander():
    g = _gsam2()
    m = np.zeros((200, 100), bool)
    m[40:180, 40:70] = True     # body
    m[100:110, 72:80] = True    # hand split off by motion blur (2 px gap)
    m[5:20, 0:10] = True        # bystander's arm, far away
    kept = g["main_object"](m)
    assert kept[105, 75] and not kept[10, 5] and kept[100, 50]


def test_clean_plate_ignores_the_target_and_fills_never_seen_pixels(monkeypatch):
    g = _gsam2()
    n, h, w = 20, 60, 40
    frames = []
    masks = np.zeros((n, h, w), np.uint8)
    for i in range(n):
        f = np.full((h, w, 3), 100, np.uint8)
        x = 2 * i                                   # a "person" walking across the frame
        f[20:50, x:x + 6] = 250
        masks[i, 20:50, x:x + 6] = 255
        f[0:10, 0:10] = 250                         # a corner always hidden by a static target
        masks[i, 0:10, 0:10] = 255
        frames.append(f)
    reader = types.SimpleNamespace(__len__=lambda: n)

    class R:
        def __init__(self, path):
            pass

        def __len__(self):
            return n

        def __getitem__(self, i):
            return types.SimpleNamespace(asnumpy=lambda: frames[i])
    g["clean_plate"].__globals__["VideoReader"] = R
    plate, unseen = g["clean_plate"]("v.mp4", masks, dilate=1)
    assert abs(int(plate[35, 20, 0]) - 100) <= 2           # the walker is gone
    assert abs(int(plate[2, 2, 0]) - 100) <= 10 and unseen > 0   # hidden corner filled from around


def test_cover_crops_to_the_exact_size():
    g = _gsam2()
    out = g["cover"](np.zeros((100, 300, 3), np.uint8), 64, 128)
    assert out.shape == (128, 64, 3)


def test_px_canvas_keeps_the_normal_480p_picture():
    """Ray's layout: bigger canvas, the original frame at exactly its plain-480p size, ~8 px border outside."""
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    import pad_replace
    h, w = 1920, 1080
    assert pad_replace.content_size(h, w, 480 * 832) == (824, 464)          # a plain 480p run
    p, canvas = pad_replace.fit_canvas(h, w, 8, "480x832")
    cw, ch = map(int, canvas.split("x"))
    assert (p, canvas) == (19, "480x856") and cw * ch > 480 * 832
    ph, pw = h + 2 * p, w + 2 * p
    oh, ow = pad_replace.content_size(ph, pw, cw * ch)                       # the pipeline's raw output
    meta = {"pad": p, "height": h, "width": w, "padded_height": ph, "padded_width": pw}
    y0, x0, y1, x1 = pad_replace.crop_box(meta, oh, ow)
    assert (y1 - y0, x1 - x0) == (824, 464) and (y0, x0) == (8, 8)


def test_pad_image_frames_the_reference_like_the_video(tmp_path):
    """The pipeline crops with the reference's framing: a padded reference must give the padded video's canvas."""
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    import pad_replace
    h, w = 192, 108
    src, ref, prefix = str(tmp_path / "d.mp4"), str(tmp_path / "r.png"), str(tmp_path / "d_canvas")
    pad_replace.write_video(src, [np.full((h, w, 3), 90, np.uint8)] * 3, 10)
    cv2.imwrite(ref, np.full((2 * h, 2 * w, 3), 50, np.uint8))
    sys.argv = ["pad_replace.py", "pad", "--video", src, "--px", "8", "--res", "96x160", "--image", ref,
                "--border", "black", "--out-prefix", prefix]
    pad_replace.main()
    meta = __import__("json").load(open(prefix + ".json"))
    padded_ref = cv2.imread(prefix + "_ref.png")
    assert padded_ref.shape[:2] == (2 * meta["padded_height"], 2 * meta["padded_width"])
    assert padded_ref[0, 0].max() == 0 and padded_ref[meta["pad"] * 2 + 1, meta["pad"] * 2 + 1].min() == 50
