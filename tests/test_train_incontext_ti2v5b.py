"""train/ TI2V-5B in-context pilot on tiny random diffusers models (CPU): forward, LoRA, loss, prep, filter, sampling."""
import os
import sys

import cv2
import numpy as np
import pytest
import torch

from conftest import REPO

pytest.importorskip("diffusers")
sys.path.insert(0, os.path.join(REPO, "train"))
import filter_clips as fc  # noqa: E402
import infer_incontext_ti2v5b as inf  # noqa: E402
import prep_incontext_ti2v5b as prep  # noqa: E402
import train_incontext_ti2v5b as tt  # noqa: E402


@pytest.fixture(autouse=True)
def _opencv_reader(monkeypatch):
    """Other tests stub decord in sys.modules; here the clips are read with OpenCV (decord is not installed on CPU)."""
    monkeypatch.setitem(sys.modules, "decord", None)


def tiny_transformer(seed=0, layers=2):
    from diffusers import WanTransformer3DModel
    torch.manual_seed(seed)
    return WanTransformer3DModel(patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=8, in_channels=48,
                                 out_channels=48, text_dim=16, freq_dim=16, ffn_dim=32, num_layers=layers,
                                 cross_attn_norm=True, qk_norm="rms_norm_across_heads")


def tiny_vae():
    from diffusers import AutoencoderKLWan
    torch.manual_seed(0)
    return AutoencoderKLWan(base_dim=16, decoder_base_dim=16, z_dim=48, dim_mult=[1, 2, 4, 4], num_res_blocks=1,
                            attn_scales=[], temperal_downsample=[False, True, True], in_channels=12, out_channels=12,
                            patch_size=2, is_residual=True, scale_factor_spatial=16, scale_factor_temporal=4,
                            latents_mean=[0.0] * 48, latents_std=[1.0] * 48)


def inputs(F_=3, h=4, w=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 48, F_, h, w, generator=g)
    drive = torch.randn(1, 48, F_, h, w, generator=g)
    t_frames = torch.full((1, F_), 700.0)
    t_frames[:, 0] = 0
    return x, drive, tt.token_timesteps(t_frames, h, w), torch.randn(1, 7, 16, generator=g)


def test_token_timesteps_are_frame_major():
    t = tt.token_timesteps(torch.tensor([[0.0, 5.0, 9.0]]), 4, 6)
    assert t.shape == (1, 3 * 2 * 3)
    assert t[0, :6].eq(0).all() and t[0, 6:12].eq(5).all() and t[0, 12:].eq(9).all()


def test_incontext_forward_without_drive_is_the_model_forward():
    m = tiny_transformer().eval()
    x, _, t_x, text = inputs()
    with torch.no_grad():
        ours = tt.incontext_forward(m, x, t_x, text)
        theirs = m(hidden_states=x, timestep=t_x, encoder_hidden_states=text, return_dict=False)[0]
    assert ours.shape == x.shape
    torch.testing.assert_close(ours, theirs, atol=1e-5, rtol=1e-5)


def test_driving_tokens_matter_and_only_lora_trains():
    m = tiny_transformer().eval()
    x, drive, t_x, text = inputs()
    t_d = torch.full_like(t_x, 400.0)
    with torch.no_grad():
        plain = tt.incontext_forward(m, x, t_x, text)
        with_drive = tt.incontext_forward(m, x, t_x, text, drive, t_d)
    assert (plain - with_drive).abs().max() > 1e-4                  # the driving tokens reach the video tokens
    params = tt.add_lora(m, rank=4)
    assert len(params) == 2 * 3 * 2                                  # A, B for q, k, v of 2 blocks
    with torch.no_grad():
        torch.testing.assert_close(tt.incontext_forward(m, x, t_x, text, drive, t_d), with_drive)  # B = 0
    loss = tt.incontext_forward(m, x, t_x, text, drive, t_d).pow(2).mean()
    loss.backward()
    assert all(p.grad is not None for p in params)
    assert any(p.grad.abs().sum() > 0 for p in params)
    assert all(p.grad is None for n, p in m.named_parameters() if not n.endswith((".A", ".B")))


def test_checkpointing_gives_the_same_gradients():
    grads = []
    for ckpt in (False, True):
        m = tiny_transformer()
        params = tt.add_lora(m, rank=4)
        for p in params:
            p.data.normal_(0, 0.1)
        x, drive, t_x, text = inputs()
        tt.incontext_forward(m, x, t_x, text, drive, torch.full_like(t_x, 300.0), use_checkpoint=ckpt).sum().backward()
        grads.append(torch.cat([p.grad.flatten() for p in params]))
    torch.testing.assert_close(grads[0], grads[1], atol=1e-5, rtol=1e-4)


def write_cache(folder, n=2, F_=3, h=4, w=4):
    os.makedirs(folder, exist_ok=True)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
    for i in range(n):
        target = torch.stack([torch.sin(3 * xx + c + 0.7 * f + i) * torch.cos(2 * yy - c) for c in range(48)
                              for f in range(F_)]).view(48, F_, h, w)
        torch.save(dict(target=target.half(), drive=(0.5 * target).half(), meta=dict(src=f"c{i}.mp4")),
                   os.path.join(folder, f"c{i}_w0.pt"))
    g = torch.Generator().manual_seed(1)
    torch.save(dict(text=torch.randn(1, 7, 16, generator=g), neg=torch.randn(1, 7, 16, generator=g)),
               os.path.join(folder, "text.pt"))


def fixed_loss(m, ds):
    g = torch.Generator().manual_seed(123)
    total = 0.0
    with torch.no_grad():
        for s in (ds[i] for i in range(len(ds))):
            x0, d = s["target"][None].float(), s["drive"][None].float()
            noise, nd = torch.randn(x0.shape, generator=g), torch.randn(x0.shape, generator=g)
            for sigma in (0.3, 0.6, 0.9):
                total += float(tt.flow_loss(m, x0, d, ds.text["text"].float(), sigma, 0.4, noise, nd))
    return total


def test_train_lowers_the_loss_and_lora_roundtrips(tmp_path):
    write_cache(tmp_path / "cache")
    ds = tt.CachedLatents(str(tmp_path / "cache"))
    assert len(ds) == 2
    m = tiny_transformer()
    before = fixed_loss(m, ds)
    hist = tt.train(m, ds, steps=80, out=str(tmp_path / "run"), lr=5e-3, rank_lora=8, warmup=1, save_every=40,
                    log=lambda *_: None)
    assert np.isfinite(hist).all()
    after = fixed_loss(m, ds)
    assert after < 0.9 * before, (before, after)
    saved = sorted(p.name for p in (tmp_path / "run").glob("*.safetensors"))
    assert saved == ["lora_000040.safetensors", "lora_000080.safetensors"]
    fresh = tt.load_lora(tiny_transformer(), str(tmp_path / "run" / saved[-1]))
    x, drive, t_x, text = inputs()
    with torch.no_grad():
        torch.testing.assert_close(tt.incontext_forward(fresh, x, t_x, text, drive, t_x),
                                   tt.incontext_forward(m.eval(), x, t_x, text, drive, t_x))


def test_pick_window_keeps_the_reference_away():
    rng = np.random.default_rng(0)
    accepted = 0
    for n in (20, 33, 45, 60, 75, 100, 300):
        for src_fps in (24, 29.97, 30):
            for _ in range(30):
                picked = prep.pick_window(n, src_fps, 24, 33, 1.0, rng)
                if picked is None:
                    continue
                idx, ref = picked
                accepted += 1
                assert len(idx) == 33 and idx == sorted(idx) and idx[-1] < n and 0 <= ref < n
                assert ref < idx[0] or ref > idx[-1]
                assert min(abs(ref - i) for i in idx) / src_fps >= 1.0
    assert accepted > 0
    for n, fps in ((20, 30), (33, 24), (45, 30)):
        assert prep.pick_window(n, fps, 24, 33, 1.0, rng) is None


def test_pick_window_retries_and_zero_gap_still_excludes_the_window():
    class Rng:
        starts = iter([34, 0])

        def uniform(self, *args):
            return next(self.starts)

        def choice(self, values):
            return values[-1]

    idx, ref = prep.pick_window(100, 24, 24, 33, 2, Rng())
    assert idx[0] == 0 and ref - idx[-1] >= 48
    assert prep.pick_window(33, 24, 24, 33, 0, np.random.default_rng(0)) is None


def test_jitter_and_destroy_appearance():
    frames = np.random.default_rng(0).integers(0, 255, (5, 64, 32, 3), dtype=np.uint8)
    j = prep.jitter(frames, 0.9, 0.05, -0.05)
    assert j.shape == frames.shape and j.dtype == np.uint8 and not np.array_equal(j, frames)
    assert np.array_equal(prep.jitter(frames, 1.0, 0.0, 0.0), frames)
    d = prep.destroy_appearance(frames, torch.Generator().manual_seed(0))
    assert d.shape == frames.shape and d.dtype == np.uint8
    assert np.array_equal(d[..., 0], d[..., 1]) and np.array_equal(d[..., 1], d[..., 2])   # grey


def write_mp4(path, n=40, w=64, h=96, fps=24, moving=True):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(n):
        f = np.full((h, w, 3), 40, np.uint8)
        x = 10 + (i % 20) if moving else 10
        f[20:80, x:x + 20] = (200, 150, 100)
        vw.write(f)
    vw.release()


def test_prep_main_end_to_end_then_train(tmp_path, monkeypatch):
    vids = tmp_path / "videos"
    vids.mkdir()
    write_mp4(vids / "a.mp4")
    write_mp4(vids / "b.mp4", n=60)
    model_dir = tmp_path / "ti2v"
    tiny_vae().save_pretrained(str(model_dir / "vae"))
    out = tmp_path / "cache"
    monkeypatch.setattr(prep, "encode_text", lambda *_: dict(text=torch.randn(1, 7, 16), neg=torch.randn(1, 7, 16)))
    monkeypatch.setattr(sys, "argv", ["prep", "--model", str(model_dir), "--videos", str(vids), "--out", str(out),
                                      "--width", "32", "--height", "64", "--frames", "9", "--windows", "2",
                                      "--ref-gap", "0.2"])
    prep.main()
    names = sorted(p.name for p in out.glob("*_w*.pt"))
    assert names == ["a_w0.pt", "a_w1.pt", "b_w0.pt", "b_w1.pt"]
    s = torch.load(out / "a_w0.pt", weights_only=True)
    assert s["target"].shape == s["drive"].shape == (48, 3, 4, 2) and s["target"].dtype == torch.float16
    assert s["meta"]["src"] == "a.mp4"
    originals = {name: (out / name).read_bytes() for name in names}
    missing = torch.load(out / "a_w1.pt", weights_only=True)
    (out / "a_w1.pt").unlink()
    prep.main()
    restored = torch.load(out / "a_w1.pt", weights_only=True)
    assert restored["meta"] == missing["meta"]
    for key in ("target", "drive"):
        torch.testing.assert_close(restored[key], missing[key], rtol=0, atol=0)
    for name in ("a_w0.pt", "b_w0.pt", "b_w1.pt"):
        assert (out / name).read_bytes() == originals[name]
    assert not list(out.glob(".tmp_*"))
    (out / ".tmp_broken.pt").write_bytes(b"incomplete")
    assert len(tt.CachedLatents(out)) == 4
    m = tiny_transformer()
    hist = tt.train(m, tt.CachedLatents(str(out)), steps=3, log=lambda *_: None)
    assert len(hist) == 3 and np.isfinite(hist).all()


def test_filter_nms_stats_motion_and_select():
    boxes = np.array([[0, 0, 10, 20], [1, 1, 10, 20], [30, 0, 40, 40]], float)
    keep = fc.nms(boxes, np.array([0.9, 0.8, 0.7]))
    assert sorted(keep) == [0, 2]                                    # the near-duplicate box goes
    count, main = fc.frame_stats(boxes, [0.9, 0.8, 0.7], (40, 40), min_area=0.05)
    assert count == 2 and list(main) == [30, 0, 40, 40]
    a = np.zeros((40, 40, 3), np.uint8)
    b = a.copy()
    b[5:15, 5:15] = 255
    assert fc.box_motion(a, a, [0, 0, 20, 20]) == 0 and fc.box_motion(a, b, [0, 0, 20, 20]) > 0.1
    rows = [dict(name=f"c{i}", person_frac=1, multi_frac=0, height=0.6, motion=i / 10) for i in range(10)]
    rows += [dict(name="crowd", person_frac=1, multi_frac=0.8, height=0.6, motion=0.9),
             dict(name="tiny", person_frac=1, multi_frac=0, height=0.1, motion=0.9),
             dict(name="empty", person_frac=0.2, multi_frac=0, height=0.6, motion=0.9)]
    picked = fc.select(rows, drop_stillest=0.35, log=lambda *_: None)
    assert "crowd" not in picked and "tiny" not in picked and "empty" not in picked
    assert picked == [f"c{i}" for i in range(4, 10)]


def test_score_clip_with_a_fake_detector(tmp_path):
    class Fake:                                                      # one person box around the moving block
        def boxes(self, img, target):
            ys, xs = np.where(img.max(-1) > 120)                     # the block (background is 40)
            return np.array([[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]], float), np.array([0.9])
    write_mp4(tmp_path / "move.mp4")
    write_mp4(tmp_path / "still.mp4", moving=False)
    move = fc.score_clip(str(tmp_path / "move.mp4"), Fake(), samples=4, gap=0.25)
    still = fc.score_clip(str(tmp_path / "still.mp4"), Fake(), samples=4, gap=0.25)
    assert move["person_frac"] == 1 and move["multi_frac"] == 0 and 0.5 < move["height"] < 0.75
    assert move["motion"] > 5 * still["motion"]


def test_sample_pins_the_reference_and_runs_with_and_without_drive():
    m = tiny_transformer().eval()
    x, drive, _, text = inputs()
    ref = x[:, :, :1]
    g = torch.Generator().manual_seed(0)
    out = inf.sample(m, ref, drive, text, text, steps=3, guide=2.0, generator=g)
    assert out.shape == drive.shape and torch.isfinite(out).all()
    torch.testing.assert_close(out[:, :, :1], ref)
    out2 = inf.sample(m, ref, drive, text, steps=3, generator=torch.Generator().manual_seed(0), use_drive=False)
    assert not torch.allclose(out[:, :, 1:], out2[:, :, 1:])
    sig = inf.flow_sigmas(4, 5.0)
    assert float(sig[0]) == 1.0 and float(sig[-1]) == 0.0 and (sig[1:] < sig[:-1]).all()


def _dist_worker(rank, world, port, folder, out, result, steps):
    import torch.distributed as dist
    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "127.0.0.1", str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    m = tiny_transformer()
    torch.manual_seed(1234 + 77 * rank)                            # undo tiny_transformer's common RNG state
    torch.rand(rank + 5)
    tt.train(m, tt.CachedLatents(folder), steps=steps, out=out, lr=1e-2, rank_lora=4, warmup=1, save_every=4,
             rank=rank, world=world, log=lambda *_: None)
    torch.save(tt.lora_state_dict(m), os.path.join(result, f"rank{rank}.pt"))
    dist.destroy_process_group()


@pytest.mark.parametrize("steps", [0, 4])
def test_two_ranks_stay_in_sync_and_only_rank0_saves(tmp_path, steps):
    import socket
    import torch.multiprocessing as mp
    write_cache(tmp_path / "cache", n=4)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    (tmp_path / "res").mkdir()
    mp.spawn(_dist_worker, args=(2, port, str(tmp_path / "cache"), str(tmp_path / "run"), str(tmp_path / "res"), steps),
             nprocs=2, join=True)
    a, b = (torch.load(tmp_path / "res" / f"rank{r}.pt", weights_only=True) for r in (0, 1))
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)       # all-reduced gradients
    if steps:
        assert any(v.abs().sum() > 0 for k, v in a.items() if k.endswith(".B"))    # and they did train
        assert sorted(os.listdir(tmp_path / "run")) == ["lora_000004.safetensors", "state.pt"]
    else:
        assert not (tmp_path / "run").exists()


def test_train_resume_matches_uninterrupted_run(tmp_path):
    write_cache(tmp_path / "cache", n=5)
    ds = tt.CachedLatents(tmp_path / "cache")
    kw = dict(lr=1e-2, rank_lora=4, warmup=5, accum=2, save_every=3, seed=17, log=lambda *_: None)
    full, first, resumed = tiny_transformer(), tiny_transformer(), tiny_transformer()
    hist = tt.train(full, ds, steps=6, out=tmp_path / "full", **kw)
    initial = tt.train(first, ds, steps=3, total_steps=6, out=tmp_path / "split", **kw)
    saved = torch.load(tmp_path / "split" / "state.pt", weights_only=True)
    assert saved["step"] == 3 and saved["cursor"] == 6 and saved["config"]["steps"] == 6
    assert saved["optimizer"]["state"] and saved["scheduler"]["last_epoch"] == 3
    assert "scaler" in saved and len(saved["rng"]) == 1
    before = (tmp_path / "split" / "lora_000003.safetensors").read_bytes()
    torch.manual_seed(999)
    rest = tt.train(resumed, ds, steps=6, out=tmp_path / "split", **kw)
    assert initial + rest == pytest.approx(hist, rel=1e-7, abs=1e-8)
    assert (tmp_path / "split" / "lora_000003.safetensors").read_bytes() == before
    a, b = tt.lora_state_dict(full), tt.lora_state_dict(resumed)
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=1e-7, atol=1e-8)
    final = torch.load(tmp_path / "split" / "state.pt", weights_only=True)
    assert final["step"] == 6 and final["cursor"] == 12
    assert final["scheduler"]["last_epoch"] == 6
    assert not list((tmp_path / "split").glob(".tmp_*"))


@pytest.mark.parametrize("change, field", [({"steps": 3}, "steps"), ({"rank_lora": 8}, "rank"),
                                          ({"lr": 0.02}, "lr"), ({"drive_noise": (0.1, 0.8)}, "drive_noise"),
                                          ({}, "data")])
def test_resume_refuses_different_run(tmp_path, change, field):
    write_cache(tmp_path / "cache")
    ds = tt.CachedLatents(tmp_path / "cache")
    kw = dict(steps=2, out=tmp_path / "run", rank_lora=4, log=lambda *_: None)
    tt.train(tiny_transformer(), ds, **kw)
    original = (tmp_path / "run" / "state.pt").read_bytes()
    if field == "data":
        write_cache(tmp_path / "other")
        ds = tt.CachedLatents(tmp_path / "other")
    with pytest.raises(ValueError, match=f"settings differ: {field}"):
        tt.train(tiny_transformer(), ds, **(kw | change))
    assert (tmp_path / "run" / "state.pt").read_bytes() == original


def test_resume_refuses_exports_without_state(tmp_path):
    write_cache(tmp_path / "cache")
    out = tmp_path / "run"
    out.mkdir()
    (out / "lora_000003.safetensors").write_bytes(b"old run")
    with pytest.raises(ValueError, match="without state.pt"):
        tt.train(tiny_transformer(), tt.CachedLatents(tmp_path / "cache"), steps=6, out=out)


def test_atomic_training_state_survives_interrupted_save(tmp_path, monkeypatch):
    write_cache(tmp_path / "cache")
    ds = tt.CachedLatents(tmp_path / "cache")
    out = tmp_path / "run"
    kw = dict(out=out, rank_lora=4, save_every=3, log=lambda *_: None)
    tt.train(tiny_transformer(), ds, steps=3, total_steps=6, **kw)
    original = (out / "state.pt").read_bytes()

    def interrupted_save(value, path):
        assert os.path.basename(path).startswith(".tmp_")
        with open(path, "wb") as fh:
            fh.write(b"partial")
        raise OSError("interrupted write")

    monkeypatch.setattr(torch, "save", interrupted_save)
    with pytest.raises(OSError, match="interrupted write"):
        tt.train(tiny_transformer(), ds, steps=6, **kw)
    assert (out / "state.pt").read_bytes() == original
    assert not (out / "lora_000006.safetensors").exists()


def test_resume_recovers_export_after_state_was_published(tmp_path, monkeypatch):
    write_cache(tmp_path / "cache")
    ds = tt.CachedLatents(tmp_path / "cache")
    out = tmp_path / "run"
    kw = dict(steps=3, out=out, rank_lora=4, log=lambda *_: None)
    export = tt.export_lora
    with monkeypatch.context() as patch:
        def fail_export(*args):
            raise OSError("interrupted export")
        patch.setattr(tt, "export_lora", fail_export)
        with pytest.raises(OSError, match="interrupted export"):
            tt.train(tiny_transformer(), ds, **kw)
    assert (out / "state.pt").exists() and not (out / "lora_000003.safetensors").exists()
    assert tt.train(tiny_transformer(), ds, **kw) == []
    assert (out / "lora_000003.safetensors").exists() and tt.export_lora is export


@pytest.mark.parametrize("field, value", [("model", "other"), ("width", 64), ("height", 128), ("frames", 13),
                                         ("fps", 12), ("windows", 3), ("ref_gap", 2), ("scale", [0.7, 1]),
                                         ("max_shift", 0.1), ("prompt", "other"), ("seed", 1)])
def test_cache_manifest_refuses_different_settings(tmp_path, field, value):
    from types import SimpleNamespace
    a = SimpleNamespace(out=str(tmp_path), model="ti2v", width=32, height=64, frames=9, fps=24, windows=2,
                        ref_gap=1, scale=(0.85, 1), max_shift=0.05, prompt="person", seed=0,
                        t5_wan=None, t5_tokenizer=None)
    prep.check_manifest(a)
    prep.check_manifest(a)                                      # same settings are restartable
    original = (tmp_path / "manifest.json").read_bytes()
    setattr(a, field, value)
    with pytest.raises(ValueError, match=f"manifest differs .*{field}"):
        prep.check_manifest(a)
    assert (tmp_path / "manifest.json").read_bytes() == original


def test_atomic_cache_write_keeps_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "text.pt"
    prep.atomic_save({"text": torch.ones(1)}, str(path))
    original = path.read_bytes()

    def interrupted_save(value, tmp):
        assert os.path.basename(tmp).startswith(".tmp_") and not str(tmp).endswith(".pt")
        with open(tmp, "wb") as fh:
            fh.write(b"partial")
        raise OSError("interrupted write")

    monkeypatch.setattr(torch, "save", interrupted_save)
    with pytest.raises(OSError, match="interrupted write"):
        prep.atomic_save({"text": torch.zeros(1)}, str(path))
    assert path.read_bytes() == original
    assert list(tmp_path.glob("*.pt")) == [path]


@pytest.mark.parametrize("empty", [False, True])
def test_filter_writes_header_for_missing_or_empty_csv(tmp_path, monkeypatch, empty):
    import csv
    from types import SimpleNamespace
    out = tmp_path / "scores.csv"
    if empty:
        out.touch()
    (tmp_path / "a.mp4").touch()
    monkeypatch.setitem(sys.modules, "grounded_sam2", SimpleNamespace(Detector=lambda *_: object()))
    row = dict(name="a.mp4", frames=48, fps=24, person_frac=1, multi_frac=0, height=0.6, motion=0.1)
    monkeypatch.setattr(fc, "score_clip", lambda *_: row)
    a = SimpleNamespace(out=str(out), videos=str(tmp_path), gdino="stub", box_threshold=0.3,
                        text_threshold=0.2, shard="0/1", samples=4, gap=0.5)
    fc.cmd_score(a)
    fc.cmd_score(a)
    with out.open() as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == fc.FIELDS
        rows = list(reader)
    assert len(rows) == 1 and rows[0]["name"] == "a.mp4"


@pytest.mark.parametrize("always_skip", [False, True])
def test_amp_skips_schedule_and_reports_stalls(tmp_path, monkeypatch, always_skip):
    class Scaler:
        value, calls = 128.0, 0

        def scale(self, loss):
            return loss

        def unscale_(self, opt):
            pass

        def get_scale(self):
            return self.value

        def step(self, opt):
            self.calls += 1
            if not always_skip and self.calls % 2 == 0:
                opt.step()

        def update(self):
            if always_skip or self.calls % 2:
                self.value /= 2

        def state_dict(self):
            return dict(scale=self.value)

    scaler, logs = Scaler(), []
    monkeypatch.setattr(torch.amp, "GradScaler", lambda *a, **kw: scaler)
    write_cache(tmp_path / "cache")
    ds = tt.CachedLatents(tmp_path / "cache")
    kw = dict(out=tmp_path / "run", rank_lora=4, log_every=10, log=logs.append)
    if always_skip:
        with pytest.raises(RuntimeError, match="50 consecutive AMP optimizer updates skipped at step 50"):
            tt.train(tiny_transformer(), ds, steps=60, **kw)
        assert scaler.calls == 50
    else:
        tt.train(tiny_transformer(), ds, steps=6, **kw)
        state = torch.load(tmp_path / "run" / "state.pt", weights_only=True)
        assert state["skipped"] == 3 and state["skip_streak"] == 0
        assert state["scheduler"]["last_epoch"] == 3
        assert state["scaler"]["scale"] == 16
        lines = [line for line in logs if line.startswith("step ")]
        assert [line.split(":")[0] for line in lines] == ["step 1", "step 2", "step 3"]
        assert all("scale " in line and "skipped " in line and "s/step" in line and "GiB peak" in line for line in lines)


@pytest.mark.parametrize("with_lora, tiling", [(False, True), (True, True), (False, False)])
def test_infer_freezes_and_separates_model_device_lifetimes(tmp_path, monkeypatch, with_lora, tiling):
    import diffusers
    import ti2v5b_replace as helpers

    class Model(torch.nn.Module):
        def __init__(self, name):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.name, self.where = name, "cpu"

        def to(self, device):
            self.where = device
            assert model.where == "cpu" or vae.where == "cpu"
            return self

    class Latent:
        def to(self, *args):
            return self

        def float(self):
            return self

    model, vae, events = Model("transformer"), Model("vae"), []
    if tiling:
        vae.enable_tiling = lambda: events.append("tiling")
    monkeypatch.setattr(diffusers.WanTransformer3DModel, "from_pretrained", lambda *a, **kw: model)
    monkeypatch.setattr(diffusers.AutoencoderKLWan, "from_pretrained", lambda *a, **kw: vae)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch, "load", lambda *a, **kw: dict(text=Latent(), neg=Latent()))
    monkeypatch.setattr(inf, "load_lora", lambda m, p: m.register_parameter("adapter", torch.nn.Parameter(torch.ones(1))))
    frames = np.zeros((9, 32, 32, 3), np.uint8)
    monkeypatch.setattr(helpers, "read_clip", lambda *a: (frames, 24))
    monkeypatch.setattr(helpers, "cover", lambda *a: frames[0])
    monkeypatch.setattr(cv2, "imread", lambda *a: frames[0])
    monkeypatch.setattr(helpers, "write_video", lambda *a: None)

    def encode(*args):
        assert model.where == "cpu" and vae.where == "cuda"
        events.append("encode")
        return Latent()

    def sample(*args, **kwargs):
        assert model.where == "cuda" and vae.where == "cpu"
        assert all(not p.requires_grad for p in model.parameters())
        assert all(not p.requires_grad for p in vae.parameters())
        assert not model.training
        events.append("sample")
        return Latent()

    def decode(*args):
        assert model.where == "cpu" and vae.where == "cuda"
        events.append("decode")
        return frames

    monkeypatch.setattr(helpers, "vae_encode", encode)
    monkeypatch.setattr(helpers, "vae_decode", decode)
    monkeypatch.setattr(inf, "sample", sample)
    argv = ["infer", "--model", "model", "--text", "text.pt", "--ref", "ref.png", "--drive", "drive.mp4",
            "--out", str(tmp_path), "--width", "32", "--height", "32", "--frames", "9"]
    monkeypatch.setattr(sys, "argv", argv + (["--lora", "lora.safetensors"] if with_lora else []))
    inf.main()
    assert events == (["tiling"] if tiling else []) + ["encode", "encode", "sample", "decode"]


def test_adapter_seed_and_broadcast_precede_optimizer(tmp_path, monkeypatch):
    write_cache(tmp_path / "cache")
    ds = tt.CachedLatents(tmp_path / "cache")
    states, broadcasts = [], []
    adam = torch.optim.AdamW

    def broadcast(p, src):
        assert p.device.type == "cpu" and p.dtype == torch.float32 and src == 0
        broadcasts.append(p)
        p.fill_(0.125)                                          # the optimizer must see the broadcast value

    def optimizer(params, **kwargs):
        assert len(broadcasts) == len(params)
        assert all(p.eq(0.125).all() for p in params)
        return adam(params, **kwargs)

    monkeypatch.setattr(tt.dist, "broadcast", broadcast)
    monkeypatch.setattr(torch.optim, "AdamW", optimizer)
    tt.train(tiny_transformer(), ds, steps=0, world=2, rank_lora=4, log=lambda *_: None)
    monkeypatch.setattr(torch.optim, "AdamW", adam)
    for seed in (123, 456):
        model = tiny_transformer()
        torch.manual_seed(seed)
        tt.train(model, ds, steps=0, seed=17, rank_lora=4, log=lambda *_: None)
        states.append(tt.lora_state_dict(model))
    for key in states[0]:
        torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)


def launcher_env(tmp_path):
    from pathlib import Path
    root, work = tmp_path / "repo", tmp_path / "work"
    (root / "train").mkdir(parents=True)
    script = root / "train" / "run_pilot_ti2v5b.sh"
    script.write_text((Path(REPO) / "train" / script.name).read_text())
    for part in ("transformer", "vae"):
        folder = tmp_path / "model" / part
        folder.mkdir(parents=True)
        (folder / "config.json").write_text("{}")
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "a.mp4").touch()
    work.mkdir()
    (work / "pass.txt").write_text("a.mp4\n")
    binary = tmp_path / "env" / "bin"
    binary.mkdir(parents=True)
    env = dict(os.environ, WORK=str(work), CKPT_5B=str(tmp_path / "model"), VIDEOS=str(videos), GPUS="0,1",
               ENV_PREFIX=str(binary.parent), GDINO=str(tmp_path / "gdino"))
    env.pop("T5_WAN", None)
    env.pop("T5_TOKENIZER", None)
    return script, work, binary, env


@pytest.mark.parametrize("stage, signal_name", [("filter", None), ("prep", None), ("prep", "SIGINT"),
                                               ("filter", "SIGTERM")])
def test_launcher_reaps_shards_on_failure_or_signal(tmp_path, stage, signal_name):
    import signal
    import subprocess
    import time
    script, work, binary, env = launcher_env(tmp_path)
    stub = binary / "python"
    stub.write_text(f"#!{sys.executable}\n" + '''import os, signal, sys, time
from pathlib import Path
if "--text-only" in sys.argv:
    sys.exit(0)
work = Path(os.environ["WORK"])
rank = sys.argv[sys.argv.index("--shard") + 1].split("/")[0]
def stop(*args):
    (work / ("stopped_" + rank)).touch()
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
(work / ("pid_" + rank)).write_text(str(os.getpid()))
if rank == "0" and os.environ["FAIL_SHARD"] == "1":
    deadline = time.monotonic() + 5
    while len(list(work.glob("pid_*"))) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    sys.exit(7)
while True:
    time.sleep(0.01)
''')
    stub.chmod(0o755)
    env["FAIL_SHARD"] = "0" if signal_name else "1"
    proc = subprocess.Popen(["bash", str(script), stage], env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    try:
        if signal_name:
            deadline = time.monotonic() + 5
            while len(list(work.glob("pid_*"))) < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            assert len(list(work.glob("pid_*"))) == 2
            proc.send_signal(getattr(signal, signal_name))
        output, _ = proc.communicate(timeout=10)
        assert proc.returncode == ({"SIGINT": 130, "SIGTERM": 143}.get(signal_name, 1)), output
        if not signal_name:
            assert "shard 0 failed" in output and ".log" in output
        for path in work.glob("pid_*"):
            with pytest.raises(ProcessLookupError):
                os.kill(int(path.read_text()), 0)
        assert (work / "stopped_1").exists()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        for path in work.glob("pid_*"):
            try:
                os.kill(int(path.read_text()), signal.SIGTERM)
            except ProcessLookupError:
                pass


def test_launcher_explains_missing_lora(tmp_path):
    import subprocess
    script, _, _, env = launcher_env(tmp_path)
    result = subprocess.run(["bash", str(script), "infer"], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 1
    assert "no LoRA in" in result.stderr and "run train first" in result.stderr


def test_offset_driving_rope_moves_only_the_driving_positions():
    m = tiny_transformer().eval()
    x, drive, t_x, text = inputs(F_=3, h=4, w=6)
    same = tt.driving_rope(m, x, "same")
    off = tt.driving_rope(m, x, "offset")
    video = m.rope(x)
    for a, b, v in zip(same, off, video):
        assert a.shape == b.shape == v.shape
        torch.testing.assert_close(a, v)
        assert (a - b).abs().amax() > 1e-3                            # driving rows sit below the video grid
    tall = m.rope(x.new_empty(1, 1, 3, 8, 6))                         # rows 0-1 of a twice-as-tall grid = the video's
    torch.testing.assert_close(tall[0].reshape(1, 3, 4, 3, 1, -1)[:, :, :2].reshape(1, 18, 1, -1), video[0])
    with torch.no_grad():
        o1 = tt.incontext_forward(m, x, t_x, text, drive, torch.full_like(t_x, 300.0), drive_rope="offset")
        o2 = tt.incontext_forward(m, x, t_x, text, drive, torch.full_like(t_x, 300.0))
    assert o1.shape == x.shape and torch.isfinite(o1).all() and not torch.allclose(o1, o2)
    with pytest.raises(ValueError):
        tt.driving_rope(m, x, "sideways")


def test_train_and_sample_with_offset_rope(tmp_path):
    write_cache(tmp_path / "cache")
    m = tiny_transformer()
    hist = tt.train(m, tt.CachedLatents(str(tmp_path / "cache")), steps=3, out=str(tmp_path / "run"), save_every=3,
                    drive_rope="offset", log=lambda *_: None)
    assert np.isfinite(hist).all()
    with pytest.raises(ValueError, match="drive_rope"):                # a run is resumed only with the same setting
        tt.train(tiny_transformer(), tt.CachedLatents(str(tmp_path / "cache")), steps=3, out=str(tmp_path / "run"),
                 save_every=3, drive_rope="same", log=lambda *_: None)
    x, drive, _, text = inputs()
    out = inf.sample(m.eval(), x[:, :, :1], drive, text, steps=2, generator=torch.Generator().manual_seed(0),
                     drive_rope="offset")
    assert torch.isfinite(out).all()


def test_val_losses_real_shuffled_none(tmp_path):
    import val_ti2v5b as vl
    write_cache(tmp_path / "cache", n=3)
    m = tiny_transformer().eval()
    res = vl.val_losses(m, tt.CachedLatents(str(tmp_path / "cache")), n=3, sigmas=(0.3, 0.9))
    assert set(res) == {"real", "shuffled", "none"} and all(set(r) == {0.3, 0.9} for r in res.values())
    assert all(np.isfinite(v) for r in res.values() for v in r.values())
    assert res["real"][0.3] != res["none"][0.3]                       # the driving tokens change the prediction
    again = vl.val_losses(m, tt.CachedLatents(str(tmp_path / "cache")), n=3, sigmas=(0.3, 0.9))
    assert again == res                                              # fixed seeds: comparable across checkpoints


def test_drive_norm_equalises_frame_brightness():
    rng = np.random.default_rng(0)
    base = rng.integers(60, 120, (1, 32, 16, 3)).astype(np.float32)
    frames = np.concatenate([base, base + 100, base - 40]).clip(0, 255).astype(np.uint8)   # a strobe: 3 brightness levels
    plain = prep.destroy_appearance(frames, torch.Generator().manual_seed(0))
    normed = prep.destroy_appearance(frames, torch.Generator().manual_seed(0), norm=True)
    assert np.ptp(plain.reshape(3, -1).mean(1)) > 30                 # without norm the flashes survive
    assert np.ptp(normed.reshape(3, -1).mean(1)) < 5                 # with norm: same brightness (up to clipping)
    assert normed.shape == frames.shape and normed.dtype == np.uint8


def test_prep_manifest_records_drive_norm(tmp_path, monkeypatch):
    vids = tmp_path / "videos"
    vids.mkdir()
    write_mp4(vids / "a.mp4")
    model_dir = tmp_path / "ti2v"
    tiny_vae().save_pretrained(str(model_dir / "vae"))
    out = tmp_path / "cache"
    monkeypatch.setattr(prep, "encode_text", lambda *_: dict(text=torch.randn(1, 7, 16), neg=torch.randn(1, 7, 16)))
    args = ["prep", "--model", str(model_dir), "--videos", str(vids), "--out", str(out), "--width", "32", "--height",
            "64", "--frames", "9", "--windows", "1", "--ref-gap", "0.2"]
    monkeypatch.setattr(sys, "argv", args + ["--drive-norm"])
    prep.main()
    import json
    assert json.load(open(out / "manifest.json"))["drive_norm"] is True
    monkeypatch.setattr(sys, "argv", args)                           # same cache without the flag: refused
    with pytest.raises(ValueError, match="drive_norm"):
        prep.main()


def test_make_teacher_jobs_writes_inputs_and_resumes(tmp_path, monkeypatch):
    import json
    import make_teacher_jobs as mtj
    vids = tmp_path / "videos"
    vids.mkdir()
    for name in ("a.mp4", "b.mp4", "c.mp4"):
        write_mp4(vids / name, n=40)
    (tmp_path / "pass.txt").write_text("a.mp4\nb.mp4\nc.mp4\n")
    args = ["mtj", "--videos", str(vids), "--list", str(tmp_path / "pass.txt"), "--out", str(tmp_path / "t"),
            "--frames", "9", "--width", "32", "--height", "64"]
    monkeypatch.setattr(sys, "argv", args + ["--n", "2"])
    mtj.main()
    jobs = [json.loads(l) for l in open(tmp_path / "t" / "jobs.jsonl")]
    assert [j["id"] for j in jobs] == ["00000", "00001"]
    assert all(j["src_a"] != j["src_b"] for j in jobs)                # A and B come from different clips
    n, fps = prep.video_info(jobs[0]["drive"])
    assert n == 9 and abs(fps - 24) < 0.5
    assert cv2.imread(jobs[0]["ref"]).shape == (64, 32, 3)
    monkeypatch.setattr(sys, "argv", args + ["--n", "3"])             # resume: the first two jobs are kept as they were
    mtj.main()
    jobs3 = [json.loads(l) for l in open(tmp_path / "t" / "jobs.jsonl")]
    assert len(jobs3) == 3 and jobs3[:2] == jobs


def test_prep_teacher_mode_builds_cross_identity_samples(tmp_path, monkeypatch):
    import json
    import make_teacher_jobs as mtj
    vids = tmp_path / "videos"
    vids.mkdir()
    for name in ("a.mp4", "b.mp4"):
        write_mp4(vids / name, n=40)
    (tmp_path / "pass.txt").write_text("a.mp4\nb.mp4\n")
    monkeypatch.setattr(sys, "argv", ["mtj", "--videos", str(vids), "--list", str(tmp_path / "pass.txt"), "--out",
                                      str(tmp_path / "t"), "--frames", "9", "--width", "32", "--height", "64", "--n", "2"])
    mtj.main()
    jobs = [json.loads(l) for l in open(tmp_path / "t" / "jobs.jsonl")]
    os.makedirs(jobs[0]["out"])
    write_mp4(os.path.join(jobs[0]["out"], "results.mp4"), n=9, w=32, h=64)   # a fake teacher video for job 0 only
    model_dir = tmp_path / "ti2v"
    tiny_vae().save_pretrained(str(model_dir / "vae"))
    monkeypatch.setattr(prep, "encode_text", lambda *_: dict(text=torch.randn(1, 7, 16), neg=torch.randn(1, 7, 16)))
    monkeypatch.setattr(sys, "argv", ["prep", "--model", str(model_dir), "--teacher", str(tmp_path / "t" / "jobs.jsonl"),
                                      "--out", str(tmp_path / "tc"), "--width", "32", "--height", "64", "--frames", "9"])
    prep.main()
    files = sorted(p.name for p in (tmp_path / "tc").glob("t*_w0.pt"))
    assert files == ["t00000_w0.pt"]                                  # job 1 has no teacher video yet: skipped
    s = torch.load(tmp_path / "tc" / "t00000_w0.pt", weights_only=True)
    assert s["target"].shape == s["drive"].shape == (48, 3, 4, 2)
    assert s["meta"]["teacher"] == "00000" and json.load(open(tmp_path / "tc" / "manifest.json"))["teacher"] is True


def test_mixed_latents_follow_the_weights(tmp_path):
    write_cache(tmp_path / "a", n=2)
    write_cache(tmp_path / "b", n=5)
    mix = tt.MixedLatents([str(tmp_path / "a"), str(tmp_path / "b")], [0.8, 0.2])
    assert len(mix) == 7 and mix.weights == [0.8, 0.2]
    srcs = [mix[k]["meta"]["src"] for k in range(400)]
    from_a = sum(s in ("c0.mp4", "c1.mp4") for s in srcs) / 400
    assert 0.65 < from_a < 0.95                                       # cache a is the smaller one, still drawn ~80 %
    assert [mix[k]["meta"]["src"] for k in range(20)] == srcs[:20]    # fixed draws (resume-safe)
    with pytest.raises(ValueError):
        tt.MixedLatents([str(tmp_path / "a"), str(tmp_path / "b")], [1.0])
    hist = tt.train(tiny_transformer(), mix, steps=2, log=lambda *_: None)
    assert np.isfinite(hist).all()


def test_batch_runner_skips_finished_jobs(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("wab", os.path.join(REPO, "infer", "wan_animate_2_batch.py"))
    wab = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "core", __import__("types").SimpleNamespace(build_object_from_config_file=None))
    spec.loader.exec_module(wab)
    jobs = [dict(id=f"{k:05d}", ref=f"r{k}.png", drive=f"d{k}.mp4", out=str(tmp_path / f"{k:05d}")) for k in range(3)]
    os.makedirs(jobs[1]["out"])
    open(os.path.join(jobs[1]["out"], "results.mp4"), "w").close()
    calls = []

    def fake_pipeline(**kw):
        calls.append(kw)
        open(os.path.join(kw["output_path"], "results.mp4"), "w").close()
    a = __import__("types").SimpleNamespace(width=32, height=64, fps=24, seed=42, clip_len=9, sample_guide_scale=1.0,
                                            step=2, prompt="p", prompt_ref="r")
    wab.run_jobs(fake_pipeline, jobs, a, log=lambda *_: None)
    assert [c["output_path"] for c in calls] == [jobs[0]["out"], jobs[2]["out"]]
    assert calls[1]["seed"] == 44 and calls[0]["clip_len"] == 9
    assert wab.pending(jobs) == []
