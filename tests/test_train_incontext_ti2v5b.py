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
    saved = sorted(os.listdir(tmp_path / "run"))
    assert saved == ["lora_000040.safetensors", "lora_000080.safetensors"]
    fresh = tt.load_lora(tiny_transformer(), str(tmp_path / "run" / saved[-1]))
    x, drive, t_x, text = inputs()
    with torch.no_grad():
        torch.testing.assert_close(tt.incontext_forward(fresh, x, t_x, text, drive, t_x),
                                   tt.incontext_forward(m.eval(), x, t_x, text, drive, t_x))


def test_pick_window_keeps_the_reference_away():
    rng = np.random.default_rng(0)
    for _ in range(20):
        idx, ref = prep.pick_window(300, 30.0, 24, 33, 1.0, rng)
        assert len(idx) == 33 and idx == sorted(idx) and idx[-1] <= 299
        assert ref < idx[0] - 30 or ref > idx[-1] + 30
    assert prep.pick_window(20, 30.0, 24, 33, 1.0, rng) is None       # shorter than the window
    idx, ref = prep.pick_window(45, 30.0, 24, 33, 1.0, np.random.default_rng(1))  # no frame 1 s away: farthest end
    assert ref in (0, 44)


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
    out.mkdir()
    torch.save(dict(text=torch.randn(1, 7, 16), neg=torch.randn(1, 7, 16)), out / "text.pt")  # no T5 on CPU
    monkeypatch.setattr(sys, "argv", ["prep", "--model", str(model_dir), "--videos", str(vids), "--out", str(out),
                                      "--width", "32", "--height", "64", "--frames", "9", "--windows", "2",
                                      "--ref-gap", "0.2"])
    prep.main()
    names = sorted(f for f in os.listdir(out) if f != "text.pt")
    assert names == ["a_w0.pt", "a_w1.pt", "b_w0.pt", "b_w1.pt"]
    s = torch.load(out / "a_w0.pt", weights_only=True)
    assert s["target"].shape == s["drive"].shape == (48, 3, 4, 2) and s["target"].dtype == torch.float16
    assert s["meta"]["src"] == "a.mp4"
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


def _dist_worker(rank, world, port, folder, out, result):
    import torch.distributed as dist
    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "127.0.0.1", str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    m = tiny_transformer()
    tt.train(m, tt.CachedLatents(folder), steps=4, out=out, lr=1e-2, rank_lora=4, warmup=1, save_every=4,
             rank=rank, world=world, log=lambda *_: None)
    torch.save(tt.lora_state_dict(m), os.path.join(result, f"rank{rank}.pt"))
    dist.destroy_process_group()


def test_two_ranks_stay_in_sync_and_only_rank0_saves(tmp_path):
    import socket
    import torch.multiprocessing as mp
    write_cache(tmp_path / "cache", n=4)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    (tmp_path / "res").mkdir()
    mp.spawn(_dist_worker, args=(2, port, str(tmp_path / "cache"), str(tmp_path / "run"), str(tmp_path / "res")),
             nprocs=2, join=True)
    a, b = (torch.load(tmp_path / "res" / f"rank{r}.pt", weights_only=True) for r in (0, 1))
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)       # all-reduced gradients
    assert any(v.abs().sum() > 0 for k, v in a.items() if k.endswith(".B"))        # and they did train
    assert os.listdir(tmp_path / "run") == ["lora_000004.safetensors"]
