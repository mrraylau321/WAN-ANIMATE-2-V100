"""WA2 port oracles, tiny-model gradients, deterministic data, and ZeRO resume (no checkpoints downloaded)."""
import ast
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import types

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from conftest import REPO
from test_train_incontext_ti2v5b import inputs, tiny_transformer, tiny_vae, write_cache as write_pilot_cache

sys.path.insert(0, os.path.join(REPO, "train"))
import infer_wa2_ti2v5b as inf
import prep_incontext_ti2v5b as pilot_prep
import prep_wa2_ti2v5b as prep
import train_incontext_ti2v5b as pilot
import train_wa2_ti2v5b as train
import wa2_ti2v5b as wa


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def write_cache(folder, n=4):
    write_pilot_cache(folder, n=n, w=6)
    (folder / "samples").mkdir()
    for path in folder.glob("c*.pt"):
        s = torch.load(path, weights_only=True)
        s["mask"] = torch.ones(s["drive"].shape[1:], dtype=torch.bool)
        torch.save(s, folder / "samples" / path.name)
        path.unlink()
    (folder / "manifest.json").write_text(json.dumps(dict(version=prep.VERSION, blur_sigma=2.0)))
    return prep.CachedLatents(folder)


@pytest.mark.parametrize("frames,hw", [(3, 6), (4, 35)])
def test_native_mask_oracle(frames, hw):
    source = Path(REPO, "wanxiang/models/wan_animate_2_model.py").read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "incontext_mask_mod")
    ns = dict(torch=torch, np=np, math=math)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "native_mask", "exec"), ns)
    native, qtotal, _ = ns["incontext_mask_mod"](4 * (frames - 1) + 1, (16, hw * 16))
    q = torch.zeros(1, 1, frames * hw, frames)
    v_d = torch.eye(frames).repeat_interleave(hw, 0)[None, None]
    out = wa.grouped_attention(q, q, q, q, v_d, hw)
    allowed = out[0, 0, ::hw] > 0
    for g in range(frames):
        for d in range(frames):
            slot = g + 1 if g else 0
            expected = native(0, 0, torch.tensor(slot * hw), torch.tensor(qtotal + d * hw))
            assert bool(allowed[g, d]) == bool(expected)
    assert not allowed[0].any()
    assert torch.equal(allowed[1:], torch.eye(frames, dtype=torch.bool)[1:])


@pytest.mark.parametrize("h,w", [(4, 6), (6, 4)])
@pytest.mark.parametrize("head_dim", [8, 128])
def test_rope_explicit_positions(h, w, head_dim):
    model = tiny_transformer()
    if head_dim != 8:
        model = type(model).from_config(model.config, attention_head_dim=head_dim)
    x, _, _, _ = inputs(h=h, w=w)
    for offset, actual in [(0, model.rope(x)), (w // 2, wa.driving_rope(model, x))]:
        for table, got in zip((model.rope.freqs_cos, model.rope.freqs_sin), actual):
            ft, fh, fw = table.split([model.rope.t_dim, model.rope.h_dim, model.rope.w_dim], 1)
            expected = torch.stack([torch.cat([ft[t], fh[r], fw[offset + c]])
                                    for t in range(3) for r in range(h // 2) for c in range(w // 2)])[None, :, None]
            torch.testing.assert_close(got, expected, atol=0, rtol=0)
    wide = model.rope(x.new_empty(1, 1, 3, h, 2 * w))
    for got, plain in zip(wide, model.rope(x)):
        left = got.reshape(1, 3, h // 2, w, 1, -1)[:, :, :, :w // 2].reshape_as(plain)
        torch.testing.assert_close(left, plain, atol=0, rtol=0)


def test_cache_is_normalized_pre_rope_and_values_unrotated(monkeypatch):
    model = tiny_transformer()
    x, drive, times, text = inputs(h=4, w=6)
    recorded = []
    original = wa.block_qkv

    def capture(*args):
        result = original(*args)
        recorded.append(result)
        return result

    monkeypatch.setattr(wa, "block_qkv", capture)
    cache = wa.forward_drive(model, drive, text)
    for (_, k, v, _), (kc, vc) in zip(recorded, cache):
        torch.testing.assert_close(kc, k, atol=0, rtol=0)
        torch.testing.assert_close(vc, v, atol=0, rtol=0)
    k0, v0 = cache[0]
    saved_k, saved_v = k0.clone(), v0.clone()
    calls = []
    grouped = wa.grouped_attention

    def attention(q, k, v, kd, vd, frame_tokens):
        calls.append((kd, vd))
        return grouped(q, k, v, kd, vd, frame_tokens)

    monkeypatch.setattr(wa, "grouped_attention", attention)
    wa.wa2_forward(model, x, times, text, drive_cache=cache)
    rope = wa.driving_rope(model, drive)
    torch.testing.assert_close(calls[0][0], wa.apply_rope(saved_k, rope).transpose(1, 2))
    torch.testing.assert_close(calls[0][1], saved_v.transpose(1, 2), atol=0, rtol=0)
    assert not torch.allclose(calls[0][0], wa.apply_rope(wa.apply_rope(saved_k, rope), rope).transpose(1, 2))
    torch.testing.assert_close(k0, saved_k, atol=0, rtol=0)
    torch.testing.assert_close(v0, saved_v, atol=0, rtol=0)


def test_grouped_attention_outputs_and_all_five_gradients():
    torch.manual_seed(17)
    frames, hw = 3, 6
    tensors = [torch.randn(2, 2, frames * hw, 128, requires_grad=True) for _ in range(5)]
    q, k, v, kd, vd = tensors
    actual = wa.grouped_attention(q, k, v, kd, vd, hw)
    gf = torch.arange(frames * hw) // hw
    mask = torch.cat([torch.ones(frames * hw, frames * hw, dtype=torch.bool),
                      (gf[:, None] == gf[None, :]) & (gf[:, None] > 0)], 1)
    expected = F.scaled_dot_product_attention(q, torch.cat([k, kd], 2), torch.cat([v, vd], 2), attn_mask=mask)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    probe = torch.randn_like(actual)
    a = torch.autograd.grad((actual * probe).sum(), tensors)
    b = torch.autograd.grad((expected * probe).sum(), tensors)
    for ours, theirs in zip(a, b):
        torch.testing.assert_close(ours, theirs, atol=2e-6, rtol=1e-4)


@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("batch", [1, 2])
def test_no_drive_is_diffusers_own_blocks(checkpoint, batch, monkeypatch):
    model, baseline = tiny_transformer(), tiny_transformer()
    wa.set_shared_qkv_trainable(model)
    wa.set_shared_qkv_trainable(baseline)
    x, _, times, text = inputs(h=4, w=6)
    x, times, text = x.repeat(batch, 1, 1, 1, 1), times.repeat(batch, 1), text.repeat(batch, 1, 1)
    x.requires_grad_()
    calls = []
    block = wa.gen_block

    def spy(*args):
        calls.append(1)
        return block(*args)

    monkeypatch.setattr(wa, "gen_block", spy)
    actual = wa.wa2_forward(model, x, times, text, use_checkpoint=checkpoint)
    expected = baseline(x, times, text).sample
    assert len(calls) == len(model.blocks)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    actual.square().mean().backward()
    gx = x.grad.clone()
    x.grad = None
    expected.square().mean().backward()
    torch.testing.assert_close(gx, x.grad, atol=1e-5, rtol=1e-5)
    for (_, p), (_, b) in zip(model.named_parameters(), baseline.named_parameters()):
        if p.requires_grad:
            torch.testing.assert_close(p.grad, b.grad, atol=1e-5, rtol=1e-5)


def test_topology_and_cache_gradient():
    model = tiny_transformer(layers=3)
    wa.set_shared_qkv_trainable(model)
    x, d, t, text = inputs(h=4, w=6)
    cache = wa.forward_drive(model, d, text)
    for k, v in cache:
        k.retain_grad()
        v.retain_grad()
    a = wa.wa2_forward(model, x, t, text, drive_cache=cache)
    wa.wa2_forward(model, x + 10, t, text, drive_cache=cache)
    rebuilt = wa.forward_drive(model, d, text)
    for pair, other in zip(cache, rebuilt):
        for left, right in zip(pair, other):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
    b = wa.wa2_forward(model, x, t, text, drive=d + torch.randn_like(d))
    assert not torch.allclose(a, b)
    a[:, :, 1:].square().mean().backward()
    assert cache[0][0].grad.abs().sum() > 0
    assert cache[0][1].grad.abs().sum() > 0
    driven_grad = model.blocks[0].attn1.to_k.weight.grad.clone()
    model.zero_grad(set_to_none=True)
    wa.wa2_forward(model, x, t, text)[:, :, 1:].square().mean().backward()
    assert not torch.allclose(driven_grad, model.blocks[0].attn1.to_k.weight.grad)


def test_exact_180_qkv_tensors_train_and_change():
    model = tiny_transformer(layers=30)
    params = wa.set_shared_qkv_trainable(model)
    assert len(params) == 180
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    x, d, t, text = inputs(F_=2, h=2, w=4)
    wa.wa2_forward(model, x, t, text, d).square().mean().backward()
    for n, p in model.named_parameters():
        assert (p.grad is not None) == bool(wa.QKV_NAME.fullmatch(n))
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
    torch.optim.AdamW(params, lr=0.01, weight_decay=0).step()
    changed = {n for n, p in model.named_parameters() if not torch.equal(p, before[n])}
    assert changed == {n for n in before if wa.QKV_NAME.fullmatch(n)}


def test_meta_parameter_report(capsys):
    from diffusers import WanTransformer3DModel
    config = json.loads(Path(REPO).parent.joinpath("review/wa2_port/scratch/transformer_config.json").read_text())
    with torch.device("meta"):
        model = WanTransformer3DModel.from_config(config)
    wa.set_shared_qkv_trainable(model)
    wa.assert_5b(model)
    print(wa.parameter_report(model))
    assert wa.parameter_report(model) == wa.COUNTS
    assert "849623040" in capsys.readouterr().out


def test_checkpoint_loss_gradients_and_repeated_updates():
    runs = []
    for use_checkpoint in (False, True):
        model = tiny_transformer()
        params = wa.set_shared_qkv_trainable(model)
        opt = torch.optim.AdamW(params, lr=0.001, weight_decay=0)
        losses, gradients = [], []
        for step in range(2):
            opt.zero_grad(set_to_none=True)
            for micro in range(2):
                x, d, _, text = inputs(h=4, w=6, seed=step * 2 + micro)
                loss = train.flow_loss(model, x, d, text, 0.7, torch.ones_like(x), use_checkpoint) / 2
                losses.append(loss.item())
                loss.backward()
            gradients.append(torch.cat([p.grad.flatten() for p in params]))
            opt.step()
        runs.append((losses, gradients, train.qkv_state_dict(model)))
    assert runs[0][0] == pytest.approx(runs[1][0], abs=1e-7)
    for a, b in zip(runs[0][1], runs[1][1]):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    for n in runs[0][2]:
        torch.testing.assert_close(runs[0][2][n], runs[1][2][n], atol=1e-6, rtol=1e-5)


def test_blur_background_constant_impulse_and_no_channel_time_mixing():
    z = torch.zeros(2, 3, 9, 7)
    z[0, 1, 4, 3] = 1
    z[1] = 3
    mask = torch.zeros(3, 9, 7, dtype=torch.bool)
    mask[:, 2:7, 1:6] = True
    out = prep.blur_person_latents(z, mask)
    assert torch.equal(out[:, ~mask], z[:, ~mask])
    assert 0 < out[0, 1, 4, 3] < 1
    assert torch.count_nonzero(out[0, [0, 2]]) == 0
    torch.testing.assert_close(out[1], z[1], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(prep.blur_person_latents(torch.ones_like(z), mask), torch.ones_like(z))


def test_temporal_groups_dilation_and_empty_skip(monkeypatch):
    masks = torch.zeros(33, 80, 80, dtype=torch.bool)
    for frame in range(33):
        masks.zero_()
        masks[frame, 32, 32] = True
        result = prep.person_mask_to_latent(masks)
        group = 0 if frame == 0 else (frame - 1) // 4 + 1
        assert result.sum() == 9 and result[group, 1:4, 1:4].all()
    called = []
    monkeypatch.setattr(prep, "vae_encode", lambda *a: called.append(1))
    assert prep.encode_drive(None, None, masks, "cpu") is None
    assert not called


def test_prep_and_inference_share_exact_drive_encode(monkeypatch):
    rng = np.random.default_rng(4)
    clip = rng.integers(0, 256, (9, 64, 32, 3), dtype=np.uint8)
    masks = np.ones((9, 64, 32), bool)
    monkeypatch.setattr(prep, "person_masks", lambda *args: masks)
    vae = tiny_vae().eval()
    first, _, _ = prep.prepare_drive(vae, clip, None, None, "cpu", width=32, height=64)
    second, _, _ = inf.prepare_drive(vae, clip, None, None, "cpu", width=32, height=64)
    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])


def test_person_detection_tracks_first_hit_and_same_cover(monkeypatch):
    seen = []
    monkeypatch.setitem(sys.modules, "grounded_sam2", types.SimpleNamespace(
        shrink=lambda f, max_side: seen.append(max_side) or f,
        main_object=lambda m: m))
    raw = np.zeros((9, 64, 64, 3), np.uint8)
    masks = np.zeros((9, 64, 64), bool)
    masks[:, 16:32, 16:32] = True

    class Detector:
        def best(self, frame, target, pick):
            seen.append((target, pick))
            return [16, 16, 32, 32] if seen.count((target, pick)) == 3 else None

    class Segmenter:
        def track(self, frames, index, box):
            assert index == 2 and box == [16, 16, 32, 32]
            return masks

    actual = prep.person_masks(raw, Detector(), Segmenter(), width=32, height=64)
    assert seen.count(512) == 9
    assert np.array_equal(actual, masks[:, :, 16:48])


def test_cache_versions_refuse_each_other(tmp_path):
    write_pilot_cache(tmp_path / "pilot")
    (tmp_path / "pilot" / "manifest.json").write_text('{}')
    with pytest.raises(ValueError, match="pilot"):
        prep.CachedLatents(tmp_path / "pilot")
    new = write_cache(tmp_path / "new")
    assert len(new) == 4
    with pytest.raises(FileNotFoundError, match="no samples"):
        pilot.CachedLatents(tmp_path / "new")


def test_windows_match_pilot_all_windows(monkeypatch):
    monkeypatch.setattr(prep, "video_info", lambda path: (300, 24.0))
    monkeypatch.setattr(pilot_prep, "video_info", lambda path: (300, 24.0))
    monkeypatch.setattr(pilot_prep, "read_frames", lambda path, idx: np.zeros((len(idx), 32, 32, 3), np.uint8))
    monkeypatch.setattr(pilot_prep, "destroy_appearance", lambda x, *args: x)
    a = types.SimpleNamespace(windows=4, fps=24, frames=33, ref_gap=1.0, width=32, height=32,
                              scale=(0.85, 1.0), max_shift=0.05)
    old = pilot_prep.make_sample("clip.mp4", a, np.random.default_rng(42), torch.Generator())
    new = list(prep.windows("clip.mp4", 4, 24, 42))
    for (_, _, before), (idx, ref, after) in zip(old, new):
        assert all(before[k] == after[k] for k in after)
        assert min(abs(ref - idx[0]), abs(ref - idx[-1])) >= 24
        assert ref < idx[0] or ref > idx[-1]


@pytest.mark.parametrize("sigma", [0.0, 0.7, 1.0])
def test_flow_excludes_reference_exact_velocity_and_times(monkeypatch, sigma):
    x, drive, _, text = inputs(h=4, w=6)
    noise = torch.randn_like(x)

    def exact(model, noisy, times, context, d, checkpoint):
        expected = (1 - sigma) * x + sigma * noise
        expected[:, :, 0] = x[:, :, 0]
        torch.testing.assert_close(noisy, expected)
        assert torch.equal(d, drive)
        assert times[:, :6].eq(0).all() and times[:, 6:].eq(1000 * sigma).all()
        prediction = noise - x
        prediction[:, :, 0] += 10000
        return prediction

    monkeypatch.setattr(train, "wa2_forward", exact)
    assert train.flow_loss(None, x, drive, text, sigma, noise).item() == 0


def test_driving_embedding_all_one(monkeypatch):
    model = tiny_transformer()
    _, drive, _, text = inputs(h=4, w=6)
    original = model.condition_embedder.forward
    calls = []

    def capture(times, *args, **kwargs):
        calls.append(times.clone())
        assert kwargs["timestep_seq_len"] == 18
        return original(times, *args, **kwargs)

    monkeypatch.setattr(model.condition_embedder, "forward", capture)
    wa.forward_drive(model, drive, text)
    assert len(calls) == 1 and calls[0].shape == (18,) and calls[0].eq(1).all()


@pytest.mark.parametrize("use_drive", [True, False])
def test_sampler_builds_once_and_clamps_reference(monkeypatch, use_drive):
    x, drive, _, text = inputs(h=4, w=6)
    cache = object()
    built, calls = [], []

    def build(*args):
        assert not torch.is_grad_enabled()
        built.append(1)
        return cache

    def predict(model, noisy, times, context, drive_cache):
        torch.testing.assert_close(noisy[:, :, :1], x[:, :, :1], atol=0, rtol=0)
        assert times[:, :6].eq(0).all()
        assert drive_cache is (cache if use_drive else None)
        calls.append(1)
        return torch.ones_like(noisy)

    monkeypatch.setattr(inf, "forward_drive", build)
    monkeypatch.setattr(inf, "wa2_forward", predict)
    actual = inf.sample(None, x[:, :, :1], drive, text, steps=3, generator=torch.Generator().manual_seed(4),
                        use_drive=use_drive)
    assert len(built) == int(use_drive) and len(calls) == 3
    torch.testing.assert_close(actual[:, :, :1], x[:, :, :1], atol=0, rtol=0)
    noise = torch.randn(drive.shape, generator=torch.Generator().manual_seed(4))
    torch.testing.assert_close(actual[:, :, 1:], noise[:, :, 1:] - 1)


def test_resume_matches_uninterrupted_and_recovers_export(tmp_path):
    dataset = write_cache(tmp_path / "cache")
    full, split, resumed = tiny_transformer(), tiny_transformer(), tiny_transformer()
    kw = dict(lr=0.001, warmup=3, accum=2, seed=7, log=lambda *_: None)
    expected = train.train(full, dataset, 4, out=tmp_path / "full", **kw)
    first = train.train(split, dataset, 2, out=tmp_path / "split", total_steps=4, **kw)
    (tmp_path / "split" / "qkv_000002.safetensors").unlink()
    rest = train.train(resumed, dataset, 4, out=tmp_path / "split", **kw)
    assert first + rest == pytest.approx(expected, abs=1e-8)
    for n, p in train.qkv_state_dict(full).items():
        torch.testing.assert_close(p, train.qkv_state_dict(resumed)[n], atol=0, rtol=0)
    assert (tmp_path / "split" / "qkv_000002.safetensors").exists()
    state = torch.load(tmp_path / "split" / "state.pt", weights_only=True)
    assert state["cursor"] == 8 and state["step"] == 4 and state["scheduler"]["last_epoch"] == 4
    assert state["optimizer"]["state"] and len(state["rng"]) == 1


@pytest.mark.parametrize("field", ["world", "lr", "manifest", "fingerprint"])
def test_resume_refuses_changed_configuration(tmp_path, field):
    dataset = write_cache(tmp_path / "cache")
    train.train(tiny_transformer(), dataset, 1, out=tmp_path / "run", total_steps=2, log=lambda *_: None)
    kw = {}
    if field in ("world", "lr"):
        kw[field] = 2
    elif field == "manifest":
        dataset.manifest = dict(dataset.manifest, blur_sigma=3.0)
    else:
        dataset.fingerprint = "changed"
    with pytest.raises(ValueError, match=field):
        train.train(tiny_transformer(), dataset, 2, out=tmp_path / "run", log=lambda *_: None, **kw)


def _dist_worker(rank, port, folder, out, result, split):
    import torch.distributed as dist
    torch.set_num_threads(1)
    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "127.0.0.1", str(port)
    dist.init_process_group("gloo", rank=rank, world_size=2)
    dataset = prep.CachedLatents(folder)
    kw = dict(lr=0.001, warmup=3, accum=2, seed=19, rank=rank, world=2, log=lambda *_: None)
    model = tiny_transformer()
    if split:
        train.train(model, dataset, 2, out=out, total_steps=4, **kw)
        model = tiny_transformer()
    train.train(model, dataset, 4, out=out, **kw)
    torch.save(train.qkv_state_dict(model), os.path.join(result, f"rank{rank}.pt"))
    dist.destroy_process_group()


def free_port_or_skip():
    try:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]
    except OSError as exc:
        pytest.skip(f"gloo requires a local socket: {exc}")


def run_distributed(tmp_path, folder, split):
    import torch.multiprocessing as mp
    port = free_port_or_skip()
    tmp_path.mkdir()
    result = tmp_path / "result"
    result.mkdir()
    mp.spawn(_dist_worker, args=(port, str(folder), str(tmp_path / "run"), str(result), split), nprocs=2, join=True)
    return [torch.load(result / f"rank{rank}.pt", weights_only=True) for rank in (0, 1)]


def test_two_rank_zero_step_agrees(tmp_path):
    free_port_or_skip()
    write_cache(tmp_path / "cache")
    a, b = run_distributed(tmp_path / "full", tmp_path / "cache", False)
    assert all(torch.equal(a[k], b[k]) for k in a)
    state = torch.load(tmp_path / "full/run/state.pt", weights_only=True)
    assert len(state["rng"]) == 2
    assert len(state["optimizer"]["state"]) == 12
    base = train.qkv_state_dict(tiny_transformer())
    assert any(not torch.equal(a[k], base[k]) for k in a)


def test_two_rank_zero_resume_matches_uninterrupted(tmp_path):
    free_port_or_skip()
    write_cache(tmp_path / "cache")
    full = run_distributed(tmp_path / "full", tmp_path / "cache", False)
    split = run_distributed(tmp_path / "split", tmp_path / "cache", True)
    for a, b in zip(full, split):
        for n in a:
            torch.testing.assert_close(a[n], b[n], atol=0, rtol=0)


@pytest.mark.parametrize("reuse", [False, True, "other_fps"])
def test_prep_joint_target_reuse_and_previews(tmp_path, monkeypatch, reuse):
    import diffusers
    out, old = tmp_path / "new", tmp_path / "pilot"
    old.mkdir()
    (old / "manifest.json").write_text(json.dumps(dict(
        width=480, height=832, frames=33, fps=12.0 if reuse == "other_fps" else 24.0, ref_gap=1.0, seed=0,
        teacher=False, model=os.path.realpath(tmp_path / "model"))))
    meta = dict(src="clip.mp4", window=0, start=30, ref=0, fps=24.0)
    old_target = torch.full((48, 9, 52, 30), 13.0)
    torch.save(dict(target=old_target, meta=meta), old / "clip_w0.pt")
    monkeypatch.setattr(prep, "windows", lambda *args: [(list(range(30, 63)), 0, meta.copy())])
    raw = np.zeros((34, 832, 480, 3), np.uint8)
    raw[:-1] = np.arange(33, dtype=np.uint8)[:, None, None, None]
    raw[-1] = 99
    monkeypatch.setattr(prep, "read_frames", lambda *args: raw)
    monkeypatch.setattr(prep, "encode_text", lambda *args: dict(text=torch.zeros(1, 2, 16)))
    monkeypatch.setattr(prep, "load_mask_models", lambda *args: (None, None))
    mask = torch.ones(9, 52, 30, dtype=torch.bool)
    drive = torch.full((48, 9, 52, 30), 17.0)
    monkeypatch.setattr(prep, "prepare_drive", lambda *args: ((drive, mask), raw[:-1],
                                                           np.ones((33, 832, 480), dtype=bool)))

    class VAE:
        def to(self, *args): return self
        def requires_grad_(self, *args): return self
        def eval(self): return self
        def enable_tiling(self): pass

    monkeypatch.setattr(diffusers.AutoencoderKLWan, "from_pretrained", lambda *args, **kw: VAE())
    encoded, previews = [], []

    def encode(vae, frames, device):
        encoded.append(frames[:, 0, 0, 0].copy())
        return torch.full((1, 48, 9, 52, 30), 23.0)

    monkeypatch.setattr(prep, "vae_encode", encode)
    monkeypatch.setattr(prep, "vae_decode", lambda *args: raw[:-1])
    monkeypatch.setattr(prep, "write_video", lambda path, *args: previews.append(path))
    listing = tmp_path / "pass.txt"
    listing.write_text("clip.mp4\n")
    argv = ["prep", "--model", str(tmp_path / "model"), "--videos", str(tmp_path), "--list", str(listing),
            "--out", str(out), "--windows", "1", "--preview", "4"]
    monkeypatch.setattr(sys, "argv", argv + (["--reuse-target", str(old)] if reuse else []))
    if reuse == "other_fps":
        with pytest.raises(ValueError, match="fps"):
            prep.main()
        return
    prep.main()
    sample = prep.CachedLatents(out)[0]
    assert torch.equal(sample["drive"], drive) and torch.equal(sample["mask"], mask)
    assert sample["meta"]["blur_sigma"] == 2.0 and sample["meta"]["dilation"] == 1
    assert len(encoded) == (0 if reuse else 1)
    if reuse:
        assert torch.equal(sample["target"], old_target)
    else:
        assert encoded[0].tolist() == [99] + list(range(1, 33))
    assert len(previews) == 1


def launcher_env(tmp_path):
    root, work, binary = tmp_path / "repo", tmp_path / "work", tmp_path / "env/bin"
    (root / "train").mkdir(parents=True)
    work.mkdir()
    binary.mkdir(parents=True)
    script = root / "train/run_wa2_ti2v5b.sh"
    script.write_text(Path(REPO, "train/run_wa2_ti2v5b.sh").read_text())
    for part in ("transformer", "vae"):
        folder = tmp_path / "model" / part
        folder.mkdir(parents=True)
        (folder / "config.json").write_text('{}')
    (work / "pass.txt").write_text("a.mp4\nb.mp4\n\nc.mp4\n")
    (root / "ckpts/sam2.1-hiera-large").mkdir(parents=True)
    env = dict(os.environ, WORK=str(work), CKPT_5B=str(tmp_path / "model"), VIDEOS=str(tmp_path / "videos"),
               HOLDOUT="2", CKPT_DIR=str(tmp_path / "ckpts"),
               GPUS="0,1", ENV_PREFIX=str(binary.parent), STEPS="4", RUN="run_wa2", CACHE="cache_wa2",
               WEIGHTS="fp16", BLUR_SIGMA="2.0", T5_WAN=str(tmp_path / "t5.pth"),
               T5_TOKENIZER=str(tmp_path / "tokenizer"), REUSE_TARGET="old_cache", FRAMES="33")
    return script, work, binary, env


def test_launcher_stubbed_stages_and_resume(tmp_path):
    script, work, binary, env = launcher_env(tmp_path)
    stub = f"#!{sys.executable}\n" + '''import json, os, sys
from pathlib import Path
work = Path(os.environ["WORK"])
with open(work / "calls.jsonl", "a") as f:
    f.write(json.dumps(dict(argv=sys.argv, gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
                           offline=os.environ.get("HF_HUB_OFFLINE"))) + "\\n")
if "train/train_wa2_ti2v5b.py" in sys.argv:
    out = Path(sys.argv[sys.argv.index("--out") + 1])
    out.mkdir(parents=True, exist_ok=True)
    (out / "qkv_000004.safetensors").touch()
'''
    for name in ("python", "torchrun"):
        (binary / name).write_text(stub)
        (binary / name).chmod(0o755)
    result = subprocess.run(["/bin/bash", str(script), "prep", "train", "infer"], env=env, capture_output=True,
                            text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [json.loads(line) for line in (work / "calls.jsonl").read_text().splitlines()]
    assert len(calls) == 12
    assert (work / "wa2_train.txt").read_text() == "a.mp4\n"
    assert (work / "wa2_holdout.txt").read_text() == "b.mp4\nc.mp4\n"
    prep_calls = [c for c in calls if "train/prep_wa2_ti2v5b.py" in c["argv"]]
    assert len(prep_calls) == 3 and "--text-only" in prep_calls[0]["argv"]
    for c in prep_calls:
        assert c["argv"][c["argv"].index("--list") + 1] == str(work / "wa2_train.txt")
        assert "--t5-wan" in c["argv"] and "--blur-sigma" in c["argv"]
        assert c["argv"][c["argv"].index("--gdino") + 1] == "IDEA-Research/grounding-dino-base"
        assert c["argv"][c["argv"].index("--sam2") + 1] == str(script.parents[1] / "ckpts/sam2.1-hiera-large")
    assert {c["gpu"] for c in prep_calls[1:]} == {"0", "1"}
    assert all(str(work / "old_cache") in c["argv"] for c in prep_calls[1:])
    training = next(c for c in calls if "train/train_wa2_ti2v5b.py" in c["argv"])
    assert training["gpu"] == "0,1" and "--nproc_per_node" in training["argv"]
    inference = [c for c in calls if "train/infer_wa2_ti2v5b.py" in c["argv"]]
    assert len(inference) == 8 and all("--ref-from-drive" in c["argv"] for c in inference)
    assert {c["gpu"] for c in inference} == {"0", "1"}
    arg = lambda c, flag: c["argv"][c["argv"].index(flag) + 1] if flag in c["argv"] else None
    names = {arg(c, "--name"): c for c in inference}
    assert set(names) == {f"{kind}_{clip}_{control}" for kind, clip in (("held", "b"), ("train", "a"))
                          for control in ("drive", "none", "reverse", "wrong")}
    assert arg(names["held_b_drive"], "--drive").endswith("videos/b.mp4")
    assert arg(names["held_b_wrong"], "--drive-clip").endswith("videos/c.mp4")
    assert arg(names["train_a_wrong"], "--drive-clip").endswith("videos/b.mp4")
    assert "--no-drive" in names["held_b_none"]["argv"] and "--reverse-drive" in names["train_a_reverse"]["argv"]
    (work / "pass.txt").write_text("a.mp4\nb.mp4\nc.mp4\nd.mp4\n")
    changed = subprocess.run(["/bin/bash", str(script), "prep"], env=env, capture_output=True, text=True, timeout=10)
    assert changed.returncode == 1 and "other clips" in changed.stderr
    assert (work / "wa2_holdout.txt").read_text() == "b.mp4\nc.mp4\n" and not list(work.glob(".wa2_split*"))
    again = subprocess.run(["/bin/bash", str(script), "train"], env=env, capture_output=True, text=True, timeout=10)
    assert again.returncode == 0 and "skipped" in again.stdout
    assert len((work / "calls.jsonl").read_text().splitlines()) == 12


@pytest.mark.parametrize("signal_name", [None, "SIGINT", "SIGTERM"])
def test_launcher_cleanup_failure_and_signals(tmp_path, signal_name):
    import signal
    import time
    script, work, binary, env = launcher_env(tmp_path)
    (binary / "python").write_text(f"#!{sys.executable}\n" + '''import os, signal, sys, time
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
    (binary / "python").chmod(0o755)
    env["FAIL_SHARD"] = "0" if signal_name else "1"
    proc = subprocess.Popen(["/bin/bash", str(script), "prep"], env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    try:
        if signal_name:
            deadline = time.monotonic() + 5
            while len(list(work.glob("pid_*"))) < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            assert len(list(work.glob("pid_*"))) == 2
            proc.send_signal(getattr(signal, signal_name))
        output, _ = proc.communicate(timeout=10)
        assert proc.returncode == {None: 1, "SIGINT": 130, "SIGTERM": 143}[signal_name], output
        assert (work / "stopped_1").exists()
        for path in work.glob("pid_*"):
            with pytest.raises(ProcessLookupError):
                os.kill(int(path.read_text()), 0)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        for path in work.glob("pid_*"):
            try:
                os.kill(int(path.read_text()), signal.SIGTERM)
            except ProcessLookupError:
                pass


def test_fp16_storage_keeps_qkv_fp32_modules_and_rope(tmp_path):
    model, ref = tiny_transformer(), tiny_transformer()
    wa.to_fp16_storage(model)
    keep = set(model._keep_in_fp32_modules)
    for (name, param), (_, original) in zip(model.named_parameters(), ref.named_parameters()):
        exact = wa.QKV_NAME.fullmatch(name) or keep & set(name.split("."))
        assert param.dtype == (torch.float32 if exact else torch.float16), name
        if exact:
            assert torch.equal(param, original)
    for (name, buf), (_, original) in zip(model.named_buffers(), ref.named_buffers()):
        assert buf.dtype == original.dtype and torch.equal(buf, original), name
    assert any(p.dtype == torch.float16 for p in model.parameters())


def test_fp16_storage_keeps_only_qkv_fp32(tmp_path):
    dataset = write_cache(tmp_path / "cache")
    model = tiny_transformer().half()
    train.train(model, dataset, 0, log=lambda *_: None)
    for name, param in model.named_parameters():
        assert param.dtype == (torch.float32 if wa.QKV_NAME.fullmatch(name) else torch.float16)


def test_qkv_export_load_and_base_identity(tmp_path):
    dataset = write_cache(tmp_path / "cache")
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text('{}')
    model = tiny_transformer()
    train.train(model, dataset, 1, out=tmp_path / "run", model_dir=str(base), log=lambda *_: None)
    path = str(tmp_path / "run/qkv_000001.safetensors")
    loaded = tiny_transformer()
    inf.load_qkv(loaded, path, str(base), 2.0)
    for name, p in train.qkv_state_dict(model).items():
        torch.testing.assert_close(p, train.qkv_state_dict(loaded)[name], atol=0, rtol=0)
    with pytest.raises(ValueError, match="blur sigma"):
        inf.load_qkv(loaded, path, str(base), 3.0)
    (base / "config.json").write_text('{"changed": true}')
    with pytest.raises(ValueError, match="different base"):
        inf.load_qkv(loaded, path, str(base), 2.0)
