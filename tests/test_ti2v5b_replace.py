"""scripts/ti2v5b_replace.py on tiny random Wan2.2-shaped models (diffusers): geometry, TI2V first frame, pinning."""
import os
import sys
import types

import numpy as np
import pytest
import torch

from conftest import REPO

pytest.importorskip("diffusers")
sys.path.insert(0, os.path.join(REPO, "scripts"))
import ti2v5b_replace as tr  # noqa: E402


def tiny_models():
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanTransformer3DModel
    torch.manual_seed(0)
    vae = AutoencoderKLWan(base_dim=16, decoder_base_dim=16, z_dim=48, dim_mult=[1, 2, 4, 4], num_res_blocks=1,
                           attn_scales=[], temperal_downsample=[False, True, True], in_channels=12, out_channels=12,
                           patch_size=2, is_residual=True, scale_factor_spatial=16, scale_factor_temporal=4,
                           latents_mean=[0.0] * 48, latents_std=[1.0] * 48)
    transformer = WanTransformer3DModel(patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=8,
                                        in_channels=48, out_channels=48, text_dim=16, freq_dim=16, ffn_dim=32,
                                        num_layers=1, cross_attn_norm=True, qk_norm="rms_norm_across_heads")
    # the published Wan2.2-TI2V-5B scheduler config
    sched = UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=5.0,
                                    num_train_timesteps=1000, solver_order=2, predict_x0=True, final_sigmas_type="zero")
    return vae, transformer, sched


def test_latent_keep_groups_frames_like_the_vae():
    masks = np.zeros((9, 64, 32), bool)
    masks[3, 16:20, 16:20] = True                       # pixel frame 3 -> latent frame 1, cell (1, 1)
    keep = tr.latent_keep(masks, grow=0)
    assert keep.shape == (1, 1, 3, 4, 2)
    assert keep[0, 0, 1, 1, 1] == 0 and keep[0, 0, 1].sum() == 7
    assert keep[0, 0, 0].min() == 1 and keep[0, 0, 2].min() == 1


def test_denoise_pins_the_background_and_keeps_the_first_frame():
    _, transformer, sched = tiny_models()
    g = torch.Generator().manual_seed(1)
    z = torch.randn(1, 48, 3, 4, 2, generator=g)
    cond0 = torch.randn(1, 48, 1, 4, 2, generator=g)
    keep = torch.ones(1, 1, 3, 4, 2)
    keep[..., 1:3, 0:1] = 0
    ctx = torch.randn(1, 8, 16)
    out = tr.denoise(transformer, sched, z, cond0, keep, ctx, ctx, steps=10, strength=0.8, guide=2.0, free_steps=0,
                     generator=torch.Generator().manual_seed(2), dtype=torch.float32, log=lambda *_: None)
    assert out.shape == z.shape and torch.isfinite(out).all()
    assert torch.allclose(out[:, :, :1], cond0)                                  # g0 slot
    bg = keep.bool().expand_as(z).clone()
    bg[:, :, 0] = False
    assert torch.allclose(out[bg], z[bg], atol=1e-4)                             # background = driving latents
    assert not torch.allclose(out[~keep.bool().expand_as(z)], z[~keep.bool().expand_as(z)], atol=1e-2)


def test_run_end_to_end_on_tiny_models(tmp_path):
    import cv2
    vae, transformer, sched = tiny_models()
    frames = [np.full((128, 72, 3), 30 + 10 * k, np.uint8) for k in range(12)]
    drive = str(tmp_path / "drive.mp4")
    tr.write_video(drive, frames, 24)
    masks = np.zeros((12, 128, 72), np.uint8)
    masks[:, 40:90, 20:50] = 255
    np.savez_compressed(tmp_path / "m.npz", masks=masks)
    cv2.imwrite(str(tmp_path / "g0.png"), np.full((256, 144, 3), 200, np.uint8))
    a = types.SimpleNamespace(drive=drive, masks=str(tmp_path / "m.npz"), g0=str(tmp_path / "g0.png"), fps=24,
                              frames=121, width=32, height=64, dilate=3, grow=0, no_blend=False, seed=0, steps=4,
                              strength=0.8, guide=1.0, free_steps=1, out=str(tmp_path / "out"))
    ctx = torch.randn(1, 8, 16)
    path = tr.run(a, transformer, vae, sched, ctx, ctx, "cpu", torch.float32, log=lambda *_: None)
    import imageio.v2 as imageio
    out = [f for f in imageio.get_reader(path)]
    assert len(out) == 9 and out[0].shape == (64, 32, 3)                         # 12 frames -> 4k+1 = 9


def test_launcher_builds_g0_then_runs_ti2v5b(tmp_path):
    import stat
    import subprocess
    fake = tmp_path / "bin"
    fake.mkdir()
    py = fake / "python"
    py.write_text('#!/bin/bash\n'
                  'if [[ "$1" == - || "$1" == -c ]]; then exec "%s" "$@"; fi\n'
                  'if [[ "$1" == scripts/grounded_sam2.py ]]; then echo "GSAM2 $*"; '
                  'while [[ $# -gt 0 ]]; do [[ "$1" == --out ]] && touch "$2"; shift; done; exit 0; fi\n'
                  'if [[ "$1" == scripts/ti2v5b_replace.py ]]; then printf "ARG %%s\\n" "$@"; exit 0; fi\n'
                  'exit 0\n' % sys.executable)
    py.chmod(py.stat().st_mode | stat.S_IEXEC)
    ck = tmp_path / "5b"
    for f in ("transformer/config.json", "vae/config.json", "text_encoder/config.json",
              "tokenizer/tokenizer_config.json", "scheduler/scheduler_config.json"):
        (ck / f).parent.mkdir(parents=True, exist_ok=True)
        (ck / f).write_text("{}")
    drive, ref = tmp_path / "clip.mp4", tmp_path / "reference.png"
    drive.write_bytes(b"")
    ref.write_bytes(b"")
    env = dict(os.environ, PATH=f"{fake}:{os.environ['PATH']}", CKPT_5B=str(ck), DRIVE=str(drive), REF_IMG=str(ref),
               PROMPT="a cat", REF_TARGET="cat. clothes", REF_PICK="union", STRENGTH="0.7",
               EXPERTS=f"{tmp_path}/a.pth@500-1000 {tmp_path}/b.pth@0-500",
               OUT_DIR=str(tmp_path / "out"), WORK_DIR=str(tmp_path / "work"))
    p = subprocess.run(["bash", os.path.join(REPO, "scripts", "run_ti2v5b_replace.sh")], env=env,
                       capture_output=True, text=True, timeout=120)
    out = p.stdout + p.stderr
    assert p.returncode == 0, out[-2000:]
    assert "GSAM2 scripts/grounded_sam2.py video" in out and "clip_gsam2_person_mask.npz" in out
    assert "--pick union" in out and "--ref-target cat. clothes" in out
    args = [line[4:] for line in out.splitlines() if line.startswith("ARG ")]
    assert args[args.index("--g0") + 1].endswith("reference_scene_union.png")
    assert args[args.index("--strength") + 1] == "0.7" and args[args.index("--model") + 1] == str(ck)
    assert [args[i + 1] for i, x in enumerate(args) if x == "--expert"] == [f"{tmp_path}/a.pth@500-1000",
                                                                          f"{tmp_path}/b.pth@0-500"]
    assert "--transformer-ckpt" not in args


def _to_original_wan_names(sd):
    """Diffusers WanTransformer3DModel keys -> original Wan checkpoint names (TI2V subset), for the loader test."""
    pairs = [("condition_embedder.time_embedder.linear_1", "time_embedding.0"),
             ("condition_embedder.time_embedder.linear_2", "time_embedding.2"),
             ("condition_embedder.text_embedder.linear_1", "text_embedding.0"),
             ("condition_embedder.text_embedder.linear_2", "text_embedding.2"),
             ("condition_embedder.time_proj", "time_projection.1"),
             ("attn2", "cross_attn"), ("attn1", "self_attn"), (".to_out.0.", ".o."), (".to_q.", ".q."),
             (".to_k.", ".k."), (".to_v.", ".v."), ("ffn.net.0.proj", "ffn.0"), ("ffn.net.2", "ffn.2"),
             ("norm2", "norm__x"), ("norm3", "norm2"), ("norm__x", "norm3")]
    out = {}
    for k, v in sd.items():
        for a, b in pairs:
            k = k.replace(a, b)
        if k == "scale_shift_table":
            k = "head.modulation"
        elif k.startswith("proj_out"):
            k = k.replace("proj_out", "head.head")
        k = k.replace(".scale_shift_table", ".modulation")
        out[k] = v
    return out


def test_transformer_ckpt_loads_original_and_prefixed_names(tmp_path):
    _, transformer, _ = tiny_models()
    cfg_dir = tmp_path / "transformer"
    transformer.save_config(cfg_dir)
    sd = transformer.state_dict()
    torch.save({"module." + k: v for k, v in sd.items()}, tmp_path / "prefixed.pth")
    torch.save({"state_dict": _to_original_wan_names(sd)}, tmp_path / "original.pth")
    for name in ("prefixed.pth", "original.pth"):
        m = tr.load_transformer_ckpt(str(tmp_path / name), str(cfg_dir), torch.float32, log=lambda *_: None)
        for k, v in m.state_dict().items():
            assert torch.equal(v, sd[k]), (name, k)
    torch.save({"blocks.0.bogus.weight": torch.zeros(1)}, tmp_path / "bad.pth")
    with pytest.raises(SystemExit):
        tr.load_transformer_ckpt(str(tmp_path / "bad.pth"), str(cfg_dir), torch.float32, log=lambda *_: None)


def test_experts_switch_by_timestep_and_keep_the_pinning():
    _, t_hi, sched = tiny_models()
    _, t_lo, _ = tiny_models()
    calls = {"hi": 0, "lo": 0}
    t_hi.register_forward_hook(lambda *_: calls.__setitem__("hi", calls["hi"] + 1))
    t_lo.register_forward_hook(lambda *_: calls.__setitem__("lo", calls["lo"] + 1))
    experts = tr.Experts([(0, 900, t_lo), (900, 1000, t_hi)], "cpu")
    g = torch.Generator().manual_seed(1)
    z, cond0 = torch.randn(1, 48, 3, 4, 2, generator=g), torch.randn(1, 48, 1, 4, 2, generator=g)
    keep = torch.ones(1, 1, 3, 4, 2)
    keep[..., 1:3, 0:1] = 0
    out = tr.denoise(experts, sched, z, cond0, keep, torch.randn(1, 8, 16), torch.randn(1, 8, 16), steps=10,
                     strength=1.0, guide=1.0, free_steps=0, generator=g, dtype=torch.float32, log=lambda *_: None)
    assert calls["hi"] > 0 and calls["lo"] > 0
    bg = keep.bool().expand_as(z).clone()
    bg[:, :, 0] = False
    assert torch.allclose(out[bg], z[bg], atol=1e-4)
    with pytest.raises(ValueError):
        tr.Experts([(0, 500, t_lo)], "cpu")(700)


def test_wan_t5_embeddings_are_padded_like_diffusers(monkeypatch):
    calls = {}

    class FakeT5:
        def __init__(self, **kw):
            calls.update(kw)

        def __call__(self, texts):
            return [torch.ones(3 + i, 8) for i in range(len(texts))]

    mod = types.ModuleType("wanxiang.eval_i2v")
    mod.T5Encoder = FakeT5
    monkeypatch.setitem(sys.modules, "wanxiang.eval_i2v", mod)
    a, b = tr.encode_prompts_wan("t5.pth", "tok", ["x", "y"], "cpu", max_len=16)
    assert a.shape == (1, 16, 8) and a[0, :3].min() == 1 and a[0, 3:].abs().sum() == 0
    assert b[0, :4].min() == 1 and b[0, 4:].abs().sum() == 0
    assert calls["checkpoint_path"] == "t5.pth" and calls["dtype"] == torch.float32 and calls["name"] == "umt5_xxl"


def test_launcher_uses_the_wan_t5_without_the_diffusers_text_encoder(tmp_path):
    import stat
    import subprocess
    fake = tmp_path / "bin"
    fake.mkdir()
    py = fake / "python"
    py.write_text('#!/bin/bash\n'
                  'if [[ "$1" == - || "$1" == -c ]]; then exec "%s" "$@"; fi\n'
                  'if [[ "$1" == scripts/grounded_sam2.py ]]; then '
                  'while [[ $# -gt 0 ]]; do [[ "$1" == --out ]] && touch "$2"; shift; done; exit 0; fi\n'
                  'if [[ "$1" == scripts/ti2v5b_replace.py ]]; then printf "ARG %%s\\n" "$@"; exit 0; fi\n'
                  'exit 0\n' % sys.executable)
    py.chmod(py.stat().st_mode | stat.S_IEXEC)
    ck = tmp_path / "5b"
    for f in ("transformer/config.json", "vae/config.json", "scheduler/scheduler_config.json"):
        (ck / f).parent.mkdir(parents=True, exist_ok=True)
        (ck / f).write_text("{}")
    t5, tok = tmp_path / "t5.pth", tmp_path / "umt5-xxl"
    t5.write_bytes(b"")
    tok.mkdir()
    drive, ref = tmp_path / "clip.mp4", tmp_path / "ref.png"
    drive.write_bytes(b"")
    ref.write_bytes(b"")
    env = dict(os.environ, PATH=f"{fake}:{os.environ['PATH']}", CKPT_5B=str(ck), DRIVE=str(drive), REF_IMG=str(ref),
               T5_WAN=str(t5), T5_TOKENIZER=str(tok), OUT_DIR=str(tmp_path / "out"), WORK_DIR=str(tmp_path / "w"))
    p = subprocess.run(["bash", os.path.join(REPO, "scripts", "run_ti2v5b_replace.sh")], env=env,
                       capture_output=True, text=True, timeout=120)
    out = p.stdout + p.stderr
    assert p.returncode == 0, out[-2000:]
    args = [line[4:] for line in out.splitlines() if line.startswith("ARG ")]
    assert args[args.index("--t5-wan") + 1] == str(t5) and args[args.index("--t5-tokenizer") + 1] == str(tok)
