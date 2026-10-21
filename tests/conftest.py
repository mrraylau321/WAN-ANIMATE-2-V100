"""Shared helpers for the CPU-only V100-port tests (no GPU, no weights needed)."""
import os
import subprocess
import sys
import types
import warnings

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM_COMMIT = "3ad2fef7d61d6200c9c653e0fe47be7616b323f3"  # Wan-Video/Wan-Animate-2 HEAD used for the port

if REPO not in sys.path:
    sys.path.insert(0, REPO)
warnings.filterwarnings("ignore", message=".*CUDA is not available.*")
warnings.filterwarnings("ignore", message=".*In CPU autocast, but the target dtype is not supported.*")

# Tiny but structurally complete config (head_dim 32 keeps the 3-way RoPE split valid).
TINY = dict(patch_size=(1, 2, 2), in_dim=36, dim=64, ffn_dim=128, freq_dim=32, text_dim=32,
            num_heads=2, num_layers=3, window_size=(-1, -1), qk_norm=True, cross_attn_norm=True,
            eps=1e-6, use_img_emb=True, refer_offset_t=1, refer_offset_h=0, refer_offset_w=-1,
            refer_stride=1, sparse_type=0, use_context_parallel=False, log_scale=-1.3)


@pytest.fixture(autouse=True)
def _fp16_check_cache(tmp_path, monkeypatch):
    """Keep fp16-check pass markers inside each test's tmp dir (never ~/.cache, never shared)."""
    monkeypatch.setenv("WAN_FP16_CHECK_CACHE", str(tmp_path / "fp16_check"))
    monkeypatch.delenv("WAN_FP16_CHECK", raising=False)


def git_show(path, commit=UPSTREAM_COMMIT):
    return subprocess.run(["git", "-C", REPO, "show", f"{commit}:{path}"], check=True,
                          capture_output=True, text=True).stdout


@pytest.fixture(scope="session")
def upstream(tmp_path_factory):
    """Import the *unmodified* upstream model + attention modules (from git) as package wan_upstream."""
    try:
        model_src = git_show("wanxiang/models/wan_animate_2_model.py")
        attn_src = git_show("wanxiang/models/attention.py")
    except Exception as e:  # pragma: no cover
        pytest.skip(f"upstream sources unavailable from git: {e}")
    root = tmp_path_factory.mktemp("upstream")
    pkg = root / "wan_upstream" / "models"
    pkg.mkdir(parents=True)
    (root / "wan_upstream" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg / "attention.py").write_text(attn_src)
    (pkg / "wan_animate_2_model.py").write_text(model_src)
    sys.path.insert(0, str(root))
    import importlib
    return importlib.import_module("wan_upstream.models.wan_animate_2_model")


def stub_heavy_video_deps():
    """Make pipelines/wan_animate_2_pipeline.py importable on a CPU-only Mac:
    decord has no macOS arm64 wheel (not used by the code under test), and wanxiang/eval_i2v.py
    evaluates torch.cuda.current_device() as a default argument at import time."""
    import importlib.machinery
    if "decord" not in sys.modules:
        mod = types.ModuleType("decord")
        mod.__spec__ = importlib.machinery.ModuleSpec("decord", None)
        mod.VideoReader = object
        sys.modules["decord"] = mod
    if not torch.cuda.is_available():
        torch.cuda.current_device = lambda: 0


def make_forward_inputs(clip_len=9, area=(48, 32), lat_hw=(4, 4), seed=0, text_dim=32):
    """Inputs mirroring inference_core for one clip. area=(width, height) of origin_area;
    lat_hw is the actual latent H, W (smaller than area/8 exercises the zero-padded positions)."""
    g = torch.Generator().manual_seed(seed)
    T = clip_len + 1
    lat_t = T // 4 + 1 + 1
    lh, lw = lat_hw
    ref_t = (clip_len - 1) // 4 + 1
    noise = [torch.randn(16, lat_t, lh, lw, generator=g)]
    y = [torch.randn(20, lat_t, lh, lw, generator=g)]
    cond_lat = torch.randn(1, 16, ref_t, lh, lw, generator=g)
    cond_y = [torch.randn(20, ref_t, lh, lw, generator=g)]
    ctx = [torch.randn(7, text_dim, generator=g)]
    ctx_ref = [torch.randn(5, text_dim, generator=g)]
    clip_fea = torch.randn(1, 257, 1280, generator=g)
    clip_fea_ref = torch.randn(1, 257, 1280, generator=g)
    seq_len = lat_t * (lh // 2) * (lw // 2)
    seq_len_ref = ref_t * (lh // 2) * (lw // 2)
    grid_sizes = torch.tensor([[lat_t, lh // 2, lw // 2]])
    grid_sizes_ref = torch.tensor([[ref_t, lh // 2, lw // 2]])
    t = torch.tensor([937])
    return dict(noise=noise, y=y, cond_lat=cond_lat, cond_y=cond_y, ctx=ctx, ctx_ref=ctx_ref,
                clip_fea=clip_fea, clip_fea_ref=clip_fea_ref, seq_len=seq_len, seq_len_ref=seq_len_ref,
                grid_sizes=grid_sizes, grid_sizes_ref=grid_sizes_ref, t=t, clip_len=clip_len,
                area=list(area))


def run_model(model, inp, call=None):
    """forward_ref then forward_gen, exactly as inference_core calls them (call wraps each forward)."""
    call = call or (lambda f, step, branch: f())
    k_cache, v_cache = {}, {}
    call(lambda: model(inp["cond_lat"], grid_sizes=inp["grid_sizes"], k_cache=k_cache, v_cache=v_cache,
                       t=inp["t"], method="forward_ref", context_ref=inp["ctx_ref"],
                       seq_len_ref=inp["seq_len_ref"], clip_fea_ref=inp["clip_fea_ref"],
                       y_ref=inp["cond_y"]), "ref", "ref")
    return call(lambda: model(inp["noise"], k_cache=k_cache, v_cache=v_cache, t=inp["t"],
                              grid_sizes_ref=inp["grid_sizes_ref"], method="forward_gen",
                              context=inp["ctx"], seq_len=inp["seq_len"], clip_fea=inp["clip_fea"],
                              y=inp["y"], origin_len=inp["clip_len"], origin_area=inp["area"]), 0, "cond")
