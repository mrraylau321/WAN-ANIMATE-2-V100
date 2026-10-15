"""'Load dict must not break': the V100 port keeps every parameter key, shape and the strict
loading path of upstream. CPU only; no weights downloaded (the release checkpoint is represented
by its safetensors header: names/dtypes/shapes)."""
import copy
import json
import os
import socket

import pytest
import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from safetensors.torch import save_file

from conftest import REPO, TINY, make_forward_inputs, run_model, stub_heavy_video_deps
from wanxiang.models import attention as A
from wanxiang.models.wan_animate_2_model import WanAnimate2Transformer
from wanxiang.wanxiang_animate_2_arch import WanxiangAnimate2Transformer

HEADER = json.load(open(os.path.join(REPO, "tests", "fixtures", "wan_animate_2_ckpt_header.json")))
YAMLS = ["wan_animate_2.yaml", "wan_animate_2_distillation.yaml",
         "wan_animate_2_v100.yaml", "wan_animate_2_distillation_v100.yaml"]


def sig(model):
    return {k: (tuple(v.shape), v.dtype) for k, v in model.state_dict().items()}


def test_tiny_key_set_identical_to_upstream(upstream):
    up = upstream.WanAnimate2Transformer(**TINY)
    assert sig(WanAnimate2Transformer(**TINY)) == sig(up)
    assert sig(WanxiangAnimate2Transformer(**TINY)) == sig(up)
    assert [n for n, _ in WanAnimate2Transformer(**TINY).named_buffers()] == [n for n, _ in up.named_buffers()]


@pytest.mark.parametrize("yaml_name", YAMLS)
def test_full_config_matches_release_checkpoint(yaml_name, upstream):
    """Full-size model (meta device) has exactly the 1303 tensors of the released bf16 checkpoints."""
    cfg = OmegaConf.to_container(OmegaConf.load(os.path.join(REPO, "infer", yaml_name)).model.transformer)
    cfg.pop("type")
    want = {k: tuple(s) for k, (_, s) in HEADER["tensors"].items()}
    assert {dt for dt, _ in HEADER["tensors"].values()} == {"BF16"}
    with torch.device("meta"):
        patched = WanxiangAnimate2Transformer(**cfg)
        ups = upstream.WanAnimate2Transformer(**cfg)
    got = {k: tuple(v.shape) for k, v in patched.state_dict().items()}
    assert got == want
    assert {k: tuple(v.shape) for k, v in ups.state_dict().items()} == want
    assert sum(v.numel() for v in patched.state_dict().values()) == 16394878784


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _save_bf16_ckpt(path, mutate=None):
    torch.manual_seed(0)
    m = WanxiangAnimate2Transformer(**TINY)
    for p in m.parameters():
        p.data.normal_(0, 0.05)
    sd = {k: v.to(torch.bfloat16).contiguous() for k, v in m.state_dict().items()}  # like the release
    if mutate:
        mutate(sd)
    save_file(sd, path)
    return sd


def _run(tmp_path, scenario, ckpt, world=2):
    from dist_helpers import load_worker
    ctx = mp.start_processes(load_worker, args=(world, _free_port(), scenario, ckpt, str(tmp_path)),
                             nprocs=world, join=False, start_method="spawn")
    for _ in range(240):
        if ctx.join(timeout=1):
            break
    else:  # pragma: no cover
        for p in ctx.processes:
            p.kill()
        pytest.fail(f"{scenario}: workers hung")
    return [json.load(open(tmp_path / f"{scenario}_rank{r}.json")) for r in range(world)]


def test_pipeline_loader_strict_roundtrip_fp16(tmp_path):
    """bf16 safetensors -> pipeline.load_safetensors -> FSDP2 (2 ranks) -> set_model_state_dict(strict)."""
    stub_heavy_video_deps()
    from pipelines.wan_animate_2_pipeline import load_safetensors
    ckpt = str(tmp_path / "tiny_bf16.safetensors")
    sd = _save_bf16_ckpt(ckpt)
    loaded = load_safetensors(ckpt)
    assert set(loaded) == set(sd) and all(torch.equal(loaded[k], sd[k]) for k in sd)
    res = _run(tmp_path, "load_fp16", ckpt)
    for r in res:
        assert r["ok"], r.get("tb")
        assert r["param_type"] == "DTensor" and r["local_rows"] == TINY["ffn_dim"] // 2
        assert r["storage_dtype"] == "torch.float32"  # upstream storage; fp16 cast happens per forward
        assert r["prec"].startswith("dit=fp16 dit_output=fp32")
    full = torch.load(tmp_path / "load_fp16_sd.pt")
    assert set(full) == set(sd)
    for k in sd:  # bit-exact: bf16 values upcast to the fp32 shards, nothing dropped or renamed
        assert torch.equal(full[k], sd[k].float()), k


@pytest.mark.parametrize("scenario,expected_dtype", [
    ("shard_bf16", "torch.bfloat16"),
    ("shard_bf16_island", "torch.float32"),
])
def test_16gb_bf16_storage_strict_roundtrip_and_fp32_island(tmp_path, scenario, expected_dtype):
    """Checkpoint-exact BF16 shards; a selected FP32 island alone retains FP32."""
    ckpt = str(tmp_path / "tiny_bf16.safetensors")
    sd = _save_bf16_ckpt(ckpt)
    res = _run(tmp_path, scenario, ckpt)
    for r in res:
        assert r["ok"], r.get("tb")
        assert r["storage_dtype"] == expected_dtype
        assert r["other_storage_dtype"] == "torch.bfloat16"
        assert r["local_allocation"].get("bfloat16", 0) > 0
        if scenario == "shard_bf16_island":
            assert r["local_allocation"].get("float32", 0) > 0
            assert "blocks.1" in r["fp32_units"]
        else:
            assert r["local_allocation"].get("float32", 0) == 0
    full = torch.load(tmp_path / f"{scenario}_sd.pt")
    assert set(full) == set(sd)
    for key, released in sd.items():
        is_island = scenario == "shard_bf16_island" and key.startswith("blocks.1.")
        assert full[key].dtype == (torch.float32 if is_island else torch.bfloat16), key
        assert torch.equal(full[key], released.float() if is_island else released), key


def test_16gb_bf16_shards_fp32_forward_matches_reference(tmp_path):
    """FSDP2 loads checkpoint BF16 shards and computes a tiny FP32 reference exactly."""
    ckpt = str(tmp_path / "tiny_bf16.safetensors")
    sd = _save_bf16_ckpt(ckpt)
    res = _run(tmp_path, "shard_bf16_forward", ckpt)
    assert all(r["ok"] for r in res), [r.get("tb") for r in res]
    ref_model = WanAnimate2Transformer(**TINY).eval()
    ref_model.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)
    A.set_attention_backend("sdpa")
    A.set_compute_dtype(torch.float32)
    try:
        with torch.no_grad():
            ref = run_model(ref_model, make_forward_inputs())[0]
    finally:
        A.set_attention_backend("auto")
        A.set_compute_dtype(torch.bfloat16)
    for rank in range(2):
        out = torch.load(tmp_path / f"shard_bf16_forward_out_rank{rank}.pt")
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_16gb_bf16_shards_fp16_compute_tiny_forward(tmp_path):
    """End-to-end tiny CPU stand-in uses BF16 FSDP shards and FP16 DiT compute.

    This cannot certify numeric quality or actual V100 CUDA kernels.
    """
    ckpt = str(tmp_path / "tiny_bf16.safetensors")
    _save_bf16_ckpt(ckpt)
    res = _run(tmp_path, "shard_bf16_fp16_forward", ckpt)
    assert all(r["ok"] for r in res), [r.get("tb") for r in res]
    for rank in range(2):
        out = torch.load(tmp_path / f"shard_bf16_fp16_forward_out_rank{rank}.pt")
        assert torch.isfinite(out).all()


def test_16gb_rejects_fp16_checkpoint_storage():
    """FP16 persistent storage would silently truncate BF16 release values."""
    from wanxiang.utils.precision import parse_dtype
    assert parse_dtype("fp16") == torch.float16
    # The pipeline checks this *before* making any FSDP or CUDA allocations.
    stub_heavy_video_deps()
    from pipelines.wan_animate_2_pipeline import build_dit
    from wanxiang.utils.precision import resolve_precision
    cfg = OmegaConf.create({"test_cfg": {"dit_shard_dtype": "fp16"}})
    with pytest.raises(ValueError, match="dit_shard_dtype"):
        build_dit(cfg, "cpu", resolve_precision(cfg.test_cfg, torch.device("cpu")), gpu=0)


def test_pipeline_loader_sharded_forward_matches_unsharded(tmp_path):
    ckpt = str(tmp_path / "tiny_bf16.safetensors")
    sd = _save_bf16_ckpt(ckpt)
    res = _run(tmp_path, "forward_fp32", ckpt)
    assert all(r["ok"] for r in res), [r.get("tb") for r in res]
    ref_model = WanAnimate2Transformer(**TINY).eval()
    ref_model.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)
    A.set_attention_backend("sdpa")
    A.set_compute_dtype(torch.float32)
    try:
        with torch.no_grad():
            ref = run_model(ref_model, make_forward_inputs())[0]
    finally:
        A.set_attention_backend("auto")
        A.set_compute_dtype(torch.bfloat16)
    for r in range(2):
        out = torch.load(tmp_path / f"forward_fp32_out_rank{r}.pt")
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("scenario,mutate,key", [
    ("extra_key", lambda sd: sd.__setitem__("blocks.0.block.extra.weight", torch.zeros(2, dtype=torch.bfloat16)),
     "blocks.0.block.extra.weight"),
    ("missing_key", lambda sd: sd.pop("blocks.2.block.ffn.2.bias"), "blocks.2.block.ffn.2.bias"),
])
def test_pipeline_loader_is_still_strict(tmp_path, scenario, mutate, key):
    """An extra or missing tensor makes the (unchanged) strict loader fail loudly, naming the key."""
    ckpt = str(tmp_path / f"{scenario}.safetensors")
    _save_bf16_ckpt(ckpt, mutate)
    res = _run(tmp_path, scenario, ckpt)
    r0 = res[0]
    assert not r0["ok"] and key in r0["error"] + r0.get("tb", ""), r0


def _spawn_json(fn, prefix, tmp_path, world=2, timeout=240):
    ctx = mp.start_processes(fn, args=(world, _free_port(), str(tmp_path)), nprocs=world, join=False,
                             start_method="spawn")
    for _ in range(timeout):
        if ctx.join(timeout=1):
            break
    else:  # pragma: no cover
        for p in ctx.processes:
            p.kill()
        pytest.fail(f"{fn.__name__}: workers hung")
    return [json.load(open(tmp_path / f"{prefix}_rank{r}.json")) for r in range(world)]


def test_t5_bf16_storage_fp32_compute_nothing_retained(tmp_path):
    """Review F3b: FSDP1 keeps the ROOT unit's unsharded params after forward (4.2 GB fp32
    for umT5-XXL). With the token embedding and final norm wrapped as their own units (V100 path)
    every unit is resharded after encoding. Weights stay bf16 as stored; compute is exact fp32."""
    from dist_helpers import t5_worker
    res = _spawn_json(t5_worker, "t5", tmp_path)
    for r in res:
        assert r["ok"], r.get("tb")
        assert r["param_dtypes"] == ["torch.bfloat16"] and r["local_dtype"] == ["torch.bfloat16"]
        assert r["out_dtype"] == "torch.float32" and r["max_diff"] < 1e-5
        assert "<root>" not in r["retained"] and set(r["retained"].values()) == {0}, r["retained"]
        assert any("token_embedding" in n for n in r["retained"]) and r["keys_equal"]


def test_context_parallel_kv_cache_owns_storage_and_matches(tmp_path):
    """Review F3a: with context parallel the cached K/V must not keep the all_to_all
    [q|k|v] buffer alive. Also: the sp=2 CP forward equals the single-process forward."""
    from dist_helpers import cp_worker
    res = _spawn_json(cp_worker, "cp", tmp_path)
    for r in res:
        assert r["ok"], r.get("tb")
        for k_store, k_bytes, v_store, v_bytes, shared in r["cache"]:
            assert (k_store, v_store, shared) == (k_bytes, v_bytes, False)
    torch.manual_seed(0)
    model = WanAnimate2Transformer(**TINY).eval()
    for p in model.parameters():
        p.data.normal_(0, 0.05)
    A.set_attention_backend("sdpa")
    A.set_compute_dtype(torch.float32)
    try:
        with torch.no_grad():
            ref = run_model(model, make_forward_inputs())[0]
    finally:
        A.set_attention_backend("auto")
        A.set_compute_dtype(torch.bfloat16)
    for r in range(2):
        torch.testing.assert_close(torch.load(tmp_path / f"cp_out_rank{r}.pt"), ref, atol=1e-5, rtol=1e-5)


# ------------------------------------------------------------------ Review F7: YAML families
@pytest.mark.parametrize("cap", [(8, 0), (9, 0), (7, 0)])
def test_v100_yaml_auto_equals_upstream_yaml(cap, monkeypatch):
    """Both YAML families resolve to the same runtime recipe on a mocked sm80/sm90 and sm70;
    on sm>=80 that recipe is exactly upstream's."""
    from wanxiang.utils.precision import resolve_precision
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d=None: cap)
    monkeypatch.setattr(A, "FLASH_VER", 2)
    A._auto_backend.cache_clear()
    dev = torch.device("cuda", 0)
    resolved = {}
    for name in YAMLS:
        tc = OmegaConf.load(os.path.join(REPO, "infer", name)).test_cfg
        p = resolve_precision(tc, dev)
        A.set_attention_backend(tc.get("attn_backend", None) or "auto")
        resolved[name] = (p.describe(), A.attention_backend(dev), str(tc.param_dtype), str(tc.reduce_dtype),
                          str(tc.buffer_dtype), tc.sp_size, tc.sharding_size, tc.sharding_strategy)
    A.set_attention_backend("auto")
    A._auto_backend.cache_clear()
    assert resolved["wan_animate_2.yaml"] == resolved["wan_animate_2_v100.yaml"]
    assert resolved["wan_animate_2_distillation.yaml"] == resolved["wan_animate_2_distillation_v100.yaml"]
    desc, backend = resolved["wan_animate_2_v100.yaml"][:2]
    if cap[0] >= 8:
        assert desc == "dit=bf16 dit_output=bf16 t5=bf16 vae=bf16 clip=fp16 guard=off clip_text_on_gpu=True"
        assert backend == "flash"
    else:
        assert desc == "dit=fp16 dit_output=fp32 t5=fp32 vae=fp32 clip=fp16 guard=first_step clip_text_on_gpu=False"
        assert backend == "sdpa"


def test_dtype_bf16_forces_upstream_recipe_with_v100_yaml(monkeypatch):
    from wanxiang.utils.precision import resolve_precision
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d=None: (7, 0))
    tc = OmegaConf.load(os.path.join(REPO, "infer", "wan_animate_2_v100.yaml")).test_cfg
    tc.dit_dtype = "bf16"
    assert resolve_precision(tc, torch.device("cuda", 0)).describe() == \
        "dit=bf16 dit_output=bf16 t5=bf16 vae=bf16 clip=fp16 guard=off clip_text_on_gpu=True"


# ------------------------------------------------------------------ Review r2 N2: no fp16 T5
@pytest.mark.parametrize("cfg", [
    {"t5_dtype": "fp16"},                                          # V100 path (fp16 DiT)
    {"t5_dtype": "float16", "dit_dtype": "bf16"},                  # upstream recipe + override
    {"t5_dtype": "fp16", "dit_dtype": "fp32"},
    {"dit_dtype": "bf16", "param_dtype": "float16"},               # T5 follows param_dtype upstream
])
def test_fp16_t5_is_refused_on_every_path(cfg, monkeypatch):
    from wanxiang.utils.precision import resolve_precision
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d=None: (7, 0))
    with pytest.raises(ValueError, match=r"T5 in fp16 is not supported.*fp16_clamp.*fp32.*bf16"):
        resolve_precision(OmegaConf.create(cfg), torch.device("cuda", 0))


@pytest.mark.parametrize("t5", ["fp32", "bf16", "auto"])
def test_fp32_and_bf16_t5_are_allowed(t5, monkeypatch):
    from wanxiang.utils.precision import resolve_precision
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d=None: (7, 0))
    p = resolve_precision(OmegaConf.create({"t5_dtype": t5}), torch.device("cuda", 0))
    assert p.t5 == {"fp32": torch.float32, "bf16": torch.bfloat16, "auto": torch.float32}[t5]


# ------------------------------------------------------------------ Review r2 R4
def test_kv_clone_only_on_the_port_path(monkeypatch):
    """flash backend (sm>=80 upstream path): the cache keeps upstream's tensors (views, no copy);
    SDPA backend (port): independent storage."""
    from wanxiang.models import wan_animate_2_model as M
    torch.manual_seed(0)
    model = WanAnimate2Transformer(**TINY).eval()
    inp = make_forward_inputs()
    A.set_compute_dtype(torch.float32)
    try:
        for backend, is_view in (("flash", True), ("sdpa", False)):
            monkeypatch.setattr(M, "attention_backend", lambda d, b=backend: b)
            if backend == "flash":  # reference math in place of the flash kernel (CPU)
                def ref_flash(q, k, v, k_lens=None, **kw):
                    L = k.size(1) if k_lens is None else int(k_lens[0])
                    return torch.nn.functional.scaled_dot_product_attention(
                        q.transpose(1, 2), k[:, :L].transpose(1, 2), v[:, :L].transpose(1, 2)).transpose(1, 2)
                monkeypatch.setattr(M, "flash_attention", ref_flash)
            else:
                monkeypatch.undo()
                A.set_attention_backend("sdpa")
            kc, vc = {}, {}
            with torch.no_grad():
                model(inp["cond_lat"], grid_sizes=inp["grid_sizes"], k_cache=kc, v_cache=vc, t=inp["t"],
                      method="forward_ref", context_ref=inp["ctx_ref"], seq_len_ref=inp["seq_len_ref"],
                      clip_fea_ref=inp["clip_fea_ref"], y_ref=inp["cond_y"])
            assert all((kc[i]._base is not None) == is_view and (vc[i]._base is not None) == is_view
                       for i in kc), backend
    finally:
        A.set_attention_backend("auto")
        A.set_compute_dtype(torch.bfloat16)
