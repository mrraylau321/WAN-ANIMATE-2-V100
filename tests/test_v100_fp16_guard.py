"""fp16 safety net: load-time castability check and runtime overflow guard fire when they should.
CPU only, tiny configs; large values / NaNs are injected on purpose."""
import json
import os

import pytest
import torch

from conftest import TINY, make_forward_inputs, run_model
from test_v100_state_dict import _run, _save_bf16_ckpt
from wanxiang.models import attention as A
from wanxiang.models.wan_animate_2_model import WanAnimate2Transformer
from wanxiang.utils.fp16_guard import (FP16OverflowError, OverflowGuard, check_fp16_castable,
                                       fp16_cast_stats)


class ListLog:
    def __init__(self):
        self.lines = []

    def info(self, m):
        self.lines.append(("info", m))

    def warning(self, m):
        self.lines.append(("warning", m))

    def error(self, m):
        self.lines.append(("error", m))

    def text(self, level=None):
        return "\n".join(m for l, m in self.lines if level in (None, l))


@pytest.fixture(autouse=True)
def _sdpa():
    A.set_attention_backend("sdpa")
    A.set_compute_dtype(torch.float32)
    yield
    A.set_attention_backend("auto")
    A.set_compute_dtype(torch.bfloat16)


def tiny_model(seed=0):
    torch.manual_seed(seed)
    m = WanAnimate2Transformer(**TINY).eval()
    for p in m.parameters():
        p.data.normal_(0, 0.05)
    return m


# ------------------------------------------------------------------ load time
def test_fp16_cast_thresholds_exact():
    t = torch.tensor([65504.0, 65519.0, 65520.0, 1e6, 2.9e-8, 3.0e-8, 1e-5, 1.0, 0.0])
    names, c, m, mf = fp16_cast_stats([("x.w", t)])
    nonfinite, overflow, flushed, subnormal, nonzero = c[0].tolist()
    # overflow tests the ORIGINAL magnitude: 65519 rounds down to 65504 in fp16 but is out of range
    assert (nonfinite, overflow, flushed, nonzero) == (0, 3, 1, 8)
    assert subnormal == 2  # 3e-8 -> 5.96e-8 and 1e-5: representable only as fp16 subnormals
    assert float(m[0]) == 1e6 and abs(float(mf[0]) - 2.9e-8) < 1e-12


def test_load_check_original_magnitude_overflow_fails():
    # Review F4a: 65519 would round to 65504 (finite) but exceeds the fp16 range
    with pytest.raises(FP16OverflowError, match=r"m: 1 values exceed the fp16 range"):
        check_fp16_castable([("m.weight", torch.tensor([65519.0]))], log=ListLog())


def test_load_check_underflow_tolerance_is_loud_and_configurable():
    # Review F4b (partially accepted): one flushed weight in 2000 is within the default
    # per-module tolerance (1e-3) but is reported as a WARNING with count and max |w|;
    # underflow_frac=0 makes any flush fatal.
    w = torch.ones(2000)
    w[0] = 1e-9
    log = ListLog()
    check_fp16_castable([("m.weight", w)], log=log)
    warn = log.text("warning")
    assert "1 nonzero weights become 0 in fp16 across 1 modules" in warn
    assert "m: 1/2000" in warn and "max flushed |w|=1e-09" in warn
    with pytest.raises(FP16OverflowError, match=r"m: 1/2000 nonzero values flush to 0"):
        check_fp16_castable([("m.weight", w)], underflow_frac=0.0, log=ListLog())


def test_load_check_passes_on_clean_weights():
    log = ListLog()
    per_mod = check_fp16_castable(tiny_model().named_parameters(), log=log)
    assert "blocks.1.block.ffn.0" in per_mod and "over_65504=0" in log.text()
    assert not log.text("warning")


def test_load_check_overflow_fails_loudly_and_allowlist_is_explicit():
    m = tiny_model()
    m.blocks[1].block.ffn[0].weight.data[5, 7] = 1e6
    with pytest.raises(FP16OverflowError, match=r"blocks\.1\.block\.ffn\.0: 1 values exceed the fp16 range"):
        check_fp16_castable(m.named_parameters(), log=ListLog())
    check_fp16_castable(m.named_parameters(), fp32_modules=["blocks.1"], log=ListLog())  # exempt


def test_load_check_nan_always_fails():
    m = tiny_model()
    m.blocks[0].block.self_attn.q.bias.data[0] = float("nan")
    for allow in ([], ["blocks.0"]):
        with pytest.raises(FP16OverflowError, match=r"blocks\.0\.block\.self_attn\.q: 1 non-finite"):
            check_fp16_castable(m.named_parameters(), fp32_modules=allow, log=ListLog())


def test_load_check_underflow():
    m = tiny_model()
    m.blocks[0].block.cross_attn.norm_q.weight.data[:] = 1e-9  # a whole scale vector would become 0
    with pytest.raises(FP16OverflowError, match=r"cross_attn\.norm_q: 64/64 nonzero values flush to 0"):
        check_fp16_castable(m.named_parameters(), log=ListLog())
    check_fp16_castable(m.named_parameters(), fp32_modules=["blocks.0.block.cross_attn"], log=ListLog())


@pytest.mark.parametrize("scenario", ["overflow", "overflow_allowlisted"])
def test_load_check_inside_pipeline_loader_two_ranks(tmp_path, scenario):
    """1e6 placed in the half of the tensor owned by rank 1: both ranks must see the violation."""
    ckpt = str(tmp_path / "ovf.safetensors")
    rows = TINY["ffn_dim"]

    def mutate(sd):
        sd["blocks.1.block.ffn.0.weight"][rows - 3, 2] = 1e6

    _save_bf16_ckpt(ckpt, mutate)
    res = _run(tmp_path, scenario, ckpt)
    if scenario == "overflow":
        for r in res:
            assert not r["ok"] and r["error_type"] == "FP16OverflowError", r
            assert "blocks.1.block.ffn.0: 1 values exceed the fp16 range" in r["error"]
    else:
        for r in res:
            assert r["ok"], r.get("tb")
            assert r["fp32_units"] == ["blocks.1"]


# ------------------------------------------------------------------ runtime
def test_runtime_hooks_cover_block_attention_ffn_modulation():
    m = tiny_model()
    g = OverflowGuard(m, mode="all", action="abort", log=ListLog())
    g.armed = True
    with torch.no_grad():
        run_model(m, make_forward_inputs())
    labels = [r[0] for r in g._records]
    for want in ["blocks.0:forward_ref[0]", "blocks.0:forward_gen[0]",
                 "blocks.0.block:pre_self_attention[3]",      # first modulation chunk e[0]
                 "blocks.0.block:post_self_attention:attn_out[0]",  # raw attention output
                 "blocks.0.block.cross_attn[0]", "blocks.0.block.ffn.0[0]", "blocks.0.block.ffn[0]",
                 "time_projection[0]", "head[0]"]:
        assert want in labels, want


def test_runtime_nan_aborts_with_first_module_and_step():
    m = tiny_model()
    m.blocks[1].block.ffn[2].bias.data[3] = float("nan")
    g = OverflowGuard(m, mode="first_step", action="abort", log=ListLog())
    with torch.no_grad(), pytest.raises(FP16OverflowError) as e:
        run_model(m, make_forward_inputs(), call=lambda f, s, b: g.run(f, s, b))
    assert "step=ref" in str(e.value) and "first offending module blocks.1.block.ffn[0]" in str(e.value)


def _overflowing_model():
    """Block 1's FFN up-projection produces ~1e5..1e6 (inf in fp16) but the block is fine in fp32."""
    m = tiny_model()
    m.blocks[1].block.ffn[0].weight.data *= 1e5  # pre-activation std ~4e4 -> many |x| > 65504
    m.blocks[1].block.ffn[2].weight.data *= 1e-5
    return m


@pytest.fixture
def cuda_like_cpu_autocast(monkeypatch):
    """The model keeps time embeddings / modulation / head in fp32 via autocast('cuda', float32)
    regions; on CPU those are no-ops, so map them to CPU the way they behave on CUDA."""
    real = torch.amp.autocast

    def shim(device_type, dtype=None, enabled=True, cache_enabled=None):
        if device_type == "cuda":
            if dtype == torch.float32:
                return real("cpu", enabled=False)
            return real("cpu", dtype=dtype, enabled=enabled)
        return real(device_type, dtype=dtype, enabled=enabled, cache_enabled=cache_enabled)

    monkeypatch.setattr(torch.amp, "autocast", shim)


def test_runtime_real_fp16_overflow_is_caught_and_fp32_fallback_recovers(cuda_like_cpu_autocast):
    inp = make_forward_inputs()
    with torch.no_grad():
        ref = run_model(_overflowing_model(), inp)[0]  # fp32 reference
    assert torch.isfinite(ref).all()
    A.set_compute_dtype(torch.float16)
    # without the guard, CPU fp16 autocast really produces inf/NaN
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.float16):
        bad = run_model(_overflowing_model(), inp)[0]
    assert not torch.isfinite(bad).all()
    # abort mode names the module where the first non-finite value appeared
    m = _overflowing_model()
    g = OverflowGuard(m, mode="first_step", action="abort", log=ListLog())
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.float16), pytest.raises(FP16OverflowError) as e:
        run_model(m, inp, call=lambda f, s, b: g.run(f, s, b))
    assert "first offending module blocks.1.block.ffn.0[0]" in str(e.value) and "step=ref" in str(e.value)
    # fp32 mode switches blocks.1 to fp32, recomputes, and the result is finite and close to fp32
    m = _overflowing_model()
    log = ListLog()
    g = OverflowGuard(m, mode="first_step", action="fp32", log=log)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.float16):
        out = run_model(m, inp, call=lambda f, s, b: g.run(f, s, b))[0]
    assert g.fp32_units == {"blocks.1"}
    assert "FALLBACK: 'blocks.1' now runs in fp32" in log.text("error")
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), ref, atol=5e-2, rtol=5e-2)


def test_runtime_warns_near_limit():
    m = tiny_model()
    m.blocks[1].block.ffn[0].weight.data *= 1e3
    log = ListLog()
    g = OverflowGuard(m, mode="first_step", action="abort", warn_at=100.0, log=log)
    with torch.no_grad():
        run_model(m, make_forward_inputs(), call=lambda f, s, b: g.run(f, s, b))
    assert "blocks.1.block.ffn.0[0]" in log.text("warning")


def test_runtime_sentinel_on_unchecked_step_relocalises():
    m = tiny_model()
    inp = make_forward_inputs()
    log = ListLog()
    g = OverflowGuard(m, mode="first_step", action="abort", log=log)
    k, v = {}, {}
    with torch.no_grad():
        g.run(lambda: m(inp["cond_lat"], grid_sizes=inp["grid_sizes"], k_cache=k, v_cache=v, t=inp["t"],
                        method="forward_ref", context_ref=inp["ctx_ref"], seq_len_ref=inp["seq_len_ref"],
                        clip_fea_ref=inp["clip_fea_ref"], y_ref=inp["cond_y"]), "ref", "ref")
        m.blocks[2].block.cross_attn.o.bias.data[0] = float("inf")  # appears only later in the run
        gen = lambda: m(inp["noise"], k_cache=k, v_cache=v, t=inp["t"], grid_sizes_ref=inp["grid_sizes_ref"],
                        method="forward_gen", context=inp["ctx"], seq_len=inp["seq_len"],
                        clip_fea=inp["clip_fea"], y=inp["y"], origin_len=inp["clip_len"],
                        origin_area=inp["area"])
        with pytest.raises(FP16OverflowError) as e:
            g.run(gen, 3, "cond")  # step 3 is not hooked in first_step mode: sentinel -> rerun hooked
    assert "re-running this forward with hooks" in log.text("error")
    assert "step=3" in str(e.value) and "blocks.2.block.cross_attn[0]" in str(e.value)


def test_runtime_ranks_agree_without_deadlock(tmp_path):
    import torch.multiprocessing as mp
    from dist_helpers import guard_worker
    from test_v100_state_dict import _free_port
    ctx = mp.start_processes(guard_worker, args=(2, _free_port(), str(tmp_path)), nprocs=2, join=False,
                             start_method="spawn")
    for _ in range(240):
        if ctx.join(timeout=1):
            break
    else:  # pragma: no cover
        for p in ctx.processes:
            p.kill()
        pytest.fail("guard workers hung")
    res = [json.load(open(tmp_path / f"guard_rank{r}.json")) for r in range(2)]
    assert all(r.get("raised") for r in res), res  # rank 0 had clean weights but must stop too
    assert res[0]["msg"] == res[1]["msg"] and "blocks.2.block.ffn[0]" in res[0]["msg"]


# ------------------------------------------------------------------ server-side checkpoint scanner
def test_check_ckpt_script(tmp_path):
    import subprocess
    import sys
    from conftest import REPO
    clean, bad = str(tmp_path / "clean.safetensors"), str(tmp_path / "bad.safetensors")
    _save_bf16_ckpt(clean)

    def mutate(sd):
        sd["blocks.2.block.ffn.0.weight"][0, 0] = 1e6
        sd["blocks.0.block.cross_attn.norm_q.weight"][:] = 1e-9

    _save_bf16_ckpt(bad, mutate)
    pth = str(tmp_path / "vae_like.pth")
    torch.save({"model_state": {"conv.weight": torch.randn(4, 4)}}, pth)
    script = [sys.executable, f"{REPO}/scripts/check_ckpt_fp16_range.py"]
    ok = subprocess.run(script + [clean, pth, "--json", str(tmp_path / "r.json")], capture_output=True, text=True)
    assert ok.returncode == 0 and "OVERALL: PASS" in ok.stdout, ok.stdout + ok.stderr
    assert json.load(open(tmp_path / "r.json"))[0]["totals"]["numel"] > 0
    fail = subprocess.run(script + [bad], capture_output=True, text=True)
    assert fail.returncode == 1 and "OVERALL: FAIL" in fail.stdout, fail.stdout + fail.stderr
    assert "blocks.2.block.ffn.0: 1 values exceed the fp16 range" in fail.stdout
    assert "fp32_modules: [blocks.0, blocks.2]" in fail.stdout
    allowed = subprocess.run(script + [bad, "--fp32-modules", "blocks.0", "blocks.2"], capture_output=True, text=True)
    assert allowed.returncode == 0, allowed.stdout


# ------------------------------------------------------------------ Review F1: concealment
class _Unit(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from wanxiang.models.wan_animate_2_model import CrossAttention
        self.block = torch.nn.Module()
        self.block.cross_attn = CrossAttention(4, 1, use_img_emb=False)

    def forward(self, x, context):  # like the real model, always entered through blocks.N
        return self.block.cross_attn(x, context, None)


class CrossModel(torch.nn.Module):
    """Review F1 repro: real CrossAttention, finite weights that pass the fp16 load check,
    but the normalised query component overflows in fp16 (65280 x 2) and, with negative keys, SDPA
    MATH returns a finite but wrong output (all zeros; the fp32 reference is all ones)."""

    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_Unit()])
        a = self.blocks[0].block.cross_attn
        with torch.no_grad():
            for p in self.parameters():
                p.zero_()
            a.q.bias.copy_(torch.tensor([2., 1., 1., 1.]))
            a.norm_q.weight.fill_(1)
            a.norm_q.weight[0] = 65280
            a.k.bias.fill_(-1)
            a.norm_k.weight.fill_(1)
            a.v.bias.fill_(1)
            a.o.weight.copy_(torch.eye(4))

    def forward(self, x, context):
        return self.blocks[0](x, context)


@pytest.mark.parametrize("mode,step", [("first_step", 3), ("all", 3), ("first_step", 0)])
def test_concealed_attention_overflow_is_caught(mode, step):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    A.set_compute_dtype(torch.float16)
    x = torch.zeros(1, 2, 4, dtype=torch.float16)
    c = torch.zeros(1, 3, 4, dtype=torch.float16)
    # (a) exactly the review's setup: fp16 weights, MATH kernel
    m = CrossModel().eval().half()
    check_fp16_castable(m.named_parameters(), log=ListLog())  # the weights are fine
    with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]):
        unguarded = m(x, c)
        assert torch.isfinite(unguarded).all() and unguarded.abs().max() == 0  # finite but wrong
        log = ListLog()
        g = OverflowGuard(m, mode=mode, action="abort", log=log)
        with pytest.raises(FP16OverflowError) as e:
            g.run(lambda: m(x, c), step=step, branch="cond")
        g.remove()
    msg = str(e.value)
    assert f"step={step}" in msg and "first offending module blocks.0.block.cross_attn:attn0.q" in msg
    if step == 3 and mode == "first_step":  # not a detailed step: kernel flag -> rerun with hooks
        assert "<attention kernel input/output> (not localised)" in log.text("error")
    # (b) fp32 fallback. CPU autocast cannot upcast fp16 weights the way CUDA autocast(fp32) does
    # for the FSDP-gathered fp16 weights, so emulate mixed precision with fp32 master weights under
    # CPU fp16 autocast: q is computed in fp32 and overflows when cast to fp16 for the kernel.
    m2 = CrossModel().eval()
    log = ListLog()
    g = OverflowGuard(m2, mode=mode, action="fp32", log=log)
    with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]), torch.autocast("cpu", dtype=torch.float16):
        assert m2(x, c).abs().max() == 0  # same concealment without the guard
        out = g.run(lambda: m2(x, c), step=step, branch="cond")
    g.remove()
    assert g.fp32_units == {"blocks.0"} and "FALLBACK: 'blocks.0'" in log.text("error")
    torch.testing.assert_close(out.float(), torch.ones(1, 2, 4))


def test_monitor_labels_kernel_tensors_in_execution_order():
    m = tiny_model()
    g = OverflowGuard(m, mode="all", action="abort", log=ListLog())
    labels = []
    orig = g._check
    g._check = lambda out, step, branch, *a: (labels.extend(r[0] for r in g._records),
                                              orig(out, step, branch, *a))[1]
    with torch.no_grad():
        run_model(m, make_forward_inputs(), call=lambda f, s, b: g.run(f, s, b))
    g.remove()
    for want in ["blocks.0:forward_ref:attn0.q", "blocks.0:forward_ref:attn0.out",
                 "blocks.0:forward_gen:masked_attn0.k", "blocks.0:forward_gen:masked_attn0.out",
                 "blocks.0.block.cross_attn:attn0.q (in blocks.0:forward_gen)",   # image context
                 "blocks.0.block.cross_attn:attn1.v (in blocks.0:forward_gen)"]:  # text context
        assert want in labels, (want, [l for l in labels if "blocks.0" in l][:20])
    i_q, i_out = labels.index("blocks.1:forward_gen:masked_attn0.q"), labels.index("blocks.1:forward_gen[0]")
    assert i_q < i_out  # kernel inputs are recorded before the block output


def _spawn(fn, args, tmp_path, n=2, timeout=120):
    import torch.multiprocessing as mp
    ctx = mp.start_processes(fn, args=args, nprocs=n, join=False, start_method="spawn")
    for _ in range(timeout):
        if ctx.join(timeout=1):
            return
    for p in ctx.processes:
        p.kill()
    pytest.fail(f"{fn.__name__}: workers hung")


def test_count_mismatch_aborts_on_every_rank(tmp_path):
    """Review F6: unequal record counts must not send ranks into different branches."""
    from dist_helpers import guard_count_worker
    from test_v100_state_dict import _free_port
    _spawn(guard_count_worker, (2, _free_port(), str(tmp_path)), tmp_path)
    res = [json.load(open(tmp_path / f"count_rank{r}.json")) for r in range(2)]
    for r in res:
        assert r["error_type"] == "RuntimeError" and "different tensor counts (min 1, max 2" in r["error"], r


def test_assert_finite_is_collective(tmp_path):
    from dist_helpers import assert_finite_worker
    from test_v100_state_dict import _free_port
    _spawn(assert_finite_worker, (2, _free_port(), str(tmp_path)), tmp_path)
    res = [json.load(open(tmp_path / f"finite_rank{r}.json")) for r in range(2)]
    for r in res:  # NaN only on rank 1, both ranks raise the same message
        assert r["raised"] and "CLIP visual (reference image)" in r["msg"] and "clip_dtype=fp32" in r["msg"], r
    assert res[0]["msg"] == res[1]["msg"]


def test_assert_finite_sees_attention_flag_and_inf():
    from wanxiang.models.attention_sdpa import MONITOR
    g = OverflowGuard(tiny_model(), mode="first_step", log=ListLog())
    g.begin_component()
    MONITOR.check("attn", q=torch.ones(2))
    clean = g.attention_flag("cpu")
    g.assert_finite("ok", {"a": torch.ones(3), "b": [torch.zeros(2), clean]})
    g.begin_component()  # e.g. CLIP: its output is finite but one attention input overflowed
    MONITOR.check("attn", q=torch.tensor([1.0, float("inf")]))
    flag = g.attention_flag("cpu")
    with pytest.raises(FP16OverflowError, match=r"in conditioning: CLIP"):
        g.assert_finite("conditioning", {"CLIP": [torch.ones(3), flag], "VAE": torch.ones(3)})
    with pytest.raises(FP16OverflowError, match=r"in VAE decode: frames"):
        g.assert_finite("VAE decode", {"frames": torch.tensor([1.0, float("inf")])})
    g.remove()


# ------------------------------------------------------------------ Review F5: CLIP order
def test_clip_checked_before_the_fp16_cast():
    """prepare_clip sees the fp32 checkpoint values (a 1e-9 weight is still nonzero, a 1e5 weight
    is still 1e5) and only then casts; failures point at clip_dtype, not the DiT allow-list."""
    from conftest import stub_heavy_video_deps
    stub_heavy_video_deps()
    from types import SimpleNamespace
    from pipelines.wan_animate_2_pipeline import prepare_clip
    from wanxiang.utils.precision import Precision

    def fake_clip(tweak=None):
        model = torch.nn.Module()
        model.visual = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.LayerNorm(8))
        model.textual = torch.nn.Linear(4, 4)
        if tweak:
            tweak(model.visual)
        return SimpleNamespace(model=model, dtype=torch.float32)

    prec = Precision(dit=torch.float16, dit_output=torch.float32, t5=torch.float32, vae=torch.float32,
                     clip=torch.float16, underflow_frac=0.0)
    clip = prepare_clip(fake_clip(), prec, log=ListLog())
    assert clip.dtype == torch.float16 and clip.model.visual[0].weight.dtype == torch.float16
    assert clip.model.textual.weight.dtype == torch.float16

    def tiny(v):
        with torch.no_grad():
            v[1].weight[:] = 1e-9  # would silently become 0 if loaded straight into fp16

    with pytest.raises(FP16OverflowError, match=r"CLIP visual weights are not safely castable") as e:
        prepare_clip(fake_clip(tiny), prec, log=ListLog())
    assert "1: 8/8 nonzero values flush to 0" in str(e.value) and "test_cfg.clip_dtype=fp32" in str(e.value)
    assert "fp32_modules" not in str(e.value)

    def big(v):
        with torch.no_grad():
            v[0].weight[0, 0] = 1e5

    with pytest.raises(FP16OverflowError, match=r"0: 1 values exceed the fp16 range"):
        prepare_clip(fake_clip(big), prec, log=ListLog())
    fp32 = Precision(dit=torch.float16, dit_output=torch.float32, t5=torch.float32, vae=torch.float32,
                     clip=torch.float32)
    clip = prepare_clip(fake_clip(big), fp32, log=ListLog())  # the documented fallback
    assert clip.model.visual[0].weight[0, 0] == 1e5 and clip.dtype == torch.float32


# ------------------------------------------------------------------ Review r2 N4: collectives
@pytest.mark.parametrize("step,expected", [(3, [(2,)]), (0, [(2,), (2,)])])
def test_guard_collectives_per_forward(step, expected):
    """Non-detailed forward: 1 all-reduce of 2 floats (sentinel + kernel flag). Detailed forward:
    2 all-reduces (record counts [n, -n], then the packed flags/values; here no hooked modules on a
    bare Linear, so the packed buffer is also 2 floats)."""
    m = torch.nn.Linear(1, 1)
    g = OverflowGuard(m, mode="first_step", action="abort", log=ListLog())
    calls = []
    real = g._all_reduce_max
    g._all_reduce_max = lambda t: (calls.append(tuple(t.shape)), real(t))[1]
    g.run(lambda: torch.ones(1), step=step, branch="cond")
    g.remove()
    assert calls == expected


# --------------------------------------------------------------- pass markers (WAN_FP16_CHECK)
def _counting(params):
    calls = []

    def named_params():
        calls.append(1)
        return iter(params)
    return named_params, calls


def test_cached_check_skips_only_the_same_files_and_settings(tmp_path, monkeypatch):
    """auto: the first run checks and records a pass; the same file + settings skip; a changed
    file, changed settings or WAN_FP16_CHECK=force check again; off skips with a warning."""
    from wanxiang.utils.fp16_guard import check_fp16_castable_cached
    ckpt = tmp_path / "w.safetensors"
    ckpt.write_bytes(b"weights")
    good = [("m.weight", torch.ones(4))]
    np_, calls = _counting(good)
    check_fp16_castable_cached(np_, [str(ckpt)], log=ListLog())
    check_fp16_castable_cached(np_, [str(ckpt)], log=ListLog())
    assert len(calls) == 1                                    # second run skipped
    check_fp16_castable_cached(np_, [str(ckpt)], fp32_modules=["m"], log=ListLog())
    assert len(calls) == 2                                    # different settings -> checked
    os.utime(ckpt, ns=(1, 1))
    check_fp16_castable_cached(np_, [str(ckpt)], log=ListLog())
    assert len(calls) == 3                                    # file changed -> checked
    monkeypatch.setenv("WAN_FP16_CHECK", "force")
    check_fp16_castable_cached(np_, [str(ckpt)], log=ListLog())
    assert len(calls) == 4                                    # force -> checked
    monkeypatch.setenv("WAN_FP16_CHECK", "off")
    log = ListLog()
    check_fp16_castable_cached(np_, [str(ckpt)], log=log)
    assert len(calls) == 4 and any(k == "warning" and "SKIPPED" in m for k, m in log.lines)
    monkeypatch.setenv("WAN_FP16_CHECK", "sometimes")
    with pytest.raises(ValueError):
        check_fp16_castable_cached(np_, [str(ckpt)], log=ListLog())


def test_cached_check_never_records_a_failure_or_an_unknown_file(tmp_path):
    from wanxiang.utils.fp16_guard import check_fp16_castable_cached
    ckpt = tmp_path / "bad.safetensors"
    ckpt.write_bytes(b"weights")
    bad = [("m.weight", torch.tensor([65519.0]))]
    np_, calls = _counting(bad)
    for _ in range(2):  # still raises on every run: nothing was recorded
        with pytest.raises(FP16OverflowError):
            check_fp16_castable_cached(np_, [str(ckpt)], log=ListLog())
    assert len(calls) == 2
    np_, calls = _counting([("m.weight", torch.ones(4))])
    for src in (None, str(tmp_path / "missing.safetensors")):  # unknown file: always checked
        check_fp16_castable_cached(np_, [src], log=ListLog())
        check_fp16_castable_cached(np_, [src], log=ListLog())
    assert len(calls) == 4
