"""SDPA attention path (V100 port) vs reference math and vs upstream flex/flash semantics. CPU only."""
import math

import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask, create_mask
from torch.nn.attention.flex_attention import flex_attention as flex_eager

from conftest import TINY, make_forward_inputs, run_model
from wanxiang.models import attention as A
from wanxiang.models import wan_animate_2_model as M
from wanxiang.models.attention_sdpa import (MaskPlan, build_mask_plan, masked_sdpa_attention,
                                            sdpa_attention, compute_dtype_override)

TOL = {torch.float32: dict(atol=2e-5, rtol=2e-5), torch.float16: dict(atol=3e-3, rtol=3e-3)}

# Upstream calls flex_attention through torch.compile(dynamic=False, fullgraph=True); plain eager
# flex in torch 2.7.1 lifts the float log_scale of the score_mod and rejects it, so the reference
# uses the same compile settings with the "eager" backend (= flex's own math implementation).
torch._dynamo.config.cache_size_limit = 256
flex_ref_kernel = torch.compile(flex_eager, backend="eager", fullgraph=True, dynamic=False)


@pytest.fixture(autouse=True)
def _sdpa_backend():
    # each test may use a different float log_scale; a dynamo recompile on a changed float would
    # un-specialise it (test artefact - upstream uses one constant log_scale per model)
    torch._dynamo.reset()
    A.set_attention_backend("sdpa")
    A.set_compute_dtype(torch.float16)
    yield
    A.set_attention_backend("auto")
    A.set_compute_dtype(torch.bfloat16)


def dense_ref(q, k, v, mask=None, bias=None, scale=None):
    """Plain softmax attention in float64. q/k/v [B, L, N, D]; mask/bias broadcastable to [B, N, Lq, Lk].
    Fully masked rows -> 0 (flex_attention convention)."""
    q, k, v = (t.double().transpose(1, 2) for t in (q, k, v))
    s = q @ k.transpose(-1, -2) * (scale if scale is not None else 1 / math.sqrt(q.size(-1)))
    if bias is not None:
        s = s + bias.double()
    if mask is not None:
        s = s.masked_fill(~mask, float("-inf"))
    p = torch.softmax(s, -1).nan_to_num(0.0)
    return (p @ v).transpose(1, 2)


def varlen_ref(q, k, v, k_lens):
    """flash_attn varlen semantics as used upstream: batch b sees keys k[b, :k_lens[b]]."""
    return torch.cat([dense_ref(q[b:b + 1], k[b:b + 1, :int(L)], v[b:b + 1, :int(L)])
                      for b, L in enumerate(k_lens)])


# ------------------------------------------------------------------ flash_attention replacement
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("k_lens", [None, [37, 37], [37, 21]])
@pytest.mark.parametrize("chunk", ["4096", "7"])
def test_sdpa_matches_flash_varlen_semantics(dtype, k_lens, chunk, monkeypatch):
    monkeypatch.setenv("WAN_SDPA_Q_CHUNK", chunk)
    torch.manual_seed(0)
    B, Lq, Lk, N, D = 2, 29, 37, 3, 32
    q, k, v = (torch.randn(B, L, N, D, dtype=dtype) for L in (Lq, Lk, Lk))
    with compute_dtype_override(dtype):
        out = A.flash_attention(q, k, v, k_lens=None if k_lens is None else torch.tensor(k_lens))
    assert out.dtype == dtype and out.shape == (B, Lq, N, D)
    ref = dense_ref(q, k, v) if k_lens is None else varlen_ref(q, k, v, k_lens)
    torch.testing.assert_close(out.double(), ref, **TOL[dtype])


def test_sdpa_fp32_inputs_cast_like_upstream():
    # upstream: fp32 q/k (after RoPE) are cast to half; v's half dtype wins. Output dtype = q dtype.
    torch.manual_seed(1)
    q, k = torch.randn(1, 16, 2, 32), torch.randn(1, 24, 2, 32)
    v = torch.randn(1, 24, 2, 32).half()
    out = A.flash_attention(q, k, v, k_lens=torch.tensor([20]))
    assert out.dtype == torch.float32
    torch.testing.assert_close(out.double(), varlen_ref(q.half(), k.half(), v, [20]), **TOL[torch.float16])


def test_sdpa_scale_qscale_qlens():
    torch.manual_seed(2)
    q, k, v = torch.randn(2, 10, 2, 16), torch.randn(2, 12, 2, 16), torch.randn(2, 12, 2, 16)
    with compute_dtype_override(torch.float32):
        out = sdpa_attention(q, k, v, softmax_scale=0.3, q_scale=1.5, q_lens=torch.tensor([10, 6]))
    ref = dense_ref(q * 1.5, k, v, scale=0.3)
    ref[1, 6:] = 0
    torch.testing.assert_close(out.double(), ref, **TOL[torch.float32])
    with pytest.raises(NotImplementedError):
        sdpa_attention(q, k, v, causal=True)
    with pytest.raises(NotImplementedError):
        sdpa_attention(q, k, v, window_size=(4, 4))


# ------------------------------------------------------------------ flex_attention replacement
def independent_mask(origin_len, area):
    """Frame-aligned In-Context mask written from its description (not from the model code):
    valid query (frame f) sees every valid generation token, plus the reference tokens of frame f
    (reference frame r is aligned to generation frame r + 1)."""
    F_ = origin_len // 4 + 1
    hw = int(area[0] * area[1] // 256)
    q_lim, k_lim = (F_ + 1) * hw, F_ * hw
    q_tot = math.ceil(q_lim / 128) * 128
    kv_tot = q_tot + math.ceil(k_lim / 128) * 128
    m = torch.zeros(q_tot, kv_tot, dtype=torch.bool)
    for qi in range(q_lim):
        f = qi // hw
        m[qi, :q_lim] = True
        if f >= 1:
            m[qi, q_tot + (f - 1) * hw: q_tot + f * hw] = True
    return m, hw


GEOMS = [(5, (32, 32)), (9, (48, 32)), (13, (80, 48)), (17, (64, 64))]


@pytest.mark.parametrize("origin_len,area", GEOMS)
def test_mask_mod_and_plan_are_exact(origin_len, area, upstream):
    mask_mod, Q, KV = M.incontext_mask_mod(origin_len, area)
    dense = create_mask(mask_mod, None, None, Q, KV, device="cpu")[0, 0]
    ind, _ = independent_mask(origin_len, area)
    assert torch.equal(dense, ind)
    # the moved mask_mod is identical to upstream's closure inside create_mask
    up = upstream.WanAnimate2Transformer.__new__(upstream.WanAnimate2Transformer)
    captured = {}
    orig = upstream.create_block_mask
    upstream.create_block_mask = lambda fn, B, H, Q_LEN, KV_LEN, device, _compile: captured.update(
        fn=fn, Q=Q_LEN, KV=KV_LEN)
    try:
        up.create_mask(origin_len, area, "cpu")
    finally:
        upstream.create_block_mask = orig
    assert (captured["Q"], captured["KV"]) == (Q, KV)
    assert torch.equal(create_mask(captured["fn"], None, None, Q, KV, device="cpu")[0, 0], dense)
    # plan reconstructs the dense mask exactly, with one run per query frame (+ padding rows)
    plan = build_mask_plan(mask_mod, Q, KV, "cpu", max_tile_elems=4096)
    rebuilt = torch.zeros_like(dense)
    for q0, q1, cols in plan.runs:
        rebuilt[q0:q1, cols] = True
    assert torch.equal(rebuilt, dense)
    hw = int(area[0] * area[1] // 256)
    assert len(plan.runs) == (origin_len // 4 + 2) + (Q > (origin_len // 4 + 2) * hw)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("log_scale", [0.0, -1.3])
@pytest.mark.parametrize("origin_len,area", GEOMS[:3])
def test_masked_sdpa_matches_upstream_flex(dtype, log_scale, origin_len, area, upstream, monkeypatch):
    monkeypatch.setenv("WAN_SDPA_Q_CHUNK", "50")
    torch.manual_seed(3)
    mask_mod, Q, KV = M.incontext_mask_mod(origin_len, area)
    hw = int(area[0] * area[1] // 256)
    N, D = 2, 32
    q = torch.randn(1, Q, N, D, dtype=dtype)
    k = torch.randn(1, KV, N, D, dtype=dtype)
    v = torch.randn(1, KV, N, D, dtype=dtype)
    score_mod = upstream._get_score_mod(hw=hw, log_scale=log_scale)  # upstream's own score_mod
    # upstream math: flex_attention (eager reference implementation) with the upstream BlockMask
    bm = create_block_mask(mask_mod, None, None, Q, KV, device="cpu", _compile=False)
    flex = flex_ref_kernel(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=bm,
                           score_mod=score_mod).transpose(1, 2)
    # dense float64 reference with the bias written out explicitly
    dense = create_mask(mask_mod, None, None, Q, KV, device="cpu")
    bias = torch.zeros(KV)
    bias[hw:2 * hw] = log_scale
    ref = dense_ref(q, k, v, mask=dense, bias=bias)
    plan = M.WanAnimate2Transformer.create_mask(None, origin_len, area, "cpu")  # backend = sdpa
    assert isinstance(plan, MaskPlan)
    with compute_dtype_override(dtype):
        out = A.flex_attention(q=q, k=k, v=v, block_mask=plan, kernel_options=None, score_mod=score_mod)
    assert out.dtype == dtype
    torch.testing.assert_close(out.double(), ref, **TOL[dtype])
    torch.testing.assert_close(flex.double(), ref, **TOL[dtype])
    torch.testing.assert_close(out.double(), flex.double(), **TOL[dtype])
    assert out[0, (origin_len // 4 + 2) * hw:].abs().max() == 0  # fully masked padding rows -> 0


def test_plan_general_paths():
    # non-frame-structured mask (causal) and a query-dependent additive score_mod still match
    Q = KV = 40
    plan = build_mask_plan(lambda b, h, q, kv: kv <= q, Q, KV, "cpu")
    score_mod = lambda s, b, h, q, kv: s + 0.01 * (q - kv)
    torch.manual_seed(4)
    q, k, v = (torch.randn(1, 40, 2, 16) for _ in range(3))
    with compute_dtype_override(torch.float32):
        out = masked_sdpa_attention(q, k, v, plan, score_mod)
    qi, ki = torch.arange(Q)[:, None], torch.arange(KV)[None]
    ref = dense_ref(q, k, v, mask=ki <= qi, bias=0.01 * (qi - ki))
    torch.testing.assert_close(out.double(), ref, **TOL[torch.float32])
    with pytest.raises(NotImplementedError):  # multiplicative score_mod is rejected, not mis-handled
        masked_sdpa_attention(q, k, v, plan, lambda s, b, h, q_, kv: s * 2)
    with pytest.raises(NotImplementedError):  # head-dependent score_mod is rejected
        masked_sdpa_attention(q, k, v, plan, lambda s, b, h, q_, kv: s + h)


# ------------------------------------------------------------------ whole model, tiny config
def _reference_kernels(monkeypatch, module):
    """Route a model module's flash/flex calls to the reference math (flex eager + dense varlen)."""
    def flash_ref(q, k, v, k_lens=None, **kw):
        out = dense_ref(q, k, v) if k_lens is None else varlen_ref(q, k, v, k_lens.tolist() * q.size(0))
        return out.to(q.dtype)

    def flex_ref(q, k, v, block_mask=None, kernel_options=None, score_mod=None, **kw):
        return flex_ref_kernel(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                               block_mask=block_mask, score_mod=score_mod).transpose(1, 2)

    real_cbm = create_block_mask
    monkeypatch.setattr(module, "flash_attention", flash_ref)
    monkeypatch.setattr(module, "flex_attention", flex_ref)
    monkeypatch.setattr(module, "create_block_mask",
                        lambda *a, _compile=True, **kw: real_cbm(*a, _compile=False, **kw))


@pytest.mark.parametrize("log_scale", [0.0, -1.3])
def test_tiny_model_sdpa_equals_upstream_model(log_scale, upstream, monkeypatch):
    cfg = dict(TINY, log_scale=log_scale)
    torch.manual_seed(5)
    patched = M.WanAnimate2Transformer(**cfg).eval()
    for p in patched.parameters():  # init_weights zeroes the head; use a non-trivial random model
        p.data.normal_(0, 0.05)
    up = upstream.WanAnimate2Transformer(**cfg).eval()
    up.load_state_dict(patched.state_dict(), strict=True)
    inp = make_forward_inputs()
    with torch.no_grad(), compute_dtype_override(torch.float32):
        out_sdpa = run_model(patched, inp)[0]
        with monkeypatch.context() as mp:
            _reference_kernels(mp, upstream)
            out_up = run_model(up, inp)[0]
    assert isinstance(next(iter(patched.block_masks.values())), MaskPlan)
    assert out_sdpa.shape == (16, 4, 4, 4) and torch.isfinite(out_sdpa).all()
    torch.testing.assert_close(out_sdpa, out_up, atol=1e-5, rtol=1e-5)


# ------------------------------------------------------------------ Review F8: varlen padding
def test_unequal_k_lens_slice_padding_away():
    """Like flash varlen packing: discarded padding (even NaN) never reaches another sample."""
    q = torch.ones(2, 1, 1, 4)
    k = torch.ones(2, 3, 1, 4)
    v = torch.ones_like(k)
    v[1, 2] = float("nan")  # padding of sample 1 (k_lens[1] == 2)
    for backends in (None, "math"):
        from torch.nn.attention import SDPBackend, sdpa_kernel
        ctx = sdpa_kernel([SDPBackend.MATH]) if backends else torch.no_grad()
        with ctx, compute_dtype_override(torch.float32):
            out = sdpa_attention(q, k, v, k_lens=[3, 2])
        torch.testing.assert_close(out, torch.ones(2, 1, 1, 4))
    torch.manual_seed(7)
    q, k, v = torch.randn(3, 5, 2, 8), torch.randn(3, 9, 2, 8), torch.randn(3, 9, 2, 8)
    with compute_dtype_override(torch.float32):
        out = sdpa_attention(q, k, v, k_lens=[9, 4, 4])
    torch.testing.assert_close(out.double(), varlen_ref(q, k, v, [9, 4, 4]), **TOL[torch.float32])


@pytest.mark.parametrize("k_lens", [[0, 3], [3, 4], [3]])
def test_k_lens_bounds_are_validated(k_lens):
    q, k = torch.ones(2, 1, 1, 4), torch.ones(2, 3, 1, 4)
    with pytest.raises(ValueError, match="k_lens"):
        sdpa_attention(q, k, k, k_lens=k_lens)


# ------------------------------------------------------------------ fused -> MATH fallback
def test_math_fallback_is_chunked_and_exact(monkeypatch):
    """If no fused kernel accepts the inputs (e.g. bf16 on sm70) the MATH kernel is used in
    budget-sized query chunks; results are unchanged."""
    import torch.nn.functional as Fn
    from wanxiang.models import attention_sdpa as S
    real = Fn.scaled_dot_product_attention
    calls = []

    def fake_sdpa(q, k, v, **kw):
        if not torch.backends.cuda.math_sdp_enabled():  # inside the fused-only attempt
            raise RuntimeError("No available kernel. Aborting execution.")
        calls.append(q.size(2))
        return real(q, k, v, **kw)

    monkeypatch.setattr(S.F, "scaled_dot_product_attention", fake_sdpa)
    # budget for 4 rows: 1 batch x 2 heads x 37 keys x 10 B x 4 rows = 2960 B
    monkeypatch.setenv("WAN_SDPA_MATH_BUDGET_MB", str(2960 / 2 ** 20))
    torch.manual_seed(8)
    q, k, v = torch.randn(1, 29, 2, 32), torch.randn(1, 37, 2, 32), torch.randn(1, 37, 2, 32)
    with compute_dtype_override(torch.float32):
        out = sdpa_attention(q, k, v)
    torch.testing.assert_close(out.double(), dense_ref(q, k, v), **TOL[torch.float32])
    assert calls == [4] * 7 + [1]


def test_outer_sdpa_kernel_context_is_respected():
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from wanxiang.models.attention_sdpa import _allowed_fused, _math_allowed
    with sdpa_kernel([SDPBackend.MATH]):
        assert _allowed_fused() == [] and _math_allowed()
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        assert _allowed_fused() == [SDPBackend.EFFICIENT_ATTENTION] and not _math_allowed()
    assert SDPBackend.EFFICIENT_ATTENTION in _allowed_fused() and _math_allowed()


@pytest.mark.parametrize("backends", [[], "efficient_only"])
def test_disabled_math_is_not_used_behind_the_callers_back(backends):
    """Review r2 N3: with MATH disabled by the caller and no allowed fused kernel accepting
    the inputs (all disabled; or EFFICIENT only, which CPU does not have), the call must fail
    instead of silently running MATH."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    allowed = [] if backends == [] else [SDPBackend.EFFICIENT_ATTENTION]
    q = torch.ones(1, 2, 1, 4)
    with sdpa_kernel(allowed), compute_dtype_override(torch.float32), pytest.raises(RuntimeError):
        sdpa_attention(q, q, q)


def test_backend_choice_is_per_call_not_cached(monkeypatch):
    """A rejected input (dropout: CPU fused kernels refuse it) must not push later, different
    calls onto MATH (review r2 N3 / repro_backend_cache.py)."""
    from wanxiang.models import attention_sdpa as S
    real, modes = S._sdpa_rows, []

    def rows(*a, **kw):
        modes.append("MATH" if torch.backends.cuda.math_sdp_enabled() else "FUSED")
        return real(*a, **kw)

    monkeypatch.setattr(S, "_sdpa_rows", rows)
    q = torch.ones(1, 2, 1, 4)
    with compute_dtype_override(torch.float32):
        S.sdpa_attention(q, q, q, dropout_p=0.1)
        assert modes == ["FUSED", "MATH"]
        modes.clear()
        out = S.sdpa_attention(q, q, q)
        assert modes == ["FUSED"]
    torch.testing.assert_close(out, torch.ones(1, 2, 1, 4))
