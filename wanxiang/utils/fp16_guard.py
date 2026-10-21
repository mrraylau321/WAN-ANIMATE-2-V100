"""fp16 safety net for the V100 port (fp16 has max 65504; upstream ran bf16, max ~3.4e38).

1. ``check_fp16_castable``  load-time: per-module stats of casting the loaded weights to fp16
   (non-finite, overflow -> inf, nonzero flushed to 0, subnormal). Raises on any violation
   unless the module is explicitly listed in ``fp32_modules`` (non-finite always raises).
   ``check_fp16_castable_cached`` skips it for weight files that already passed with the same
   settings (fingerprint marker); env WAN_FP16_CHECK=auto (default) | force | off.
2. ``force_fp32``           runs one module in fp32 (args upcast, fp32 autocast on CUDA, fp32 SDPA).
3. ``OverflowGuard``        runtime: forward hooks record max|x| of block / attention / FFN /
   modulation tensors (one host sync per checked forward), a cheap sentinel checks every model
   output, and on a non-finite value the first offending module + step is logged and the run
   either aborts or re-runs that forward with the offending block switched to fp32.
"""
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field

import torch
import torch.distributed as dist

from wanxiang.models.attention_sdpa import MONITOR, compute_dtype_override

FP16_MAX = 65504.0
FP16_MIN_NORMAL = 2.0 ** -14   # below: subnormal, reduced precision
FP16_MIN_SUBNORMAL = 2.0 ** -24  # 5.96e-8; |x| <= 2**-25 (2.98e-8) rounds to 0

_log = logging.getLogger("wan.fp16_guard")


class FP16OverflowError(RuntimeError):
    pass


def _matches(name, prefixes):
    return any(name == p or name.startswith(p + ".") for p in prefixes)


# ----------------------------------------------------------------------------- load time
@torch.no_grad()
def fp16_cast_stats(named_params, group=None):
    """Per-tensor stats of casting to fp16, on local shards (DTensor), reduced over ``group``.

    Returns (names, counts [N, 5] = nonfinite, overflow, flushed_to_zero, subnormal, nonzero,
    maxabs [N], max_flushed [N]). overflow tests the ORIGINAL magnitude |w| > 65504 (values in
    (65504, 65520) would round down to 65504 but are still counted); flushed = nonzero values
    that become 0 in fp16 (|w| <= 2**-25); subnormal = fp16 result below 2**-14."""
    names, counts, maxabs, maxflush = [], [], [], []
    for name, p in named_params:
        t = p.to_local() if hasattr(p, "to_local") else p
        t = t.detach()
        zero = torch.zeros((), dtype=torch.float32, device=t.device)
        finite = torch.isfinite(t)
        a = torch.where(finite, t.float().abs(), zero)
        nonzero = (t != 0) & finite
        h = t.to(torch.float16)
        flushed = nonzero & (h == 0)
        counts.append(torch.stack([
            (~finite).sum(),
            (a > FP16_MAX).sum(),
            flushed.sum(),
            ((h != 0) & torch.isfinite(h) & (h.abs() < FP16_MIN_NORMAL)).sum(),
            nonzero.sum(),
        ]))
        maxabs.append(a.max() if a.numel() else zero)
        maxflush.append(torch.where(flushed, a, zero).max() if a.numel() else zero)
        names.append(name)
    if not names:
        z = torch.zeros(0, dtype=torch.float32)
        return [], torch.zeros(0, 5, dtype=torch.long), z, z
    c = torch.stack(counts).long()
    m = torch.stack([torch.stack(maxabs), torch.stack(maxflush)]).float()
    if group is not None and dist.is_available() and dist.is_initialized():
        dist.all_reduce(c, group=group)
        dist.all_reduce(m, op=dist.ReduceOp.MAX, group=group)
    m = m.cpu()
    return names, c.cpu(), m[0], m[1]


def aggregate_per_module(names, counts, maxabs, maxflush=None, per_mod=None):
    """Fold per-tensor stats into
    {module: [nonfinite, overflow, flushed, subnormal, nonzero, max|w|, max flushed |w|]}."""
    per_mod = {} if per_mod is None else per_mod
    for i, name in enumerate(names):
        d = per_mod.setdefault(name.rsplit(".", 1)[0], [0, 0, 0, 0, 0, 0.0, 0.0])
        for j in range(5):
            d[j] += int(counts[i, j])
        d[5] = max(d[5], float(maxabs[i]))
        if maxflush is not None:
            d[6] = max(d[6], float(maxflush[i]))
    return per_mod


def fp16_violations(per_mod, fp32_modules=(), underflow_frac=1e-3):
    """Pass/fail rules shared by the load-time check and scripts/check_ckpt_fp16_range.py:
    any non-finite value -> fail (always); any |w| > 65504 -> fail; a module whose flushed-to-0
    fraction of nonzero values exceeds ``underflow_frac`` -> fail (0 = strict: any flush fails).
    fp32_modules are exempt from the last two rules. Subnormals are reported, never fatal.

    Why a tolerance for underflow: a flushed weight has |w| <= 2**-25 ~ 3e-8, so the absolute error
    it introduces is ~300x smaller than ordinary fp16 rounding of a typical 0.02 weight
    (2**-11 * 0.02 ~ 1e-5); failing on any flush would push nearly every Linear into fp32.
    Every flush is still logged (see flush_report)."""
    out = []
    for mod, (nonfin, over, flush, sub, nz, mx, mxf) in per_mod.items():
        exempt = _matches(mod, fp32_modules)
        why = []
        if nonfin:
            why.append(f"{nonfin} non-finite values in the checkpoint")
        if over and not exempt:
            why.append(f"{over} values exceed the fp16 range (max|w|={mx:.6g} > {FP16_MAX:g})")
        if nz and flush / nz > underflow_frac and not exempt:
            why.append(f"{flush}/{nz} nonzero values flush to 0 in fp16 (max flushed |w|={mxf:.3g}; "
                       f"tolerance {underflow_frac:g})")
        if why:
            out.append((mod, why))
    return out


def flush_report(per_mod, limit=None):
    """Lines for every module with any value flushed to 0 in fp16 (count, fraction, max |w|)."""
    rows = [(mod, d[2], d[4], d[6]) for mod, d in per_mod.items() if d[2]]
    rows.sort(key=lambda r: -r[1] / max(r[2], 1))
    total = sum(r[1] for r in rows)
    lines = [f"{mod}: {fl}/{nz} ({fl / max(nz, 1):.2e}) flushed, max flushed |w|={mf:.3g}"
             for mod, fl, nz, mf in rows[:limit]]
    return total, len(rows), lines


def check_fp16_castable(named_params, fp32_modules=(), underflow_frac=1e-3, group=None,
                        log=None, what="DiT", hint=None):
    """Raise FP16OverflowError if casting the weights to fp16 is unsafe (see fp16_violations).
    Modules listed in ``fp32_modules`` are exempt from the range rules (they run in fp32).
    Every flushed value is logged as a WARNING even when within tolerance."""
    log = log or _log
    names, c, m, mf = fp16_cast_stats(named_params, group=group)
    per_mod = aggregate_per_module(names, c, m, mf)
    tot = c.sum(0).tolist() if len(names) else [0] * 5
    violations = fp16_violations(per_mod, fp32_modules, underflow_frac)
    top = sorted(per_mod.items(), key=lambda kv: -kv[1][5])[:5]
    log.info(f"[fp16-check] {what}: {len(names)} tensors, nonzero={tot[4]}, max|w|="
             f"{float(m.max()) if len(names) else 0:.4g}, over_65504={tot[1]}, flushed_to_0={tot[2]}, "
             f"subnormal={tot[3]}, non-finite={tot[0]}; underflow tolerance={underflow_frac:g}; "
             f"fp32 allow-list={list(fp32_modules)}")
    log.info("[fp16-check] largest |w|: " + ", ".join(f"{k}={v[5]:.4g}" for k, v in top))
    total, nmods, lines = flush_report(per_mod)
    if total:
        log.warning(f"[fp16-check] {what}: {total} nonzero weights become 0 in fp16 across {nmods} "
                    f"modules (all |w| <= 2.98e-8):\n  " + "\n  ".join(lines))
    if violations:
        msg = (f"[fp16-check] {what} weights are not safely castable to fp16:\n  "
               + "\n  ".join(f"{mod}: " + "; ".join(why) for mod, why in violations)
               + "\nFix: " + (hint or "add the module(s) to test_cfg.fp32_modules (they will be kept "
                                       "and run in fp32), or run with --dtype fp32."))
        log.error(msg)
        raise FP16OverflowError(msg)
    return per_mod


def fp16_check_mode():
    """WAN_FP16_CHECK: auto (default: check, skip weight files that already passed), force, off."""
    mode = os.environ.get("WAN_FP16_CHECK", "auto").strip().lower()
    if mode not in ("auto", "force", "off"):
        raise ValueError(f"WAN_FP16_CHECK must be auto, force or off (got {mode!r})")
    return mode


def fp16_check_marker(what, sources, settings):
    """Path of the 'passed' marker for these weight files (path, size, mtime) + check settings.
    Any change to a file or to the settings gives a different path, so the check runs again.
    None (= always check, never cache) when a weight file is unknown or missing."""
    if not sources or not all(src and os.path.isfile(str(src)) for src in sources):
        return None
    items = [what, settings]
    for src in sources:
        st = os.stat(src)
        items.append([os.path.abspath(src), st.st_size, st.st_mtime_ns])
    key = hashlib.sha256(json.dumps(items, sort_keys=True, default=str).encode()).hexdigest()[:20]
    cache = os.environ.get("WAN_FP16_CHECK_CACHE",
                           os.path.join(os.path.expanduser("~"), ".cache", "wan_animate2_v100", "fp16_check"))
    tag = re.sub(r"[^A-Za-z0-9]+", "_", what).strip("_")
    return os.path.join(cache, f"{tag}-{key}.ok")


def _all_agree(flag, group):
    """True only if every rank of ``group`` has flag=True (keeps collective calls in lockstep)."""
    if group is None or not (dist.is_available() and dist.is_initialized()):
        return flag
    dev = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    t = torch.tensor([1 if flag else 0], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MIN, group=group)
    return bool(t.item())


def check_fp16_castable_cached(named_params, sources, fp32_modules=(), underflow_frac=1e-3, group=None,
                               log=None, what="DiT", hint=None):
    """check_fp16_castable, skipped when these exact weight files already passed with the same
    settings. ``named_params`` is a callable returning the (name, param) iterator, so a skipped
    check does not touch the weights. A failed check never writes a marker (it raises)."""
    log = log or _log
    mode = fp16_check_mode()
    if mode == "off":
        log.warning(f"[fp16-check] {what}: SKIPPED (WAN_FP16_CHECK=off); weights that do not fit fp16 "
                    "would fail silently")
        return
    marker = fp16_check_marker(what, sources, {"fp32_modules": sorted(fp32_modules),
                                               "underflow_frac": underflow_frac})
    if _all_agree(mode == "auto" and marker is not None and os.path.exists(marker), group):
        log.info(f"[fp16-check] {what}: these weight files already passed ({marker}); skipped. "
                 "WAN_FP16_CHECK=force re-checks")
        return
    check_fp16_castable(named_params(), fp32_modules=fp32_modules, underflow_frac=underflow_frac,
                        group=group, log=log, what=what, hint=hint)
    if marker is None:
        return
    try:  # passed: remember it (atomic, so concurrent ranks never see a half-written marker)
        os.makedirs(os.path.dirname(marker), exist_ok=True)
        tmp = f"{marker}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump({"what": what, "sources": [os.path.abspath(s) for s in sources]}, f)
        os.replace(tmp, marker)
    except OSError as e:  # a read-only cache only costs the next run its 3 minutes
        log.warning(f"[fp16-check] could not write {marker}: {e}")


# ----------------------------------------------------------------------------- fp32 islands
def _to_fp32(x):
    if torch.is_tensor(x):
        return x.float() if x.is_floating_point() and x.dtype != torch.float32 else x
    if isinstance(x, (list, tuple)) and x and all(torch.is_tensor(t) for t in x):
        return type(x)(_to_fp32(t) for t in x)
    return x


def force_fp32(module, device_type=None):
    """Run ``module.forward`` in fp32: floating args upcast, fp32 autocast on CUDA (also upcasts
    the fp16 FSDP-gathered weights), SDPA attention in fp32. Idempotent."""
    if getattr(module, "_wan_fp32", False):
        return module
    if device_type is None:
        p = next(module.parameters(), None)
        device_type = p.device.type if p is not None else "cuda"
    inner = module.forward

    def forward(*args, **kwargs):
        args = tuple(_to_fp32(a) for a in args)
        kwargs = {k: _to_fp32(v) for k, v in kwargs.items()}
        if device_type == "cuda":
            ctx = torch.autocast("cuda", dtype=torch.float32)
        else:  # CPU path (tests): weights are fp32 already, just leave autocast
            ctx = torch.autocast(device_type, enabled=False)
        with ctx, compute_dtype_override(torch.float32):
            return inner(*args, **kwargs)

    module.forward = forward
    module._wan_fp32 = True
    return module


# ----------------------------------------------------------------------------- runtime
_WATCH = re.compile(
    r"^(patch_embedding|text_embedding|time_projection|img_emb|head)$"
    r"|^blocks\.\d+(\.block(\.(cross_attn|ffn|ffn\.0))?)?$")


def module_of(label):
    return re.split(r"[:\[]", label)[0]


def unit_of(label):
    """Top-level unit that can be switched to fp32: 'blocks.N' or a root child."""
    m = re.match(r"^(blocks\.\d+)", label)
    return m.group(1) if m else re.split(r"[.:\[]", label)[0]


def _tensors(obj):
    if torch.is_tensor(obj):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for o in obj:
            yield from _tensors(o)
    elif hasattr(obj, "tensors"):  # TensorList
        yield from _tensors(obj.tensors)


@dataclass
class Finding:
    step: object
    branch: str
    label: str = None          # None: non-finite seen by a sentinel, module not localised yet
    value: float = float("nan")
    dtype: str = ""
    source: str = "model output"

    @property
    def unit(self):
        return unit_of(self.label) if self.label else None

    def message(self):
        where = self.label or f"<{self.source}> (not localised)"
        return (f"[fp16-guard] non-finite (inf/NaN) activation at step={self.step} branch={self.branch}: "
                f"first offending module {where} (dtype={self.dtype})")


def _maxabs(t):
    if not t.numel():
        return torch.zeros((), dtype=torch.float32, device=t.device)
    lo, hi = torch.aminmax(t.detach())
    return torch.maximum(hi.float(), -lo.float())  # NaN propagates


def _dist_on():
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


class OverflowGuard:
    """Runtime fp16 guard around every DiT forward.

    mode: 'off' | 'first_step' (hooks on the reference pass + step 0 of every clip) | 'all'.
    action: 'abort' (raise FP16OverflowError) | 'fp32' (switch the offending block to fp32, rerun).

    Always on (mode != off), every step: the model-output sentinel and the attention-kernel monitor
    (q/k/v/out of every SDPA call, after cast + RoPE + slicing) are device-side flags; a
    non-detailed forward then costs 1 all-reduce of 2 floats + 1 host readback. On a detailed step
    forward hooks additionally record max|x| of block / attention / FFN / modulation outputs and
    the monitor records each kernel tensor with its module label, in execution order; that
    forward costs 2 all-reduces (record counts, then all flags/values) + 2 readbacks. Every rank
    takes the same decision; nothing raises inside a hook or kernel."""

    def __init__(self, model, mode="first_step", action="fp32", warn_at=2.0 ** 14,
                 max_promotions=8, fp32_modules=(), log=None):
        assert mode in ("off", "first_step", "all"), mode
        assert action in ("abort", "fp32"), action
        self.model, self.mode, self.action = model, mode, action
        self.warn_at, self.max_promotions = float(warn_at), int(max_promotions)
        self.log = log or _log
        self.armed = False
        self.fp32_units = set(fp32_modules)  # load-time allow-list + runtime promotions
        self._records = []
        self.handles = []
        if mode != "off":
            MONITOR.enabled = True
            for name, mod in model.named_modules():
                if _WATCH.match(name):
                    self.handles.append(mod.register_forward_pre_hook(
                        self._make_pre_hook(name), with_kwargs=True))
                    self.handles.append(mod.register_forward_hook(
                        self._make_hook(name), with_kwargs=True))

    def remove(self):
        for h in self.handles:
            h.remove()
        MONITOR.enabled = False

    @staticmethod
    def _label(name, kwargs):
        method = kwargs.get("method")
        return f"{name}:{method}" if method else name

    def _make_pre_hook(self, name):
        def pre_hook(module, args, kwargs):
            if self.armed:
                MONITOR.push(self._label(name, kwargs))
        return pre_hook

    def _make_hook(self, name):
        def hook(module, args, kwargs, output):
            if not self.armed:
                return
            MONITOR.pop()
            label = self._label(name, kwargs)
            if kwargs.get("method") == "post_self_attention" and args:
                self._add(label + ":attn_out", args[0])
            self._add(label, output)
        return hook

    def _add(self, label, obj):
        for i, t in enumerate(_tensors(obj)):
            if t.is_floating_point():
                self._records.append((f"{label}[{i}]", str(t.dtype).replace("torch.", ""), _maxabs(t)))

    def detailed(self, step):
        return self.mode == "all" or (self.mode == "first_step" and step in ("ref", 0))

    @staticmethod
    def _all_reduce_max(t):
        # every rank runs the same forward on its own sequence shard: decisions must agree,
        # otherwise a rank that re-runs a forward alone would deadlock the collectives.
        if _dist_on():
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return t

    def _device(self, out, records):
        dev = next((t.device for t in _tensors(out)), None)
        if dev is None and records:
            dev = records[0][2].device
        return dev if dev is not None else next(self.model.parameters()).device

    def _check(self, out, step, branch, armed=True):
        """Collectives per call: detailed (armed) forward = 2 all-reduces (record counts, then the
        packed flags/values) + 2 device->host readbacks; non-detailed forward = 1 all-reduce of 2
        floats + 1 readback. 'armed' is identical on all ranks (step schedule + collective rerun
        decisions), and only armed forwards produce records, so the count check is skipped there."""
        records, self._records = self._records, []
        dev = self._device(out, records)
        n = len(records)
        if not armed and n:
            raise RuntimeError(f"[fp16-guard] internal: {n} records on a non-detailed forward")
        if armed:
            # identical code paths give identical record counts; if not, every rank must take the
            # same error branch BEFORE the packed reduction (min and max in one collective)
            cnt = self._all_reduce_max(torch.tensor([float(n), -float(n)], dtype=torch.float32,
                                                    device=dev)).cpu().tolist()  # one readback
            mx, mn = int(cnt[0]), int(-cnt[1])
        else:
            mx = mn = 0
        if mx != mn:
            rank = dist.get_rank() if _dist_on() else 0
            raise RuntimeError(f"[fp16-guard] ranks recorded different tensor counts (min {mn}, max {mx}; "
                               f"rank {rank} recorded {n}) at step={step} branch={branch}; cannot agree "
                               f"on a finding - aborting on all ranks")
        sentinel = torch.zeros(1, dtype=torch.float32, device=dev)
        for t in _tensors(out):
            if t.is_floating_point():
                sentinel = torch.maximum(sentinel, (~torch.isfinite(_maxabs(t))).float().view(1))
        kernel = MONITOR.take(dev).view(1)
        if records:
            vals = torch.stack([r[2].to(dev) for r in records])
            bad = ~torch.isfinite(vals)
            packed = torch.cat([bad.float(), torch.where(bad, torch.zeros_like(vals), vals), sentinel, kernel])
        else:
            packed = torch.cat([sentinel, kernel])
        packed = self._all_reduce_max(packed).cpu().tolist()  # single host sync
        bad, vals, sentinel, kernel = packed[:n], packed[n:2 * n], packed[-2], packed[-1]
        finding, warn = None, []
        for (label, dtype, _), b, v in zip(records, bad, vals):
            if b:
                if finding is None:
                    finding = Finding(step, branch, label, float("inf"), dtype)
            elif v >= self.warn_at and not _matches(module_of(label), self.fp32_units):
                warn.append((label, dtype, v))
        if warn and (not _dist_on() or dist.get_rank() == 0):
            warn.sort(key=lambda r: -r[2])
            self.log.warning(
                f"[fp16-guard] step={step} branch={branch}: {len(warn)} tensors with max|x| >= "
                f"{self.warn_at:g} (fp16 max {FP16_MAX:g}); largest: "
                + ", ".join(f"{l}({d})={v:.4g}" for l, d, v in warn[:5]))
        if finding is None and (sentinel or kernel):
            finding = Finding(step, branch, None, float("nan"), "",
                              source="attention kernel input/output" if kernel else "model output")
        return finding

    def run(self, call, step, branch=""):
        """Run ``call()`` (one model forward) under the guard; returns its output."""
        if self.mode == "off":
            return call()
        armed = self.detailed(step)
        for _ in range(self.max_promotions + 2):
            self.armed = armed
            self._records = []  # never carry records over from an interrupted forward
            MONITOR.begin(sink=self._records if armed else None)
            try:
                out = call()
            finally:
                self.armed = False
                MONITOR.end()
            finding = self._check(out, step, branch, armed)
            if finding is None:
                return out
            if finding.label is None:
                if armed:
                    raise FP16OverflowError(finding.message() + " (no hooked tensor was non-finite)")
                self.log.error(finding.message() + "; re-running this forward with hooks to localise")
                armed = True
                continue
            self.log.error(finding.message())
            unit = finding.unit
            already = _matches(module_of(finding.label), self.fp32_units)
            try:
                target = self.model.get_submodule(unit)
            except AttributeError:
                target = None
            if self.action == "abort" or already or target is None:
                hint = ("already running in fp32 - this is not an fp16 range problem" if already else
                        f"'{unit}' cannot be switched to fp32 on its own; run with --dtype fp32"
                        if target is None else
                        f"add '{unit}' to test_cfg.fp32_modules, set test_cfg.overflow_action=fp32, "
                        f"or run with --dtype fp32")
                raise FP16OverflowError(f"{finding.message()}; {hint}")
            if len(self.fp32_units) >= self.max_promotions:
                raise FP16OverflowError(f"{finding.message()}; already {len(self.fp32_units)} units "
                                        f"in fp32 ({sorted(self.fp32_units)}): use --dtype fp32")
            force_fp32(target)
            self.fp32_units.add(unit)
            self.log.error(f"[fp16-guard] FALLBACK: '{unit}' now runs in fp32 activations for the rest "
                           f"of this run (fp32 units: {sorted(self.fp32_units)}); re-running step={step} "
                           f"branch={branch}")
            armed = True
        raise FP16OverflowError("[fp16-guard] gave up after repeated fp32 promotions")

    # ------------------------------------------------------------- other components
    def begin_component(self):
        """Clear the attention monitor before running a non-DiT component (e.g. CLIP)."""
        if self.mode != "off":
            MONITOR.begin()

    def attention_flag(self, device):
        """inf if any attention kernel since begin_component() saw inf/NaN, else 0 (a device
        scalar that assert_finite() treats like any other tensor)."""
        if self.mode == "off":
            return None
        f = MONITOR.take(device)
        return torch.where(f > 0, torch.full_like(f, float("inf")), torch.zeros_like(f))

    def assert_finite(self, what, items, hints=None):
        """Collective inf/NaN check of named tensors (or lists of tensors / 0-d flag tensors).
        All ranks reduce the same flags and raise the same FP16OverflowError."""
        if self.mode == "off" or not items:
            return
        names, flags = list(items), []
        dev = None
        for name in names:
            ts = [t for t in _tensors(items[name]) if t.is_floating_point()]
            f = torch.zeros((), dtype=torch.float32, device=ts[0].device if ts else "cpu")
            for t in ts:
                f = torch.maximum(f, (~torch.isfinite(_maxabs(t))).float().to(f.device))
            dev = dev or f.device
            flags.append(f)
        flags = self._all_reduce_max(torch.stack([f.to(dev) for f in flags])).cpu().tolist()
        bad = [n for n, f in zip(names, flags) if f]
        if bad:
            hints = hints or {}
            msg = (f"[fp16-guard] non-finite (inf/NaN) values in {what}: " + ", ".join(bad) + ". "
                   + " ".join(dict.fromkeys(hints[n] for n in bad if n in hints)))
            self.log.error(msg)
            raise FP16OverflowError(msg)
