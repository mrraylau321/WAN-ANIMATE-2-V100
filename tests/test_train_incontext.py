"""train/train_incontext_qkv.py on the tiny Wan-Animate-2 model (CPU): freezing, LoRA, driving gradient, overfit."""
import os
import sys
import time

import torch

from conftest import REPO, TINY

sys.path.insert(0, os.path.join(REPO, "train"))
import train_incontext_qkv as tq  # noqa: E402
from wanxiang.models.wan_animate_2_model import WanAnimate2Transformer  # noqa: E402


def tiny_model(seed=0):
    torch.manual_seed(seed)
    m = WanAnimate2Transformer(**TINY)
    for p in m.parameters():  # init_weights zeroes the head (no gradient at all): use a non-trivial random model
        p.data.normal_(0, 0.05)
    return m


def tiny_sample(seed=0, T=3, h=4, w=4):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
    target = torch.stack([torch.sin(3 * xx + k + 0.5 * t) * torch.cos(2 * yy - k) for k in range(16)
                          for t in range(T)]).view(16, T, h, w)
    return dict(target=target, ref=target[:, :1] + 0.1 * r(16, 1, h, w), drive=target + 0.3 * r(16, T, h, w),
                zero=0.1 * r(16, T, h, w), clip_ref=r(257, 1280), clip_drive=r(257, 1280), text=r(7, 32),
                text_ref=r(5, 32))


def probe_loss(model, s, n=4, seed=123):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        return sum(float(tq.flow_loss(model, s, torch.tensor([0.2 + 0.2 * i]), torch.randn(16, 4, 4, 4, generator=g)))
                   for i in range(n)) / n


def test_lora_starts_identical_and_merges_exactly():
    m, s = tiny_model(), tiny_sample()
    base = probe_loss(m, s)
    params = tq.set_trainable(m, "lora", rank=4)
    assert abs(probe_loss(m, s) - base) < 1e-6                            # B = 0: same model
    assert all(n.endswith((".A", ".B")) for n, p in m.named_parameters() if p.requires_grad)
    assert sum(p.numel() for p in params) == 3 * 3 * 4 * (64 + 64)        # 3 blocks x q,k,v x rank 4
    for p in params:
        p.data.normal_(0, 0.1)
    before = probe_loss(m, s)
    tq.merge_lora(m)
    assert abs(probe_loss(m, s) - before) < 1e-5


def test_only_shared_qkv_learns_and_the_driving_clip_matters():
    m, s = tiny_model(), tiny_sample()
    params = tq.set_trainable(m, "qkv")
    names = {n for n, p in m.named_parameters() if p.requires_grad}
    assert names and all(".self_attn.q." in n or ".self_attn.k." in n or ".self_attn.v." in n for n in names)
    s["drive"] = s["drive"].clone().requires_grad_(True)
    loss = tq.flow_loss(m, s, torch.tensor([0.5]), torch.randn(16, 4, 4, 4))
    loss.backward()
    assert s["drive"].grad.abs().sum() > 0                                 # the in-context path is used
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in params)
    assert all(p.grad is None for p in m.parameters() if not p.requires_grad)


def _memorise(m, s, params, lr, steps):
    g = torch.Generator().manual_seed(1)
    fixed = [(torch.tensor([0.3 + 0.2 * i]), torch.randn(16, 4, 4, 4, generator=g)) for i in range(3)]
    probe = lambda: sum(float(tq.flow_loss(m, s, sg, n)) for sg, n in fixed) / len(fixed)
    opt = torch.optim.Adam(params, lr=lr)
    before = probe()
    for step in range(steps):
        sg, n = fixed[step % len(fixed)]
        loss = tq.flow_loss(m, s, sg, n)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return before, probe()


def test_overfits_one_clip():
    """Loss, inputs and gradients are wired right: with every weight free the tiny model memorises one clip.
    With the QKV only it still learns, but a 64-wide random model has little capacity there (real models do)."""
    m, s = tiny_model(), tiny_sample()
    before, after = _memorise(m, s, list(m.parameters()), 3e-3, 300)
    assert after < 0.05 * before, (before, after)
    m = tiny_model()
    before, after = _memorise(m, s, tq.set_trainable(m, "qkv"), 3e-2, 150)
    assert after < 0.85 * before, (before, after)


def test_train_loop_runs_and_saves(tmp_path):
    m, s = tiny_model(), tiny_sample()
    hist = tq.train(m, [s], steps=4, lr=1e-3, mode="lora", rank=4, out=str(tmp_path), save_every=2,
                    log=lambda *_: None)
    assert len(hist) == 4 and all(torch.isfinite(torch.tensor(hist)))
    from safetensors.torch import load_file
    sd = load_file(str(tmp_path / "trainable_000004.safetensors"))
    assert sd and all(k.endswith((".A", ".B")) for k in sd)


def test_augment_driving_removes_colour_keeps_shape():
    frames = torch.rand(3, 5, 32, 32) * 2 - 1
    out = tq.augment_driving(frames, torch.Generator().manual_seed(0))
    assert out.shape == frames.shape and torch.equal(out[0], out[1]) and out.abs().max() <= 1
