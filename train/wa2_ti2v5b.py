"""Shared-QKV Wan-Animate-2 topology on the TI2V-5B reference-slot layout."""
import re

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

QKV_NAME = re.compile(r"blocks\.\d+\.attn1\.to_[qkv]\.(weight|bias)\Z")
COUNTS = dict(total=4_999_787_712, trainable=849_623_040, frozen=4_150_164_672, tensors=180)


def set_shared_qkv_trainable(model):
    """Freeze everything except the shared, unfused self-attention Q/K/V weights and biases."""
    if any(b.attn1.fused_projections for b in model.blocks):
        raise ValueError("WA2 requires unfused QKV projections")
    model.requires_grad_(False)
    params = []
    for name, p in model.named_parameters():
        if QKV_NAME.fullmatch(name):
            p.requires_grad_(True)
            params.append(p)
    if len(params) != 6 * len(model.blocks):
        raise ValueError("expected six QKV tensors per block")
    return params


def parameter_report(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return dict(total=total, trainable=trainable, frozen=total - trainable,
                tensors=sum(p.requires_grad for p in model.parameters()))


def assert_5b(model):
    expected = dict(in_channels=48, out_channels=48, num_layers=30, num_attention_heads=24,
                    attention_head_dim=128, text_dim=4096, freq_dim=256, ffn_dim=14336,
                    qk_norm="rms_norm_across_heads", cross_attn_norm=True, image_dim=None,
                    added_kv_proj_dim=None, rope_max_seq_len=1024, pos_embed_seq_len=None, eps=1e-6)
    for key, value in expected.items():
        if getattr(model.config, key) != value:
            raise ValueError(f"not TI2V-5B: {key} != {value}")
    assert tuple(model.config.patch_size) == (1, 2, 2)
    assert parameter_report(model) == COUNTS, parameter_report(model)


def to_fp16_storage(model):
    """fp16 weights except the shared QKV (exact fp32 start) and diffusers' _keep_in_fp32_modules (as
    from_pretrained(torch_dtype=fp16) keeps them); buffers such as the RoPE cos/sin tables stay untouched, since
    .to(fp16) would round them for good."""
    keep = set(model._keep_in_fp32_modules or ())
    for name, p in model.named_parameters():
        if not QKV_NAME.fullmatch(name) and not keep & set(name.split(".")):
            p.data = p.data.half()
    return model


def driving_rope(model, x):
    """Positions (t, r, W_tok + c), from the right half of a width-doubled grid."""
    _, _, frames, height, width = x.shape
    pt, ph, pw = model.config.patch_size
    f, h, w = frames // pt, height // ph, width // pw
    wide = model.rope(x.new_empty(1, 1, frames, height, 2 * width))
    return tuple(r.reshape(1, f, h, 2 * w, 1, -1)[:, :, :, w:].reshape(1, f * h * w, 1, -1) for r in wide)


def apply_rope(x, rope):
    x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
    cos, sin = rope[0][..., 0::2], rope[1][..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


def grouped_attention(q, k, v, k_d=None, v_d=None, frame_tokens=390):
    """[B, heads, N, dim]: one softmax over own keys plus the aligned driving frame."""
    if k_d is None:
        return F.scaled_dot_product_attention(q, k, v)
    if q.shape != k_d.shape or v_d.shape != v.shape or q.shape[2] % frame_tokens:
        raise ValueError("generation and driving token grids must match")
    out = []
    for start in range(0, q.shape[2], frame_tokens):
        stop = start + frame_tokens
        keys = k if start == 0 else torch.cat([k, k_d[:, :, start:stop]], 2)
        values = v if start == 0 else torch.cat([v, v_d[:, :, start:stop]], 2)
        out.append(F.scaled_dot_product_attention(q[:, :, start:stop], keys, values))
    return torch.cat(out, 2)


def block_qkv(block, h, temb_proj):
    """Diffusers modulation and full-width Q/K normalization, before head reshape and RoPE."""
    modulation = tuple(t.squeeze(2) for t in
                       (block.scale_shift_table.unsqueeze(0) + temb_proj.float()).chunk(6, dim=2))
    shift, scale = modulation[:2]
    u = (block.norm1(h.float()) * (1 + scale) + shift).type_as(h)
    attn = block.attn1
    q = attn.norm_q(attn.to_q(u)).unflatten(2, (attn.heads, -1))
    k = attn.norm_k(attn.to_k(u)).unflatten(2, (attn.heads, -1))
    v = attn.to_v(u).unflatten(2, (attn.heads, -1))
    return q, k, v, modulation


def block_tail(block, h, text, out, modulation):
    _, _, gate, shift, scale, ffn_gate = modulation
    out = out.transpose(1, 2).flatten(2, 3)
    out = block.attn1.to_out[1](block.attn1.to_out[0](out))
    h = (h.float() + out * gate).type_as(h)
    u = block.norm2(h.float()).type_as(h)
    h = h + block.attn2(u, text, None, None)
    u = (block.norm3(h.float()) * (1 + scale) + shift).type_as(h)
    return (h.float() + block.ffn(u).float() * ffn_gate).type_as(h)


def drive_block(block, h, text, temb_proj, rope_d):
    q, k, v, modulation = block_qkv(block, h, temb_proj)
    out = F.scaled_dot_product_attention(apply_rope(q, rope_d).transpose(1, 2),
                                        apply_rope(k, rope_d).transpose(1, 2), v.transpose(1, 2))
    return block_tail(block, h, text, out, modulation), k, v


def gen_block(block, h, text, temb_proj, rope_g, k_d, v_d, rope_d, frame_tokens=390):
    q, k, v, modulation = block_qkv(block, h, temb_proj)
    kd = apply_rope(k_d, rope_d).transpose(1, 2) if k_d is not None else None
    vd = v_d.transpose(1, 2) if v_d is not None else None
    out = grouped_attention(apply_rope(q, rope_g).transpose(1, 2), apply_rope(k, rope_g).transpose(1, 2),
                            v.transpose(1, 2), kd, vd, frame_tokens)
    return block_tail(block, h, text, out, modulation)


def forward_drive(model, drive, text, use_checkpoint=False):
    """Build graph-bearing pre-RoPE K/V for every block; no generation inputs enter this pass."""
    h = model.patch_embedding(drive).flatten(2).transpose(1, 2).float()
    times = torch.ones(h.shape[:2], device=h.device)
    _, tproj, enc, _ = model.condition_embedder(times.flatten(), text, None, timestep_seq_len=h.shape[1])
    tproj = tproj.unflatten(2, (6, -1))
    rope = driving_rope(model, drive)
    cache = []
    for block in model.blocks:
        args = (block, h, enc, tproj, rope)
        if use_checkpoint and torch.is_grad_enabled():
            h, k, v = checkpoint(drive_block, *args, use_reentrant=False)
        else:
            h, k, v = drive_block(*args)
        cache.append((k, v))
    return tuple(cache)


def wa2_forward(model, x, t_x, text, drive=None, use_checkpoint=False, drive_cache=None):
    """Two shared passes, or the same generation block code without driving retrieval."""
    if drive is not None and drive_cache is not None:
        raise ValueError("pass drive or drive_cache, not both")
    if drive is not None:
        if drive.shape != x.shape:
            raise ValueError("generation and driving latent shapes must match")
        drive_cache = forward_drive(model, drive, text, use_checkpoint)
    if drive_cache is not None and len(drive_cache) != len(model.blocks):
        raise ValueError("one driving K/V pair is required per block")
    b, _, frames, height, width = x.shape
    pt, ph, pw = model.config.patch_size
    rope, rope_d = model.rope(x), driving_rope(model, x) if drive_cache is not None else None
    h = model.patch_embedding(x).flatten(2).transpose(1, 2).float()
    temb, tproj, enc, _ = model.condition_embedder(t_x.flatten(), text, None, timestep_seq_len=t_x.shape[1])
    tproj = tproj.unflatten(2, (6, -1))
    for i, block in enumerate(model.blocks):
        k, v = drive_cache[i] if drive_cache is not None else (None, None)
        args = (block, h, enc, tproj, rope, k, v, rope_d, (height // ph) * (width // pw))
        if use_checkpoint and torch.is_grad_enabled():
            h = checkpoint(gen_block, *args, use_reentrant=False)
        else:
            h = gen_block(*args)
    shift, scale = (model.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
    h = (model.norm_out(h.float()) * (1 + scale.squeeze(2)) + shift.squeeze(2)).type_as(h)
    h = model.proj_out(h)
    h = h.reshape(b, frames // pt, height // ph, width // pw, pt, ph, pw, -1).permute(0, 7, 1, 4, 2, 5, 3, 6)
    return h.flatten(6, 7).flatten(4, 5).flatten(2, 3)
