"""scripts/convert_to_wan5b.py: a DiT-style checkpoint (fused qkv / kv, other names) converts back exactly."""
import os
import re
import sys

import pytest
import torch

from conftest import REPO

pytest.importorskip("diffusers")
sys.path.insert(0, os.path.join(REPO, "scripts"))
import convert_to_wan5b as cv  # noqa: E402


def tiny():
    from diffusers import WanTransformer3DModel
    torch.manual_seed(0)
    return WanTransformer3DModel(patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=8, in_channels=48,
                                 out_channels=48, text_dim=16, freq_dim=16, ffn_dim=32, num_layers=2,
                                 cross_attn_norm=True, qk_norm="rms_norm_across_heads")


def to_dit_style(sd):
    """Diffusers names -> a DiT-style layout with fused qkv and kv (what RENAME/SPLIT expect)."""
    inv = [(r"^patch_embedding\.", "x_embedder.proj."),
           (r"^condition_embedder\.time_embedder\.linear_1\.", "t_embedder.mlp.0."),
           (r"^condition_embedder\.time_embedder\.linear_2\.", "t_embedder.mlp.2."),
           (r"^condition_embedder\.time_proj\.", "t_block.1."),
           (r"^condition_embedder\.text_embedder\.linear_1\.", "y_embedder.y_proj.fc1."),
           (r"^condition_embedder\.text_embedder\.linear_2\.", "y_embedder.y_proj.fc2."),
           (r"\.attn1\.to_out\.0\.", ".attn.proj."), (r"\.attn1\.norm_q\.", ".attn.q_norm."),
           (r"\.attn1\.norm_k\.", ".attn.k_norm."), (r"\.attn2\.to_q\.", ".cross_attn.q_linear."),
           (r"\.attn2\.to_out\.0\.", ".cross_attn.proj."), (r"\.attn2\.norm_q\.", ".cross_attn.q_norm."),
           (r"\.attn2\.norm_k\.", ".cross_attn.k_norm."), (r"\.ffn\.net\.0\.proj\.", ".mlp.fc1."),
           (r"\.ffn\.net\.2\.", ".mlp.fc2."), (r"^proj_out\.", "final_layer.linear."),
           (r"^scale_shift_table$", "final_layer.scale_shift_table")]
    out = {}
    for k, v in sd.items():
        m = re.match(r"^(blocks\.\d+)\.attn1\.to_([qkv])\.(weight|bias)$", k)
        if m:
            if m.group(2) == "q":
                b, s = m.group(1), m.group(3)
                out[f"{b}.attn.qkv.{s}"] = torch.cat([sd[f"{b}.attn1.to_{x}.{s}"] for x in "qkv"])
            continue
        m = re.match(r"^(blocks\.\d+)\.attn2\.to_([kv])\.(weight|bias)$", k)
        if m:
            if m.group(2) == "k":
                b, s = m.group(1), m.group(3)
                out[f"{b}.cross_attn.kv_linear.{s}"] = torch.cat([sd[f"{b}.attn2.to_{x}.{s}"] for x in "kv"])
            continue
        for pat, rep in inv:
            k = re.sub(pat, rep, k)
        out[k] = v
    return out


def test_dit_style_checkpoint_converts_back_exactly(tmp_path):
    m = tiny()
    sd = m.state_dict()
    src = {"module." + k: v for k, v in to_dit_style(sd).items()}
    assert len(src) < len(sd)                                   # fused layers: fewer tensors, like the real case
    m.save_config(tmp_path / "cfg")
    shapes = cv.target_shapes(str(tmp_path / "cfg"))
    out, mapped, _ = cv.convert(src)
    missing, extra, bad = cv.check(out, shapes)
    assert not missing and not extra and not bad
    for k, v in sd.items():
        assert torch.equal(out[k], v), k
    text, _ = cv.report("x", out, mapped, shapes)
    assert "missing 0 | left over 0 | shape mismatches 0" in text


def test_report_lists_what_is_left(tmp_path):
    m = tiny()
    m.save_config(tmp_path / "cfg")
    shapes = cv.target_shapes(str(tmp_path / "cfg"))
    src = to_dit_style(m.state_dict())
    src["blocks.0.weird.weight"] = src.pop("blocks.0.mlp.fc1.weight")
    src["x_embedder.proj.weight"] = src["x_embedder.proj.weight"].reshape(32, -1)   # a Linear patch embedding
    out, mapped, _ = cv.convert(src)
    text, (missing, extra, bad) = cv.report("x", out, mapped, shapes)
    assert missing == ["blocks.0.ffn.net.0.proj.weight"] and extra == ["blocks.0.weird.weight"]
    assert list(bad) == ["patch_embedding.weight"] and "numel same" in text


def test_cli_writes_a_file_the_5b_script_loads(tmp_path):
    import ti2v5b_replace as tr
    m = tiny()
    m.save_config(tmp_path / "cfg")
    torch.save({"state_dict": to_dit_style(m.state_dict())}, tmp_path / "expert.pth")
    sys.argv = ["convert_to_wan5b.py", "--src", str(tmp_path / "expert.pth"), "--config", str(tmp_path / "cfg"),
                "--report", str(tmp_path / "r.txt"), "--out", str(tmp_path / "expert_wan.safetensors"), "--dtype", "fp32"]
    cv.main()
    loaded = tr.load_transformer_ckpt(str(tmp_path / "expert_wan.safetensors"), str(tmp_path / "cfg"), torch.float32,
                                      log=lambda *_: None)
    for k, v in m.state_dict().items():
        assert torch.equal(loaded.state_dict()[k], v), k
