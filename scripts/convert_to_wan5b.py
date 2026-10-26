"""Convert a TI2V-5B-architecture transformer checkpoint from another codebase's key layout into the public
diffusers WanTransformer3DModel layout (Wan-AI/Wan2.2-TI2V-5B-Diffusers), so scripts/ti2v5b_replace.py can load it.

  report:  python scripts/convert_to_wan5b.py --src expert.pth --config <5B-Diffusers>/transformer --report r.txt
  convert: python scripts/convert_to_wan5b.py --src expert.pth --config <5B-Diffusers>/transformer --out expert_wan.safetensors

The report lists, side by side, which source keys are mapped (and to what), which target keys are still missing
and which source keys are left over, with shapes. Edit RENAME / SPLIT below until the report is clean
(0 missing, 0 left over, 0 shape mismatches); only then --out writes a file. A clean report proves names and shapes,
not semantics (modulation order, patch flatten order, RoPE, time embedding): check those with a numerical
equivalence test against the original model code (same random input -> same output).
"""
import argparse
import os
import re

import torch

# (regex on the source key, replacement) applied in order; starting guesses for DiT-style codebases - edit freely
RENAME = [
    (r"^x_embedder\.proj\.", "patch_embedding."),
    (r"^t_embedder\.mlp\.0\.", "condition_embedder.time_embedder.linear_1."),
    (r"^t_embedder\.mlp\.2\.", "condition_embedder.time_embedder.linear_2."),
    (r"^t_block\.1\.", "condition_embedder.time_proj."),
    (r"^y_embedder\.y_proj\.fc1\.", "condition_embedder.text_embedder.linear_1."),
    (r"^y_embedder\.y_proj\.fc2\.", "condition_embedder.text_embedder.linear_2."),
    (r"^blocks\.(\d+)\.attn\.proj\.", r"blocks.\1.attn1.to_out.0."),
    (r"^blocks\.(\d+)\.attn\.q_norm\.", r"blocks.\1.attn1.norm_q."),
    (r"^blocks\.(\d+)\.attn\.k_norm\.", r"blocks.\1.attn1.norm_k."),
    (r"^blocks\.(\d+)\.cross_attn\.q_linear\.", r"blocks.\1.attn2.to_q."),
    (r"^blocks\.(\d+)\.cross_attn\.proj\.", r"blocks.\1.attn2.to_out.0."),
    (r"^blocks\.(\d+)\.cross_attn\.q_norm\.", r"blocks.\1.attn2.norm_q."),
    (r"^blocks\.(\d+)\.cross_attn\.k_norm\.", r"blocks.\1.attn2.norm_k."),
    (r"^blocks\.(\d+)\.mlp\.fc1\.", r"blocks.\1.ffn.net.0.proj."),
    (r"^blocks\.(\d+)\.mlp\.fc2\.", r"blocks.\1.ffn.net.2."),
    (r"^final_layer\.linear\.", "proj_out."),
    (r"^final_layer\.scale_shift_table$", "scale_shift_table"),
]
# fused tensors split along dim 0 into equal parts: (regex, [replacement per part])
SPLIT = [
    (r"^blocks\.(\d+)\.attn\.qkv\.", [r"blocks.\1.attn1.to_q.", r"blocks.\1.attn1.to_k.", r"blocks.\1.attn1.to_v."]),
    (r"^blocks\.(\d+)\.cross_attn\.kv_linear\.", [r"blocks.\1.attn2.to_k.", r"blocks.\1.attn2.to_v."]),
]
PREFIXES = ("module.", "model.", "_orig_mod.", "dit.", "transformer.", "net.")


def read_state_dict(path):
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    for k in ("state_dict", "model", "module", "ema"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict) and len(sd) <= 4:
            sd = sd[k]
    return sd


def target_shapes(config_dir):
    """Key -> shape of the public diffusers model built from its config (no weights needed)."""
    from diffusers import WanTransformer3DModel
    with torch.device("meta"):
        m = WanTransformer3DModel.from_config(WanTransformer3DModel.load_config(config_dir))
    return {k: tuple(v.shape) for k, v in m.state_dict().items()}


def convert(sd, rename=RENAME, split=SPLIT):
    """Returns (converted dict, {source key: [target keys]}, [source keys left over])."""
    for pre in PREFIXES:
        if sd and all(k.startswith(pre) for k in sd):
            sd = {k[len(pre):]: v for k, v in sd.items()}
    if any(".self_attn.q." in k for k in sd):  # original Wan names: diffusers has the converter
        from diffusers.loaders.single_file_utils import convert_wan_transformer_to_diffusers
        sd = convert_wan_transformer_to_diffusers(dict(sd))
    out, mapped, left = {}, {}, []
    for k, v in sd.items():
        for pat, reps in split:
            if re.search(pat, k):
                parts = v.chunk(len(reps), dim=0)
                names = [re.sub(pat, r, k) for r in reps]
                out.update(zip(names, parts))
                mapped[k] = names
                break
        else:
            new = k
            for pat, rep in rename:
                new = re.sub(pat, rep, new)
            out[new] = v
            if new != k:
                mapped[k] = [new]
    return out, mapped, left


def check(out, shapes):
    """missing target keys, left-over keys, shape mismatches {key: (got, want)} (reshapable if numel matches)."""
    missing = [k for k in shapes if k not in out]
    extra = [k for k in out if k not in shapes]
    bad = {k: (tuple(out[k].shape), shapes[k]) for k in shapes if k in out and tuple(out[k].shape) != shapes[k]}
    return missing, extra, bad


def report(src, out, mapped, shapes, path=None):
    missing, extra, bad = check(out, shapes)
    lines = [f"source: {src}", f"target tensors {len(shapes)}, converted {len(out)}, renamed/split {len(mapped)}",
             f"missing {len(missing)} | left over {len(extra)} | shape mismatches {len(bad)}", "",
             "== missing target keys (shape) =="] + [f"  {k}  {shapes[k]}" for k in missing] + \
            ["", "== left-over source keys after renaming (shape) =="] + [f"  {k}  {tuple(out[k].shape)}" for k in extra] + \
            ["", "== shape mismatches (got -> want; same numel = maybe just a reshape) =="] + \
            [f"  {k}  {g} -> {w}  numel {'same' if torch.Size(g).numel() == torch.Size(w).numel() else 'DIFFERENT'}"
             for k, (g, w) in bad.items()] + ["", "== mapped =="] + [f"  {k} -> {', '.join(v)}" for k, v in mapped.items()]
    text = "\n".join(lines) + "\n"
    if path:
        open(path, "w").write(text)
    return text, (missing, extra, bad)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True)
    p.add_argument("--config", required=True, help="<Wan2.2-TI2V-5B-Diffusers>/transformer (only config.json is read)")
    p.add_argument("--report", help="write the side-by-side report here")
    p.add_argument("--out", help="write the converted .safetensors (only when the report is clean)")
    p.add_argument("--reshape", action="store_true", help="reshape tensors whose numel matches (e.g. a Linear patch "
                                                          "embedding to Conv3d) - verify numerically afterwards")
    p.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    a = p.parse_args()
    shapes = target_shapes(a.config)
    out, mapped, _ = convert(read_state_dict(a.src))
    if a.reshape:
        for k, (g, w) in check(out, shapes)[2].items():
            if torch.Size(g).numel() == torch.Size(w).numel():
                out[k] = out[k].reshape(w)
    text, (missing, extra, bad) = report(a.src, out, mapped, shapes, a.report)
    print(text.split("\n\n")[0])
    if a.out:
        if missing or extra or bad:
            raise SystemExit("not clean: fix RENAME/SPLIT (see the report) before writing --out")
        from safetensors.torch import save_file
        dt = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[a.dtype]
        save_file({k: v.to(dt).contiguous() for k, v in out.items()}, a.out)
        print(f"-> {a.out} ({os.path.getsize(a.out) / 1e9:.1f} GB)")


if __name__ == "__main__":
    main()
