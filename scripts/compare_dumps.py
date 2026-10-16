"""Compare two WAN_DUMP_DIR folders stage by stage and report where they start to diverge.

    python scripts/compare_dumps.py <dump_dir_A> <dump_dir_B> [rel_threshold=0.05]

Each stage prints the relative L2 difference ||A-B|| / ||B||, the largest absolute difference,
the largest |value| in each run and the number of non-finite values. Runs with different dtypes
(e.g. bf16 vs fp32) always differ a little; look for the first stage whose difference jumps.
"""
import os
import re
import sys

import torch


def load(d):
    out = {}
    for f in os.listdir(d):
        m = re.match(r"^(\d+)_(.+)\.pt$", f)
        if m:
            out[m.group(2)] = (int(m.group(1)), os.path.join(d, f))
    return out


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    a, b = load(sys.argv[1]), load(sys.argv[2])
    thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
    first = None
    for name, (_, pa) in sorted(a.items(), key=lambda kv: kv[1][0]):
        if name not in b:
            print(f"{name:28s} missing in B")
            continue
        x, y = torch.load(pa).float(), torch.load(b[name][1]).float()
        if x.shape != y.shape:
            print(f"{name:28s} shape {tuple(x.shape)} vs {tuple(y.shape)}")
            continue
        nonfinite = int((~torch.isfinite(x)).sum() + (~torch.isfinite(y)).sum())
        d = torch.nan_to_num(x - y)
        rel = (d.norm() / (torch.nan_to_num(y).norm() + 1e-12)).item()
        flag = ""
        if first is None and rel > thr:
            first, flag = name, "   <== first big divergence"
        print(f"{name:28s} rel={rel:9.3e} max|diff|={d.abs().max().item():9.3e} "
              f"max|A|={torch.nan_to_num(x).abs().max().item():9.3e} max|B|={torch.nan_to_num(y).abs().max().item():9.3e} "
              f"nonfinite={nonfinite}{flag}")
    for name in sorted(set(b) - set(a)):
        print(f"{name:28s} missing in A")
    print(f"\nfirst stage with rel > {thr}: {first}")


if __name__ == "__main__":
    main()
