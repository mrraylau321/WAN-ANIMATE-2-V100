"""Does the LoRA actually use the driving video? Flow loss on cached samples at fixed noise levels, three ways:
  real      the sample's own driving clip
  shuffled  another sample's driving clip (wrong motion)
  none      no driving tokens at all
If the model uses the driving clip: real < none < shuffled (roughly). The training-loss curve cannot show this: its
per-step noise level is random, so it stays noisy and flat even while the model gets better at using the condition.
Fixed seeds and noise levels make the numbers comparable across checkpoints and runs.
  python train/val_ti2v5b.py --model <TI2V-5B diffusers dir> --data <cache> --lora run/lora_002000.safetensors
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_incontext_ti2v5b import CachedLatents, flow_loss, incontext_forward, load_lora, token_timesteps  # noqa: E402


@torch.no_grad()
def val_losses(model, dataset, n=32, sigmas=(0.3, 0.6, 0.9), drive_level=0.4, seed=0, device="cpu", amp=False,
               drive_rope="same"):
    """-> {'real'|'shuffled'|'none': {sigma: mean loss}} over the first n samples (shuffled = sample i+1's drive)."""
    n = min(n, len(dataset))
    text = dataset.text["text"].to(device).float()
    out = {k: {s: 0.0 for s in sigmas} for k in ("real", "shuffled", "none")}
    for i in range(n):
        s, other = dataset[i], dataset[(i + 1) % len(dataset)]
        x0 = s["target"][None].to(device).float()
        drives = {"real": s["drive"][None].to(device).float(), "shuffled": other["drive"][None].to(device).float()}
        if drives["shuffled"].shape != x0.shape:
            drives["shuffled"] = drives["real"].flip(2)            # different shape: time-reversed own drive instead
        g = torch.Generator().manual_seed(seed + i)
        noise, noise_d = torch.randn(x0.shape, generator=g).to(device), torch.randn(x0.shape, generator=g).to(device)
        for sigma in sigmas:
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                for k, d in drives.items():
                    out[k][sigma] += float(flow_loss(model, x0, d, text, sigma, drive_level, noise, noise_d,
                                                     drive_rope=drive_rope)) / n
                B, _, F_, h, w = x0.shape
                x_s = (1 - sigma) * x0 + sigma * noise
                x_s[:, :, 0] = x0[:, :, 0]
                t = torch.full((B, F_), sigma * 1000, device=device)
                t[:, 0] = 0
                v = incontext_forward(model, x_s, token_timesteps(t, h, w), text)
                out["none"][sigma] += float(F.mse_loss(v[:, :, 1:].float(), (noise - x0)[:, :, 1:].float())) / n
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--lora", help="trained LoRA; none = the untrained base")
    p.add_argument("--n", type=int, default=32, help="samples (the first n of the cache, fixed)")
    p.add_argument("--drive-rope", choices=["same", "offset"], default="same", help="must match the LoRA's training")
    p.add_argument("--weights", choices=["fp32", "fp16"], default="fp32")
    a = p.parse_args()
    from diffusers import WanTransformer3DModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WanTransformer3DModel.from_pretrained(os.path.join(a.model, "transformer"),
                                                  torch_dtype=torch.float32 if a.weights == "fp32" else torch.float16)
    if a.lora:
        load_lora(model, a.lora)
    model.requires_grad_(False).eval().to(device)
    res = val_losses(model, CachedLatents(a.data), a.n, device=device, amp=device != "cpu", drive_rope=a.drive_rope)
    name = os.path.basename(a.lora) if a.lora else "base"
    print(f"{name} ({a.n} samples, drive noise 0.4)")
    for k, row in res.items():
        print(f"  {k:9s} " + "  ".join(f"sigma {s}: {v:.4f}" for s, v in row.items()) +
              f"   mean {sum(row.values()) / len(row):.4f}")


if __name__ == "__main__":
    main()
