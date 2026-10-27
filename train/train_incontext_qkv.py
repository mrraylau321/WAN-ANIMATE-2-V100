"""Training skeleton: teach a video DiT to take its motion from a driving video through in-context tokens, training
only the shared self-attention QKV (everything else frozen), the recipe described for Wan-Animate-2.

Reference architecture: wanxiang/models/wan_animate_2_model.py. Its forward_ref (driving tokens -> per-block K/V cache)
and forward_gen (the generated tokens attend to their own K/V plus the cached driving K/V) are the in-context path, and
the QKV are shared between the two. Porting to another backbone means rewriting `model_inputs`, `predict` and
`qkv_layers` only; the loss, the freezing and the loop stay.

One training sample = one clip, already encoded (CACHE_KEYS; latents use the VAE of the backbone, Wan2.1 here):
  target      [C, T, h, w]  the clip to reconstruct (T latent frames)
  ref         [C, 1, h, w]  the reference image (the character)
  drive       [C, T, h, w]  the driving clip: another person doing the motion (teacher pair), or the target clip
                            itself with its appearance degraded (self-reconstruction, see augment_driving), making
                            copying harder; cross-identity evaluation must establish motion transfer
  zero        [C, T, h, w]  an all-grey clip (the unknown frames of the image-to-video condition), one per shape
  clip_ref    [257, 1280]   CLIP image features of the reference;  clip_drive: of the first driving frame
  text [L, 4096]            T5 of the prompt;  text_ref [L', 4096]: T5 of the driving prompt
Flow matching, Wan convention: x_s = (1 - s) * x0 + s * noise, the model predicts v = noise - x0, timestep t = 1000 s.

    python train/train_incontext_qkv.py --ckpt dit.safetensors --config dit_config.json --data cache_dir \
        --mode lora --rank 64 --steps 2000 --out runs/qkv_lora
"""
import argparse
import glob
import json
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CACHE_KEYS = ("target", "ref", "drive", "zero", "clip_ref", "clip_drive", "text", "text_ref")


# ------------------------------------------------------------------ what is trained
class LoRALinear(nn.Module):
    """y = base(x) + scale * B(A(x)); base frozen, B zero-initialised (the model starts exactly as before)."""

    def __init__(self, base: nn.Linear, rank=64, alpha=None):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.A = nn.Parameter(torch.empty(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = (alpha or rank) / rank

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A.to(x.dtype)), self.B.to(x.dtype)) * self.scale

    def merged(self):
        lin = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        lin.weight.data = self.base.weight.data + (self.B @ self.A).to(self.base.weight.dtype) * self.scale
        if self.base.bias is not None:
            lin.bias.data = self.base.bias.data.clone()
        return lin


def qkv_layers(model):
    """(module owning the layer, attribute name) of every shared self-attention Q, K and V."""
    return [(blk.block.self_attn, n) for blk in model.blocks for n in ("q", "k", "v")]


def set_trainable(model, mode="lora", rank=64, alpha=None):
    """Freeze everything, then make only the shared QKV trainable: 'qkv' = the full Q/K/V weights (as in the paper),
    'lora' = a LoRA on each. Returns the trainable parameters."""
    model.requires_grad_(False)
    for owner, name in qkv_layers(model):
        layer = getattr(owner, name)
        if mode == "qkv":
            layer.requires_grad_(True)
        elif mode == "lora":
            setattr(owner, name, LoRALinear(layer, rank, alpha))
        else:
            raise ValueError(f"mode must be qkv or lora, not {mode}")
    return [p for p in model.parameters() if p.requires_grad]


def merge_lora(model):
    """Fold every LoRA back into a plain nn.Linear (for inference with the unchanged pipeline)."""
    for owner, name in qkv_layers(model):
        layer = getattr(owner, name)
        if isinstance(layer, LoRALinear):
            setattr(owner, name, layer.merged())
    return model


def trainable_state_dict(model):
    return {k: v.detach().cpu() for k, v in model.named_parameters() if v.requires_grad}


# ------------------------------------------------------------------ data
def augment_driving(frames, generator=None):
    """Make appearance harder to copy: grey, blurred, random contrast and brightness.
    frames: float [3, n, H, W] in [-1, 1] (pixels, before VAE encoding). The same random draw for the whole clip."""
    g = generator
    x = frames.mean(0, keepdim=True)                                         # grey: no colours, no clothes colour
    k = int(torch.randint(2, 6, (1,), generator=g)) * 2 + 1                  # blur: reduce fine texture
    x = F.avg_pool2d(x.permute(1, 0, 2, 3), k, stride=1, padding=k // 2, count_include_pad=False).permute(1, 0, 2, 3)
    c = 0.5 + torch.rand(1, generator=g) * 1.0
    b = (torch.rand(1, generator=g) - 0.5) * 0.6
    return (x * c + b).clamp(-1, 1).expand(3, -1, -1, -1).contiguous()


class CachedClipDataset(torch.utils.data.Dataset):
    """*.pt files, each a dict with CACHE_KEYS (written by the data-prep step)."""

    def __init__(self, folder):
        self.files = sorted(glob.glob(os.path.join(folder, "*.pt")))
        if not self.files:
            raise FileNotFoundError(f"no *.pt samples in {folder}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        s = torch.load(self.files[i], map_location="cpu", weights_only=True)
        missing = [k for k in CACHE_KEYS if k not in s]
        if missing:
            raise KeyError(f"{self.files[i]} lacks {missing}")
        return s


# ------------------------------------------------------------------ the backbone-specific part (Wan-Animate-2)
def model_inputs(s, x_s, t):
    """forward_ref / forward_gen arguments for one clip, laid out exactly as inference_core does for its first clip
    (move-mode condition: latent frame 0 = the reference, known; the clip frames unknown, grey)."""
    C, T, h, w = s["target"].shape
    ones = lambda n: torch.ones(4, n, h, w, dtype=x_s.dtype, device=x_s.device)
    y = torch.cat([torch.cat([ones(1), s["ref"]]), torch.cat([ones(T) * 0, s["zero"]])], dim=1)
    cond_y = torch.cat([ones(T), s["drive"]])                                  # driving frames all known
    tok = (h // 2) * (w // 2)
    return dict(noise=[x_s], y=[y], cond_lat=s["drive"][None], cond_y=[cond_y], ctx=[s["text"]],
                ctx_ref=[s["text_ref"]], clip_fea=s["clip_ref"][None], clip_fea_ref=s["clip_drive"][None],
                seq_len=(T + 1) * tok, seq_len_ref=T * tok, grid_sizes=torch.tensor([[T + 1, h // 2, w // 2]]),
                grid_sizes_ref=torch.tensor([[T, h // 2, w // 2]]), t=t, clip_len=(T - 1) * 4 + 1,
                area=[w * 8, h * 8])


def predict(model, inp):
    """Driving pass (fills the shared-QKV K/V cache, with gradients) then the generation pass -> v [C, T+1, h, w]."""
    k_cache, v_cache = {}, {}
    model(inp["cond_lat"], grid_sizes=inp["grid_sizes"], k_cache=k_cache, v_cache=v_cache, t=inp["t"],
          method="forward_ref", context_ref=inp["ctx_ref"], seq_len_ref=inp["seq_len_ref"],
          clip_fea_ref=inp["clip_fea_ref"], y_ref=inp["cond_y"])
    return model(inp["noise"], k_cache=k_cache, v_cache=v_cache, t=inp["t"], grid_sizes_ref=inp["grid_sizes_ref"],
                 method="forward_gen", context=inp["ctx"], seq_len=inp["seq_len"], clip_fea=inp["clip_fea"],
                 y=inp["y"], origin_len=inp["clip_len"], origin_area=inp["area"])[0]


# ------------------------------------------------------------------ loss and loop (backbone-independent)
def sample_sigma(shift=5.0, generator=None):
    """Logit-normal noise level, then the same time shift as sampling uses."""
    s = torch.sigmoid(torch.randn(1, generator=generator))
    return shift * s / (1 + (shift - 1) * s)


def flow_loss(model, s, sigma, noise):
    """MSE between the predicted and the true velocity on the clip frames (latent frame 0 = the given reference)."""
    x0 = torch.cat([s["ref"], s["target"]], dim=1)
    x_s = (1 - sigma) * x0 + sigma * noise
    v = predict(model, model_inputs(s, x_s, (sigma * 1000).reshape(1)))
    return F.mse_loss(v[:, 1:].float(), (noise - x0)[:, 1:].float())


def train(model, dataset, steps, lr=1e-4, mode="lora", rank=64, shift=5.0, log_every=10, save_every=500,
          out=None, seed=0, device="cpu", log=print):
    params = set_trainable(model, mode, rank)
    model.to(device).train()
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(seed)
    log(f"trainable {sum(p.numel() for p in params) / 1e6:.2f} M params ({mode}), {len(dataset)} clips, {steps} steps")
    history = []
    for step in range(1, steps + 1):
        s = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in dataset[(step - 1) % len(dataset)].items()}
        x0_shape = (s["target"].shape[0], s["target"].shape[1] + 1, *s["target"].shape[2:])
        noise = torch.randn(x0_shape, generator=g).to(device)
        loss = flow_loss(model, s, sample_sigma(shift, g).to(device), noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        history.append(float(loss))
        if step % log_every == 0:
            log(f"step {step}: loss {sum(history[-log_every:]) / log_every:.4f}")
        if out and (step % save_every == 0 or step == steps):
            from safetensors.torch import save_file
            os.makedirs(out, exist_ok=True)
            save_file(trainable_state_dict(model), os.path.join(out, f"trainable_{step:06d}.safetensors"))
    return history


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="DiT weights (.safetensors)")
    p.add_argument("--config", required=True, help="json with the WanAnimate2Transformer constructor arguments")
    p.add_argument("--data", required=True, help="folder of cached *.pt clips")
    p.add_argument("--mode", choices=["lora", "qkv"], default="lora")
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--out", default="runs/qkv")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    from safetensors.torch import load_file
    from wanxiang.models.wan_animate_2_model import WanAnimate2Transformer
    model = WanAnimate2Transformer(**json.load(open(a.config)))
    model.load_state_dict(load_file(a.ckpt), strict=True)
    train(model, CachedClipDataset(a.data), a.steps, a.lr, a.mode, a.rank, a.shift, out=a.out, device=a.device)


if __name__ == "__main__":
    main()
