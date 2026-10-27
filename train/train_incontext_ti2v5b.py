"""Pilot: teach Wan2.2-TI2V-5B (diffusers, public weights) to take its motion from a driving video through in-context
tokens, training only a LoRA on the self-attention Q/K/V (everything else frozen). TI2V-5B cannot follow a driving
video out of the box, so a working pilot shows the method itself works (the reason for using it, not Wan-Animate-2).

In-context layout (no new modules): the driving clip's latents are patch-embedded by the same layer and appended to the
video tokens; both go through every block together (joint self-attention, shared Q/K/V), with the SAME RoPE positions
(driving frame j sits on video frame j) and their own per-token timestep. Only the video tokens are decoded.
  video tokens   frame 0 = the reference image, clean (timestep 0, TI2V's own first-frame scheme); frames 1.. noised
  driving tokens the driving latents noised to a random level s (timestep 1000 s), making appearance harder to copy
                 from the same clip used as the target; cross-identity evaluation must establish motion transfer
Flow matching, Wan convention: x_sigma = (1 - sigma) x0 + sigma noise, the model predicts v = noise - x0.

Training samples are written by train/prep_incontext_ti2v5b.py: <data>/*.pt with "target" [48, F, h, w] (latent
frame 0 = the reference image) and "drive" [48, F, h, w] (appearance-degraded driving clip), plus <data>/text.pt.
  torchrun --nproc_per_node 4 train/train_incontext_ti2v5b.py --model <TI2V-5B diffusers dir> --data <cache> --out <run>
"""
import argparse
import glob
import math
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_incontext_qkv import LoRALinear, sample_sigma  # noqa: E402


# ------------------------------------------------------------------ model: in-context forward and LoRA
def token_timesteps(t_frames, h, w, patch=(1, 2, 2)):
    """Per-latent-frame timesteps [B, F] -> per-token timesteps [B, F * h/2 * w/2] (TI2V-5B's expanded timesteps)."""
    return t_frames[:, :, None].expand(-1, -1, (h // patch[1]) * (w // patch[2])).flatten(1)


def incontext_forward(model, x, t_x, text, drive=None, t_drive=None, use_checkpoint=False):
    """WanTransformer3DModel forward with the driving tokens appended (drive=None: exactly the model's own forward).
    x, drive [B, C, F, H, W] (same shape); t_x, t_drive per-token timesteps [B, S]; text [B, L, 4096]. Returns v of x.
    The residual stream is kept in float32, so fp16 autocast only touches the matmuls."""
    B, _, F_, H, W = x.shape
    p_t, p_h, p_w = model.config.patch_size
    rope = model.rope(x)
    h = model.patch_embedding(x).flatten(2).transpose(1, 2).float()
    n = h.shape[1]
    ts = t_x
    if drive is not None:
        hd = model.patch_embedding(drive).flatten(2).transpose(1, 2).float()
        h = torch.cat([h, hd], 1)
        rope = tuple(torch.cat([r, r], 1) for r in rope)       # driving frame j on the positions of video frame j
        ts = torch.cat([t_x, t_drive], 1)
    temb, tproj, enc, _ = model.condition_embedder(ts.flatten(), text, None, timestep_seq_len=ts.shape[1])
    tproj = tproj.unflatten(2, (6, -1))
    for block in model.blocks:
        if use_checkpoint and torch.is_grad_enabled():
            h = checkpoint(block, h, enc, tproj, rope, use_reentrant=False)
        else:
            h = block(h, enc, tproj, rope)
    h, temb = h[:, :n], temb[:, :n]
    shift, scale = (model.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
    h = (model.norm_out(h.float()) * (1 + scale.squeeze(2)) + shift.squeeze(2)).type_as(h)
    h = model.proj_out(h)
    h = h.reshape(B, F_ // p_t, H // p_h, W // p_w, p_t, p_h, p_w, -1).permute(0, 7, 1, 4, 2, 5, 3, 6)
    return h.flatten(6, 7).flatten(4, 5).flatten(2, 3)


def qkv_layers(model):
    """(module owning the layer, attribute name) of every self-attention Q, K and V."""
    return [(blk.attn1, n) for blk in model.blocks for n in ("to_q", "to_k", "to_v")]


def add_lora(model, rank=64, alpha=None):
    """Freeze everything, put a LoRA (B = 0: output unchanged) on every self-attention Q/K/V. Returns its parameters."""
    model.requires_grad_(False)
    for owner, name in qkv_layers(model):
        layer = getattr(owner, name)
        if not isinstance(layer, LoRALinear):
            setattr(owner, name, LoRALinear(layer, rank, alpha))
    return [p for p in model.parameters() if p.requires_grad]


def lora_state_dict(model):
    return {k: v.detach().float().cpu() for k, v in model.named_parameters() if k.endswith((".A", ".B"))}


def load_lora(model, path, rank=None, alpha=None):
    """add_lora + load a saved LoRA (rank read from the file unless given); every LoRA key must match."""
    from safetensors.torch import load_file
    sd = load_file(path)
    rank = rank or next(v.shape[0] for k, v in sd.items() if k.endswith(".A"))
    add_lora(model, rank, alpha)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    lora_missing = [k for k in missing if k.endswith((".A", ".B"))]
    if lora_missing or unexpected:
        raise KeyError(f"LoRA mismatch: missing {lora_missing[:4]}, unexpected {unexpected[:4]}")
    return model


# ------------------------------------------------------------------ data
class CachedLatents(torch.utils.data.Dataset):
    """<folder>/*.pt from prep_incontext_ti2v5b.py (each: target, drive) and <folder>/text.pt (text, neg)."""

    def __init__(self, folder):
        self.folder = os.path.realpath(folder)
        self.files = sorted(f for f in glob.glob(os.path.join(folder, "*.pt"))
                            if os.path.basename(f) != "text.pt" and not os.path.basename(f).startswith(".tmp_"))
        if not self.files:
            raise FileNotFoundError(f"no samples in {folder}")
        self.text = torch.load(os.path.join(folder, "text.pt"), map_location="cpu", weights_only=True)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        s = torch.load(self.files[i], map_location="cpu", weights_only=True)
        if s["target"].shape != s["drive"].shape:
            raise ValueError(f"{self.files[i]}: target {tuple(s['target'].shape)} != drive {tuple(s['drive'].shape)}")
        return s


# ------------------------------------------------------------------ loss
def flow_loss(model, x0, drive, text, sigma, drive_level, noise, noise_drive, use_checkpoint=False):
    """x0, drive [B, C, F, h, w]; latent frame 0 of x0 is the reference image (kept clean, timestep 0).
    MSE between predicted and true velocity on frames 1.. ."""
    B, _, F_, h, w = x0.shape
    x_s = (1 - sigma) * x0 + sigma * noise
    x_s[:, :, 0] = x0[:, :, 0]
    d_s = (1 - drive_level) * drive + drive_level * noise_drive
    t_frames = torch.full((B, F_), float(sigma) * 1000, device=x0.device)
    t_frames[:, 0] = 0
    t_drive = torch.full((B, F_), float(drive_level) * 1000, device=x0.device)
    v = incontext_forward(model, x_s, token_timesteps(t_frames, h, w), text, d_s, token_timesteps(t_drive, h, w),
                          use_checkpoint)
    return F.mse_loss(v[:, :, 1:].float(), (noise - x0)[:, :, 1:].float())


# ------------------------------------------------------------------ loop
def setup_dist():
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        rank, world = dist.get_rank(), dist.get_world_size()
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
        return rank, world
    return 0, 1


def read_training_state(out, config, steps, device):
    """Load a matching run or refuse to overwrite its checkpoints."""
    if not out:
        return None
    path = os.path.join(out, "state.pt")
    if os.path.exists(path):
        state = torch.load(path, map_location=device, weights_only=True)
        changed = [key for key in config if state["config"].get(key) != config[key]]
        if changed:
            raise ValueError(f"cannot resume {path}: run settings differ: {', '.join(changed)}")
        if state["step"] > steps:
            raise ValueError(f"cannot resume step {state['step']} with stop step {steps}")
        return state
    if glob.glob(os.path.join(out, "lora_*.safetensors")):
        raise ValueError(f"cannot resume {out}: LoRA checkpoints exist without state.pt; use a new output directory")
    return None


def export_lora(lora, out, step):
    """Publish an inference adapter atomically."""
    from safetensors.torch import save_file
    name = f"lora_{step:06d}.safetensors"
    tmp = os.path.join(out, f".tmp_{name}")
    save_file(lora, tmp)
    os.replace(tmp, os.path.join(out, name))


def save_training_state(out, state):
    """Publish resume state before its matching inference export."""
    os.makedirs(out, exist_ok=True)
    tmp = os.path.join(out, ".tmp_state.pt")
    torch.save(state, tmp)
    os.replace(tmp, os.path.join(out, "state.pt"))
    export_lora(state["lora"], out, state["step"])


def train(model, dataset, steps, out=None, lr=1e-4, rank_lora=64, shift=5.0, drive_noise=(0.2, 0.7), warmup=50,
          accum=1, log_every=10, save_every=500, seed=0, device="cpu", amp=False, use_checkpoint=False,
          rank=0, world=1, log=print, total_steps=None, model_dir=None):
    """All-reduced LoRA training; total_steps preserves the run plan when a call stops early at steps."""
    total_steps = steps if total_steps is None else total_steps
    if steps > total_steps:
        raise ValueError("steps cannot exceed the planned total_steps")
    config = dict(steps=total_steps, rank=rank_lora, lr=lr, drive_noise=list(drive_noise), data=dataset.folder,
                  warmup=warmup, accum=accum, seed=seed, world=world, shift=shift, amp=amp,
                  model=os.path.realpath(model_dir) if model_dir else None, weights=str(next(model.parameters()).dtype))
    state_path = os.path.join(out, "state.pt") if out else None
    state = read_training_state(out, config, steps, device)
    torch.manual_seed(seed)
    params = add_lora(model, rank_lora)
    model.to(device).train()
    for p in params:                                             # LoRA weights stay float32 (AMP master weights)
        p.data = p.data.float()
    if state:
        keys = {k for k, _ in model.named_parameters() if k.endswith((".A", ".B"))}
        if state["lora"].keys() != keys:
            raise ValueError(f"cannot resume {state_path}: LoRA keys differ")
        model.load_state_dict(state["lora"], strict=False)
    if world > 1:
        for p in params:
            dist.broadcast(p.data, src=0)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, warmup)))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    g = torch.Generator().manual_seed(seed + 1000 * rank)
    start, k, skipped, skip_streak = 0, 0, 0, 0
    if state:
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start, k = state["step"], state["cursor"]
        skipped, skip_streak = state["skipped"], state["skip_streak"]
        rng = state["rng"][rank]
        g.set_state(rng["sample"].cpu())
        torch.set_rng_state(rng["torch"].cpu())
        if rng["cuda"] is not None:
            torch.cuda.set_rng_state(rng["cuda"].cpu(), device)
        del state
        if rank == 0 and not os.path.exists(os.path.join(out, f"lora_{start:06d}.safetensors")):
            export_lora(lora_state_dict(model), out, start)       # recover interruption between state and export
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))
    text = dataset.text["text"].to(device).float()
    if rank == 0:
        log(f"LoRA r{rank_lora}: {sum(p.numel() for p in params) / 1e6:.2f} M trainable, {len(dataset)} samples, "
            f"{world} rank(s) x {accum} accum, {steps} steps, amp {amp}, checkpoint {use_checkpoint}")
        if start:
            log(f"resuming {state_path} at step {start}, sample cursor {k}")
    history, t0 = [], time.time()
    for step in range(start + 1, steps + 1):
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(accum):
            s = dataset[int(order[(k * world + rank) % len(dataset)])]
            k += 1
            x0 = s["target"][None].to(device).float()
            drive = s["drive"][None].to(device).float()
            sigma = float(sample_sigma(shift, g))
            level = drive_noise[0] + (drive_noise[1] - drive_noise[0]) * float(torch.rand(1, generator=g))
            noise = torch.randn(x0.shape, generator=g).to(device)
            noise_d = torch.randn(x0.shape, generator=g).to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = flow_loss(model, x0, drive, text, sigma, level, noise, noise_d, use_checkpoint) / accum
            scaler.scale(loss).backward()
            step_loss += float(loss)
        if world > 1:
            for p in params:
                dist.all_reduce(p.grad)
                p.grad /= world
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        scale = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        if scaler.get_scale() < scale:
            skipped += 1
            skip_streak += 1
        else:
            skip_streak = 0
            sched.step()
        if skip_streak >= 50:
            raise RuntimeError(f"50 consecutive AMP optimizer updates skipped at step {step}; check fp16 gradients")
        if not math.isfinite(step_loss):
            raise RuntimeError(f"non-finite loss at step {step}: inspect overflowing projections; "
                               "fp32 weight storage still uses fp16 autocast")
        history.append(step_loss)
        if rank == 0 and (step <= start + 3 or step % log_every == 0):
            mem = torch.cuda.max_memory_allocated(device) / 2**30 if torch.device(device).type == "cuda" else 0.0
            recent = history[-log_every:]
            log(f"step {step}: loss {sum(recent) / len(recent):.4f}, "
                f"{(time.time() - t0) / (step - start):.1f} s/step, {mem:.1f} GiB peak, "
                f"scale {scaler.get_scale():g}, skipped {skipped}, updates {step - skipped}, grad norm {grad_norm:.4f}")
        if out and (step % save_every == 0 or step == steps):
            rng = dict(sample=g.get_state(), torch=torch.get_rng_state(),
                       cuda=torch.cuda.get_rng_state(device) if torch.device(device).type == "cuda" else None)
            rngs = [None] * world
            if world > 1:
                dist.all_gather_object(rngs, rng)
            else:
                rngs[0] = rng
            if rank == 0:
                state = dict(config=config, lora=lora_state_dict(model), optimizer=opt.state_dict(), scheduler=sched.state_dict(),
                             scaler=scaler.state_dict(), step=step, cursor=k, rng=rngs,
                             skipped=skipped, skip_streak=skip_streak)
                save_training_state(out, state)
                del state
            if world > 1:
                dist.barrier()
    return history


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Wan2.2-TI2V-5B-Diffusers folder (uses its transformer/)")
    p.add_argument("--data", required=True, help="folder written by prep_incontext_ti2v5b.py")
    p.add_argument("--out", required=True, help="LoRA checkpoints go here")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=64, help="LoRA rank")
    p.add_argument("--accum", type=int, default=1, help="gradient accumulation (samples per rank per step)")
    p.add_argument("--drive-noise", type=float, nargs=2, default=(0.2, 0.7), help="driving latent noise level range")
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--weights", choices=["fp32", "fp16"], default="fp32",
                   help="frozen weight storage; compute is fp16 autocast either way (fp32 = 20 GB, safer on V100)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rank, world = setup_dist()
    device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    from diffusers import WanTransformer3DModel
    dtype = torch.float32 if a.weights == "fp32" else torch.float16
    model = WanTransformer3DModel.from_pretrained(os.path.join(a.model, "transformer"), torch_dtype=dtype)
    log = lambda m: print(f"[rank {rank}] {m}", flush=True)
    train(model, CachedLatents(a.data), a.steps, a.out, a.lr, a.rank, a.shift, tuple(a.drive_noise), accum=a.accum,
          save_every=a.save_every, seed=a.seed, device=device, amp=device != "cpu", use_checkpoint=device != "cpu",
          rank=rank, world=world, log=log, model_dir=a.model)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
