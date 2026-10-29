"""Full shared-QKV training for WA2 TI2V-5B, with explicit gradient reduction and ZeRO Adam state."""
import argparse
import glob
import json
import math
import os
import shutil
import threading
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

from prep_wa2_ti2v5b import CachedLatents, VERSION, atomic_save, file_identity
from train_incontext_ti2v5b import read_training_state, sample_sigma, setup_dist, token_timesteps
from wa2_ti2v5b import QKV_NAME, assert_5b, parameter_report, set_shared_qkv_trainable, to_fp16_storage, wa2_forward


def flow_loss(model, x0, drive, text, sigma, noise, use_checkpoint=False):
    x = (1 - sigma) * x0 + sigma * noise
    x[:, :, 0] = x0[:, :, 0]
    b, _, frames, h, w = x.shape
    times = torch.full((b, frames), float(sigma) * 1000, device=x.device)
    times[:, 0] = 0
    v = wa2_forward(model, x, token_timesteps(times, h, w), text, drive, use_checkpoint)
    return F.mse_loss(v[:, :, 1:].float(), (noise - x0)[:, :, 1:].float())


def qkv_state_dict(model):
    return {n: p.detach().float().cpu() for n, p in model.named_parameters() if QKV_NAME.fullmatch(n)}


def load_qkv_state(model, state):
    expected = {n for n, _ in model.named_parameters() if QKV_NAME.fullmatch(n)}
    if state.keys() != expected:
        raise ValueError("QKV checkpoint keys differ")
    model.load_state_dict(state, strict=False)


def export_qkv(qkv, out, step, config):
    from safetensors.torch import save_file
    name = f"qkv_{step:06d}.safetensors"
    tmp = os.path.join(out, f".tmp_{name}")
    save_file(qkv, tmp, metadata={"config": json.dumps(config, sort_keys=True)})
    os.replace(tmp, os.path.join(out, name))


def save_training_state(out, state):
    os.makedirs(out, exist_ok=True)
    atomic_save(state, os.path.join(out, "state.pt"))
    export_qkv(state["qkv"], out, state["step"], state["config"])


def copy_file(src, dst):
    """One file between a folder and a folder or obs:// (ModelArts MoXing, preinstalled in ModelArts images)."""
    if src.startswith("obs://") or dst.startswith("obs://"):
        import moxing as mox
        mox.file.copy(src, dst)
        return
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    shutil.copyfile(src, dst + ".tmp")
    os.replace(dst + ".tmp", dst)


def exists(path):
    if path.startswith("obs://"):
        import moxing as mox
        return mox.file.exists(path)
    return os.path.exists(path)


def upload(out, sync_to, names, log=print):
    """Copy what a save just wrote to sync_to in the background: the ranks keep training meanwhile. A failed upload
    is reported, not fatal (the local copy is intact)."""
    def run():
        for name in names:
            try:
                copy_file(os.path.join(out, name), f"{sync_to.rstrip('/')}/{name}")
            except Exception as exc:  # noqa: BLE001 (network/OBS errors must not stop training)
                log(f"sync of {name} to {sync_to} FAILED: {exc!r} (training continues; local copy in {out})")
                return
        log(f"synced {', '.join(names)} -> {sync_to}")
    thread = threading.Thread(target=run)
    thread.start()
    return thread


def restore_state(out, sync_to):
    """A fresh machine: fetch state.pt from sync_to, so train() resumes where the synced run stopped."""
    local, remote = os.path.join(out, "state.pt"), f"{sync_to.rstrip('/')}/state.pt"
    if os.path.exists(local) or not exists(remote):
        return False
    copy_file(remote, local)
    return True


def train(model, dataset, steps, out=None, lr=1e-4, shift=5.0, warmup=50, accum=1, log_every=10,
          save_every=250, seed=0, device="cpu", amp=False, use_checkpoint=True, rank=0, world=1,
          log=print, total_steps=None, model_dir=None, sync_to=None):
    """Resume the same sample/RNG cursor; total_steps keeps a split run's original schedule. sync_to: after every
    save, state.pt and that step's QKV export are also copied there (obs:// or a folder)."""
    total_steps = steps if total_steps is None else total_steps
    if steps > total_steps:
        raise ValueError("steps cannot exceed total_steps")
    config = dict(version=VERSION, steps=total_steps, lr=lr, data=dataset.folder,
                  manifest=dataset.manifest, fingerprint=dataset.fingerprint, warmup=warmup, accum=accum,
                  seed=seed, world=world, shift=shift, amp=amp,
                  model=file_identity(model_dir) if model_dir else None,
                  # "_" keys (e.g. _use_default_values) are ordered by the per-process hash seed: ranks differ
                  model_config={k: v for k, v in model.config.items() if not k.startswith("_")},
                  weights=str(next(model.parameters()).dtype))
    if out and not os.path.exists(os.path.join(out, "state.pt")) and glob.glob(os.path.join(out, "qkv_*.safetensors")):
        raise ValueError("QKV exports exist without state.pt; use a new output directory")
    state = read_training_state(out, config, steps, "cpu")
    torch.manual_seed(seed)
    params = set_shared_qkv_trainable(model)
    model.to(device).train()
    for p in params:
        p.data = p.data.float()
    if state:
        load_qkv_state(model, state["qkv"])
    if world > 1:
        for p in params:
            dist.broadcast(p.data, src=0)
        from torch.distributed.optim import ZeroRedundancyOptimizer
        opt = ZeroRedundancyOptimizer(params, optimizer_class=torch.optim.AdamW, lr=lr, weight_decay=0, foreach=False)
    else:
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0, foreach=False)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, warmup)))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    g = torch.Generator().manual_seed(seed + 1000 * rank)
    start = cursor = skipped = skip_streak = 0
    if state:
        opt.load_state_dict(state["optimizer"])
        if world > 1:  # ZeRO's load_state_dict also copies this rank's moments into the wrapper: ~1.6 GiB unused
            opt.state.clear()
        sched.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start, cursor = state["step"], state["cursor"]
        skipped, skip_streak = state["skipped"], state["skip_streak"]
        rng = state["rng"][rank]
        g.set_state(rng["sample"].cpu())
        torch.set_rng_state(rng["torch"].cpu())
        if rng["cuda"] is not None:
            torch.cuda.set_rng_state(rng["cuda"].cpu(), device)
        del state
        if rank == 0 and not os.path.exists(os.path.join(out, f"qkv_{start:06d}.safetensors")):
            export_qkv(qkv_state_dict(model), out, start, config)
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))
    text = dataset.text["text"].to(device).float()
    if rank == 0:
        log(f"parameters: {parameter_report(model)}")
        log(f"{len(dataset)} samples, {world} ranks x {accum} accum, amp {amp}, checkpoint {use_checkpoint}")
        if start:
            log(f"resuming step {start}, sample cursor {cursor}")
    history, t0, uploading = [], time.time(), None
    for step in range(start + 1, steps + 1):
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(accum):
            s = dataset[int(order[(cursor * world + rank) % len(dataset)])]
            cursor += 1
            x0 = s["target"][None].to(device).float()
            drive = s["drive"][None].to(device).float()
            sigma = float(sample_sigma(shift, g))
            noise = torch.randn(x0.shape, generator=g).to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = flow_loss(model, x0, drive, text, sigma, noise, use_checkpoint) / accum
            scaler.scale(loss).backward()
            step_loss += float(loss.detach())
        if world > 1:
            for p in params:
                dist.all_reduce(p.grad)
                p.grad /= world
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not amp and not torch.isfinite(grad_norm):
            raise RuntimeError(f"non-finite gradient at step {step}")
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
            raise RuntimeError(f"50 consecutive AMP updates skipped at step {step}")
        if not math.isfinite(step_loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        history.append(step_loss)
        if rank == 0 and (step <= start + 3 or step % log_every == 0):
            cuda = torch.device(device).type == "cuda"
            allocated = torch.cuda.max_memory_allocated(device) / 2**30 if cuda else 0.0
            reserved = torch.cuda.max_memory_reserved(device) / 2**30 if cuda else 0.0
            recent = history[-log_every:]
            log(f"step {step}: loss {sum(recent) / len(recent):.4f}, "
                f"{(time.time() - t0) / (step - start):.1f} s/step, "
                f"max_memory_allocated {allocated:.2f} GiB, max_memory_reserved {reserved:.2f} GiB, "
                f"scale {scaler.get_scale():g}, skipped {skipped}, updates {step - skipped}, grad norm {grad_norm:.4f}")
        if out and (step % save_every == 0 or step == steps):
            rng = dict(sample=g.get_state(), torch=torch.get_rng_state(),
                       cuda=torch.cuda.get_rng_state(device) if torch.device(device).type == "cuda" else None)
            rngs = [None] * world
            if world > 1:
                dist.all_gather_object(rngs, rng)
                opt.consolidate_state_dict(to=0)
            else:
                rngs[0] = rng
            if rank == 0:
                if uploading:
                    uploading.join()  # never replace state.pt while the previous one is still uploading
                state = dict(config=config, qkv=qkv_state_dict(model), optimizer=opt.state_dict(),
                             scheduler=sched.state_dict(), scaler=scaler.state_dict(), step=step, cursor=cursor,
                             rng=rngs, skipped=skipped, skip_streak=skip_streak)
                save_training_state(out, state)
                del state
                if sync_to:
                    uploading = upload(out, sync_to, ["state.pt", f"qkv_{step:06d}.safetensors"], log)
            if world > 1:
                dist.barrier(device_ids=[torch.cuda.current_device()] if torch.cuda.is_available() else None)
    if uploading:
        uploading.join()
    return history


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--save-every", type=int, default=500, help="state.pt ~10 GB + a 3.4 GB QKV export each time")
    p.add_argument("--weights", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sync-to", help="obs://bucket/path (MoXing) or a folder: each save is copied there, and a fresh "
                                     "machine without --out/state.pt resumes from it")
    a = p.parse_args()
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    rank, world = setup_dist()
    if a.sync_to and int(os.environ.get("LOCAL_RANK", rank)) == 0 and restore_state(a.out, a.sync_to):
        print(f"[rank {rank}] restored {a.out}/state.pt from {a.sync_to}", flush=True)
    if world > 1:  # every rank reads state.pt only after it is there
        dist.barrier(device_ids=[torch.cuda.current_device()] if torch.cuda.is_available() else None)
    device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    from diffusers import WanTransformer3DModel
    # fp32 load, as in the pilot (fp16 from_pretrained needs accelerate), then fp16 storage
    model = WanTransformer3DModel.from_pretrained(os.path.join(a.model, "transformer"), torch_dtype=torch.float32,
                                                local_files_only=True)
    if a.weights == "fp16":
        to_fp16_storage(model)
    set_shared_qkv_trainable(model)
    assert_5b(model)
    log = lambda msg: print(f"[rank {rank}] {msg}", flush=True)
    train(model, CachedLatents(a.data), a.steps, a.out, lr=a.lr, shift=a.shift, accum=a.accum,
          save_every=a.save_every, seed=a.seed, device=device, amp=device != "cpu", use_checkpoint=True,
          rank=rank, world=world, log=log, model_dir=a.model, sync_to=a.sync_to)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
