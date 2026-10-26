"""Training-free replace for rectified-flow video samplers that use the "t=0 noise, t=1 data" convention
(x_t = t * x1 + (1 - t) * x0, the model predicts v = x1 - x0), with a 16x16-pixel, 4-frame latent grid (Wan2.2 VAE).
Framework-free: plain torch, so any sampler can call it. Three insertion points in the sampling loop:

    keep = latent_keep(person_masks, grow=1).to(x)               # before the loop, (T, h, w), 1 = pin to the driving
    x, grid = sdedit_start(z_drive, x, grid, t_start)            # start from the noised driving latents
    for i, t in enumerate(grid[:-1]):
        v = <model + CFG as before>
        if i < len(grid) - 1 - free_steps:
            v = pin_velocity(v, x, z_drive, t, keep)             # after CFG: background predicts the driving latent
        x = x + (grid[i + 1] - t) * v                            # the existing Euler step, unchanged

z_drive: the driving clip encoded by the same VAE and normalised exactly like the sampler's latents, same shape as x.
Latent frame 0 is never pinned (keep[0] = 0): it belongs to the first-frame image condition (g0).
self_test() checks the whole thing numerically on random tensors; run it once after wiring.
"""
import numpy as np
import torch
import torch.nn.functional as F

VAE_STRIDE, T_STRIDE = 16, 4


def latent_keep(masks, grow=1, dilate=0):
    """Person masks (n, H, W; bool/0-1/0-255; H, W multiples of 16) of the frames being generated -> keep (T, h, w),
    1 where the driving latent is kept. Latent frame 0 <- pixel frame 0 (never pinned), latent k <- pixel frames
    4k-3 .. 4k; a latent cell is generated if any of its pixels (after `dilate` px) is the person, plus `grow` cells."""
    m = torch.as_tensor(np.asarray(masks)).float()
    m = (m > (127 if m.max() > 1 else 0.5)).float()[:, None]                      # n, 1, H, W
    if dilate > 0:
        m = F.max_pool2d(m, 2 * dilate + 1, stride=1, padding=dilate)
    gen = F.max_pool2d(m, VAE_STRIDE)
    if grow > 0:
        gen = F.max_pool2d(gen, 2 * grow + 1, stride=1, padding=grow)
    gen = gen[:, 0]
    n_lat = (len(gen) - 1) // T_STRIDE + 1
    groups = [gen[:1]] + [gen[1 + T_STRIDE * (k - 1):1 + T_STRIDE * k] for k in range(1, n_lat)]
    keep = 1 - torch.stack([g.amax(0) for g in groups])
    keep[0] = 0
    return keep


def sdedit_start(z_drive, noise, grid, t_start):
    """x at t_start (t_start * z_drive + (1 - t_start) * noise) and the part of the time grid from t_start on.
    grid: increasing times from ~0 (noise) to 1 (data); the first kept point is moved exactly to t_start."""
    grid = torch.as_tensor(grid, dtype=torch.float64)
    rest = grid[grid > t_start]
    new_grid = torch.cat([torch.tensor([float(t_start)], dtype=torch.float64), rest])
    return t_start * z_drive + (1 - t_start) * noise, new_grid


def pin_velocity(v, x, z_drive, t, keep):
    """Where keep = 1, replace v so that the predicted data x + (1 - t) * v equals z_drive.
    v, x, z_drive: (..., T, h, w) latents (any leading dims, e.g. B, C); keep: (T, h, w)."""
    t = float(t)
    pinned = (z_drive - x) / max(1.0 - t, 1e-6)
    k = keep.to(v.dtype)
    return v * (1 - k) + pinned.to(v.dtype) * k


def self_test(steps=30, t_start=0.12, seed=0):
    """Euler on random tensors with a fake model that wants x1 = a: the pinned region must end exactly on z_drive,
    the rest on a. Returns the two max errors (both should be < 1e-4)."""
    g = torch.Generator().manual_seed(seed)
    z, a, noise = (torch.randn(1, 4, 5, 4, 6, generator=g, dtype=torch.float64) for _ in range(3))
    masks = np.zeros((17, 64, 96), np.uint8)
    masks[:, 16:40, 32:64] = 255
    keep = latent_keep(masks, grow=0).double()
    x, grid = sdedit_start(z, noise, torch.linspace(0, 1, steps + 1), t_start)
    for i in range(len(grid) - 1):
        t = grid[i]
        v = (a - x) / (1 - t)                       # fake model: predicts x1 = a
        v = pin_velocity(v, x, z, t, keep)
        x = x + (grid[i + 1] - t) * v
    kb = keep.bool().expand_as(x)
    return float((x - z)[kb].abs().max()), float((x - a)[~kb].abs().max())


if __name__ == "__main__":
    e_bg, e_fg = self_test()
    print(f"pinned region vs driving: {e_bg:.2e}   free region vs model: {e_fg:.2e}   "
          f"{'PASS' if max(e_bg, e_fg) < 1e-4 else 'FAIL'}")
