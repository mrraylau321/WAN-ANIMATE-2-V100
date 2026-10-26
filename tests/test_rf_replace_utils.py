"""scripts/rf_replace_utils.py: t=0-noise rectified-flow replace helpers (keep mask, SDEdit start, pinning)."""
import os
import sys

import numpy as np
import torch

from conftest import REPO

sys.path.insert(0, os.path.join(REPO, "scripts"))
import rf_replace_utils as ru  # noqa: E402


def test_keep_mask_groups_like_the_wan22_vae_and_never_pins_frame_0():
    masks = np.zeros((9, 32, 48), np.uint8)
    masks[3, 16:20, 16:20] = 255                      # pixel frame 3 -> latent frame 1, cell (1, 1)
    keep = ru.latent_keep(masks, grow=0)
    assert keep.shape == (3, 2, 3)
    assert keep[0].max() == 0 and keep[1, 1, 1] == 0 and keep[1].sum() == 5 and keep[2].min() == 1


def test_sdedit_start_moves_the_grid_to_t_start():
    z, n = torch.ones(2), torch.zeros(2)
    x, grid = ru.sdedit_start(z, n, torch.linspace(0, 1, 11), 0.25)
    assert torch.allclose(x, torch.full((2,), 0.25)) and float(grid[0]) == 0.25 and float(grid[-1]) == 1.0
    assert all(float(g) > 0.25 for g in grid[1:])


def test_euler_with_pinning_ends_on_the_driving_latent():
    e_bg, e_fg = ru.self_test()
    assert e_bg < 1e-4 and e_fg < 1e-4
