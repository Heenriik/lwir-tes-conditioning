"""
Shared helpers for hybrid-variant experiments (run_hybrid*.py).

The canonical baseline lives in run_hybrid.py (inverse-space residual +
spectral_norm Lipschitz bound). Variant scripts (run_hybrid_gated.py,
run_hybrid_forward.py, ...) reuse the same dataset generator, constants,
and evaluation primitives to keep results comparable.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # To avoid OpenMP-library-collision warning.
import sys
import numpy as np
import random
import torch

# Allow scripts/ files to import src.* regardless of how the entry script is run.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.forward_model import blackbody_environment, compute_radiance_batch
from src.simulation.generator import (
    generate_wavelength_sets, generate_emissivity_curve, MATERIAL_LIBRARY,
)
from src.simulation.noise import add_noise_batch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(ROOT, "results", "figures")
DATA_DIR = os.path.join(ROOT, "results", "data")

# --- Experiment constants (shared across all variants) ---
N_BANDS = 10
WL_RANGE = (8e-6, 14e-6)
MATERIALS = list(MATERIAL_LIBRARY.keys())
T_TRAIN_RANGE = (250.0, 380.0)
T_OOD_OFFSETS = np.array([0, 10, 20, 30, 40, 50])  # K above training range
N_TRAIN = 5000
N_TEST = 500
N_OOD_PER_OFFSET = 100
NEDT_K = 0.1
EMISSIVITY_VARIABILITY = 0.02
L_ENV_T = 280.0

# Shared training-init seed. Must be called BEFORE any nn.Module is constructed
# (PyTorch uses the global RNG for default weight init). Setting it here keeps
# all four hybrid variants (baseline / gated / forward / phys) trained from the
# same initial weights, so variant differences reflect algorithm choices rather
# than RNG luck — especially important for OOD numbers, which are very
# sensitive to initialisation.
SEED = 42


def set_seed(seed=SEED):
    """Pin all PyTorch + numpy + Python RNGs to one seed. To keep OOD numbers reproducible"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_dataset(n_samples, T_range, seed=0):
    """
    Generate noisy radiance + ground-truth (T, ε) for a hybrid experiment.

    Materials cycle deterministically through MATERIALS via i % len(MATERIALS),
    so reproducible across variants for fixed (n_samples, seed).

    Returns:
        wl (N,), L_env (N,), T_batch (M,), eps_batch (M, N),
        L_noisy (M, N), material_idx (M,)
    """
    rng = np.random.default_rng(seed)
    wl = generate_wavelength_sets(min_wl=WL_RANGE[0], max_wl=WL_RANGE[1],
                                   n_bands=N_BANDS, strategy="uniform")
    T_batch = rng.uniform(T_range[0], T_range[1], n_samples)
    material_idx = np.array([i % len(MATERIALS) for i in range(n_samples)])

    eps_batch = np.stack([
        generate_emissivity_curve(
            N_BANDS, material=MATERIALS[material_idx[i]],
            wavelength_range=WL_RANGE, variability=EMISSIVITY_VARIABILITY,
            seed=int(rng.integers(0, 2**31)),
        )
        for i in range(n_samples)
    ])
    L_env = blackbody_environment(wl, L_ENV_T)
    L_clean = compute_radiance_batch(wl, eps_batch, T_batch, L_env)
    L_noisy = add_noise_batch(L_clean, wl, T_batch, nedt_K=NEDT_K, seed=seed + 1)
    return wl, L_env, T_batch, eps_batch, L_noisy, material_idx


def params_to_array(T_batch, eps_batch):
    return np.column_stack([T_batch, eps_batch])


def print_per_material_breakdown(T_true, T_phys, T_hyb, mat_idx):
    """Print the in-dist physics/hybrid |ΔT| table split by material."""
    print(f"  In-dist mean |ΔT|:  physics={np.mean(np.abs(T_phys - T_true)):.2f} K"
          f"  hybrid={np.mean(np.abs(T_hyb - T_true)):.2f} K")
    print(f"  {'material':>12}  {'phys |ΔT|':>10}  {'hyb |ΔT|':>10}  {'n':>4}")
    for j, mat in enumerate(MATERIALS):
        mask = mat_idx == j
        if mask.any():
            print(f"  {mat:>12}"
                  f"  {np.mean(np.abs(T_phys[mask] - T_true[mask])):>10.2f}"
                  f"  {np.mean(np.abs(T_hyb[mask] - T_true[mask])):>10.2f}"
                  f"  {int(mask.sum()):>4d}")


def print_kappa_distribution(p_kap, h_kap, label="κ"):
    """Print percentile table for physics/hybrid condition numbers."""
    finite = np.isfinite(p_kap) & np.isfinite(h_kap)
    print(f"\nFig 10 data — {label} distribution (in-distribution test set):")
    print(f"  {'pctile':>7}  {'physics':>10}  {'hybrid':>10}  {'h/p ratio':>10}")
    for q in [10, 25, 50, 75, 90]:
        print(f"  {q:>5}th  {np.nanpercentile(p_kap, q):>10.3f}"
              f"  {np.nanpercentile(h_kap, q):>10.3f}"
              f"  {np.nanpercentile(h_kap[finite] / p_kap[finite], q):>10.3f}")
    print(f"  mean   {np.nanmean(p_kap):>10.3f}  {np.nanmean(h_kap):>10.3f}"
          f"  {np.nanmean(h_kap[finite] / p_kap[finite]):>10.3f}")
    n_better = int(np.sum(h_kap[finite] < p_kap[finite]))
    n_total = int(finite.sum())
    print(f"  hybrid {label} < physics {label} in {n_better}/{n_total} samples"
          f"  ({100*n_better/max(n_total, 1):.1f}%)")
