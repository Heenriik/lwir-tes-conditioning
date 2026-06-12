"""
Multi-seed L-curve sweep for the physics-consistency variant.

Re-runs the full L-curve over lambda_phys ∈ {0.01, 0.1, 1, 10, 100} for
three model-init seeds {7, 42, 123} — matching the §4.1.4 reproducibility
protocol used in the noise study. Training and test data are held fixed
across seeds (seed=42 for training data, seed=99 for test data); only the
model-init RNG varies. This isolates training-stochasticity contributions
to the L-curve from data-side variability.

Outputs:
  results/data/fig13_lcurve_phys_multiseed.csv
      one row per (seed, lambda_phys): accuracy_T_mae, phys_violation,
      final_loss, lipschitz
  results/data/fig13_lcurve_phys_summary.csv
      one row per lambda_phys: median + p25/p75 of accuracy and violation,
      plus an is_knee_median flag computed on the median curve.

The single-seed fig13_lcurve_phys.csv is NOT overwritten; that file
remains the seed=42 reference used by the rest of the pipeline.

Run after activating the Mthesis env:
    python scripts/run_lcurve_multiseed.py
"""

import os
import sys

# Ensure scripts/ helpers are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch

from _hybrid_common import (
    DATA_DIR, N_BANDS, N_TEST, N_TRAIN, T_TRAIN_RANGE,
    generate_dataset, params_to_array,
)
from src.inversion.tex_inversion import inversion_ensemble
from src.hybrid.residual_model import ResidualMLP, train_residual_model
from src.hybrid.hybrid_model import HybridTeXModel
from src.forward_model import compute_radiance


LAMBDA_PHYS_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0]
INIT_SEEDS = [7, 42, 123, 314, 1729]


def physics_consistency_violation(wl, T_est, eps_est, L_env, L_obs):
    """Match run_hybrid_phys.physics_consistency_violation exactly."""
    M = len(T_est)
    L_scale_sq = float(np.abs(L_obs).max()) ** 2
    err_sq = 0.0
    for i in range(M):
        L_pred = compute_radiance(wl, T_est[i], eps_est[i], L_env)
        err_sq += float(np.mean((L_pred - L_obs[i]) ** 2))
    return (err_sq / M) / L_scale_sq


def set_init_seed(seed):
    """Pin only what controls model init + training stochasticity."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_knee(accuracy, consistency):
    """Hansen corner-distance heuristic — match run_hybrid_phys.pick_lcurve_knee."""
    log_c = np.log10(np.asarray(consistency))
    log_a = np.log10(np.asarray(accuracy))
    log_c_n = (log_c - log_c.min()) / (log_c.max() - log_c.min() + 1e-30)
    log_a_n = (log_a - log_a.min()) / (log_a.max() - log_a.min() + 1e-30)
    dist = np.sqrt(log_c_n ** 2 + log_a_n ** 2)
    return int(np.argmin(dist))


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # --- Shared data (fixed across seeds) ---
    print(f"Generating {N_TRAIN} training samples (data seed=42)...")
    wl, L_env, T_true, eps_true, L_noisy, _ = generate_dataset(
        N_TRAIN, T_TRAIN_RANGE, seed=42
    )
    true_arr = params_to_array(T_true, eps_true)

    print("Running physics inversion on training set...")
    T_phys, eps_phys, _ = inversion_ensemble(L_noisy, wl, L_env)
    phys_arr = params_to_array(T_phys, eps_phys)

    print(f"Generating {N_TEST} test samples (data seed=99)...")
    _, _, T_test, eps_test, L_test, mat_idx_test = generate_dataset(
        N_TEST, T_TRAIN_RANGE, seed=99
    )
    T_phys_test, _, _ = inversion_ensemble(
        L_test, wl, L_env, T_bounds=(200.0, 500.0)
    )
    phys_acc = float(np.mean(np.abs(T_phys_test - T_test)))
    print(f"  Physics-only in-dist |ΔT| = {phys_acc:.3f} K")

    # --- Sweep over (seed, lambda) ---
    # Incremental: if a multiseed CSV already exists, keep existing
    # (seed, lambda) rows and only train the missing combinations.
    out_path = os.path.join(DATA_DIR, "fig13_lcurve_phys_multiseed.csv")
    existing_rows = []
    done_pairs = set()
    if os.path.exists(out_path):
        prev = pd.read_csv(out_path)
        for _, r in prev.iterrows():
            existing_rows.append(r.to_dict())
            done_pairs.add((int(r["seed"]), float(r["lambda_phys"])))
        print(
            f"\nFound existing {out_path}: "
            f"{len(done_pairs)} (seed, λ) pairs already computed; will skip them."
        )

    rows = list(existing_rows)
    for seed in INIT_SEEDS:
        for lam in LAMBDA_PHYS_VALUES:
            if (seed, float(lam)) in done_pairs:
                print(f"[seed={seed}, λ_phys={lam:g}] already in CSV — skipped")
                continue
            print(f"\n[seed={seed}, λ_phys={lam:g}] training...")
            set_init_seed(seed)
            model = ResidualMLP(
                input_dim=N_BANDS, output_dim=N_BANDS + 1,
                hidden_dims=[64, 64], use_spectral_norm=True,
            )
            history = train_residual_model(
                model, L_noisy, true_arr, phys_arr,
                n_epochs=200, lr=1e-3, weight_decay=1e-4,
                lambda_phys=lam, wavelengths=wl, L_env=L_env,
            )
            hybrid = HybridTeXModel(
                wl, L_env, model, T_bounds=(200.0, 500.0)
            )
            hyb_results = [hybrid.forward(L_test[i]) for i in range(N_TEST)]
            T_hyb = np.array([r["T_est"] for r in hyb_results])
            eps_hyb = np.stack([r["emissivity_est"] for r in hyb_results])
            acc_mae = float(np.mean(np.abs(T_hyb - T_test)))
            viol = physics_consistency_violation(
                wl, T_hyb, eps_hyb, L_env, L_test
            )
            lip = float(model.lipschitz_upper_bound)
            final_loss = float(history[-1])
            print(
                f"  final_loss={final_loss:.4e}  Lipschitz={lip:.3f}"
                f"  |ΔT|={acc_mae:.3f} K  viol={viol:.4e}"
            )
            rows.append({
                "seed": seed,
                "lambda_phys": lam,
                "accuracy_T_mae": acc_mae,
                "phys_violation": viol,
                "final_loss": final_loss,
                "lipschitz": lip,
            })

    df = pd.DataFrame(rows).sort_values(["seed", "lambda_phys"]).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")

    # --- Summary per lambda ---
    summary_rows = []
    for lam in LAMBDA_PHYS_VALUES:
        sub = df[df.lambda_phys == lam]
        summary_rows.append({
            "lambda_phys": lam,
            "accuracy_median": sub.accuracy_T_mae.median(),
            "accuracy_p25": sub.accuracy_T_mae.quantile(0.25),
            "accuracy_p75": sub.accuracy_T_mae.quantile(0.75),
            "violation_median": sub.phys_violation.median(),
            "violation_p25": sub.phys_violation.quantile(0.25),
            "violation_p75": sub.phys_violation.quantile(0.75),
            "lipschitz_median": sub.lipschitz.median(),
        })
    summary = pd.DataFrame(summary_rows)

    # Knee on the median curve
    knee_idx = pick_knee(
        summary.accuracy_median.values, summary.violation_median.values
    )
    summary["is_knee_median"] = [i == knee_idx for i in range(len(summary))]

    summary_path = os.path.join(DATA_DIR, "fig13_lcurve_phys_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")

    print("\nL-curve summary (median across seeds):")
    cols = ["lambda_phys", "accuracy_median", "accuracy_p25", "accuracy_p75",
            "violation_median", "violation_p25", "violation_p75",
            "lipschitz_median", "is_knee_median"]
    print(summary[cols].to_string(index=False))
    print(f"\nMedian-curve knee: λ_phys = {summary.lambda_phys.iloc[knee_idx]:g}")


if __name__ == "__main__":
    main()
