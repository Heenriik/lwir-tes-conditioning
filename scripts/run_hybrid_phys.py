"""
SENSE Eq. 10 — physical-consistency loss term.

    L_total = ‖x̂ - x_GT‖²   +  λ_phys · ‖L - P(x̂)‖²   +  (Lipschitz via spectral_norm)

The accuracy term is unchanged from the baseline; the physics-consistency
term penalises parameter estimates that, when forward-projected, fail to
reproduce the measured radiance. Lipschitz regularisation is still provided
architecturally by spectral_norm (so the third SENSE Eq. 10 term is satisfied
via construction, not a soft Frobenius penalty).

The script performs an L-curve sweep over λ_phys (Hansen 1992 logic, same
citation as your existing Tikhonov section §1.6) to pick a defensible value,
then runs the full eval pipeline at the knee λ to produce figs 10/11/12
with the `_phys` suffix.

Reference: SENSE proposal Eq. (10), p.3.

Writes tidy CSVs to results/data/ for fig10/11/12/13 (phys variant). Rendering
is done by scripts/plot_all.py.

Usage:
    conda activate Mthesis
    python scripts/run_hybrid_phys.py
"""

import os
import numpy as np
import pandas as pd

from _hybrid_common import (
    DATA_DIR, MATERIALS,
    N_BANDS, N_TEST, N_OOD_PER_OFFSET, N_TRAIN, T_OOD_OFFSETS, T_TRAIN_RANGE,
    generate_dataset, params_to_array, set_seed,
    print_kappa_distribution, print_per_material_breakdown,
)

from src.forward_model import compute_radiance
from src.inversion.tex_inversion import inversion_ensemble
from src.inversion.jacobian import jacobian_svd
from src.hybrid.residual_model import ResidualMLP, train_residual_model
from src.hybrid.hybrid_model import HybridTeXModel, condition_number_comparison

VARIANT = "phys"


# L-curve sweep range. phys term in the loss is
#   λ_phys · (‖P(x̂) − L‖² / L_scale²) / acc_loss_init
# so lambda_phys=1 means "phys term initially comparable to accuracy term."
# Sweep covers under-regularised (λ=0.01, no effect) through over-regularised
# (λ=100, physics term dominates) regimes.
LAMBDA_PHYS_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0]


def physics_consistency_violation(wl, T_est, eps_est, L_env, L_obs):
    """Mean relative ‖L − P(x̂)‖² / ‖L_scale‖² across a test batch."""
    M = len(T_est)
    L_scale_sq = float(np.abs(L_obs).max()) ** 2
    err_sq = 0.0
    for i in range(M):
        L_pred = compute_radiance(wl, T_est[i], eps_est[i], L_env)
        err_sq += float(np.mean((L_pred - L_obs[i]) ** 2))
    return (err_sq / M) / L_scale_sq


def pick_lcurve_knee(accuracy, consistency):
    """
    Return the index of the L-curve knee — the point closest to the origin
    after log10 normalisation of both axes to [0, 1]. This is Hansen's
    'corner distance' heuristic for picking regularisation strength.
    """
    log_c = np.log10(np.asarray(consistency))
    log_a = np.log10(np.asarray(accuracy))
    log_c_n = (log_c - log_c.min()) / (log_c.max() - log_c.min() + 1e-30)
    log_a_n = (log_a - log_a.min()) / (log_a.max() - log_a.min() + 1e-30)
    dist = np.sqrt(log_c_n ** 2 + log_a_n ** 2)
    return int(np.argmin(dist))


def train_and_eval(lambda_phys, wl, L_env, L_train, true_train_arr, phys_train_arr,
                    T_test, eps_test, L_test, mat_idx_test, n_test):
    """Train at one lambda, evaluate in-distribution. Returns metrics + hybrid."""
    # Reset RNG so every λ in the sweep starts from the same model weights —
    # otherwise the L-curve mixes two confounded variables (λ_phys and init).
    set_seed()
    model = ResidualMLP(input_dim=N_BANDS, output_dim=N_BANDS + 1,
                         hidden_dims=[64, 64], use_spectral_norm=True)
    history = train_residual_model(
        model, L_train, true_train_arr, phys_train_arr,
        n_epochs=200, lr=1e-3, weight_decay=1e-4,
        lambda_phys=lambda_phys, wavelengths=wl, L_env=L_env,
    )
    hybrid = HybridTeXModel(wl, L_env, model, T_bounds=(200.0, 500.0))

    hyb_results = [hybrid.forward(L_test[i]) for i in range(n_test)]
    T_hyb = np.array([r["T_est"] for r in hyb_results])
    eps_hyb = np.stack([r["emissivity_est"] for r in hyb_results])

    acc_mae = float(np.mean(np.abs(T_hyb - T_test)))
    phys_viol = physics_consistency_violation(wl, T_hyb, eps_hyb, L_env, L_test)
    return {
        "lambda_phys": lambda_phys,
        "model": model,
        "hybrid": hybrid,
        "final_loss": float(history[-1]),
        "lipschitz": float(model.lipschitz_upper_bound),
        "accuracy_mae": acc_mae,
        "phys_violation": phys_viol,
        "T_hyb": T_hyb,
        "eps_hyb": eps_hyb,
    }


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # --- Shared training + test data (all λ_phys runs use the same data) ---
    print(f"Generating {N_TRAIN} training samples...")
    wl, L_env, T_true, eps_true, L_noisy, _ = generate_dataset(
        N_TRAIN, T_TRAIN_RANGE, seed=42
    )
    true_arr = params_to_array(T_true, eps_true)

    print("Running physics inversion on training set (parallel)...")
    T_phys, eps_phys, _ = inversion_ensemble(L_noisy, wl, L_env)
    phys_arr = params_to_array(T_phys, eps_phys)

    print(f"Generating {N_TEST} in-distribution test samples...")
    _, _, T_test, eps_test, L_test, mat_idx_test = generate_dataset(
        N_TEST, T_TRAIN_RANGE, seed=99
    )
    T_phys_test, _, _ = inversion_ensemble(L_test, wl, L_env,
                                             T_bounds=(200.0, 500.0))
    phys_acc = float(np.mean(np.abs(T_phys_test - T_test)))
    print(f"  Physics-only in-dist |ΔT| = {phys_acc:.3f} K")

    # --- L-curve sweep ---
    print(f"\n--- L-curve sweep over lambda_phys = {LAMBDA_PHYS_VALUES} ---")
    runs = []
    for lam in LAMBDA_PHYS_VALUES:
        print(f"\nTraining at λ_phys = {lam:g} ...")
        run = train_and_eval(lam, wl, L_env, L_noisy, true_arr, phys_arr,
                              T_test, eps_test, L_test, mat_idx_test, N_TEST)
        print(f"  final_loss={run['final_loss']:.4e}"
              f"  Lipschitz={run['lipschitz']:.3f}"
              f"  in-dist |ΔT|={run['accuracy_mae']:.3f} K"
              f"  phys_violation={run['phys_violation']:.4e}")
        runs.append(run)

    accuracy = np.array([r["accuracy_mae"] for r in runs])
    consistency = np.array([r["phys_violation"] for r in runs])
    knee_idx = pick_lcurve_knee(accuracy, consistency)
    knee_lambda = LAMBDA_PHYS_VALUES[knee_idx]

    # If a multi-seed summary exists, prefer its median-curve knee — the
    # single-seed Hansen pick can flip between adjacent lambdas under
    # training noise (see scripts/run_lcurve_multiseed.py).
    summary_path = os.path.join(DATA_DIR, "fig13_lcurve_phys_summary.csv")
    if os.path.exists(summary_path):
        df_sum = pd.read_csv(summary_path)
        if "is_knee_median" in df_sum.columns and df_sum["is_knee_median"].any():
            median_knee_lambda = float(
                df_sum[df_sum["is_knee_median"]].iloc[0]["lambda_phys"]
            )
            if median_knee_lambda != knee_lambda:
                print(
                    f"  [info] single-seed Hansen picked λ={knee_lambda:g}; "
                    f"multi-seed median picks λ={median_knee_lambda:g}. "
                    f"Using multi-seed knee for downstream eval."
                )
            knee_lambda = median_knee_lambda
            knee_idx = LAMBDA_PHYS_VALUES.index(knee_lambda)

    print("\nL-curve table:")
    print(f"  {'lambda':>10}  {'accuracy |ΔT|':>14}  {'phys viol.':>12}  {'knee?':>5}")
    for i, (lam, a, c) in enumerate(zip(LAMBDA_PHYS_VALUES, accuracy, consistency)):
        mark = "  <-" if i == knee_idx else ""
        print(f"  {lam:>10.3g}  {a:>14.3f}  {c:>12.4e}{mark}")
    print(f"\nKnee: λ_phys = {knee_lambda:g}")

    pd.DataFrame({
        "lambda_phys": np.asarray(LAMBDA_PHYS_VALUES, dtype=float),
        "accuracy_T_mae": accuracy,
        "phys_violation": consistency,
        "is_knee": [i == knee_idx for i in range(len(LAMBDA_PHYS_VALUES))],
    }).to_csv(os.path.join(DATA_DIR, "fig13_lcurve_phys.csv"), index=False)
    print("Wrote fig13 L-curve CSV")

    # --- Full eval at knee lambda (figs 10/11/12 with _phys suffix) ---
    knee_run = runs[knee_idx]
    hybrid = knee_run["hybrid"]
    print(f"\n--- Full evaluation at λ_phys = {knee_lambda:g} (knee) ---")

    print_per_material_breakdown(T_test, T_phys_test, knee_run["T_hyb"], mat_idx_test)

    cmp = condition_number_comparison(L_test, hybrid)
    p_kap = cmp["physics_kappas"]
    h_kap = cmp["hybrid_kappas"]
    print_kappa_distribution(p_kap, h_kap, label="κ")

    pd.DataFrame({
        "sample_idx": np.arange(len(p_kap)),
        "kappa_physics": p_kap, "kappa_hybrid": h_kap,
    }).to_csv(os.path.join(DATA_DIR, f"fig10_hybrid_conditioning_{VARIANT}.csv"),
              index=False)
    print(f"Wrote fig10 CSV ({VARIANT})")

    # Figure 11
    mid_idx = int(np.nanargmin(np.abs(p_kap - np.nanmedian(p_kap))))
    L_rep = L_test[mid_idx]
    J_phys = hybrid.physics_jacobian(L_rep)
    J_hyb_t = hybrid.numerical_output_jacobian(L_rep).T
    svd_phys = jacobian_svd(J_phys)
    svd_hyb = jacobian_svd(J_hyb_t)

    pd.DataFrame({
        "index": np.arange(1, len(svd_phys["s"]) + 1),
        "sigma_physics": svd_phys["s"], "sigma_hybrid": svd_hyb["s"],
    }).to_csv(os.path.join(DATA_DIR, f"fig11_hybrid_singular_values_{VARIANT}.csv"),
              index=False)
    pd.DataFrame([{
        "kappa_physics": float(svd_phys["condition_number"]),
        "kappa_hybrid": float(svd_hyb["condition_number"]),
        "sample_idx": int(mid_idx),
        "T_true": float(T_test[mid_idx]),
        "material": MATERIALS[mat_idx_test[mid_idx]],
        "lambda_phys": float(knee_lambda),
    }]).to_csv(os.path.join(DATA_DIR, f"fig11_hybrid_singular_values_{VARIANT}_meta.csv"),
              index=False)
    print(f"Wrote fig11 CSVs ({VARIANT})")
    print(f"\nFig 11 data — singular values at median-κ sample (idx={mid_idx},"
          f" T_true={T_test[mid_idx]:.1f} K,"
          f" material={MATERIALS[mat_idx_test[mid_idx]]}):")
    print(f"  {'index':>5}  {'σ_physics':>12}  {'σ_hybrid':>12}")
    for k, (sp, sh) in enumerate(zip(svd_phys["s"], svd_hyb["s"])):
        print(f"  {k+1:>5}  {sp:>12.4e}  {sh:>12.4e}")
    print(f"  κ_physics = {svd_phys['condition_number']:.3e}"
          f"   κ_hybrid = {svd_hyb['condition_number']:.3e}")

    # --- OOD test ---
    print("Running distribution shift tests...")
    physics_errs_ood, hybrid_errs_ood = [], []
    for offset in T_OOD_OFFSETS:
        T_lo = T_TRAIN_RANGE[1] + offset
        T_hi = T_lo + 10.0
        _, _, T_ood, eps_ood, L_ood, _ = generate_dataset(
            N_OOD_PER_OFFSET, (T_lo, T_hi), seed=int(offset) + 200
        )
        T_phys_ood, _, _ = inversion_ensemble(L_ood, wl, L_env,
                                               T_bounds=(200.0, 500.0))
        T_hyb_ood = np.array([
            hybrid.forward(L_ood[i])["T_est"] for i in range(N_OOD_PER_OFFSET)
        ])
        physics_errs_ood.append(float(np.mean(np.abs(T_phys_ood - T_ood))))
        hybrid_errs_ood.append(float(np.mean(np.abs(T_hyb_ood - T_ood))))
        print(f"  Offset +{offset:.0f} K:  physics |ΔT|={physics_errs_ood[-1]:.2f} K"
              f"  hybrid |ΔT|={hybrid_errs_ood[-1]:.2f} K")

    pd.DataFrame({
        "offset_K": np.asarray(T_OOD_OFFSETS, dtype=float),
        "physics_T_mae": np.array(physics_errs_ood),
        "hybrid_T_mae": np.array(hybrid_errs_ood),
    }).to_csv(os.path.join(DATA_DIR, f"fig12_distribution_shift_{VARIANT}.csv"),
              index=False)
    print(f"Wrote fig12 CSV ({VARIANT})")

    print(f"\nAll CSVs saved to {DATA_DIR}")
    print(f"Knee-selected λ_phys = {knee_lambda:g}"
          f"  → in-dist |ΔT| = {knee_run['accuracy_mae']:.3f} K"
          f"  vs physics-only {phys_acc:.3f} K")


if __name__ == "__main__":
    main()
