"""
Canonical hybrid baseline — inverse-space residual + spectral_norm Lipschitz bound.

x̂ = f_TeX(L) + r_θ(L)  (state-space residual)

Writes tidy CSVs to results/data/ for figures 10, 11, 12 (baseline variant).
Rendering is done by scripts/plot_all.py.

CSVs produced:
  fig10_hybrid_conditioning_baseline.csv     sample_idx, kappa_physics, kappa_hybrid
  fig11_hybrid_singular_values_baseline.csv  index, sigma_physics, sigma_hybrid
  fig11_hybrid_singular_values_baseline_meta.csv  kappa_physics, kappa_hybrid, sample_idx, T_true, material
  fig12_distribution_shift_baseline.csv      offset_K, physics_T_mae, hybrid_T_mae

Variant scripts (run_hybrid_gated.py, run_hybrid_forward.py) reuse the
shared helpers from _hybrid_common.py. Do not change the constants or
dataset generator here — change them in _hybrid_common.py instead.

Usage:
    conda activate Mthesis
    python scripts/run_hybrid.py
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

from src.inversion.tex_inversion import inversion_ensemble
from src.inversion.jacobian import jacobian_svd
from src.hybrid.residual_model import ResidualMLP, train_residual_model
from src.hybrid.hybrid_model import HybridTeXModel, condition_number_comparison

VARIANT = "baseline"


def main():
    set_seed()  # pin model init RNG so OOD numbers are reproducible
    os.makedirs(DATA_DIR, exist_ok=True)

    # --- Generate training dataset ---
    print(f"Generating {N_TRAIN} training samples...")
    wl, L_env, T_true, eps_true, L_noisy, _ = generate_dataset(
        N_TRAIN, T_TRAIN_RANGE, seed=42
    )
    true_params = params_to_array(T_true, eps_true)

    # --- Physics inversion on training set ---
    print("Running physics inversion on training set (parallel)...")
    T_phys, eps_phys, _ = inversion_ensemble(L_noisy, wl, L_env)
    phys_params = params_to_array(T_phys, eps_phys)

    # --- Train residual MLP ---
    print("Training ResidualMLP with spectral normalization...")
    model = ResidualMLP(input_dim=N_BANDS, output_dim=N_BANDS + 1,
                         hidden_dims=[64, 64], use_spectral_norm=True)
    history = train_residual_model(
        model, L_noisy, true_params, phys_params,
        n_epochs=200, lr=1e-3, weight_decay=1e-4,
    )
    print(f"  Final training loss: {history[-1]:.6f}")
    print(f"  Lipschitz upper bound: {model.lipschitz_upper_bound:.4f}")

    # Wide T_bounds so OOD evaluation isn't truncated.
    # In-distribution inversion is unaffected since training T ∈ [250, 380] lies well inside.
    hybrid = HybridTeXModel(wl, L_env, model, T_bounds=(200.0, 500.0))

    # --- In-distribution test set (Figure 10 & 11) ---
    print(f"Evaluating on {N_TEST} in-distribution test samples...")
    _, _, T_test, eps_test, L_test, mat_idx_test = generate_dataset(
        N_TEST, T_TRAIN_RANGE, seed=99
    )

    T_phys_test, _, _ = inversion_ensemble(L_test, wl, L_env,
                                             T_bounds=(200.0, 500.0))
    T_hyb_test = np.array([hybrid.forward(L_test[i])["T_est"]
                            for i in range(N_TEST)])
    print_per_material_breakdown(T_test, T_phys_test, T_hyb_test, mat_idx_test)

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

    # Figure 11: representative SVD comparison (pick median-κ test sample)
    mid_idx = int(np.nanargmin(np.abs(p_kap - np.nanmedian(p_kap))))
    L_rep = L_test[mid_idx]
    J_phys = hybrid.physics_jacobian(L_rep)
    J_hyb_t = hybrid.numerical_output_jacobian(L_rep).T  # → (N, N+1)
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
    }]).to_csv(os.path.join(DATA_DIR, f"fig11_hybrid_singular_values_{VARIANT}_meta.csv"),
              index=False)
    print(f"Wrote fig11 CSVs ({VARIANT})")
    print(f"\nFig 11 data — singular values at median-κ sample (idx={mid_idx},"
          f" T_true={T_test[mid_idx]:.1f} K, material={MATERIALS[mat_idx_test[mid_idx]]}):")
    print(f"  {'index':>5}  {'σ_physics':>12}  {'σ_hybrid':>12}")
    for k, (sp, sh) in enumerate(zip(svd_phys["s"], svd_hyb["s"])):
        print(f"  {k+1:>5}  {sp:>12.4e}  {sh:>12.4e}")
    print(f"  κ_physics = {svd_phys['condition_number']:.3e}"
          f"   κ_hybrid = {svd_hyb['condition_number']:.3e}")

    # --- Distribution shift test (Figure 12) ---
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


if __name__ == "__main__":
    main()
