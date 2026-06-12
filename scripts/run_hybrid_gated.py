"""
SENSE Eq. 9 — density-gated residual.

    δ̃ (L) = δ(L) · σ(γ · (log q(L) - log q_threshold))

The residual is multiplied by a sigmoid of the radiance log-density under
a learned q(L). In-distribution → gate ≈ 1, full residual applied.
OOD → gate → 0, falls back to pure physics.

Reference: SENSE proposal Eq. (9), p.3.
The original suggests a Normalizing Flow for q(y); we use a Gaussian
Mixture Model for simplicity — the gating mechanism is the load-bearing
part, q's form is exchangeable.

Writes tidy CSVs to results/data/ for fig10/11/12 (gated variant). Rendering
is done by scripts/plot_all.py.

Usage:
    conda activate Mthesis
    python scripts/run_hybrid_gated.py
"""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.mixture import GaussianMixture

from _hybrid_common import (
    DATA_DIR, MATERIALS,
    N_BANDS, N_TEST, N_OOD_PER_OFFSET, N_TRAIN, T_OOD_OFFSETS, T_TRAIN_RANGE,
    generate_dataset, params_to_array, set_seed,
    print_kappa_distribution, print_per_material_breakdown,
)

from src.inversion.tex_inversion import inversion_ensemble, invert_tex
from src.inversion.jacobian import jacobian_svd
from src.hybrid.residual_model import ResidualMLP, train_residual_model
from src.hybrid.hybrid_model import HybridTeXModel, condition_number_comparison

VARIANT = "gated"


# Gating hyperparameters — calibrated to training log-density distribution
GMM_N_COMPONENTS = 8           # one per material; keeps GMM small
GATE_PCTILE = 5.0              # gate ≈ 0.5 at this training-set log-density
                                # (≈ "5% of training inputs look this OOD")


class GatedHybridTeXModel(HybridTeXModel):
    """
    Hybrid model where the residual is multiplied by an OOD likelihood gate.

    Inherits forward / numerical_output_jacobian / physics_jacobian from
    HybridTeXModel — we only override the per-sample forward to insert the
    gate factor.
    """

    def __init__(self, wavelengths, L_env, residual_model, density_estimator,
                 log_q_threshold, log_q_scale, T_bounds=(200.0, 500.0)):
        super().__init__(wavelengths, L_env, residual_model, T_bounds=T_bounds)
        self.density_estimator = density_estimator
        self.log_q_threshold = log_q_threshold
        self.log_q_scale = log_q_scale

    def gate(self, L):
        """σ((log q(L) − threshold) / scale)  ∈ (0, 1)"""
        log_q = self.density_estimator.score_samples(L.reshape(1, -1))[0]
        z = (log_q - self.log_q_threshold) / self.log_q_scale
        return float(1.0 / (1.0 + np.exp(-z))), log_q

    def forward(self, L):
        phys = invert_tex(self.wavelengths, L, self.L_env, T_bounds=self.T_bounds)
        T_phys = phys["T_est"]
        eps_phys = phys["emissivity_est"]

        device = next(self.residual_model.parameters()).device
        L_tensor = torch.tensor(L.astype(np.float32), device=device)
        self.residual_model.eval()
        with torch.no_grad():
            raw_correction = self.residual_model(L_tensor).cpu().numpy()

        g, log_q = self.gate(L)
        correction = g * raw_correction  # gated residual

        T_scale = 300.0
        T_est = np.clip(T_phys + correction[0] * T_scale, *self.T_bounds)
        eps_est = np.clip(eps_phys + correction[1:], 0.01, 1.0)
        return {
            "T_est": float(T_est),
            "emissivity_est": eps_est,
            "physics_T": T_phys,
            "physics_emissivity": eps_phys,
            "residual_correction": correction,
            "raw_correction": raw_correction,
            "gate": g,
            "log_q": log_q,
        }


def main():
    set_seed()  # pin model init RNG so OOD numbers are reproducible
    os.makedirs(DATA_DIR, exist_ok=True)

    # --- Training data + base residual training (identical to baseline) ---
    print(f"Generating {N_TRAIN} training samples...")
    wl, L_env, T_true, eps_true, L_noisy, _ = generate_dataset(
        N_TRAIN, T_TRAIN_RANGE, seed=42
    )
    true_params = params_to_array(T_true, eps_true)

    print("Running physics inversion on training set (parallel)...")
    T_phys, eps_phys, _ = inversion_ensemble(L_noisy, wl, L_env)
    phys_params = params_to_array(T_phys, eps_phys)

    print("Training ResidualMLP with spectral normalization...")
    model = ResidualMLP(input_dim=N_BANDS, output_dim=N_BANDS + 1,
                         hidden_dims=[64, 64], use_spectral_norm=True)
    history = train_residual_model(
        model, L_noisy, true_params, phys_params,
        n_epochs=200, lr=1e-3, weight_decay=1e-4,
    )
    print(f"  Final training loss: {history[-1]:.6f}")
    print(f"  Lipschitz upper bound: {model.lipschitz_upper_bound:.4f}")

    # --- Fit GMM on training radiance and calibrate gate ---
    print(f"Fitting GMM density estimator (n_components={GMM_N_COMPONENTS})...")
    gmm = GaussianMixture(n_components=GMM_N_COMPONENTS, covariance_type="full",
                          random_state=42, reg_covar=1e-6, max_iter=200)
    gmm.fit(L_noisy)
    log_q_train = gmm.score_samples(L_noisy)
    log_q_threshold = float(np.percentile(log_q_train, GATE_PCTILE))
    log_q_scale = float(np.std(log_q_train))
    print(f"  log q(L) train range: [{log_q_train.min():.1f}, {log_q_train.max():.1f}]"
          f"  median={np.median(log_q_train):.1f}")
    print(f"  Gate calibration: threshold={log_q_threshold:.1f}  scale={log_q_scale:.2f}"
          f"  (gate≈0.5 at {GATE_PCTILE:.0f}th-pctile training input)")

    hybrid = GatedHybridTeXModel(wl, L_env, model, gmm,
                                  log_q_threshold, log_q_scale,
                                  T_bounds=(200.0, 500.0))

    # --- In-distribution test ---
    print(f"Evaluating on {N_TEST} in-distribution test samples...")
    _, _, T_test, eps_test, L_test, mat_idx_test = generate_dataset(
        N_TEST, T_TRAIN_RANGE, seed=99
    )

    T_phys_test, _, _ = inversion_ensemble(L_test, wl, L_env,
                                             T_bounds=(200.0, 500.0))
    hyb_results = [hybrid.forward(L_test[i]) for i in range(N_TEST)]
    T_hyb_test = np.array([r["T_est"] for r in hyb_results])
    gates_in = np.array([r["gate"] for r in hyb_results])
    print(f"  In-dist gate stats: mean={gates_in.mean():.3f}"
          f"  median={np.median(gates_in):.3f}"
          f"  min={gates_in.min():.3f}  max={gates_in.max():.3f}")
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
        "gate": float(hyb_results[mid_idx]["gate"]),
    }]).to_csv(os.path.join(DATA_DIR, f"fig11_hybrid_singular_values_{VARIANT}_meta.csv"),
              index=False)
    print(f"Wrote fig11 CSVs ({VARIANT})")
    print(f"\nFig 11 data — singular values at median-κ sample (idx={mid_idx},"
          f" T_true={T_test[mid_idx]:.1f} K,"
          f" material={MATERIALS[mat_idx_test[mid_idx]]},"
          f" gate={hyb_results[mid_idx]['gate']:.3f}):")
    print(f"  {'index':>5}  {'σ_physics':>12}  {'σ_hybrid':>12}")
    for k, (sp, sh) in enumerate(zip(svd_phys["s"], svd_hyb["s"])):
        print(f"  {k+1:>5}  {sp:>12.4e}  {sh:>12.4e}")
    print(f"  κ_physics = {svd_phys['condition_number']:.3e}"
          f"   κ_hybrid = {svd_hyb['condition_number']:.3e}")

    # --- OOD with gate-tracking ---
    print("Running distribution shift tests...")
    physics_errs_ood, hybrid_errs_ood, gate_means_ood = [], [], []

    for offset in T_OOD_OFFSETS:
        T_lo = T_TRAIN_RANGE[1] + offset
        T_hi = T_lo + 10.0
        _, _, T_ood, eps_ood, L_ood, _ = generate_dataset(
            N_OOD_PER_OFFSET, (T_lo, T_hi), seed=int(offset) + 200
        )
        T_phys_ood, _, _ = inversion_ensemble(L_ood, wl, L_env,
                                               T_bounds=(200.0, 500.0))
        ood_results = [hybrid.forward(L_ood[i]) for i in range(N_OOD_PER_OFFSET)]
        T_hyb_ood = np.array([r["T_est"] for r in ood_results])
        gates_ood = np.array([r["gate"] for r in ood_results])
        physics_errs_ood.append(float(np.mean(np.abs(T_phys_ood - T_ood))))
        hybrid_errs_ood.append(float(np.mean(np.abs(T_hyb_ood - T_ood))))
        gate_means_ood.append(float(gates_ood.mean()))
        print(f"  Offset +{offset:.0f} K:  physics |ΔT|={physics_errs_ood[-1]:.2f} K"
              f"  hybrid |ΔT|={hybrid_errs_ood[-1]:.2f} K"
              f"  gate (mean)={gate_means_ood[-1]:.3f}")

    pd.DataFrame({
        "offset_K": np.asarray(T_OOD_OFFSETS, dtype=float),
        "physics_T_mae": np.array(physics_errs_ood),
        "hybrid_T_mae": np.array(hybrid_errs_ood),
        "mean_gate": np.array(gate_means_ood),
    }).to_csv(os.path.join(DATA_DIR, f"fig12_distribution_shift_{VARIANT}.csv"),
              index=False)
    print(f"Wrote fig12 CSV ({VARIANT})")
    print(f"\nGate behaviour: in-dist mean={gates_in.mean():.3f},"
          f" OOD offsets {list(T_OOD_OFFSETS)} K → gates"
          f" {[f'{g:.3f}' for g in gate_means_ood]}")

    print(f"\nAll CSVs saved to {DATA_DIR}")


if __name__ == "__main__":
    main()
