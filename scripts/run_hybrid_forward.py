"""
SENSE Eq. 8 — forward-space (radiance-space) residual.

    ŷ = P(x̂_phys; α) + δ(y; ϕ)

The residual operates on radiance space, not parameter space. Interpretation:
δ(L) predicts the part of L that physics cannot reconstruct after a first
inversion pass. At inference, the corrected radiance (L - δ(L)) is re-inverted
to obtain final parameter estimates.

The baseline (run_hybrid.py) corrects in parameter space; this script tests
whether the forward-space gives different conditioning/OOD behaviour.

Reference: SENSE proposal Eq. (8), p.2; restated Eq. (17) p.7, Eq. (26) p.9.

Writes tidy CSVs to results/data/ for fig10/11/12 (forward variant). Rendering
is done by scripts/plot_all.py.

Usage:
    conda activate Mthesis
    python scripts/run_hybrid_forward.py
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from _hybrid_common import (
    DATA_DIR, MATERIALS,
    N_BANDS, N_TEST, N_OOD_PER_OFFSET, N_TRAIN, T_OOD_OFFSETS, T_TRAIN_RANGE,
    generate_dataset, params_to_array, set_seed,
    print_kappa_distribution, print_per_material_breakdown,
)

from src.forward_model import compute_radiance, blackbody_environment
from src.inversion.tex_inversion import inversion_ensemble, invert_tex
from src.inversion.jacobian import compute_jacobian, jacobian_svd
from src.hybrid.residual_model import ResidualMLP

VARIANT = "forward"


def physics_reconstruction(wl, T_batch, eps_batch, L_env):
    """Forward radiance from a (possibly imperfect) physics inversion."""
    M, N = eps_batch.shape
    L_rec = np.zeros((M, N))
    for i in range(M):
        L_rec[i] = compute_radiance(wl, T_batch[i], eps_batch[i], L_env)
    return L_rec


def train_radiance_residual(model, L_noisy, L_phys_reconstruction,
                              n_epochs=200, lr=1e-3, weight_decay=1e-4,
                              device=None):
    """
    Train δ(L) ≈ L_noisy − L_phys_reconstruction.

    Both input and target are in raw radiance units. The ResidualMLP divides
    its input by L_scale internally and we treat its output as also normalised
    by the same L_scale, so train/eval are consistent.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    L_scale = float(np.abs(L_noisy).max())
    if L_scale < 1e-30:
        L_scale = 1.0
    model.L_scale.fill_(L_scale)

    L_t = torch.tensor(L_noisy.astype(np.float32), device=device)
    target_raw = (L_noisy - L_phys_reconstruction).astype(np.float32)
    target_t = torch.tensor(target_raw / L_scale, device=device)  # normalised

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr,
                                   weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history = []

    model.train()
    for _ in range(n_epochs):
        optimiser.zero_grad()
        pred = model(L_t)  # internally divides by L_scale
        loss = loss_fn(pred, target_t)
        loss.backward()
        optimiser.step()
        history.append(loss.item())

    model.eval()
    return history, L_scale


class ForwardHybridTeXModel:
    """
    Forward-space hybrid: invert physics on (L − δ_θ(L)) instead of L.

    Has the same public surface as HybridTeXModel so condition_number_comparison
    and the rest of the eval pipeline work without modification:
        forward(L) → dict with T_est, emissivity_est
        physics_jacobian(L) → (N, N+1)
        numerical_output_jacobian(L) → (N+1, N)
    """

    def __init__(self, wavelengths, L_env, residual_model,
                 T_bounds=(200.0, 500.0)):
        self.wavelengths = np.asarray(wavelengths, dtype=float)
        self.L_env = np.asarray(L_env, dtype=float)
        self.residual_model = residual_model
        self.T_bounds = T_bounds
        self.N = len(wavelengths)

    def _delta(self, L):
        """Network output in raw radiance units."""
        device = next(self.residual_model.parameters()).device
        L_tensor = torch.tensor(L.astype(np.float32), device=device)
        self.residual_model.eval()
        with torch.no_grad():
            d_norm = self.residual_model(L_tensor).cpu().numpy()
        return d_norm * float(self.residual_model.L_scale.item())

    def forward(self, L):
        delta = self._delta(L)
        L_corr = L - delta
        phys = invert_tex(self.wavelengths, L_corr, self.L_env,
                          T_bounds=self.T_bounds)
        return {
            "T_est": float(phys["T_est"]),
            "emissivity_est": phys["emissivity_est"],
            "delta": delta,
            "L_corrected": L_corr,
        }

    def _forward_as_vector(self, L):
        result = self.forward(L)
        return np.concatenate([[result["T_est"]], result["emissivity_est"]])

    def numerical_output_jacobian(self, L, eps_frac=0.01):
        N = self.N
        J = np.zeros((N + 1, N))
        for i in range(N):
            delta = max(abs(L[i]) * eps_frac, 1e-10)
            L_plus = L.copy(); L_plus[i] += delta
            L_minus = L.copy(); L_minus[i] -= delta
            x_plus = self._forward_as_vector(L_plus)
            x_minus = self._forward_as_vector(L_minus)
            J[:, i] = (x_plus - x_minus) / (2 * delta)
        return J

    def physics_jacobian(self, L):
        """Physics Jacobian at the corrected-radiance inversion point."""
        result = self.forward(L)
        T_est = result["T_est"]
        eps_est = result["emissivity_est"]
        return compute_jacobian(self.wavelengths, T_est, eps_est, self.L_env)


def condition_number_comparison_forward(L_test_batch, hybrid_model):
    """Same shape as src.hybrid.hybrid_model.condition_number_comparison."""
    M = len(L_test_batch)
    p_kap = np.zeros(M)
    h_kap = np.zeros(M)
    for i, L in enumerate(L_test_batch):
        svd_phys = jacobian_svd(hybrid_model.physics_jacobian(L))
        svd_hyb = jacobian_svd(hybrid_model.numerical_output_jacobian(L).T)
        p_kap[i] = svd_phys["condition_number"]
        h_kap[i] = svd_hyb["condition_number"]
    return {"physics_kappas": p_kap, "hybrid_kappas": h_kap}


def main():
    set_seed()  # pin model init RNG so OOD numbers are reproducible
    os.makedirs(DATA_DIR, exist_ok=True)

    # --- Training data + first-pass physics inversion ---
    print(f"Generating {N_TRAIN} training samples...")
    wl, L_env, T_true, eps_true, L_noisy, _ = generate_dataset(
        N_TRAIN, T_TRAIN_RANGE, seed=42
    )

    print("Running physics inversion on training set (parallel)...")
    T_phys, eps_phys, _ = inversion_ensemble(L_noisy, wl, L_env)

    # P(x̂_phys): forward radiance from the (imperfect) physics estimate
    print("Computing physics reconstructions L̂ = P(x̂_phys)...")
    L_reconstruction = physics_reconstruction(wl, T_phys, eps_phys, L_env)
    train_residual = L_noisy - L_reconstruction
    print(f"  Training residual stats:"
          f"  mean={train_residual.mean():.3e}  std={train_residual.std():.3e}"
          f"  max|·|={np.abs(train_residual).max():.3e}")

    # --- Train radiance-space residual ---
    print("Training ResidualMLP (radiance-space residual)...")
    model = ResidualMLP(input_dim=N_BANDS, output_dim=N_BANDS,
                         hidden_dims=[64, 64], use_spectral_norm=True)
    history, L_scale = train_radiance_residual(
        model, L_noisy, L_reconstruction,
        n_epochs=200, lr=1e-3, weight_decay=1e-4,
    )
    print(f"  Final training loss (normalised): {history[-1]:.6e}")
    print(f"  L_scale: {L_scale:.3e}")
    print(f"  Lipschitz upper bound: {model.lipschitz_upper_bound:.4f}")

    hybrid = ForwardHybridTeXModel(wl, L_env, model, T_bounds=(200.0, 500.0))

    # --- In-distribution test ---
    print(f"Evaluating on {N_TEST} in-distribution test samples...")
    _, _, T_test, eps_test, L_test, mat_idx_test = generate_dataset(
        N_TEST, T_TRAIN_RANGE, seed=99
    )

    T_phys_test, _, _ = inversion_ensemble(L_test, wl, L_env,
                                             T_bounds=(200.0, 500.0))
    hyb_results = [hybrid.forward(L_test[i]) for i in range(N_TEST)]
    T_hyb_test = np.array([r["T_est"] for r in hyb_results])
    delta_mags = np.array([np.linalg.norm(r["delta"]) for r in hyb_results])
    print(f"  In-dist ‖δ(L)‖ stats: mean={delta_mags.mean():.3e}"
          f"  median={np.median(delta_mags):.3e}"
          f"  max={delta_mags.max():.3e}")
    print_per_material_breakdown(T_test, T_phys_test, T_hyb_test, mat_idx_test)

    cmp = condition_number_comparison_forward(L_test, hybrid)
    p_kap = cmp["physics_kappas"]
    h_kap = cmp["hybrid_kappas"]
    print_kappa_distribution(p_kap, h_kap, label="κ")

    pd.DataFrame({
        "sample_idx": np.arange(len(p_kap)),
        "kappa_physics": p_kap, "kappa_hybrid": h_kap,
    }).to_csv(os.path.join(DATA_DIR, f"fig10_hybrid_conditioning_{VARIANT}.csv"),
              index=False)
    print(f"Wrote fig10 CSV ({VARIANT})")

    # Figure 11 — both spectra plotted from the *inverse* operator ∂x/∂L so
    # they share units (the forward variant headline is "spectra overlay").
    mid_idx = int(np.nanargmin(np.abs(p_kap - np.nanmedian(p_kap))))
    L_rep = L_test[mid_idx]

    def _physics_inverse_vec(L):
        out = invert_tex(wl, L, L_env, T_bounds=(200.0, 500.0))
        return np.concatenate([[out["T_est"]], out["emissivity_est"]])
    N_in = len(L_rep)
    J_phys_inv = np.zeros((N_in + 1, N_in))
    for i in range(N_in):
        d = max(abs(L_rep[i]) * 0.01, 1e-10)
        Lp = L_rep.copy(); Lp[i] += d
        Lm = L_rep.copy(); Lm[i] -= d
        J_phys_inv[:, i] = (_physics_inverse_vec(Lp) - _physics_inverse_vec(Lm)) / (2 * d)

    J_hyb_inv = hybrid.numerical_output_jacobian(L_rep)
    svd_phys = jacobian_svd(J_phys_inv.T)
    svd_hyb = jacobian_svd(J_hyb_inv.T)

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
          f" T_true={T_test[mid_idx]:.1f} K,"
          f" material={MATERIALS[mat_idx_test[mid_idx]]}):")
    print(f"  {'index':>5}  {'σ_physics':>12}  {'σ_hybrid':>12}")
    for k, (sp, sh) in enumerate(zip(svd_phys["s"], svd_hyb["s"])):
        print(f"  {k+1:>5}  {sp:>12.4e}  {sh:>12.4e}")
    print(f"  κ_physics = {svd_phys['condition_number']:.3e}"
          f"   κ_hybrid = {svd_hyb['condition_number']:.3e}")

    # --- OOD ---
    print("Running distribution shift tests...")
    physics_errs_ood, hybrid_errs_ood, delta_means_ood = [], [], []

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
        delta_norms = np.array([np.linalg.norm(r["delta"]) for r in ood_results])
        physics_errs_ood.append(float(np.mean(np.abs(T_phys_ood - T_ood))))
        hybrid_errs_ood.append(float(np.mean(np.abs(T_hyb_ood - T_ood))))
        delta_means_ood.append(float(delta_norms.mean()))
        print(f"  Offset +{offset:.0f} K:  physics |ΔT|={physics_errs_ood[-1]:.2f} K"
              f"  hybrid |ΔT|={hybrid_errs_ood[-1]:.2f} K"
              f"  mean ‖δ(L)‖={delta_means_ood[-1]:.3e}")

    pd.DataFrame({
        "offset_K": np.asarray(T_OOD_OFFSETS, dtype=float),
        "physics_T_mae": np.array(physics_errs_ood),
        "hybrid_T_mae": np.array(hybrid_errs_ood),
        "mean_delta_norm": np.array(delta_means_ood),
    }).to_csv(os.path.join(DATA_DIR, f"fig12_distribution_shift_{VARIANT}.csv"),
              index=False)
    print(f"Wrote fig12 CSV ({VARIANT})")

    print(f"\nAll CSVs saved to {DATA_DIR}")


if __name__ == "__main__":
    main()
