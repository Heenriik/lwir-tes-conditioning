import numpy as np
import torch


class HybridTeXModel:
    """
    Hybrid TeX inversion: x̂ = f_TeX(L) + r_θ(L)

    Combines a physics-based inversion with a Lipschitz-constrained neural
    residual corrector. The forward method returns the combined estimate;
    numerical_output_jacobian computes the Jacobian of x̂ w.r.t. L via
    central differences (authoritative for conditioning analysis).
    """

    def __init__(self, wavelengths, L_env, residual_model, T_bounds=(200.0, 400.0)):
        """
        Args:
            wavelengths (np.ndarray): meters, shape (N,)
            L_env (np.ndarray): downwelling radiance, shape (N,)
            residual_model (ResidualMLP): trained residual model
            T_bounds (tuple): (T_min, T_max) clipping bounds in Kelvin
        """
        self.wavelengths = np.asarray(wavelengths, dtype=float)
        self.L_env = np.asarray(L_env, dtype=float)
        self.residual_model = residual_model
        self.T_bounds = T_bounds
        self.N = len(wavelengths)

    def forward(self, L):
        """
        Predict [T, ε_1,...,ε_N] from measured radiance.

        Args:
            L (np.ndarray): measured radiance, shape (N,)

        Returns:
            dict with:
                T_est, emissivity_est (N,)    — final combined estimate
                physics_T, physics_emissivity — physics-only estimate
                residual_correction (N+1,)    — raw neural correction (unnormalised)
        """
        from ..inversion.tex_inversion import invert_tex

        phys = invert_tex(self.wavelengths, L, self.L_env, T_bounds=self.T_bounds)
        T_phys = phys["T_est"]
        eps_phys = phys["emissivity_est"]

        device = next(self.residual_model.parameters()).device
        L_tensor = torch.tensor(L.astype(np.float32), device=device)
        self.residual_model.eval()
        with torch.no_grad():
            correction = self.residual_model(L_tensor).cpu().numpy()  # (N+1,)

        # correction[0] is in normalised T units (T/300); convert back
        T_scale = 300.0
        T_est = np.clip(T_phys + correction[0] * T_scale, *self.T_bounds)
        eps_est = np.clip(eps_phys + correction[1:], 0.01, 1.0)

        return {
            "T_est": float(T_est),
            "emissivity_est": eps_est,
            "physics_T": T_phys,
            "physics_emissivity": eps_phys,
            "residual_correction": correction,
        }

    def numerical_output_jacobian(self, L, eps_frac=0.01):
        """
        Jacobian of x̂ w.r.t. L via central differences.

        J_hybrid[k, i] = ∂x̂_k / ∂L_i

        Shape: (N+1, N) --> N+1 output parameters x N input bands.

        This is the authoritative hybrid Jacobian for conditioning analysis.
        J_physics and J_residual have incompatible shapes and cannot be directly
        added; use this numerical version for Figures 10 and 11.

        Args:
            L (np.ndarray): radiance, shape (N,)
            eps_frac (float): perturbation fraction of each L_i

        Returns:
            np.ndarray: shape (N+1, N)
        """
        N = self.N
        x0 = self._forward_as_vector(L)
        J = np.zeros((N + 1, N))

        for i in range(N):
            delta = max(abs(L[i]) * eps_frac, 1e-10)
            L_plus = L.copy()
            L_minus = L.copy()
            L_plus[i] += delta
            L_minus[i] -= delta
            x_plus = self._forward_as_vector(L_plus)
            x_minus = self._forward_as_vector(L_minus)
            J[:, i] = (x_plus - x_minus) / (2 * delta)

        return J

    def _forward_as_vector(self, L):
        result = self.forward(L)
        return np.concatenate([[result["T_est"]], result["emissivity_est"]])

    def physics_jacobian(self, L):
        """
        Analytical Jacobian of the physics forward model at the estimated point.

        Shape: (N, N+1)

        Args:
            L (np.ndarray): radiance, shape (N,)

        Returns:
            np.ndarray: shape (N, N+1)
        """
        from ..inversion.jacobian import compute_jacobian
        from ..inversion.tex_inversion import invert_tex

        phys = invert_tex(self.wavelengths, L, self.L_env, T_bounds=self.T_bounds)
        return compute_jacobian(
            self.wavelengths, phys["T_est"], phys["emissivity_est"], self.L_env
        )


def condition_number_comparison(L_test_batch, hybrid_model, T_true_batch=None):
    """
    Compute κ(J_physics) and κ(J_hybrid) for each test sample.

    Args:
        L_test_batch (np.ndarray): shape (M, N)
        hybrid_model (HybridTeXModel): fitted hybrid model
        T_true_batch (np.ndarray or None): true temperatures (unused; for logging)

    Returns:
        dict:
            physics_kappas (M,)
            hybrid_kappas (M,)
            ratio (M,)  = κ_physics / κ_hybrid  (>1 means hybrid improved)
    """
    from ..inversion.jacobian import jacobian_svd

    M = len(L_test_batch)
    physics_kappas = np.zeros(M)
    hybrid_kappas = np.zeros(M)

    for i, L in enumerate(L_test_batch):
        J_phys = hybrid_model.physics_jacobian(L)
        svd_phys = jacobian_svd(J_phys)
        physics_kappas[i] = svd_phys["condition_number"]

        J_hyb = hybrid_model.numerical_output_jacobian(L)
        # J_hybrid is (N+1, N); compute condition number of the output Jacobian directly
        svd_hyb = jacobian_svd(J_hyb.T)  # transpose to (N, N+1) for consistency
        hybrid_kappas[i] = svd_hyb["condition_number"]

    ratio = np.where(hybrid_kappas > 0, physics_kappas / hybrid_kappas, np.nan)
    return {
        "physics_kappas": physics_kappas,
        "hybrid_kappas": hybrid_kappas,
        "ratio": ratio,
    }
