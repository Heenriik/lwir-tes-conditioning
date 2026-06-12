import numpy as np
from ..forward_model import (
    planck_radiance,
    analytical_planck_derivative,
    numerical_planck_derivative,
)


def compute_jacobian(wavelengths, temperature, emissivity, L_env):
    """
    Analytical Jacobian of L w.r.t. x = [T, ε_1, ..., ε_N].

    J shape: (N, N+1)
      J[:, 0]   = ε · dB/dT          (temperature column)
      J[:, 1:]  = diag(B - L_env)    (emissivity columns, diagonal)

    Derived by differentiating the forward model L_i = ε_i·B(λ_i,T) + (1-ε_i)·L_env(λ_i)
    (Kirchhoff's law of thermal radiation) w.r.t. each parameter:
        ∂L_i/∂T    = ε_i · dB/dT(λ_i, T)
        ∂L_i/∂ε_j = [B(λ_i,T) - L_env(λ_i)] · δ_ij

    Forward model reference:
        Gillespie et al. (1998) "A temperature and emissivity separation algorithm
        for ASTER images", IEEE Trans. Geosci. Remote Sens. 36(4), 1113-1126.

    Args:
        wavelengths (np.ndarray): meters, shape (N,)
        temperature (float): Kelvin
        emissivity (np.ndarray): dimensionless, shape (N,)
        L_env (float or np.ndarray): downwelling radiance, scalar or (N,)

    Returns:
        np.ndarray: shape (N, N+1)
    """
    N = len(wavelengths)
    L_env = np.broadcast_to(np.asarray(L_env, dtype=float), (N,))

    dB_dT = analytical_planck_derivative(wavelengths, temperature)
    B = planck_radiance(wavelengths, temperature)

    J = np.zeros((N, 1 + N))
    J[:, 0] = emissivity * dB_dT
    J[:, 1:] = np.diag(B - L_env)
    return J


def compute_jacobian_numerical(wavelengths, temperature, emissivity, L_env,
                               eps_T=0.01, eps_e=1e-4):
    """
    Central-difference Jacobian — used to validate compute_jacobian.

    Args:
        eps_T (float): temperature perturbation in K
        eps_e (float): emissivity perturbation (dimensionless)

    Returns:
        np.ndarray: shape (N, N+1)
    """
    from ..forward_model.radiance_model import compute_radiance

    N = len(wavelengths)
    J = np.zeros((N, 1 + N))

    # Temperature column
    L_plus = compute_radiance(wavelengths, temperature + eps_T, emissivity, L_env)
    L_minus = compute_radiance(wavelengths, temperature - eps_T, emissivity, L_env)
    J[:, 0] = (L_plus - L_minus) / (2 * eps_T)

    # Emissivity columns
    for k in range(N):
        e_plus = emissivity.copy()
        e_minus = emissivity.copy()
        e_plus[k] += eps_e
        e_minus[k] -= eps_e
        L_plus = compute_radiance(wavelengths, temperature, e_plus, L_env)
        L_minus = compute_radiance(wavelengths, temperature, e_minus, L_env)
        J[:, k + 1] = (L_plus - L_minus) / (2 * eps_e)

    return J


def jacobian_svd(J):
    """
    Full SVD analysis of the Jacobian.

    Condition number:
        κ(J) = σ_max / σ_min
    Standard definition from numerical linear algebra.
    Ref: Golub & Van Loan (2013) "Matrix Computations", 4th ed., Johns Hopkins UP.
         §2.6.2 p.87 eq. (2.6.3) defines κ(A) = ‖A‖·‖A⁻¹‖;
         eq. (2.6.5) p.88 gives the 2-norm form κ₂(A) = σ_max(A)/σ_min(A).

    Numerical-zero guard (s_min < 1e-12 * s_max → κ = inf):
    Treats J as effectively rank-deficient when the smallest singular value falls
    below 1e-12 of the largest. This is a conservative version of the threshold
    used internally by numpy.linalg.matrix_rank (which uses rcond * max(m,n) * σ_max,
    approximately 1e-14 for a 50x51 double-precision matrix). The 1e-12 factor provides
    a safety margin for near-singular configurations without over-penalising well-scaled
    problems. Standard numerical analysis practice; see Golub & Van Loan ibid., §5.5
    "The Rank-Deficient Least Squares Problem", p. 288.

    Args:
        J (np.ndarray): shape (N, N+1)

    Returns:
        dict with keys: U, s, Vt, condition_number, rank
    """
    U, s, Vt = np.linalg.svd(J, full_matrices=True)
    s_min = s[-1]
    s_max = s[0]
    if s_min < 1e-12 * s_max:
        condition_number = np.inf
    else:
        condition_number = s_max / s_min
    rank = int(np.linalg.matrix_rank(J))
    return {"U": U, "s": s, "Vt": Vt, "condition_number": condition_number, "rank": rank}


def validate_jacobian(wavelengths, temperature, emissivity, L_env, rtol=1e-6):
    """
    Compare analytical and numerical Jacobians as a regression smoke test.

    Threshold rtol=1e-6 (1 ppm):
    Empirically the analytical (sympy-derived) and central-difference numerical
    Jacobians agree to ~1e-9 across all materials and band counts (see notebook
    01 §5). The 1e-6 threshold leaves three orders of magnitude headroom over
    that floating-point floor while remaining tight enough that real coding
    errors (off-by-one column indices, sign flips, wrong matrix structure)
    produce relative errors of order 1 and trigger an obvious failure.

    Returns:
        dict with max_relative_error (float) and passed (bool)
    """
    J_analytic = compute_jacobian(wavelengths, temperature, emissivity, L_env)
    J_numeric = compute_jacobian_numerical(wavelengths, temperature, emissivity, L_env)

    denom = np.abs(J_analytic)
    denom[denom < 1e-30] = 1e-30
    rel_err = np.abs(J_analytic - J_numeric) / denom
    max_err = float(np.max(rel_err))
    return {"max_relative_error": max_err, "passed": max_err < rtol}


def parameter_sensitivity(J):
    """
    Per-parameter identifiability strength: L2 norm of each Jacobian column.

    For a linear system y = J·x, the L2 norm of column k of J equals the gain of
    the observations to a unit perturbation in parameter k. A small norm means
    measurements barely respond to that parameter — it is poorly identifiable.

    Note: T-sensitivity ∝ mean(ε) because J[:,0] = ε·dB/dT.
          ε-sensitivity = |B(λ_k)-L_env(λ_k)| (thermal contrast) — independent of ε.

    Returns:
        dict with temperature_sensitivity (float) and emissivity_sensitivity (np.ndarray shape (N,))
    """
    N = J.shape[1] - 1
    T_sens = float(np.linalg.norm(J[:, 0]))
    e_sens = np.array([np.linalg.norm(J[:, k + 1]) for k in range(N)])
    return {"temperature_sensitivity": T_sens, "emissivity_sensitivity": e_sens}


def scale_jacobian(J, T_scale, eps_scale):
    """
    Column-scale the Jacobian by prior 1σ parameter uncertainties.

    Raw J has unit-mismatched columns: J[:,0] is per-Kelvin while J[:,1:] is
    per-(unit emissivity). Multiplying each column by its prior 1σ uncertainty
    converts perturbations into dimensionless "1σ prior change" units, making
    κ(J_scaled) a physically meaningful identifiability metric.

    This corresponds to evaluating κ on J·C_M^{1/2}, where C_M is the prior
    covariance matrix on the model parameters. Tarantola (2005) §3.2 introduces
    C_M generally as the metric of the model-space norm in the misfit function
    (eq. 3.32, p. 64). When the prior uncertainties on different parameters are
    uncorrelated — which is our default assumption (T and each ε_i have
    independent priors) — Tarantola's footnote 46 (p. 64) gives the diagonal
    form (C_M)^αβ = (σ_M^α)² δ^αβ. With σ_T on the temperature axis and σ_ε on
    each emissivity axis this becomes C_M = diag(σ_T², σ_ε² I_N), and
    C_M^{1/2} = diag(σ_T, σ_ε I_N) is exactly the column scaling applied below.

    The mismatch is not just stylistic. In §3.4.1 Tarantola contrasts a correct
    gradient step that includes the inverse metric tensor g^αβ (eq. 3.80) with
    a pseudoalgorithm that drops it (eq. 3.81). The footnote on eq. 3.81 (p. 76)
    notes that the pseudoalgorithm is "dimensionally wrong (if the parameters
    have different physical dimensions), and the algorithm is coordinate
    dependent". In the least-squares/Gaussian setting of §3.2 the natural
    metric IS the inverse prior covariance — i.e., g^αβ ↔ C_M and g_αβ ↔ C_M⁻¹
    — so dropping C_M-weighting reproduces the same dimensional problem at the
    level of κ. Computing κ on J·C_M^{1/2} works in prior-whitened coordinates
    where the metric is the identity, restoring coordinate invariance. This
    matters precisely in our case: T is in Kelvin, ε is dimensionless.

    Ref:
        Tarantola, A. (2005) "Inverse Problem Theory and Methods for Model
        Parameter Estimation", SIAM.
            §3.2 — misfit function with general C_M (eq. 3.32, p. 64)
            §3.2 footnote 46, p. 64 — diagonal C_M for uncorrelated priors
            §3.4.1 footnote on eq. 3.81, p. 76 — dimensional invariance argument
        URL: https://www.geologie.ens.fr/~jolivet/Research_files/Tarantola.pdf

    Args:
        J (np.ndarray): shape (N, N+1)
        T_scale (float): prior 1σ on temperature, in Kelvin (e.g. 5.0)
        eps_scale (float): prior 1σ on emissivity, dimensionless (e.g. 0.05)

    Returns:
        np.ndarray: shape (N, N+1), column-scaled copy of J
    """
    J_scaled = J.copy()
    J_scaled[:, 0] *= T_scale
    J_scaled[:, 1:] *= eps_scale
    return J_scaled
