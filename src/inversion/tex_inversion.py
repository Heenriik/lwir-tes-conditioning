import numpy as np
from scipy.optimize import least_squares

from ..forward_model.radiance_model import compute_radiance


def invert_tex(wavelengths, L_obs, L_env, initial_guess=None,
               T_bounds=(200.0, 400.0), reg_lambda=0.0):
    """
    Solve the TeX inverse problem: find [T, ε_1,...,ε_N] given L_obs.

    Solver: scipy least_squares with method='trf' (Trust Region Reflective).
    TRF natively handles box constraints (bounds on each parameter), unlike
    Levenberg-Marquardt which requires a separate penalty formulation.
    Ref: Branch, Coleman & Li (1999) "A Subspace, Interior, and Conjugate Gradient
         Method for Large-Scale Bound-Constrained Minimization Problems",
         https://epubs.siam.org/doi/epdf/10.1137/S1064827595289108

    T_bounds (200, 400) K:
    Physical bounds covering virtually all natural Earth surface temperatures.
    Consistent with the operational range used in ASTER TES retrievals.
    Ref: Gillespie et al. (1998) IEEE Trans. Geosci. Remote Sens. 36(4), 1113-1126.

    ε_bounds (0.01, 1.0):
    Physical bounds: ε = 0 is a perfect reflector (no natural surface), ε = 1 is a
    perfect blackbody. Lower bound 0.01 accommodates highly polished metals (brass in
    the HADAR library reaches ε ≈ 0.11). Most natural surfaces have ε > 0.7 in LWIR.

    initial_guess T=300 K, ε=0.9:
    Standard initialisation: 300 K ≈ room temperature / typical Earth surface;
    ε=0.9 ≈ mean emissivity of natural surfaces in LWIR. No specific citation —
    this is a conventional starting point in TES literature.

    reg_lambda (Tikhonov regularization):
    When reg_lambda > 0, the residual vector is augmented from r = (L - L_obs)
    to r' = [L - L_obs;  reg_lambda · x]. Solving the augmented least-squares
    problem ‖r'‖² → min is mathematically equivalent to the canonical Tikhonov
    minimisation
        min_x  ‖A x - b‖² + reg_lambda² · ‖L x‖²
    with L = I (no derivative seminorm), as defined in Hansen (1992) eq. (5),
    p. 563. The equivalence comes from [A; λI]^T [A; λI] = A^T A + λ²I and
    [A; λI]^T [b; 0] = A^T b yielding the same normal equations
    (A^T A + λ²I) x = A^T b — solving the augmented least-squares system is the
    standard numerical realisation of the Tikhonov problem.
    Ref: Hansen, P.C. (1992) "Analysis of Discrete Ill-Posed Problems by Means
        of the L-Curve", SIAM Review 34(4), 561-580. Tikhonov problem stated
        as equation (5), p. 563.
        https://epubs.siam.org/doi/epdf/10.1137/1034115

        
    Args:
        wavelengths (np.ndarray): meters, shape (N,)
        L_obs (np.ndarray): measured radiance, shape (N,)
        L_env (np.ndarray): downwelling radiance, shape (N,)
        initial_guess (np.ndarray or None): shape (N+1,); defaults to T=300 K, ε=0.9
        T_bounds (tuple): (T_min, T_max) in Kelvin
        reg_lambda (float): Tikhonov regularization weight; 0 disables it

    Returns:
        dict: T_est, emissivity_est (N,), residual_norm, success, n_iter
    """
    N = len(wavelengths)

    if initial_guess is None:
        initial_guess = np.concatenate([[300.0], np.full(N, 0.9)])


    lower = np.concatenate([[T_bounds[0]], np.full(N, 0.01)])
    upper = np.concatenate([[T_bounds[1]], np.full(N, 1.0)])

    def residual(x):
        T = x[0]
        eps = x[1:]
        r = compute_radiance(wavelengths, T, eps, L_env) - L_obs
        if reg_lambda > 0:
            r = np.concatenate([r, reg_lambda * x])
        return r

    result = least_squares(residual, initial_guess, bounds=(lower, upper), method='trf')

    return {
        "T_est": float(result.x[0]),
        "emissivity_est": result.x[1:],
        "residual_norm": float(np.linalg.norm(result.fun)),
        "success": result.success,
        "n_iter": result.nfev,
    }


def inversion_ensemble(L_batch, wavelengths, L_env, T_bounds=(200.0, 400.0),
                       reg_lambda=0.0, n_jobs=-1):
    """
    Run invert_tex over a batch of radiance measurements.

    Args:
        L_batch (np.ndarray): shape (M, N)
        wavelengths (np.ndarray): meters, shape (N,)
        L_env (np.ndarray): shape (N,)
        n_jobs (int): joblib parallel workers (-1 = all CPUs)

    Returns:
        tuple: T_est (M,), emissivity_est (M, N), residuals (M,)
    """
    try:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs)(
            delayed(invert_tex)(wavelengths, L_batch[i], L_env,
                                T_bounds=T_bounds, reg_lambda=reg_lambda)
            for i in range(len(L_batch))
        )
    except ImportError:
        results = [
            invert_tex(wavelengths, L_batch[i], L_env,
                       T_bounds=T_bounds, reg_lambda=reg_lambda)
            for i in range(len(L_batch))
        ]

    T_est = np.array([r["T_est"] for r in results])
    emissivity_est = np.stack([r["emissivity_est"] for r in results], axis=0)
    residuals = np.array([r["residual_norm"] for r in results])
    return T_est, emissivity_est, residuals
