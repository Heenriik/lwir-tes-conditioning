import numpy as np


def apply_wavelength_shift(wavelengths, delta_lambda):
    """
    Uniform wavelength shift: λ → λ + δλ (meters).

    Args:
        wavelengths (np.ndarray): meters, shape (N,)
        delta_lambda (float): shift in meters (can be negative)

    Returns:
        np.ndarray: shifted wavelengths, shape (N,)
    """
    return wavelengths + delta_lambda


def apply_wavelength_shift_random(wavelengths, sigma_lambda, seed=None):
    """
    Per-band random wavelength shift from N(0, sigma_lambda).

    Args:
        wavelengths (np.ndarray): meters, shape (N,)
        sigma_lambda (float): std of shift in meters
        seed (int or None): random seed

    Returns:
        np.ndarray: perturbed wavelengths (sorted), shape (N,)
    """
    rng = np.random.default_rng(seed)
    shifts = rng.normal(0.0, sigma_lambda, size=wavelengths.shape)
    return np.sort(wavelengths + shifts)


def apply_fwhm_broadening(L, wavelengths, fwhm_original, fwhm_perturbed):
    """
    Simulate spectral FWHM broadening via Gaussian convolution.

    Args:
        L (np.ndarray): radiance, shape (N,)
        wavelengths (np.ndarray): meters, shape (N,) — must be uniform spacing
        fwhm_original (float): original FWHM in meters
        fwhm_perturbed (float): perturbed FWHM in meters

    Returns:
        np.ndarray: broadened radiance, shape (N,)
    """
    N = len(wavelengths)
    if N < 2:
        return L.copy()
    band_spacing = (wavelengths[-1] - wavelengths[0]) / (N - 1)
    if band_spacing < 1e-30:
        return L.copy()
    delta_fwhm = max(0.0, fwhm_perturbed - fwhm_original)
    if delta_fwhm < 1e-30:
        return L.copy()
    sigma_pixels = delta_fwhm / (2.355 * band_spacing)
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(L.astype(float), sigma=sigma_pixels)


def calibration_error_study(wavelengths, emissivity, T, L_env,
                             delta_lambda_range, n_samples=100):
    """
    Sweep wavelength shifts and measure κ and inversion reconstruction error.

    Produces Figure 6 data.

    Args:
        wavelengths (np.ndarray): meters, shape (N,)
        emissivity (np.ndarray): shape (N,)
        T (float): true surface temperature in Kelvin
        L_env (np.ndarray): shape (N,)
        delta_lambda_range (np.ndarray): shift values in meters to test

    Returns:
        dict:
            delta_lambda_values, condition_numbers, reconstruction_errors
    """
    from ..forward_model.radiance_model import compute_radiance
    from ..inversion.jacobian import compute_jacobian, jacobian_svd
    from ..inversion.tex_inversion import invert_tex

    kappas = []
    T_errors = []

    for delta in delta_lambda_range:
        wl_shifted = apply_wavelength_shift(wavelengths, delta)
        J = compute_jacobian(wl_shifted, T, emissivity, L_env)
        svd = jacobian_svd(J)
        kappas.append(svd["condition_number"])

        L_obs = compute_radiance(wl_shifted, T, emissivity, L_env)
        result = invert_tex(wl_shifted, L_obs, L_env,
                             initial_guess=None, T_bounds=(200.0, 400.0))
        T_errors.append(abs(result["T_est"] - T))

    return {
        "delta_lambda_values": np.array(delta_lambda_range),
        "condition_numbers": np.array(kappas),
        "reconstruction_errors": np.array(T_errors),
    }


def combined_perturbation_analysis(wavelengths, emissivity, T, L_env,
                                    nedt_values, delta_lambda_values,
                                    n_trials=200, seed=42):
    """
    2-D Monte Carlo grid combining NEDT noise and wavelength shift.

    Produces Figure 7 data.

    Args:
        wavelengths (np.ndarray): meters, shape (N,)
        emissivity (np.ndarray): shape (N,)
        T (float): true temperature in Kelvin
        L_env (np.ndarray): shape (N,)
        nedt_values (np.ndarray): NEDT levels in Kelvin
        delta_lambda_values (np.ndarray): shift levels in meters
        n_trials (int): Monte Carlo trials per (NEDT, δλ) cell
        seed (int): base random seed

    Returns:
        dict:
            nedt_values, delta_lambda_values,
            mean_errors (len_nedt, len_dlam),
            std_errors (len_nedt, len_dlam)
    """
    from ..forward_model.radiance_model import compute_radiance
    from ..inversion.tex_inversion import invert_tex
    from .noise import add_nedt_noise

    n_nedt = len(nedt_values)
    n_dlam = len(delta_lambda_values)
    mean_errors = np.zeros((n_nedt, n_dlam))
    std_errors = np.zeros((n_nedt, n_dlam))
    rng = np.random.default_rng(seed)

    for i, nedt in enumerate(nedt_values):
        for j, delta in enumerate(delta_lambda_values):
            wl_shifted = apply_wavelength_shift(wavelengths, delta)
            L_true = compute_radiance(wl_shifted, T, emissivity, L_env)
            T_errs = []
            for _ in range(n_trials):
                trial_seed = int(rng.integers(0, 2**31))
                L_noisy = add_nedt_noise(L_true, wl_shifted, T,
                                         nedt_K=nedt, seed=trial_seed)
                result = invert_tex(wl_shifted, L_noisy, L_env,
                                     T_bounds=(200.0, 400.0))
                T_errs.append(abs(result["T_est"] - T))
            mean_errors[i, j] = np.mean(T_errs)
            std_errors[i, j] = np.std(T_errs)

    return {
        "nedt_values": np.array(nedt_values),
        "delta_lambda_values": np.array(delta_lambda_values),
        "mean_errors": mean_errors,
        "std_errors": std_errors,
    }
