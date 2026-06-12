import numpy as np


def add_noise(L, snr_db, seed=None):
    """
    Add Gaussian radiance noise scaled to a target SNR.

    sigma_i = L_i / 10^(snr_db/20)
    σ ∝ L

    Args:
        L (np.ndarray): radiance, shape (N,)
        snr_db (float): signal-to-noise ratio in dB
        seed (int or None): random seed

    Returns:
        np.ndarray: noisy radiance, shape (N,)
    """
    rng = np.random.default_rng(seed)
    sigma = np.abs(L) / (10 ** (snr_db / 20))
    return L + rng.normal(0.0, sigma)


# Backwards-compatible alias
add_gaussian_noise = add_noise


def add_nedt_noise(L, wavelengths, T, nedt_K=0.1, seed=None):
    """
    Add Gaussian noise calibrated to a target Noise Equivalent Differential Temperature (NEDT).

    sigma_i = NEDT_K * dB/dT(lambda_i, T)
    σ ∝ dB/dT

    NEDT is defined as the temperature difference that produces a detector output
    equal to the RMS noise: NEDT = σ / (dB/dT), so σ = NEDT · dB/dT.
    This is physically grounded: a detector with NEDT = 0.1 K has per-band
    radiance noise equal to 0.1 x the Planck derivative at that wavelength.

    Refs:
        Holst (1998) "Testing and Evaluation of Infrared Imaging Systems", SPIE Press,
            §3.2 — formal NEDT definition and conversion to radiance noise.
        Shao et al. (2020) Sensors 20(7):2109, doi:10.3390/s20072109 — uses this
            exact σ = NEDT·(dB/dT) formulation for LWIR band noise simulation.

    Args:
        L (np.ndarray): radiance, shape (N,)
        wavelengths (np.ndarray): meters, shape (N,)
        T (float): surface temperature in Kelvin (operating point for derivative)
        nedt_K (float): noise equivalent differential temperature in Kelvin
        seed (int or None): random seed

    Returns:
        np.ndarray: noisy radiance, shape (N,)
    """
    from ..forward_model.planck import analytical_planck_derivative

    rng = np.random.default_rng(seed)
    dB_dT = analytical_planck_derivative(wavelengths, T)
    sigma = nedt_K * dB_dT
    return L + rng.normal(0.0, sigma)


def add_noise_batch(L_batch, wavelengths, T_batch, nedt_K=0.1, seed=None):
    """
    Vectorised NEDT noise over M samples.

    Args:
        L_batch (np.ndarray): shape (M, N)
        wavelengths (np.ndarray): meters, shape (N,)
        T_batch (np.ndarray): surface temperatures in Kelvin, shape (M,)
        nedt_K (float): NEDT in Kelvin
        seed (int or None): random seed for reproducibility

    Returns:
        np.ndarray: noisy radiance, shape (M, N)
    """
    from ..forward_model.planck import analytical_planck_derivative

    rng = np.random.default_rng(seed)
    M, N = L_batch.shape
    noise = np.zeros_like(L_batch)
    for i in range(M):
        dB_dT = analytical_planck_derivative(wavelengths, T_batch[i])
        sigma = nedt_K * dB_dT
        noise[i] = rng.normal(0.0, sigma)
    return L_batch + noise


def snr_from_nedt(nedt_K, wavelengths, T, L):
    """
    Convert NEDT to per-band SNR in dB.

    snr_i = 20 * log10(L_i / sigma_i)  where  sigma_i = NEDT * dB/dT(lambda_i, T)

    The factor 20 (not 10) applies because radiance is a linear amplitude quantity;
    SNR_dB = 20·log10(signal/noise) is the standard amplitude-ratio definition.
    Ref: IEEE Std 1241-2010 (for amplitude-type SNR); standard signal processing.

    Args:
        nedt_K (float): NEDT in Kelvin
        wavelengths (np.ndarray): meters, shape (N,)
        T (float): Kelvin
        L (np.ndarray): radiance, shape (N,)

    Returns:
        np.ndarray: SNR in dB, shape (N,)
    """
    from ..forward_model.planck import analytical_planck_derivative

    dB_dT = analytical_planck_derivative(wavelengths, T)
    sigma = nedt_K * dB_dT
    sigma = np.where(sigma < 1e-30, 1e-30, sigma)
    L_safe = np.where(np.abs(L) < 1e-30, 1e-30, np.abs(L))
    return 20 * np.log10(L_safe / sigma)
