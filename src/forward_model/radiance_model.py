import numpy as np
from .planck import planck_radiance


def compute_radiance(wavelengths, temperature, emissivity, L_env):
    """
    LWIR radiance model: L = ε·B(T) + (1-ε)·L_env

    Args:
        wavelengths (np.ndarray): meters, shape (N,)
        temperature (float): Kelvin
        emissivity (np.ndarray): dimensionless [0,1], shape (N,)
        L_env (float or np.ndarray): downwelling radiance, scalar or shape (N,)

    Returns:
        np.ndarray: shape (N,)
    """
    B = planck_radiance(wavelengths, temperature)
    return emissivity * B + (1 - emissivity) * L_env


def blackbody_environment(wavelengths, T_env):
    """
    Blackbody downwelling environment radiance at temperature T_env.

    Convenient alternative to a flat or zero L_env assumption.

    Args:
        wavelengths (np.ndarray): meters, shape (N,)
        T_env (float): environment temperature in Kelvin

    Returns:
        np.ndarray: shape (N,)
    """
    return planck_radiance(wavelengths, T_env)


def compute_radiance_batch(wavelengths, emissivity_batch, T_batch, L_env):
    """
    Vectorised radiance over M samples.

    Args:
        wavelengths (np.ndarray): meters, shape (N,)
        emissivity_batch (np.ndarray): shape (M, N)
        T_batch (np.ndarray): Kelvin, shape (M,)
        L_env (np.ndarray): shape (N,) or (M, N)

    Returns:
        np.ndarray: shape (M, N)
    """
    M = T_batch.shape[0]
    # B_batch: (M, N) — one Planck spectrum per temperature
    B_batch = np.stack([planck_radiance(wavelengths, T) for T in T_batch], axis=0)
    L_env_2d = np.broadcast_to(np.asarray(L_env), (M, len(wavelengths)))
    return emissivity_batch * B_batch + (1 - emissivity_batch) * L_env_2d
