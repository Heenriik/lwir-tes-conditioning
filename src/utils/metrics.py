import numpy as np


def reconstruction_error(T_true, T_est, emissivity_true, emissivity_est):
    """
    Temperature and emissivity reconstruction error.

    Args:
        T_true (float or np.ndarray): true temperature(s) in Kelvin
        T_est (float or np.ndarray): estimated temperature(s) in Kelvin
        emissivity_true (np.ndarray): shape (N,) or (M, N)
        emissivity_est (np.ndarray): same shape as emissivity_true

    Returns:
        dict:
            T_rmse: scalar K
            emissivity_rmse: scalar
            emissivity_rmse_per_band: np.ndarray shape (N,)
            combined_error: weighted combination T_rmse/10 + emissivity_rmse/0.1
    """
    T_true = np.asarray(T_true, dtype=float)
    T_est = np.asarray(T_est, dtype=float)
    emissivity_true = np.asarray(emissivity_true, dtype=float)
    emissivity_est = np.asarray(emissivity_est, dtype=float)

    T_rmse = float(np.sqrt(np.mean((T_true - T_est) ** 2)))
    e_diff = emissivity_true - emissivity_est
    emissivity_rmse = float(np.sqrt(np.mean(e_diff ** 2)))
    if emissivity_true.ndim == 1:
        per_band = np.abs(e_diff)
    else:
        per_band = np.sqrt(np.mean(e_diff ** 2, axis=0))

    combined = T_rmse / 10.0 + emissivity_rmse / 0.1

    return {
        "T_rmse": T_rmse,
        "emissivity_rmse": emissivity_rmse,
        "emissivity_rmse_per_band": per_band,
        "combined_error": combined,
    }


def temperature_bias(T_true_batch, T_est_batch):
    """
    Mean signed temperature error (bias) and RMSE over a batch.

    Args:
        T_true_batch (np.ndarray): shape (M,)
        T_est_batch (np.ndarray): shape (M,)

    Returns:
        dict: bias_K (float), rmse_K (float)
    """
    diff = np.asarray(T_est_batch) - np.asarray(T_true_batch)
    return {
        "bias_K": float(np.mean(diff)),
        "rmse_K": float(np.sqrt(np.mean(diff ** 2))),
    }


def spectral_angle_mapper(emissivity_true, emissivity_est):
    """
    Spectral angle between true and estimated emissivity vectors.

    Args:
        emissivity_true (np.ndarray): shape (N,) or (M, N)
        emissivity_est (np.ndarray): same shape

    Returns:
        float or np.ndarray: angle(s) in radians
    """
    e_true = np.asarray(emissivity_true, dtype=float)
    e_est = np.asarray(emissivity_est, dtype=float)
    dot = np.sum(e_true * e_est, axis=-1)
    norm_true = np.linalg.norm(e_true, axis=-1)
    norm_est = np.linalg.norm(e_est, axis=-1)
    cos_theta = np.clip(dot / (norm_true * norm_est + 1e-30), -1.0, 1.0)
    return np.arccos(cos_theta)


def bootstrap_error_ci(T_true, T_est_batch, confidence=0.95, n_bootstrap=1000,
                        seed=None):
    """
    Bootstrap confidence interval on mean absolute temperature error.

    Args:
        T_true (float): true temperature in Kelvin
        T_est_batch (np.ndarray): shape (M,) from Monte Carlo trials
        confidence (float): e.g. 0.95 for 95% CI
        n_bootstrap (int): bootstrap resamples
        seed (int or None): random seed

    Returns:
        dict: mean (K), ci_lower (K), ci_upper (K)
    """
    rng = np.random.default_rng(seed)
    errors = np.abs(np.asarray(T_est_batch) - T_true)
    means = []
    M = len(errors)
    for _ in range(n_bootstrap):
        sample = rng.choice(errors, size=M, replace=True)
        means.append(np.mean(sample))
    means = np.sort(means)
    alpha = 1 - confidence
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return {"mean": float(np.mean(errors)), "ci_lower": float(lo), "ci_upper": float(hi)}


def condition_to_error_correlation(condition_numbers, reconstruction_errors):
    """
    Pearson and Spearman correlation between κ values and reconstruction errors.

    Validates that κ is a good predictor of inversion difficulty.

    Args:
        condition_numbers (np.ndarray): shape (M,)
        reconstruction_errors (np.ndarray): shape (M,)

    Returns:
        dict: pearson_r, pearson_p, spearman_r, spearman_p
    """
    kappas = np.asarray(condition_numbers, dtype=float)
    errors = np.asarray(reconstruction_errors, dtype=float)

    # Remove inf/nan pairs
    mask = np.isfinite(kappas) & np.isfinite(errors)
    kappas = kappas[mask]
    errors = errors[mask]

    if len(kappas) < 3:
        return {"pearson_r": np.nan, "pearson_p": np.nan,
                "spearman_r": np.nan, "spearman_p": np.nan}

    from scipy.stats import pearsonr, spearmanr  # lazy import
    pr, pp = pearsonr(np.log1p(kappas), errors)
    sr, sp = spearmanr(kappas, errors)
    return {
        "pearson_r": float(pr), "pearson_p": float(pp),
        "spearman_r": float(sr), "spearman_p": float(sp),
    }
