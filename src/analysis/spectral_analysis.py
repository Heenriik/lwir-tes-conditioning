import numpy as np

from ..inversion.jacobian import compute_jacobian, jacobian_svd
from ..simulation.generator import generate_wavelength_sets, get_material_emissivity


def optimal_band_selection(wavelength_candidates, emissivity_func, T,
                            L_env_func, n_bands_target):
    """
    Greedy κ-optimal band selection.

    Iteratively picks the wavelength from the candidate pool that maximally
    reduces κ(J). Starts by selecting the single best wavelength, then adds
    one at a time. O(N · M) Jacobian evaluations where N = n_bands_target
    and M = len(wavelength_candidates).

    Args:
        wavelength_candidates (np.ndarray): pool of candidate wavelengths (m)
        emissivity_func (callable): wavelengths → emissivity (N,)
        T (float): Kelvin
        L_env_func (callable): wavelengths → L_env (N,)
        n_bands_target (int): desired band count

    Returns:
        dict: selected_wavelengths (np.ndarray), achieved_condition_number (float)
    """
    candidates = np.sort(wavelength_candidates)

    def _kappa(wl_subset):
        wl = np.sort(wl_subset)
        J = compute_jacobian(wl, T, emissivity_func(wl), L_env_func(wl))
        return jacobian_svd(J)["condition_number"]

    selected = []
    remaining = list(candidates)

    # Seed: pick the single band giving the lowest κ
    best_kappa = np.inf
    best_wl = remaining[0]
    for wl in remaining:
        k = _kappa(np.array([wl]))
        if k < best_kappa:
            best_kappa = k
            best_wl = wl
    selected.append(best_wl)
    remaining.remove(best_wl)

    while len(selected) < n_bands_target and remaining:
        best_kappa = np.inf
        best_add = remaining[0]
        for wl in remaining:
            k = _kappa(np.array(selected + [wl]))
            if k < best_kappa:
                best_kappa = k
                best_add = wl
        selected.append(best_add)
        remaining.remove(best_add)

    selected = np.sort(np.array(selected))
    return {
        "selected_wavelengths": selected,
        "achieved_condition_number": _kappa(selected),
    }


def emissivity_variability_sweep(wavelength_range, n_bands, base_material,
                                  variability_levels, T, L_env_func,
                                  n_trials=30, seed=None,
                                  T_scale=5.0, eps_scale=0.05):
    """
    Sweep emissivity variability (noise std) and measure scaled κ + inversion error.

    Produces Figure 9 data (emissivity variability vs spectral resolution).

    Args:
        wavelength_range (tuple): (wl_min, wl_max) in meters
        n_bands (int): fixed band count
        base_material (str): 'water', 'vegetation', 'metal', 'concrete'
        variability_levels (np.ndarray): std values for emissivity noise
        T (float): Kelvin
        L_env_func (callable): wavelengths → L_env (N,)
        n_trials (int): Monte Carlo trials per variability level
        seed (int or None): random seed
        T_scale (float): prior 1σ on T (K), used for scale_jacobian
        eps_scale (float): prior 1σ on ε (dimensionless)

    Returns:
        dict:
            variability_levels: np.ndarray
            mean_condition_numbers: np.ndarray   (scaled κ, mean over trials)
            std_condition_numbers: np.ndarray
            mean_reconstruction_errors: np.ndarray   (T MAE in K)
    """
    from ..simulation.generator import generate_emissivity_curve
    from ..forward_model.radiance_model import compute_radiance
    from ..inversion.tex_inversion import invert_tex
    from ..inversion.jacobian import scale_jacobian

    rng = np.random.default_rng(seed)
    wl = generate_wavelength_sets(
        min_wl=wavelength_range[0], max_wl=wavelength_range[1],
        n_bands=n_bands, strategy="uniform",
    )
    L_env = L_env_func(wl)

    mean_kappas, std_kappas, mean_errors = [], [], []

    for variability in variability_levels:
        kappas = []
        T_errors = []
        for _ in range(n_trials):
            trial_seed = int(rng.integers(0, 2**31))
            emissivity = generate_emissivity_curve(
                n_bands, material=base_material,
                wavelength_range=wavelength_range,
                variability=variability, seed=trial_seed,
            )
            J = compute_jacobian(wl, T, emissivity, L_env)
            J_scaled = scale_jacobian(J, T_scale, eps_scale)
            svd = jacobian_svd(J_scaled)
            kappas.append(svd["condition_number"])

            L_obs = compute_radiance(wl, T, emissivity, L_env)
            result = invert_tex(wl, L_obs, L_env,
                                 initial_guess=None, T_bounds=(200.0, 400.0))
            T_errors.append(abs(result["T_est"] - T))

        mean_kappas.append(np.nanmean(kappas))
        std_kappas.append(np.nanstd(kappas))
        mean_errors.append(np.mean(T_errors))

    return {
        "variability_levels": np.array(variability_levels),
        "mean_condition_numbers": np.array(mean_kappas),
        "std_condition_numbers": np.array(std_kappas),
        "mean_reconstruction_errors": np.array(mean_errors),
    }


def stability_map_band_snr(band_counts, snr_db_values, material, T,
                            wavelength_range=(8e-6, 14e-6)):
    """
    2-D sweep: condition number as a function of band count vs SNR.

    Produces Figure 8 data.

    Args:
        band_counts (list[int]): x-axis values
        snr_db_values (np.ndarray): y-axis values in dB
        material (str): emissivity material
        T (float): Kelvin
        wavelength_range (tuple): (wl_min, wl_max) in meters

    Returns:
        dict:
            band_counts, snr_db_values,
            condition_numbers: np.ndarray shape (len(snr_db_values), len(band_counts))
    """
    from ..forward_model.planck import analytical_planck_derivative
    from ..forward_model.radiance_model import blackbody_environment, compute_radiance
    from ..simulation.noise import add_nedt_noise
    from ..inversion.tex_inversion import invert_tex

    n_snr = len(snr_db_values)
    n_bands_list = list(band_counts)
    kappa_map = np.full((n_snr, len(n_bands_list)), np.nan)

    for j, n in enumerate(n_bands_list):
        wl = generate_wavelength_sets(
            min_wl=wavelength_range[0], max_wl=wavelength_range[1],
            n_bands=n, strategy="uniform",
        )
        emissivity = get_material_emissivity(material, wl)
        L_env = blackbody_environment(wl, 280.0)
        J = compute_jacobian(wl, T, emissivity, L_env)
        svd = jacobian_svd(J)
        base_kappa = svd["condition_number"]

        for i, snr_db in enumerate(snr_db_values):
            # Convert SNR to NEDT-equivalent: sigma_L = L / 10^(snr/20)
            L = compute_radiance(wl, T, emissivity, L_env)
            sigma_L = L / (10 ** (snr_db / 20))
            # Scale J columns by noise floor to get effective condition number
            noise_weight = np.mean(sigma_L)
            dB_dT = analytical_planck_derivative(wl, T)
            sigma_eff = noise_weight / (np.mean(dB_dT) + 1e-30)
            # Effective condition number: nominal κ inflated by noise-to-signal ratio
            kappa_map[i, j] = base_kappa * (1 + sigma_eff)

    return {
        "band_counts": np.array(n_bands_list),
        "snr_db_values": np.array(snr_db_values),
        "condition_numbers": kappa_map,
    }


def stability_map_variability_resolution(variability_levels, band_counts,
                                          material, T, L_env_func,
                                          wavelength_range=(8e-6, 14e-6),
                                          n_trials=20, seed=None,
                                          T_scale=5.0, eps_scale=0.05):
    """
    2-D sweep: scaled κ AND empirical T_mae as a function of emissivity
    variability and band count.

    Produces Figure 9 (scaled κ heatmap) and Figure 9b (empirical T_mae heatmap).

    Args:
        variability_levels (np.ndarray): emissivity noise std (y-axis)
        band_counts (list[int]): x-axis values
        material (str): base emissivity material
        T (float): Kelvin
        L_env_func (callable): wavelengths → L_env (N,)
        wavelength_range (tuple): (wl_min, wl_max) in meters
        n_trials (int): Monte Carlo trials per cell
        seed (int or None): random seed
        T_scale, eps_scale (float): prior 1σ for the scaled κ

    Returns:
        dict:
            variability_levels, band_counts,
            condition_numbers: np.ndarray shape (V, B) — scaled κ
            error_map:         np.ndarray shape (V, B) — mean |ΔT| (K)
    """
    V, B = len(variability_levels), len(band_counts)
    kappa_map = np.full((V, B), np.nan)
    error_map = np.full((V, B), np.nan)

    for j, n in enumerate(band_counts):
        sweep = emissivity_variability_sweep(
            wavelength_range, n, material, variability_levels, T,
            L_env_func=L_env_func, n_trials=n_trials, seed=seed,
            T_scale=T_scale, eps_scale=eps_scale,
        )
        kappa_map[:, j] = sweep["mean_condition_numbers"]
        error_map[:, j] = sweep["mean_reconstruction_errors"]

    return {
        "variability_levels": np.array(variability_levels),
        "band_counts": np.array(band_counts),
        "condition_numbers": kappa_map,
        "error_map": error_map,
    }
