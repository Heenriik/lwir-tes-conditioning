import numpy as np

from ..inversion.jacobian import compute_jacobian, jacobian_svd
from ..simulation.generator import generate_wavelength_sets, get_material_emissivity


def compute_condition_number(J):
    """
    Condition number and SVD summary of the Jacobian.

    effective_rank threshold (s > 1e-3 * s_max):
    Counts singular values larger than 0.1% of the dominant singular value.
    This is a pragmatic/practical "working rank" (not strict numerical rank) intended to
    measure how many independent directions in the measurement space carry useful
    signal above a practical noise floor. The 1e-3 factor is a thesis-defined
    operational choice since there seems to be no single canonical value in literature.
    Compare with the stricter threshold in jacobian_svd (1e-12 * s_max) which
    is used for strict rank-deficiency detection.

    Args:
        J (np.ndarray): shape (N, N+1)

    Returns:
        dict: condition_number, singular_values, rank, effective_rank
    """
    svd = jacobian_svd(J)
    s = svd["s"]
    effective_rank = int(np.sum(s > 1e-3 * s[0]))
    return {
        "condition_number": svd["condition_number"],
        "singular_values": s,
        "rank": svd["rank"],
        "effective_rank": effective_rank,
    }


def sweep_band_count(wavelength_range, emissivity_func, T, L_env_func,
                     band_counts, strategy="uniform", seed=None):
    """
    Sweep over band counts and compute Jacobian conditioning for each.

    Produces Figure 1 (condition number vs band count) and
    Figure 2 (singular value spectra) data.

    Args:
        wavelength_range (tuple): (wl_min, wl_max) in meters
        emissivity_func (callable): wavelengths (N,) → emissivity (N,)
        T (float): surface temperature in Kelvin
        L_env_func (callable): wavelengths (N,) → L_env (N,)
        band_counts (list[int]): e.g. [3, 5, 7, 10, 15, 20, 30, 50]
        strategy (str): 'uniform', 'random', 'targeted'
        seed (int or None): random seed for non-deterministic strategies

    Returns:
        dict:
            band_counts: list[int]
            condition_numbers: np.ndarray shape (len(band_counts),)
            singular_value_spectra: list of np.ndarray (one per band count)
            jacobians: list of np.ndarray shape (N, N+1)
    """
    condition_numbers = []
    singular_value_spectra = []
    jacobians = []

    for n in band_counts:
        wl = generate_wavelength_sets(
            min_wl=wavelength_range[0], max_wl=wavelength_range[1],
            n_bands=n, strategy=strategy, seed=seed,
        )
        emissivity = emissivity_func(wl)
        L_env = L_env_func(wl)
        J = compute_jacobian(wl, T, emissivity, L_env)
        svd = jacobian_svd(J)

        condition_numbers.append(svd["condition_number"])
        singular_value_spectra.append(svd["s"])
        jacobians.append(J)

    return {
        "band_counts": band_counts,
        "condition_numbers": np.array(condition_numbers),
        "singular_value_spectra": singular_value_spectra,
        "jacobians": jacobians,
    }


def sweep_spectral_placement(n_bands, wavelength_range, emissivity_func, T, L_env_func,
                              strategies, n_random_trials=50, seed=None):
    """
    Fixed band count; compare condition numbers across placement strategies.

    Produces Figure 3 data.

    Args:
        n_bands (int): number of spectral bands
        wavelength_range (tuple): (wl_min, wl_max) in meters
        emissivity_func (callable): wavelengths → emissivity
        T (float): Kelvin
        L_env_func (callable): wavelengths → L_env
        strategies (list[str]): e.g. ['uniform', 'random', 'targeted']
        n_random_trials (int): Monte Carlo trials for 'random' strategy
        seed (int or None): base seed

    Returns:
        dict keyed by strategy name, each with:
            condition_number: float (mean for random)
            condition_number_std: float (std for random, 0 for deterministic)
            singular_values: np.ndarray
    """
    results = {}
    rng = np.random.default_rng(seed)

    for strat in strategies:
        if strat == "random":
            kappas = []
            svs = []
            for trial in range(n_random_trials):
                trial_seed = int(rng.integers(0, 2**31))
                wl = generate_wavelength_sets(
                    min_wl=wavelength_range[0], max_wl=wavelength_range[1],
                    n_bands=n_bands, strategy="random", seed=trial_seed,
                )
                emissivity = emissivity_func(wl)
                L_env = L_env_func(wl)
                J = compute_jacobian(wl, T, emissivity, L_env)
                svd = jacobian_svd(J)
                kappas.append(svd["condition_number"])
                svs.append(svd["s"])
            results[strat] = {
                "condition_number": float(np.nanmean(kappas)),
                "condition_number_std": float(np.nanstd(kappas)),
                "singular_values": np.mean(svs, axis=0),
            }
        else:
            wl = generate_wavelength_sets(
                min_wl=wavelength_range[0], max_wl=wavelength_range[1],
                n_bands=n_bands, strategy=strat,
            )
            emissivity = emissivity_func(wl)
            L_env = L_env_func(wl)
            J = compute_jacobian(wl, T, emissivity, L_env)
            svd = jacobian_svd(J)
            results[strat] = {
                "condition_number": svd["condition_number"],
                "condition_number_std": 0.0,
                "singular_values": svd["s"],
            }

    return results


def identifiability_threshold_analysis(wavelength_range, materials, T, nedt_values,
                                        band_counts, kappa_threshold=100.0):
    """
    For each (material, NEDT) pair, find the minimum band count with κ < kappa_threshold.

    Produces Figure 8 data (band count vs SNR stability map).

    κ threshold (default 100.0):
    Conservative operational guard. The formal definition (Golub & Van Loan,
    2013, "Matrix Computations", 4th ed., §3.4.10 p.135) is:
        well-conditioned  ⇔  κ(A) ≈ O(1)
        ill-conditioned   ⇔  κ(A) ≈ O(1/u)
    where u is unit roundoff (≈ 10⁻¹⁶ for IEEE double). The boundary is therefore
    enormous (~10¹⁶); intermediate values map to digit loss via Heuristic II of
    §3.5 p.138: if u ≈ 10⁻ᵈ and κ ≈ 10ᵠ, the solution carries about d - q correct
    decimal digits. κ = 100 corresponds to losing 2 of 16 digits — comfortably
    well-conditioned by the strict criterion, but a sensible engineering threshold
    beyond which retrievals start to feel uncomfortable. Non-binding in our
    8-14 µm regime (raw κ ≈ 2); kept as a guard for future configurations.
    https://www.ee.iitb.ac.in/~belur/uplod/golub-van-loan-matrix-computations-2012-edition-4th.pdf

    Noise-limited check: σ_eff = s[-1] / (nedt · dB/dT_mean · √N)
    This is a thesis-derived metric (no direct citation). Rationale: the minimum
    singular value σ_N governs sensitivity in the worst-constrained direction.
    The expected radiance noise amplitude in that direction is ≈ nedt·dB/dT_mean·√N
    (RMS of N independent noise contributions of magnitude nedt·dB/dT_mean each).
    If σ_N falls below this noise floor (σ_eff < 1), the configuration is
    noise-limited regardless of κ. Conceptually related to Shao et al. (2020) - (NEDT · dB/dT), 
    but the specific formula is original here.

    Args:
        wavelength_range (tuple): (wl_min, wl_max) in meters
        materials (list[str]): material names
        T (float): Kelvin
        nedt_values (np.ndarray): NEDT values in Kelvin
        band_counts (list[int]): candidate band counts to search
        kappa_threshold (float): condition number target

    Returns:
        dict keyed by material, each value is np.ndarray of shape (len(nedt_values),)
        containing the threshold band count (np.nan if never achieved)
    """
    from ..forward_model.planck import analytical_planck_derivative

    result = {}
    for mat in materials:
        thresholds = []
        for nedt in nedt_values:
            found = np.nan
            for n in sorted(band_counts):
                wl = generate_wavelength_sets(
                    min_wl=wavelength_range[0], max_wl=wavelength_range[1],
                    n_bands=n, strategy="uniform",
                )
                emissivity = get_material_emissivity(mat, wl)
                # Approximate L_env as blackbody at 280 K
                from ..forward_model.radiance_model import blackbody_environment
                L_env = blackbody_environment(wl, 280.0)
                J = compute_jacobian(wl, T, emissivity, L_env)
                svd = jacobian_svd(J)
                kappa = svd["condition_number"]

                # Noise-limited check: smallest singular value vs NEDT noise floor
                dB_dT_mean = float(np.mean(analytical_planck_derivative(wl, T)))
                sigma_L = nedt * dB_dT_mean   # NEDT = σ_L / (dB/dT), this is rearranged (for mean sensitivity over all bands)
                sigma_eff = svd["s"][-1] / (sigma_L * np.sqrt(n))

                if np.isfinite(kappa) and kappa < kappa_threshold and sigma_eff >= 1.0:
                    found = n
                    break
            thresholds.append(found)
        result[mat] = np.array(thresholds)
    return result
