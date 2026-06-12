"""
Month 2 conditioning experiment script.

Writes tidy CSVs to results/data/ for figures 1, 1b, 2, 3, 8, 8b. Rendering
is done by scripts/plot_all.py.

CSVs produced:
  fig1_condition_numbers.csv            band_count, material, kappa
  fig1b_condition_numbers_scaled.csv    band_count, material, kappa
  fig2_singular_values.csv              n_bands, index, sigma
  fig3_spectral_placement.csv           strategy, kappa, kappa_std
  fig8_stability_band_snr.csv           n_bands, snr_db, kappa
  fig8b_stability_band_snr_mc.csv       n_bands, snr_db, mean_T_mae

Usage:
    conda activate Mthesis
    python scripts/run_conditioning.py
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.forward_model import blackbody_environment
from src.simulation.generator import get_material_emissivity
from src.inversion.jacobian import scale_jacobian, jacobian_svd, compute_jacobian
from src.analysis.conditioning import (
    sweep_band_count, sweep_spectral_placement, identifiability_threshold_analysis,
)
from src.analysis.spectral_analysis import stability_map_band_snr

# Prior 1σ uncertainties for the scaled Jacobian — Tarantola (2005) §3.1
T_SCALE = 5.0    # K
E_SCALE = 0.05   # dimensionless

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "results", "data")
CONFIG_PATH = os.path.join(ROOT, "configs", "simulation.yaml")

BAND_COUNTS = [3, 5, 7, 10, 15, 20, 30, 50]
MATERIALS = ["water", "vegetation", "metal", "concrete"]
T = 300.0
WL_RANGE = (8e-6, 14e-6)
STRATEGIES = ["uniform", "random", "targeted"]


def _emissivity_func(material):
    def f(wl):
        return get_material_emissivity(material, wl)
    return f


def _L_env_func(wl):
    return blackbody_environment(wl, 280.0)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # --- Figure 1 & 1b: band count sweep per material (raw + scaled κ) ---
    print("Running band count sweep...")
    rows_raw, rows_scaled = [], []

    for material in MATERIALS:
        result = sweep_band_count(
            wavelength_range=WL_RANGE,
            emissivity_func=_emissivity_func(material),
            T=T,
            L_env_func=_L_env_func,
            band_counts=BAND_COUNTS,
            strategy="uniform",
        )
        raw = result["condition_numbers"]
        scaled = np.array([
            jacobian_svd(scale_jacobian(J, T_SCALE, E_SCALE))["condition_number"]
            for J in result["jacobians"]
        ])
        for n, k in zip(BAND_COUNTS, raw):
            rows_raw.append({"band_count": int(n), "material": material, "kappa": float(k)})
        for n, k in zip(BAND_COUNTS, scaled):
            rows_scaled.append({"band_count": int(n), "material": material, "kappa": float(k)})
        print(f"  {material}: raw κ [{raw.min():.2e}, {raw.max():.2e}]"
              f"  scaled κ [{scaled.min():.2e}, {scaled.max():.2e}]")

    pd.DataFrame(rows_raw).to_csv(
        os.path.join(DATA_DIR, "fig1_condition_numbers.csv"), index=False)
    pd.DataFrame(rows_scaled).to_csv(
        os.path.join(DATA_DIR, "fig1b_condition_numbers_scaled.csv"), index=False)
    print("Wrote fig1 / fig1b CSVs")

    # --- Figure 2: singular value spectra for vegetation (representative) ---
    # Includes N=30 to show the Jacobian spectrum at Gillespie's (1998)
    # MMD-scatter plateau point (see THESIS_Draft.md §2.2.1).
    veg_band_counts = [3, 5, 10, 20, 30, 50]
    veg_result = sweep_band_count(
        wavelength_range=WL_RANGE,
        emissivity_func=_emissivity_func("vegetation"),
        T=T,
        L_env_func=_L_env_func,
        band_counts=veg_band_counts,
        strategy="uniform",
    )
    rows_sv = []
    for n, sv in zip(veg_band_counts, veg_result["singular_value_spectra"]):
        for k, s in enumerate(sv, start=1):
            rows_sv.append({"n_bands": int(n), "index": int(k), "sigma": float(s)})
    pd.DataFrame(rows_sv).to_csv(
        os.path.join(DATA_DIR, "fig2_singular_values.csv"), index=False)
    print("Wrote fig2 CSV")

    # --- Figure 3: spectral placement comparison (incl. greedy κ-optimal) ---
    print("Running spectral placement sweep...")
    placement_results = sweep_spectral_placement(
        n_bands=10,
        wavelength_range=WL_RANGE,
        emissivity_func=_emissivity_func("vegetation"),
        T=T,
        L_env_func=_L_env_func,
        strategies=STRATEGIES,
        n_random_trials=50,
        seed=0,
    )

    print("Running greedy κ-optimal placement (this may take a few seconds)...")
    from src.analysis.spectral_analysis import optimal_band_selection
    candidates = np.linspace(WL_RANGE[0], WL_RANGE[1], 80)
    greedy = optimal_band_selection(
        wavelength_candidates=candidates,
        emissivity_func=_emissivity_func("vegetation"),
        T=T,
        L_env_func=_L_env_func,
        n_bands_target=10,
    )
    placement_results["greedy"] = {
        "condition_number": greedy["achieved_condition_number"],
        "condition_number_std": 0.0,
    }

    rows_pl = []
    for strat, res in placement_results.items():
        rows_pl.append({
            "strategy": strat,
            "kappa": float(res["condition_number"]),
            "kappa_std": float(res["condition_number_std"]),
        })
        print(f"  {strat}: κ = {res['condition_number']:.3e} ± {res['condition_number_std']:.3e}")
    pd.DataFrame(rows_pl).to_csv(
        os.path.join(DATA_DIR, "fig3_spectral_placement.csv"), index=False)
    print("Wrote fig3 CSV")

    # --- Identifiability threshold analysis (console-only diagnostic) ---
    print("Running identifiability threshold analysis...")
    nedt_values = np.linspace(0.05, 2.0, 10)
    thresholds = identifiability_threshold_analysis(
        wavelength_range=WL_RANGE,
        materials=MATERIALS,
        T=T,
        nedt_values=nedt_values,
        band_counts=BAND_COUNTS,
        kappa_threshold=100.0,
    )
    print("  Minimum bands for κ < 100 (at NEDT=0.1 K):")
    nedt_idx = int(np.argmin(np.abs(nedt_values - 0.1)))
    for mat, thrs in thresholds.items():
        val = thrs[nedt_idx]
        print(f"    {mat:<12s}  N_min = {val}")

    # --- Figure 8: analytical κ_eff stability map (band count × SNR) ---
    print("Running stability map (band count × SNR)...")
    snr_db_values = np.linspace(10, 60, 15)
    map_result = stability_map_band_snr(
        band_counts=BAND_COUNTS,
        snr_db_values=snr_db_values,
        material="vegetation",
        T=T,
        wavelength_range=WL_RANGE,
    )
    rows_f8 = []
    cond = map_result["condition_numbers"]  # shape (n_snr, n_bands)
    for i, snr in enumerate(snr_db_values):
        for j, n in enumerate(BAND_COUNTS):
            rows_f8.append({
                "n_bands": int(n), "snr_db": float(snr),
                "kappa": float(cond[i, j]),
            })
    pd.DataFrame(rows_f8).to_csv(
        os.path.join(DATA_DIR, "fig8_stability_band_snr.csv"), index=False)
    print("Wrote fig8 CSV")

    # --- Figure 8b: empirical Monte Carlo T_mae map ---
    print("Running Monte Carlo stability map (band count × SNR, n_trials=30)...")
    from src.forward_model import compute_radiance
    from src.simulation.generator import generate_wavelength_sets
    from src.simulation.noise import add_noise
    from src.inversion.tex_inversion import invert_tex

    N_TRIALS = 30
    rng_mc = np.random.default_rng(42)
    snr_grid = np.linspace(10, 60, 8)
    rmse_map = np.full((len(snr_grid), len(BAND_COUNTS)), np.nan)

    for j, n in enumerate(BAND_COUNTS):
        wl = generate_wavelength_sets(min_wl=WL_RANGE[0], max_wl=WL_RANGE[1],
                                       n_bands=n, strategy="uniform")
        emi = get_material_emissivity("vegetation", wl)
        L_env = _L_env_func(wl)
        L_true = compute_radiance(wl, T, emi, L_env)
        for i, snr_db in enumerate(snr_grid):
            errs = []
            for _ in range(N_TRIALS):
                seed = int(rng_mc.integers(0, 2**31))
                L_noisy = add_noise(L_true, snr_db, seed=seed)
                res = invert_tex(wl, L_noisy, L_env, T_bounds=(200.0, 400.0))
                errs.append(abs(res["T_est"] - T))
            rmse_map[i, j] = float(np.mean(errs))
        print(f"  N={n:3d} done")

    rows_f8b = []
    for i, snr in enumerate(snr_grid):
        for j, n in enumerate(BAND_COUNTS):
            rows_f8b.append({
                "n_bands": int(n), "snr_db": float(snr),
                "mean_T_mae": float(rmse_map[i, j]),
            })
    pd.DataFrame(rows_f8b).to_csv(
        os.path.join(DATA_DIR, "fig8b_stability_band_snr_mc.csv"), index=False)
    print("Wrote fig8b CSV")

    print(f"\nAll CSVs saved to {DATA_DIR}")


if __name__ == "__main__":
    main()
