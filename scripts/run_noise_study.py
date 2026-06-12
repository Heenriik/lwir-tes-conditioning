"""
Noise and calibration sensitivity script.

Writes tidy CSVs to results/data/ for figures 4, 5, 6, 7, 9, 9b. Rendering
is done by scripts/plot_all.py.

CSVs produced:
  fig4_noise_sensitivity.csv             snr_db, n_bands, T_mae, emissivity_mae
  fig5_nedt_sensitivity.csv              nedt_K, n_bands, T_mae
  fig6_wavelength_shift.csv              delta_lambda_m, n_bands, T_mae
  fig7_combined_perturbation.csv         nedt_K, delta_lambda_m, mean_T_mae
  fig9_stability_variability.csv         n_bands, variability, scaled_kappa
  fig9b_stability_variability_mae.csv    n_bands, variability, mean_T_mae

Usage:
    conda activate Mthesis
    python scripts/run_noise_study.py
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.forward_model import blackbody_environment, compute_radiance
from src.simulation.generator import get_material_emissivity, generate_wavelength_sets
from src.simulation.noise import add_noise, add_nedt_noise
from src.simulation.calibration import calibration_error_study, combined_perturbation_analysis
from src.inversion.tex_inversion import invert_tex
from src.utils.metrics import bootstrap_error_ci
from src.analysis.spectral_analysis import stability_map_variability_resolution

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "results", "data")
NOISE_CONFIG = os.path.join(ROOT, "configs", "noise.yaml")

T = 300.0
MATERIAL = "vegetation"
WL_RANGE = (8e-6, 14e-6)
BAND_COUNTS = [5, 10, 20]
N_BANDS_HEATMAP = 10
N_TRIALS = 200


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    with open(NOISE_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    snr_cfg = cfg["sweep"]
    cal_cfg = cfg["calibration"]
    mc_cfg = cfg["monte_carlo"]

    rng = np.random.default_rng(mc_cfg.get("seed", 42))

    # Pre-generate per-band wl / emissivity / L_env / L_true once
    per_band = {}
    for n in BAND_COUNTS:
        wl_n = generate_wavelength_sets(min_wl=WL_RANGE[0], max_wl=WL_RANGE[1],
                                         n_bands=n, strategy="uniform")
        emi_n = get_material_emissivity(MATERIAL, wl_n)
        L_env_n = blackbody_environment(wl_n, 280.0)
        L_true_n = compute_radiance(wl_n, T, emi_n, L_env_n)
        per_band[n] = dict(wl=wl_n, emi=emi_n, L_env=L_env_n, L_true=L_true_n)

    # --- Figure 4: SNR sweep over multiple band counts ---
    print("Running SNR sensitivity sweep...")
    snr_values = np.linspace(snr_cfg["snr_range"][0], snr_cfg["snr_range"][1],
                              snr_cfg["snr_steps"])
    rows_f4 = []
    for n in BAND_COUNTS:
        d = per_band[n]
        T_mae, e_mae = [], []
        for snr_db in snr_values:
            T_errs, e_errs = [], []
            for _ in range(N_TRIALS):
                seed = int(rng.integers(0, 2**31))
                L_noisy = add_noise(d["L_true"], snr_db, seed=seed)
                result = invert_tex(d["wl"], L_noisy, d["L_env"], T_bounds=(200.0, 400.0))
                T_errs.append(abs(result["T_est"] - T))
                e_errs.append(float(np.mean(np.abs(result["emissivity_est"] - d["emi"]))))
            T_mae.append(np.mean(T_errs))
            e_mae.append(np.mean(e_errs))
        for snr, tm, em in zip(snr_values, T_mae, e_mae):
            rows_f4.append({"snr_db": float(snr), "n_bands": int(n),
                            "T_mae": float(tm), "emissivity_mae": float(em)})
        print(f"  N={n} done (Mean |ΔT| range: {min(T_mae):.3f}-{max(T_mae):.3f} K)")
    pd.DataFrame(rows_f4).to_csv(
        os.path.join(DATA_DIR, "fig4_noise_sensitivity.csv"), index=False)
    print("Wrote fig4 CSV")

    # --- Figure 5: NEDT sweep over multiple band counts ---
    print("Running NEDT sensitivity sweep...")
    nedt_values = np.linspace(cfg["sweep"]["nedt_range"][0],
                               cfg["sweep"]["nedt_range"][1],
                               cfg["sweep"]["nedt_steps"])
    rows_f5 = []
    for n in BAND_COUNTS:
        d = per_band[n]
        T_mean = []
        for nedt in nedt_values:
            T_ests = []
            for _ in range(N_TRIALS):
                seed = int(rng.integers(0, 2**31))
                L_noisy = add_nedt_noise(d["L_true"], d["wl"], T, nedt_K=nedt, seed=seed)
                result = invert_tex(d["wl"], L_noisy, d["L_env"], T_bounds=(200.0, 400.0))
                T_ests.append(result["T_est"])
            ci = bootstrap_error_ci(T, np.array(T_ests), seed=0)
            T_mean.append(ci["mean"])
        for nedt, tm in zip(nedt_values, T_mean):
            rows_f5.append({"nedt_K": float(nedt), "n_bands": int(n),
                            "T_mae": float(tm)})
        print(f"  N={n} done (mean |ΔT| range: {min(T_mean):.3f}-{max(T_mean):.3f} K)")
    pd.DataFrame(rows_f5).to_csv(
        os.path.join(DATA_DIR, "fig5_nedt_sensitivity.csv"), index=False)
    print("Wrote fig5 CSV")

    # --- Figure 6: Wavelength shift over multiple band counts ---
    print("Running wavelength shift study...")
    n_steps = cal_cfg["delta_lambda_steps"]
    dl_range = np.linspace(cal_cfg["delta_lambda_range_m"][0],
                            cal_cfg["delta_lambda_range_m"][1],
                            n_steps)
    rows_f6 = []
    for n in BAND_COUNTS:
        d = per_band[n]
        cal_result = calibration_error_study(d["wl"], d["emi"], T, d["L_env"], dl_range)
        for dl, err in zip(dl_range, cal_result["reconstruction_errors"]):
            rows_f6.append({"delta_lambda_m": float(dl), "n_bands": int(n),
                            "T_mae": float(err)})
        print(f"  N={n} done")
    pd.DataFrame(rows_f6).to_csv(
        os.path.join(DATA_DIR, "fig6_wavelength_shift.csv"), index=False)
    print("Wrote fig6 CSV")

    # --- Figure 7: Combined perturbation heatmap (single N) ---
    print(f"Running combined perturbation analysis (may take a few minutes, fixed N={N_BANDS_HEATMAP})...")
    d = per_band[N_BANDS_HEATMAP]
    nedt_sub = np.linspace(0.05, 1.0, 8)
    dlam_sub = np.linspace(dl_range[0], dl_range[-1], 8)
    combined = combined_perturbation_analysis(
        d["wl"], d["emi"], T, d["L_env"], nedt_sub, dlam_sub,
        n_trials=50, seed=mc_cfg.get("seed", 42),
    )
    rows_f7 = []
    me = combined["mean_errors"]  # shape (n_nedt, n_dlam)
    for i, nedt in enumerate(combined["nedt_values"]):
        for j, dl in enumerate(combined["delta_lambda_values"]):
            rows_f7.append({
                "nedt_K": float(nedt), "delta_lambda_m": float(dl),
                "mean_T_mae": float(me[i, j]),
            })
    pd.DataFrame(rows_f7).to_csv(
        os.path.join(DATA_DIR, "fig7_combined_perturbation.csv"), index=False)
    print(f"Wrote fig7 CSV (N={N_BANDS_HEATMAP} fixed)")

    # --- Figure 9 / 9b: Stability map (variability × band count) ---
    print("Running emissivity variability vs band count stability map...")
    variability_levels = np.linspace(0.0, 0.15, 12)
    band_counts9 = [5, 10, 15, 20, 30, 50]

    def _L_env_func(wl_arr):
        return blackbody_environment(wl_arr, 280.0)

    map9 = stability_map_variability_resolution(
        variability_levels=variability_levels,
        band_counts=band_counts9,
        material=MATERIAL,
        T=T,
        L_env_func=_L_env_func,
        wavelength_range=WL_RANGE,
        n_trials=50,
        seed=mc_cfg.get("seed", 42),
    )

    rows_f9 = []
    rows_f9b = []
    kap = map9["condition_numbers"]
    err = map9["error_map"]
    for i, v in enumerate(map9["variability_levels"]):
        for j, n in enumerate(map9["band_counts"]):
            rows_f9.append({"n_bands": int(n), "variability": float(v),
                            "scaled_kappa": float(kap[i, j])})
            rows_f9b.append({"n_bands": int(n), "variability": float(v),
                             "mean_T_mae": float(err[i, j])})
    pd.DataFrame(rows_f9).to_csv(
        os.path.join(DATA_DIR, "fig9_stability_variability.csv"), index=False)
    pd.DataFrame(rows_f9b).to_csv(
        os.path.join(DATA_DIR, "fig9b_stability_variability_mae.csv"), index=False)
    print("Wrote fig9 / fig9b CSVs")

    print(f"\nAll CSVs saved to {DATA_DIR}")


if __name__ == "__main__":
    main()
