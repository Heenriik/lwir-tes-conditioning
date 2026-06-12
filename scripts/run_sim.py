"""
Validates the forward model and Jacobian, generates synthetic spectra for
all four material classes, and produces a quick conditioning preview.

Usage:
    conda activate Mthesis
    python scripts/run_sim.py
"""

import os
import sys
import yaml
import numpy as np

# Allow running from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.forward_model import planck_radiance, blackbody_environment, compute_radiance
from src.inversion.jacobian import validate_jacobian, parameter_sensitivity, compute_jacobian
from src.simulation.generator import MATERIAL_LIBRARY, get_material_emissivity, generate_wavelength_sets

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "month1")
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "configs", "simulation.yaml")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    T = float(cfg["simulation"]["temperature"])
    wl_range = cfg["simulation"]["wavelength_range"]
    wl_min, wl_max = float(wl_range[0]), float(wl_range[1])
    band_counts = cfg["simulation"]["band_counts"]

    print(f"Surface temperature: {T} K")
    print(f"Wavelength range: {wl_min*1e6:.1f}–{wl_max*1e6:.1f} µm")
    print()

    materials = list(MATERIAL_LIBRARY.keys())

    # --- Synthetic spectra and Jacobian validation ---
    for material in materials:
        print(f"=== {material.upper()} ===")
        for n in band_counts:
            wl = generate_wavelength_sets(min_wl=wl_min, max_wl=wl_max,
                                          n_bands=n, strategy="uniform")
            emissivity = get_material_emissivity(material, wl)
            L_env = blackbody_environment(wl, 280.0)
            L = compute_radiance(wl, T, emissivity, L_env)

            # Save radiance
            fname = os.path.join(RESULTS_DIR, f"{material}_N{n}_radiance.npy")
            np.save(fname, {"wavelengths": wl, "L": L,
                             "emissivity": emissivity, "T": T})

            # Validate Jacobian
            val = validate_jacobian(wl, T, emissivity, L_env)
            status = "PASS" if val["passed"] else "FAIL"
            print(f"  N={n:2d}  Jacobian validation: {status}"
                  f"  max_rel_err={val['max_relative_error']:.2e}")

            # Parameter sensitivity: T column vs emissivity columns
            J = compute_jacobian(wl, T, emissivity, L_env)
            sens = parameter_sensitivity(J)
            print(f"         T-sensitivity={sens['temperature_sensitivity']:.3e}"
                  f"  ε-sensitivity(mean)={sens['emissivity_sensitivity'].mean():.3e}")

        print()

    # --- Quick conditioning preview (N=5, uniform, all materials) ---
    from src.analysis.conditioning import sweep_band_count
    from src.utils.plotting import plot_condition_numbers

    print(f"Conditioning preview (N={band_counts[0]} uniform):")
    cond_dict = {}
    for material in materials:
        def emissivity_func(wl, mat=material):
            return get_material_emissivity(mat, wl)
        def L_env_func(wl):
            return blackbody_environment(wl, 280.0)

        result = sweep_band_count(
            wavelength_range=(wl_min, wl_max),
            emissivity_func=emissivity_func,
            T=T,
            L_env_func=L_env_func,
            band_counts=[band_counts[0]],
            strategy="uniform",
        )
        kappa = result["condition_numbers"][0]
        cond_dict[material] = np.array([kappa])
        print(f"  {material:<12s}  κ = {kappa:.3e}")

    print(f"\nResults saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
