"""
Render all thesis figures from the tidy CSVs in results/data/.

The data-producing scripts (run_conditioning.py, run_noise_study.py, the
four run_hybrid*.py scripts) write tidy CSVs only; this script reads them
and produces PDFs in results/figures/. Edit the calls below (or the
plot_* functions in src/utils/plotting.py) to change titles, colors,
font sizes, etc., without re-running the experiments.

Any figure whose CSV is missing is skipped with a warning, so partial
data is OK during iteration.

Usage:
    conda activate Mthesis
    python scripts/plot_all.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.plotting import (
    plot_condition_numbers, plot_singular_values, plot_spectral_placement,
    plot_noise_sensitivity, plot_nedt_sensitivity,
    plot_wavelength_shift_sensitivity, plot_combined_perturbation,
    plot_stability_map, plot_hybrid_conditioning,
    plot_hybrid_singular_values, plot_distribution_shift,
    plot_distribution_shift_cross_variant, plot_lcurve,
    plot_lcurve_multiseed,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "results", "data")
FIG_DIR = os.path.join(ROOT, "results", "figures")

HYBRID_VARIANTS = ["baseline", "forward", "gated", "phys"]


def _path(name):
    return os.path.join(DATA_DIR, name)


def _fig(name):
    return os.path.join(FIG_DIR, name)


def _exists(name):
    p = _path(name)
    if not os.path.isfile(p):
        print(f"  skip — missing {name}")
        return False
    return True


def _pivot_dict(df, key_col, value_col, sort_key=None):
    """Helper: {key: np.array(values)} preserving original ordering of key_col."""
    keys = df[key_col].unique() if sort_key is None else sort_key
    return {k: df[df[key_col] == k][value_col].to_numpy() for k in keys}


# ---------------------------------------------------------------------------
# fig1 & fig1b — band count sweep, raw / scaled κ
# ---------------------------------------------------------------------------

def render_fig1():
    if not _exists("fig1_condition_numbers.csv"):
        return
    df = pd.read_csv(_path("fig1_condition_numbers.csv"))
    band_counts = sorted(df["band_count"].unique())
    cond_dict = {
        mat: df[df["material"] == mat].sort_values("band_count")["kappa"].to_numpy()
        for mat in df["material"].unique()
    }
    plot_condition_numbers(
        band_counts, condition_number_dict=cond_dict, reference_lines=(),
        save_path=_fig("fig1_condition_numbers.pdf"),
        plain_y_ticks=True,
    )
    print("Rendered fig1")


def render_fig1b():
    if not _exists("fig1b_condition_numbers_scaled.csv"):
        return
    df = pd.read_csv(_path("fig1b_condition_numbers_scaled.csv"))
    band_counts = sorted(df["band_count"].unique())
    cond_dict = {
        mat: df[df["material"] == mat].sort_values("band_count")["kappa"].to_numpy()
        for mat in df["material"].unique()
    }
    plot_condition_numbers(
        band_counts, condition_number_dict=cond_dict, reference_lines=(10,),
        save_path=_fig("fig1b_condition_numbers_scaled.pdf"),
    )
    print("Rendered fig1b")


# ---------------------------------------------------------------------------
# fig2 — singular value spectra
# ---------------------------------------------------------------------------

def render_fig2():
    if not _exists("fig2_singular_values.csv"):
        return
    df = pd.read_csv(_path("fig2_singular_values.csv"))
    svd_dict = {}
    for n in sorted(df["n_bands"].unique()):
        sub = df[df["n_bands"] == n].sort_values("index")
        svd_dict[int(n)] = sub["sigma"].to_numpy()
    plot_singular_values(svd_dict, save_path=_fig("fig2_singular_values.pdf"))
    print("Rendered fig2")


# ---------------------------------------------------------------------------
# fig3 — spectral placement
# ---------------------------------------------------------------------------

def render_fig3():
    if not _exists("fig3_spectral_placement.csv"):
        return
    df = pd.read_csv(_path("fig3_spectral_placement.csv"))
    placement_results = {
        row["strategy"]: {
            "condition_number": float(row["kappa"]),
            "condition_number_std": float(row["kappa_std"]),
        }
        for _, row in df.iterrows()
    }
    plot_spectral_placement(placement_results,
                            save_path=_fig("fig3_spectral_placement.pdf"))
    print("Rendered fig3")


# ---------------------------------------------------------------------------
# fig4 / fig5 / fig6 — multi-N sensitivity sweeps
# ---------------------------------------------------------------------------

def render_fig4():
    if not _exists("fig4_noise_sensitivity.csv"):
        return
    df = pd.read_csv(_path("fig4_noise_sensitivity.csv"))
    snr_values = np.sort(df["snr_db"].unique())
    errors_per_band = {}
    for n in sorted(df["n_bands"].unique()):
        sub = df[df["n_bands"] == n].sort_values("snr_db")
        errors_per_band[int(n)] = {
            "T_mae": sub["T_mae"].to_numpy(),
            "emissivity_mae": sub["emissivity_mae"].to_numpy(),
        }
    plot_noise_sensitivity(snr_values, errors_per_band=errors_per_band,
                           save_path=_fig("fig4_noise_sensitivity.pdf"))
    print("Rendered fig4")


def render_fig5():
    if not _exists("fig5_nedt_sensitivity.csv"):
        return
    df = pd.read_csv(_path("fig5_nedt_sensitivity.csv"))
    nedt_values = np.sort(df["nedt_K"].unique())
    errors_per_band = {}
    for n in sorted(df["n_bands"].unique()):
        sub = df[df["n_bands"] == n].sort_values("nedt_K")
        errors_per_band[int(n)] = sub["T_mae"].to_numpy()
    plot_nedt_sensitivity(nedt_values, errors_per_band=errors_per_band,
                          save_path=_fig("fig5_nedt_sensitivity.pdf"))
    print("Rendered fig5")


def render_fig6():
    if not _exists("fig6_wavelength_shift.csv"):
        return
    df = pd.read_csv(_path("fig6_wavelength_shift.csv"))
    dl_values = np.sort(df["delta_lambda_m"].unique())
    errors_per_band = {}
    for n in sorted(df["n_bands"].unique()):
        sub = df[df["n_bands"] == n].sort_values("delta_lambda_m")
        errors_per_band[int(n)] = sub["T_mae"].to_numpy()
    plot_wavelength_shift_sensitivity(dl_values, errors_per_band=errors_per_band,
                                      save_path=_fig("fig6_wavelength_shift.pdf"))
    print("Rendered fig6")


# ---------------------------------------------------------------------------
# fig7 — combined perturbation heatmap
# ---------------------------------------------------------------------------

def render_fig7():
    if not _exists("fig7_combined_perturbation.csv"):
        return
    df = pd.read_csv(_path("fig7_combined_perturbation.csv"))
    nedt_vals = np.sort(df["nedt_K"].unique())
    dl_vals = np.sort(df["delta_lambda_m"].unique())
    pivot = df.pivot_table(index="nedt_K", columns="delta_lambda_m",
                           values="mean_T_mae").sort_index()
    pivot = pivot.reindex(columns=dl_vals)
    plot_combined_perturbation(nedt_vals, dl_vals, pivot.to_numpy(),
                               save_path=_fig("fig7_combined_perturbation.pdf"))
    print("Rendered fig7")


# ---------------------------------------------------------------------------
# fig8 / fig8b — stability maps over band count × SNR
# ---------------------------------------------------------------------------

def render_fig8():
    if not _exists("fig8_stability_band_snr.csv"):
        return
    df = pd.read_csv(_path("fig8_stability_band_snr.csv"))
    band_counts = np.sort(df["n_bands"].unique())
    snr_db = np.sort(df["snr_db"].unique())
    pivot = df.pivot_table(index="snr_db", columns="n_bands",
                           values="kappa").reindex(index=snr_db, columns=band_counts)
    plot_stability_map(
        x_values=band_counts, y_values=snr_db,
        condition_numbers=pivot.to_numpy(),
        xlabel="Number of Bands", ylabel="SNR (dB)",
        title="Stability Map: Band Count × SNR",
        save_path=_fig("fig8_stability_band_snr.pdf"),
    )
    print("Rendered fig8")


def render_fig8b():
    if not _exists("fig8b_stability_band_snr_mc.csv"):
        return
    df = pd.read_csv(_path("fig8b_stability_band_snr_mc.csv"))
    band_counts = np.sort(df["n_bands"].unique())
    snr_db = np.sort(df["snr_db"].unique())
    pivot = df.pivot_table(index="snr_db", columns="n_bands",
                           values="mean_T_mae").reindex(index=snr_db, columns=band_counts)
    plot_stability_map(
        x_values=band_counts, y_values=snr_db,
        condition_numbers=pivot.to_numpy(),
        xlabel="Number of Bands", ylabel="SNR (dB)",
        title="Empirical Stability Map: Mean |ΔT| (K)",
        log_scale=False, cbar_label="Mean |ΔT| (K)", contour_value=1.0,
        save_path=_fig("fig8b_stability_band_snr_mc.pdf"),
    )
    print("Rendered fig8b")


# ---------------------------------------------------------------------------
# fig9 / fig9b — stability maps over variability × band count
# ---------------------------------------------------------------------------

def render_fig9():
    if not _exists("fig9_stability_variability.csv"):
        return
    df = pd.read_csv(_path("fig9_stability_variability.csv"))
    band_counts = np.sort(df["n_bands"].unique())
    variability = np.sort(df["variability"].unique())
    pivot = df.pivot_table(index="variability", columns="n_bands",
                           values="scaled_kappa").reindex(index=variability,
                                                          columns=band_counts)
    plot_stability_map(
        x_values=band_counts, y_values=variability,
        condition_numbers=pivot.to_numpy(),
        xlabel="Number of Bands", ylabel="Emissivity Variability (std)",
        title="Stability Map: Scaled κ — Variability × Resolution",
        save_path=_fig("fig9_stability_variability.pdf"),
    )
    print("Rendered fig9")


def render_fig9b():
    if not _exists("fig9b_stability_variability_mae.csv"):
        return
    df = pd.read_csv(_path("fig9b_stability_variability_mae.csv"))
    band_counts = np.sort(df["n_bands"].unique())
    variability = np.sort(df["variability"].unique())
    pivot = df.pivot_table(index="variability", columns="n_bands",
                           values="mean_T_mae").reindex(index=variability,
                                                        columns=band_counts)
    plot_stability_map(
        x_values=band_counts, y_values=variability,
        condition_numbers=pivot.to_numpy(),
        xlabel="Number of Bands", ylabel="Emissivity Variability (std)",
        title="Empirical Stability Map: Mean |ΔT| (K)",
        log_scale=False, cbar_label="Mean |ΔT| (K)", contour_value=1.0,
        save_path=_fig("fig9b_stability_variability_mae.pdf"),
    )
    print("Rendered fig9b")


# ---------------------------------------------------------------------------
# fig10 / fig11 / fig12 — hybrid variants
# ---------------------------------------------------------------------------

def _hybrid_suffix(variant):
    """Match historical naming: baseline omits suffix, variants append _{name}."""
    return "" if variant == "baseline" else f"_{variant}"


def render_fig10_variant(variant):
    name = f"fig10_hybrid_conditioning_{variant}.csv"
    if not _exists(name):
        return
    df = pd.read_csv(_path(name))
    plot_hybrid_conditioning(
        df["kappa_physics"].to_numpy(), df["kappa_hybrid"].to_numpy(),
        save_path=_fig(f"fig10_hybrid_conditioning{_hybrid_suffix(variant)}.pdf"),
    )
    print(f"Rendered fig10 ({variant})")


def render_fig11_variant(variant):
    sig_name = f"fig11_hybrid_singular_values_{variant}.csv"
    meta_name = f"fig11_hybrid_singular_values_{variant}_meta.csv"
    if not (_exists(sig_name) and _exists(meta_name)):
        return
    sig = pd.read_csv(_path(sig_name)).sort_values("index")
    meta = pd.read_csv(_path(meta_name)).iloc[0]
    svd_phys = {"s": sig["sigma_physics"].to_numpy(),
                "condition_number": float(meta["kappa_physics"])}
    svd_hyb = {"s": sig["sigma_hybrid"].to_numpy(),
               "condition_number": float(meta["kappa_hybrid"])}
    plot_hybrid_singular_values(
        svd_phys, svd_hyb,
        save_path=_fig(f"fig11_hybrid_singular_values{_hybrid_suffix(variant)}.pdf"),
    )
    print(f"Rendered fig11 ({variant})")


def render_fig12_variant(variant):
    name = f"fig12_distribution_shift_{variant}.csv"
    if not _exists(name):
        return
    df = pd.read_csv(_path(name)).sort_values("offset_K")
    plot_distribution_shift(
        df["offset_K"].to_numpy(),
        df["physics_T_mae"].to_numpy(),
        df["hybrid_T_mae"].to_numpy(),
        save_path=_fig(f"fig12_distribution_shift{_hybrid_suffix(variant)}.pdf"),
    )
    print(f"Rendered fig12 ({variant})")


def render_fig12_cross_variant():
    """Cross-variant fig 12 used in §5.10 — physics + all 4 hybrid OOD curves."""
    variant_label = {
        "baseline": "baseline",
        "phys": "physics-consistency",
        "gated": "density-gated",
        "forward": "forward-space",
    }
    shift = None
    phys = None
    variant_errors = {}
    for variant in ["baseline", "phys", "gated", "forward"]:
        name = f"fig12_distribution_shift_{variant}.csv"
        if not _exists(name):
            return
        df = pd.read_csv(_path(name)).sort_values("offset_K")
        if shift is None:
            shift = df["offset_K"].to_numpy()
            phys = df["physics_T_mae"].to_numpy()
        variant_errors[variant_label[variant]] = df["hybrid_T_mae"].to_numpy()
    plot_distribution_shift_cross_variant(
        shift, phys, variant_errors,
        save_path=_fig("fig12_distribution_shift_cross_variant.pdf"),
    )
    print("Rendered fig12 cross-variant")


# ---------------------------------------------------------------------------
# fig13 — phys L-curve
# ---------------------------------------------------------------------------

def render_fig13():
    """Render fig13. Prefers the multi-seed L-curve if available."""
    multi_path = _path("fig13_lcurve_phys_multiseed.csv")
    summary_path = _path("fig13_lcurve_phys_summary.csv")
    if os.path.exists(multi_path) and os.path.exists(summary_path):
        per_seed = pd.read_csv(multi_path)
        summary = pd.read_csv(summary_path)
        plot_lcurve_multiseed(
            per_seed, summary,
            save_path=_fig("fig13_lcurve_phys.pdf"),
        )
        print("Rendered fig13 (multi-seed)")
        return

    if not _exists("fig13_lcurve_phys.csv"):
        return
    df = pd.read_csv(_path("fig13_lcurve_phys.csv"))
    knee_idx = int(np.argmax(df["is_knee"].to_numpy()))
    plot_lcurve(
        df["lambda_phys"].tolist(),
        df["accuracy_T_mae"].to_numpy(),
        df["phys_violation"].to_numpy(),
        knee_idx,
        save_path=_fig("fig13_lcurve_phys.pdf"),
    )
    print("Rendered fig13 (single-seed fallback)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    print(f"Reading CSVs from {DATA_DIR}")
    print(f"Writing figures to {FIG_DIR}\n")

    render_fig1()
    render_fig1b()
    render_fig2()
    render_fig3()
    render_fig4()
    render_fig5()
    render_fig6()
    render_fig7()
    render_fig8()
    render_fig8b()
    render_fig9()
    render_fig9b()
    for variant in HYBRID_VARIANTS:
        render_fig10_variant(variant)
        render_fig11_variant(variant)
        render_fig12_variant(variant)
    render_fig12_cross_variant()
    render_fig13()

    print("\nDone.")


if __name__ == "__main__":
    main()
