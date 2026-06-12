"""
Figure generation

Each function accepts a save_path argument; when given, the figure is saved
to that path (PDF at 300 dpi) instead of being displayed.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Publication defaults.
# Titles are deliberately dropped (captions live in LaTeX); fonts are sized
# to render at ~11pt on-page (Computer Modern Roman, thesis body) when the
# figure is embedded at typical thesis widths (\textwidth ≈ 0.85× figure
# native). Matplotlib falls back gracefully if CM is not installed system-wide.
_RC = {
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 12,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 150,
    # PDF text-layer hygiene: matplotlib's default Type 3 fonts encode
    # glyphs (especially \Delta) as Private Use Area characters, which
    # integrity checkers flag as "white characters". Type 42 (TrueType)
    # embedding preserves real Unicode codepoints in the extracted text.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
}
_MATERIAL_COLORS = {
    "water": "#1f77b4",
    "vegetation": "#2ca02c",
    "metal": "#d62728",
    "concrete": "#ff7f0e",
}


def _save_or_show(fig, save_path):
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Figure 1: Condition number vs band count (by material)
# ---------------------------------------------------------------------------

def plot_condition_numbers(band_counts, condition_number_dict=None,
                            condition_numbers=None, save_path=None,
                            reference_lines=(10, 100, 1000),
                            plain_y_ticks=False):
    """
    Figure 1: κ vs band count, one line per material.

    Shows how spectral conditioning of the forward model changes as more
    bands are added. Reference lines at κ=10/100/1000 mark common stability
    thresholds. With raw J, materials nearly overlap (the emissivity block
    dominates the SVD); use a scaled Jacobian to expose material differences.

    What to look for:
      - Overall trend with N: roughly flat or slowly growing for raw κ
      - Material spread: small for raw κ, large for scaled κ
      - Crossings of the κ=100 threshold mark identifiability limits

    Args:
        band_counts (list[int] or np.ndarray): x-axis
        condition_number_dict (dict): {material: np.ndarray of κ values}
            If None, falls back to single-series mode via condition_numbers.
        condition_numbers (list or np.ndarray): single series (legacy API)
        save_path (str or None): output file path
    """
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(7, 4))

        if condition_number_dict is not None:
            for mat, kappas in condition_number_dict.items():
                color = _MATERIAL_COLORS.get(mat, None)
                ax.plot(band_counts, kappas, marker="o", label=mat.capitalize(),
                        color=color)
        else:
            ax.plot(band_counts, condition_numbers, marker="o", color="steelblue")

        _ls_for = {10: ":", 100: "--", 1000: "-."}
        for ref in reference_lines:
            ax.axhline(ref, color="gray", linestyle=_ls_for.get(ref, "--"),
                       linewidth=0.8, label=f"κ = {ref}")

        ax.set_yscale("log")
        if plain_y_ticks:
            # Show tick labels as plain numbers (e.g. "1.91") instead of
            # LogFormatter's "1.91 × 10⁰" power notation. Apply to both major
            # and minor — for data tightly clustered inside a single decade,
            # only minor ticks land in the visible range.
            fmt = plt.matplotlib.ticker.ScalarFormatter()
            fmt.set_scientific(False)
            ax.yaxis.set_major_formatter(fmt)
            ax.yaxis.set_minor_formatter(fmt)
        ax.set_xlabel("Number of Spectral Bands")
        ax.set_ylabel("Condition Number κ(J)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Figure 2: Singular value spectra
# ---------------------------------------------------------------------------

def plot_singular_values(svd_dict, band_count_labels=None, save_path=None):
    """
    Figure 2: sorted singular values on log scale, one series per band count.

    Shows the full SVD spectrum of the Jacobian rather than just the κ ratio.
    σ_max is the best-constrained inversion direction; σ_min is the worst.
    A flat spectrum means a well-balanced problem; a steep drop-off means
    one or more parameter directions are nearly invisible to the data.

    What to look for:
      - Drop-off from σ_1 to σ_N — gentle is good, cliff-like is bad
      - As N grows, more singular values appear (spectrum elongates) —
        the smallest typically does NOT shrink dramatically with N
      - Effective rank: where the spectrum drops below ~1% of σ_max

    Args:
        svd_dict (dict): {n_bands (int): singular_values (np.ndarray)}
            or a list of arrays (legacy API).
        band_count_labels (list or None): display labels; defaults to dict keys
        save_path (str or None)
    """
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(7, 4))

        if isinstance(svd_dict, dict):
            items = list(svd_dict.items())
        else:
            # Legacy: list of arrays
            items = [(i, sv) for i, sv in enumerate(svd_dict)]

        cmap = plt.get_cmap("viridis", len(items))
        for idx, (key, sv) in enumerate(items):
            label = str(key) if band_count_labels is None else str(band_count_labels[idx])
            ax.plot(np.arange(1, len(sv) + 1), sv, marker="o",
                    label=f"N={label}", color=cmap(idx))

        ax.set_yscale("log")
        ax.set_xlabel("Singular Value Index")
        ax.set_ylabel("Singular Value σ")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(title="Band count")
        _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Figure 3: Spectral placement comparison
# ---------------------------------------------------------------------------

def plot_spectral_placement(placement_results, save_path=None):
    """
    Figure 3: bar chart of κ by placement strategy (with error bars for random).

    Compares uniform vs random vs targeted band placement at fixed N. Targeted
    placement aligns bands with known emissivity features (e.g. the silicate
    reststrahlen dip in soil); random samples N wavelengths uniformly across
    the window. Error bars on 'random' show variability across trials.

    What to look for:
      - Targeted vs uniform: targeted should match or beat uniform if the
        emissivity has structure to exploit
      - Random spread: small spread means placement matters little at this N;
        large spread means placement is a key design lever

    Args:
        placement_results (dict): from sweep_spectral_placement —
            {strategy: {condition_number, condition_number_std, singular_values}}
        save_path (str or None)
    """
    with plt.rc_context(_RC):
        strategies = list(placement_results.keys())
        kappas = [placement_results[s]["condition_number"] for s in strategies]
        stds = [placement_results[s]["condition_number_std"] for s in strategies]

        fig, ax = plt.subplots(figsize=(6, 4))
        colors = plt.get_cmap("tab10")(np.linspace(0, 0.5, len(strategies)))
        ax.bar(strategies, kappas, yerr=stds, color=colors,
               capsize=5, edgecolor="black", linewidth=0.7)
        ax.set_yscale("log")
        ax.set_xlabel("Spectral Placement Strategy")
        ax.set_ylabel("Condition Number κ(J)")
        ax.grid(True, axis="y", alpha=0.3)
        _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Figure 4: Noise (SNR) sensitivity
# ---------------------------------------------------------------------------

def plot_noise_sensitivity(snr_values, errors_dict=None, save_path=None,
                            errors_per_band=None):
    """
    Figure 4: reconstruction error vs SNR in dB.

    Plots mean absolute T error (Mean |ΔT|), not RMSE. Two display modes:
      - Single-N mode (legacy): pass errors_dict={"T_mae":..., "emissivity_mae":...}
        Twin y-axes for T and ε.
      - Multi-N mode: pass errors_per_band={N: {"T_mae":..., "emissivity_mae":...}}
        One T-line per band count; ε omitted to keep figure readable.

    Args:
        snr_values (np.ndarray): SNR in dB
        errors_dict (dict): legacy single-N input
        errors_per_band (dict): {N: {"T_mae": array, "emissivity_mae": array}}
        save_path (str or None)
    """
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(6.5, 4))
        if errors_per_band is not None:
            cmap = plt.get_cmap("viridis", max(2, len(errors_per_band)))
            for i, (n, errs) in enumerate(sorted(errors_per_band.items())):
                ax.plot(snr_values, errs["T_mae"], marker="o",
                        label=f"N = {n}", color=cmap(i))
        else:
            if errors_dict is None:
                raise ValueError("Either errors_dict or errors_per_band must be given.")
            if "T_mae" in errors_dict:
                ax.plot(snr_values, errors_dict["T_mae"], marker="o",
                        label="Mean |ΔT|", color="steelblue")
            if "emissivity_mae" in errors_dict:
                ax2 = ax.twinx()
                ax2.plot(snr_values, errors_dict["emissivity_mae"], marker="s",
                         linestyle="--", label="ε mean |Δ|", color="darkorange")
                ax2.set_ylabel("Emissivity mean |Δ|")
                ax2.legend(loc="upper right")
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("Mean |ΔT| (K)")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="upper right")
        _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Figure 5: NEDT sensitivity
# ---------------------------------------------------------------------------

def plot_nedt_sensitivity(nedt_values, T_errors=None, ci_lower=None, ci_upper=None,
                           save_path=None, errors_per_band=None):
    """
    Figure 5: temperature RMSE vs NEDT.

    NEDT (Noise Equivalent Differential Temperature) is the detector-grounded
    noise model: σ_L = NEDT · dB/dT. Two display modes:
      - Single-N (legacy): pass T_errors and optional CI bands; shows shaded 95% CI.
      - Multi-N: pass errors_per_band={N: T_errors_array}; one line per band count.

    Args:
        nedt_values (np.ndarray): NEDT in Kelvin
        T_errors (np.ndarray): legacy single-N mean |ΔT|
        ci_lower, ci_upper (np.ndarray or None): legacy single-N CI bounds
        errors_per_band (dict): {N: T_errors_array} for multi-N mode
        save_path (str or None)
    """
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(6.5, 4))
        if errors_per_band is not None:
            cmap = plt.get_cmap("viridis", max(2, len(errors_per_band)))
            for i, (n, errs) in enumerate(sorted(errors_per_band.items())):
                ax.plot(nedt_values, errs, marker="o", label=f"N = {n}",
                        color=cmap(i))
        else:
            if T_errors is None:
                raise ValueError("Either T_errors or errors_per_band must be given.")
            ax.plot(nedt_values, T_errors, marker="o", color="steelblue",
                    label="Mean |ΔT|")
            if ci_lower is not None and ci_upper is not None:
                ax.fill_between(nedt_values, ci_lower, ci_upper,
                                alpha=0.25, color="steelblue", label="95% CI")
        ax.set_xlabel("NEDT (K)")
        ax.set_ylabel("Mean |ΔT| (K)")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend()
        _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Figure 6: Wavelength shift sensitivity
# ---------------------------------------------------------------------------

def plot_wavelength_shift_sensitivity(delta_lambda_values, errors=None, save_path=None,
                                       errors_per_band=None):
    """
    Figure 6: reconstruction error vs wavelength shift δλ.

    Tests sensitivity to spectrometer mis-calibration. The model assumes
    bands are centered at nominal λ_i, but δλ shifts them by a constant
    offset. Shao et al. (2020) recommend |δλ| < one band-spacing — at N=10
    over 8-14 µm that's ≈ 667 nm, so ±500 nm is within the operational range.

    What to look for:
      - Approximate symmetry around δλ = 0 (well-calibrated reference)
      - Asymmetry indicates the emissivity has structure shifting bands across
      - Slope at zero gives the calibration tolerance budget

    Args:
        delta_lambda_values (np.ndarray): shift values in meters
        errors (np.ndarray): temperature reconstruction error in K
        save_path (str or None)
        errors_per_band (dict): {N: errors_array} for multi-N mode
    """
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(6.5, 4))
        dlam_nm = delta_lambda_values * 1e9
        if errors_per_band is not None:
            cmap = plt.get_cmap("viridis", max(2, len(errors_per_band)))
            for i, (n, errs) in enumerate(sorted(errors_per_band.items())):
                ax.plot(dlam_nm, errs, marker="o", label=f"N = {n}",
                        color=cmap(i))
            ax.legend()
        else:
            ax.plot(dlam_nm, errors, marker="o", color="darkorange")
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Wavelength Shift δλ (nm)")
        ax.set_ylabel("Mean |ΔT| (K)")
        ax.grid(True, alpha=0.3)
        _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Figure 7: Combined perturbation heatmap
# ---------------------------------------------------------------------------

def plot_combined_perturbation(nedt_values, delta_lambda_values, mean_errors,
                                save_path=None):
    """
    Figure 7: 2-D heatmap of T error vs (NEDT, δλ).

    Joint perturbation Monte Carlo: noise and miscalibration applied
    together to test interaction effects. Green = stable (low error),
    red = unstable. The boundary between green and red regions defines
    the operating envelope: any (NEDT, δλ) inside it gives acceptable T_rmse.

    What to look for:
      - Asymmetry along axes — does noise dominate or miscalibration?
      - Synergy: does combined perturbation amplify error beyond the sum?
      - Iso-error contours give engineering tolerance budgets

    Args:
        nedt_values (np.ndarray): shape (n_nedt,)
        delta_lambda_values (np.ndarray): shape (n_dlam,)
        mean_errors (np.ndarray): shape (n_nedt, n_dlam)
        save_path (str or None)
    """
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(7, 5))
        dlam_nm = delta_lambda_values * 1e9
        pcm = ax.pcolormesh(dlam_nm, nedt_values, mean_errors,
                            cmap="RdYlGn_r", shading="auto")
        plt.colorbar(pcm, ax=ax, label="Mean |ΔT| (K)")
        # 1 K iso-error contour — community LST accuracy target (cf. Shao 2020).
        cs = ax.contour(dlam_nm, nedt_values, mean_errors,
                        levels=[1.0], colors="black", linewidths=1.4)
        ax.clabel(cs, inline=True, fontsize=10, fmt="%.0f K")
        ax.set_xlabel("Wavelength Shift δλ (nm)")
        ax.set_ylabel("NEDT (K)")
        _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Figures 8 & 9: General stability heatmap
# ---------------------------------------------------------------------------

def plot_stability_map(x_values, y_values, condition_numbers, xlabel, ylabel,
                        title, log_scale=True, save_path=None,
                        cbar_label=None, contour_value=None):
    """
    General-purpose 2-D stability heatmap.

    Reused for Figures 8 (band count x SNR) and 9 (variability x resolution).
    Colormap goes red→green→red so 'good' regions visually pop. The white
    contour line at κ=100 marks the conventional threshold between
    well-conditioned and ill-conditioned regimes.

    What to look for:
      - Phase boundary at κ=100 — the operational frontier of the design
      - Direction of the gradient — which axis dominates the conditioning
      - Plateau regions — where adding more (bands/SNR) yields diminishing returns

    Args:
        x_values (np.ndarray): shape (Nx,) — horizontal axis
        y_values (np.ndarray): shape (Ny,) — vertical axis
        condition_numbers (np.ndarray): shape (Ny, Nx) — note row=y convention
        xlabel, ylabel, title (str)
        log_scale (bool): apply log10 to condition numbers before plotting
        save_path (str or None)
    """
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(7, 5))
        data = np.log10(np.clip(condition_numbers, 1, None)) if log_scale else condition_numbers
        pcm = ax.pcolormesh(x_values, y_values, data,
                            cmap="RdYlGn_r", shading="auto")
        label = cbar_label if cbar_label is not None else (
            "log₁₀(κ)" if log_scale else "κ"
        )
        plt.colorbar(pcm, ax=ax, label=label)

        # Optional contour line at a meaningful threshold
        if contour_value is not None:
            cv = np.log10(contour_value) if log_scale else contour_value
            try:
                cs = ax.contour(x_values, y_values, data, levels=[cv],
                                colors="white", linewidths=1.5)
                ax.clabel(cs, fmt=f"{contour_value:g}", fontsize=8)
            except Exception:
                pass
        elif log_scale:
            try:
                cs = ax.contour(x_values, y_values, data, levels=[2.0],
                                colors="white", linewidths=1.5)
                ax.clabel(cs, fmt="κ=100", fontsize=8)
            except Exception:
                pass

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Figure 10: Hybrid vs physics κ comparison
# ---------------------------------------------------------------------------

def plot_hybrid_conditioning(physics_kappas, hybrid_kappas, save_path=None):
    """
    Figure 10: scatter or paired bar of κ_physics vs κ_hybrid.

    One point per test sample. The diagonal y=x line is the break-even
    reference: points below it mean the hybrid model regularised the
    inversion (lower κ); points above mean the residual network *worsened*
    conditioning. The Lipschitz bound on the residual MLP guarantees a
    bounded effect, but doesn't guarantee improvement on every sample.

    What to look for:
      - Cloud below diagonal = consistent hybrid improvement
      - Cloud spanning diagonal = sample-dependent benefit
      - Outliers above diagonal = adversarial cases worth investigating

    Args:
        physics_kappas (np.ndarray): shape (M,)
        hybrid_kappas (np.ndarray): shape (M,)
        save_path (str or None)
    """
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(6, 5))
        finite = np.isfinite(physics_kappas) & np.isfinite(hybrid_kappas)
        p = physics_kappas[finite]
        h = hybrid_kappas[finite]
        if p.size == 0:
            raise ValueError(
                "plot_hybrid_conditioning: no finite (κ_physics, κ_hybrid) pairs. "
                "Hybrid Jacobian likely diverged — check training loss and "
                "ResidualMLP.L_scale."
            )
        ax.scatter(p, h, alpha=0.5, s=15, color="steelblue", label="Samples")
        lim_max = max(np.nanmax(p), np.nanmax(h))
        lim_min = min(np.nanmin(p), np.nanmin(h))
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", linewidth=0.8,
                label="κ_hybrid = κ_physics")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("κ (Physics only)")
        ax.set_ylabel("κ (Hybrid)")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Figure 11: Hybrid Jacobian singular value comparison
# ---------------------------------------------------------------------------

def plot_hybrid_singular_values(physics_svd, hybrid_svd, save_path=None):
    """
    Figure 11: side-by-side singular value spectra for physics and hybrid.

    Two bar charts in matching y-axis scale, showing how the residual model
    redistributes singular values. The κ value is in each panel title.
    Useful to see WHERE the regularisation acts: whether hybrid lifts the
    smallest σ (boosts identifiability) or compresses the largest σ
    (reduces sensitivity to dominant directions).

    What to look for:
      - σ_min: hybrid should lift it if regularisation works
      - Spectrum flattening: smaller drop-off = better conditioning
      - Both panels for one representative sample (median-κ test point)

    Args:
        physics_svd (dict): from jacobian_svd — must have key 's'
        hybrid_svd (dict): same structure for the hybrid Jacobian
        save_path (str or None)
    """
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, svd, label in [(axes[0], physics_svd, "Physics"),
                                (axes[1], hybrid_svd, "Hybrid")]:
            s = svd["s"]
            ax.bar(np.arange(1, len(s) + 1), s, color="steelblue", edgecolor="black",
                   linewidth=0.5)
            ax.set_yscale("log")
            ax.set_xlabel("Singular Value Index")
            ax.grid(True, axis="y", alpha=0.3)
        axes[0].set_ylabel("Singular Value σ")
        _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Figure 12: Distribution shift robustness
# ---------------------------------------------------------------------------

def plot_distribution_shift_cross_variant(shift_levels, physics_errors,
                                            variant_errors, save_path=None):
    """
    Cross-variant fig 12: one OOD curve per hybrid variant against the shared
    physics-only reference. `variant_errors` is an ordered dict
    {label: hybrid_T_mae_array}; labels are used verbatim in the legend.
    """
    style = {
        "baseline":           {"color": "#ff7f0e", "marker": "s", "ls": "--"},
        "physics-consistency":{"color": "#30ad30", "marker": "^", "ls": "--"},
        "density-gated":      {"color": "#f91414", "marker": "D", "ls": "--"},
        "forward-space":      {"color": "#28e1f5", "marker": "v", "ls": "--"},
    }
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(shift_levels, physics_errors, marker="o", linestyle="-",
                color="black", label="Physics only", linewidth=1.6)
        for label, errs in variant_errors.items():
            s = style.get(label, {"color": "gray", "marker": "x", "ls": "--"})
            ax.plot(shift_levels, errs, marker=s["marker"], linestyle=s["ls"],
                    color=s["color"], label=label.capitalize(), linewidth=1.4)
        ax.set_xlabel("Distribution shift (K above training range)")
        ax.set_ylabel("Mean |ΔT| (K)")
        ax.legend(loc="best", frameon=True)
        ax.grid(True, alpha=0.3)
        _save_or_show(fig, save_path)


def plot_distribution_shift(shift_levels, physics_errors, hybrid_errors,
                              save_path=None):
    """
    Figure 12: T error vs distribution shift level, physics vs hybrid.

    Out-of-distribution test: train the hybrid on T ∈ [250, 380] K, then
    evaluate on T ∈ [380, 430] K (the offset axis). Physics-only is the
    distribution-free baseline (it doesn't learn from data, so OOD doesn't
    hurt it). Hybrid is expected to lose accuracy as shift grows — the
    question is *how fast*. Spectral-norm bounding caps the worst-case
    degradation, which should show up as a slow rather than catastrophic rise.

    What to look for:
      - In-distribution (offset=0): hybrid ≤ physics (regularisation gain)
      - Crossover point: where hybrid becomes worse than physics
      - Slope of the hybrid line: steepness reflects Lipschitz tightness

    Args:
        shift_levels (np.ndarray): e.g. OOD temperature offset in K
        physics_errors (np.ndarray): mean |ΔT| for physics-only
        hybrid_errors (np.ndarray): mean |ΔT| for hybrid
        save_path (str or None)
    """
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(shift_levels, physics_errors, marker="o", label="Physics only",
                color="steelblue")
        ax.plot(shift_levels, hybrid_errors, marker="s", linestyle="--",
                label="Hybrid", color="darkorange")
        ax.set_xlabel("Distribution Shift (K above training range)")
        ax.set_ylabel("Mean |ΔT| (K)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Figure 13: L-curve for lambda_phys sweep (Hansen 1992 corner-distance)
# ---------------------------------------------------------------------------

def plot_lcurve(lambdas, accuracy, consistency, knee_idx, save_path=None):
    """
    Figure 13: log-log L-curve of accuracy error vs physics-consistency violation
    for a sweep over the physics-loss weight λ_phys.

    The 'knee' (Hansen 1992 corner-distance heuristic) is highlighted; it
    marks the regularisation weight that minimises distance to the origin
    after log10-normalising both axes to [0, 1].

    Args:
        lambdas (sequence): λ_phys values swept
        accuracy (np.ndarray): mean |ΔT| per λ
        consistency (np.ndarray): ‖L − P(x̂)‖² / ‖L_scale‖² per λ
        knee_idx (int): index of the knee in `lambdas`
        save_path (str or None)
    """
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.loglog(consistency, accuracy, "o-", color="steelblue", linewidth=1.2)
        for lam, c, a in zip(lambdas, consistency, accuracy):
            ax.annotate(f"λ={lam:g}", (c, a),
                        textcoords="offset points", xytext=(7, 5), fontsize=9)
        ax.plot(consistency[knee_idx], accuracy[knee_idx], "o",
                color="red", markersize=14, fillstyle="none", linewidth=1.5,
                label=f"knee: λ = {lambdas[knee_idx]:g}")
        ax.set_xlabel("Physical consistency violation  ‖L − P(x̂)‖² / ‖L_scale‖²")
        ax.set_ylabel("Accuracy error  mean |ΔT| (K)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best")
        _save_or_show(fig, save_path)


def plot_lcurve_multiseed(per_seed_df, summary_df, save_path=None):
    """
    Figure 13 (multi-seed variant): regularisation sweep with Hansen-
    convention axes — accuracy (residual) on x, physics violation
    (regulariser term) on y, both log scale. The median curve carries
    2D interquartile error bars; individual seeds are shown as a faint
    scatter to expose raw spread.

    Args:
        per_seed_df (pd.DataFrame): rows of (seed, lambda_phys, accuracy_T_mae,
            phys_violation, ...).
        summary_df (pd.DataFrame): rows of (lambda_phys, accuracy_median,
            accuracy_p25, accuracy_p75, violation_median, violation_p25,
            violation_p75, is_knee_median).
        save_path (str or None)
    """
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(7, 5))

        summary_df = summary_df.sort_values("lambda_phys").reset_index(drop=True)
        n_seeds = per_seed_df["seed"].nunique()

        # Hansen-convention orientation: residual on x, regulariser on y
        ax.scatter(
            per_seed_df.accuracy_T_mae, per_seed_df.phys_violation,
            s=18, color="lightsteelblue", alpha=0.6, zorder=1,
            label=f"individual seeds (n={n_seeds})",
        )

        x_med = summary_df.accuracy_median.values
        y_med = summary_df.violation_median.values
        x_err = [
            x_med - summary_df.accuracy_p25.values,
            summary_df.accuracy_p75.values - x_med,
        ]
        y_err = [
            y_med - summary_df.violation_p25.values,
            summary_df.violation_p75.values - y_med,
        ]
        ax.errorbar(
            x_med, y_med, xerr=x_err, yerr=y_err,
            fmt="o-", color="steelblue", linewidth=1.6, markersize=7,
            capsize=4, elinewidth=1.1, ecolor="steelblue",
            zorder=3, label="median, p25-p75",
        )

        # Per-lambda label placement to avoid overlap with markers/error bars
        label_offsets = {
            0.01: (-20, 10),
            0.1:  (-5, 14),
            1.0:  (-15, 8),
            10.0: (12, 8),
            100.0: (10, 8),
        }
        for lam, x, y in zip(summary_df.lambda_phys, x_med, y_med):
            dx, dy = label_offsets.get(float(lam), (10, 8))
            ax.annotate(
                f"λ={lam:g}", (x, y),
                textcoords="offset points", xytext=(dx, dy), fontsize=11,
                color="black",
            )

        knee = summary_df[summary_df.is_knee_median].iloc[0]
        ax.plot(
            knee.accuracy_median, knee.violation_median, "o",
            color="red", markersize=16, fillstyle="none", linewidth=1.8,
            zorder=4, label=f"knee (median): λ = {knee.lambda_phys:g}",
        )

        ax.set_yscale("log")
        ax.set_xlabel("Accuracy error  mean |ΔT| (K)")
        ax.set_ylabel("Physics violation  ‖L − P(x̂)‖² / ‖L‖²")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="upper left", framealpha=0.9)
        _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Convenience: save all figures
# ---------------------------------------------------------------------------

def save_all_figures(results, output_dir, dpi=300, fmt="pdf"):
    """
    Save all thesis figures from a results dictionary.

    Args:
        results (dict): pre-computed data for each figure (see individual functions)
        output_dir (str): directory for output files
        dpi (int)
        fmt (str): 'pdf' or 'png'
    """
    os.makedirs(output_dir, exist_ok=True)

    def _path(name):
        return os.path.join(output_dir, f"{name}.{fmt}")

    if "fig1" in results:
        d = results["fig1"]
        plot_condition_numbers(d["band_counts"], d["condition_number_dict"],
                               save_path=_path("fig1_condition_numbers"))
    if "fig2" in results:
        d = results["fig2"]
        plot_singular_values(d["svd_dict"], save_path=_path("fig2_singular_values"))
    if "fig3" in results:
        d = results["fig3"]
        plot_spectral_placement(d["placement_results"],
                                save_path=_path("fig3_spectral_placement"))
    if "fig4" in results:
        d = results["fig4"]
        plot_noise_sensitivity(d["snr_values"], d["errors_dict"],
                               save_path=_path("fig4_noise_sensitivity"))
    if "fig5" in results:
        d = results["fig5"]
        plot_nedt_sensitivity(d["nedt_values"], d["T_errors"],
                              d.get("ci_lower"), d.get("ci_upper"),
                              save_path=_path("fig5_nedt_sensitivity"))
    if "fig6" in results:
        d = results["fig6"]
        plot_wavelength_shift_sensitivity(d["delta_lambda_values"], d["errors"],
                                          save_path=_path("fig6_wavelength_shift"))
    if "fig7" in results:
        d = results["fig7"]
        plot_combined_perturbation(d["nedt_values"], d["delta_lambda_values"],
                                   d["mean_errors"],
                                   save_path=_path("fig7_combined_perturbation"))
    if "fig8" in results:
        d = results["fig8"]
        plot_stability_map(d["band_counts"], d["snr_db_values"],
                           d["condition_numbers"],
                           xlabel="Band Count", ylabel="SNR (dB)",
                           title="Stability Map: Band Count × SNR",
                           save_path=_path("fig8_stability_band_snr"))
    if "fig9" in results:
        d = results["fig9"]
        plot_stability_map(d["band_counts"], d["variability_levels"],
                           d["condition_numbers"],
                           xlabel="Band Count", ylabel="Emissivity Variability (std)",
                           title="Stability Map: Variability × Resolution",
                           save_path=_path("fig9_stability_variability"))
    if "fig10" in results:
        d = results["fig10"]
        plot_hybrid_conditioning(d["physics_kappas"], d["hybrid_kappas"],
                                 save_path=_path("fig10_hybrid_conditioning"))
    if "fig11" in results:
        d = results["fig11"]
        plot_hybrid_singular_values(d["physics_svd"], d["hybrid_svd"],
                                    save_path=_path("fig11_hybrid_singular_values"))
    if "fig12" in results:
        d = results["fig12"]
        plot_distribution_shift(d["shift_levels"], d["physics_errors"],
                                d["hybrid_errors"],
                                save_path=_path("fig12_distribution_shift"))
