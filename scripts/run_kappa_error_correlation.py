"""
Empirical κ-vs-T_mae correlation diagnostic — raw vs scaled Jacobian.

----------------------------------------------------------------------
What this script answers
----------------------------------------------------------------------
The thesis claim is that the condition number κ of the forward-model
Jacobian is the right diagnostic for inversion stability — i.e. samples
with high κ should also have high T_mae after running the unconstrained
TRF inversion. This script measures that empirically by computing
(κ, |ΔT|) pairs over a held-out test set and reporting Pearson and
Spearman correlations.

We run the correlation for two versions of κ:

1. Raw κ: jacobian_svd(J).condition_number on the unscaled Jacobian.
   Raw J has unit-mismatched columns: J[:,0] is per-Kelvin, J[:,1:] is
   per-(unit emissivity). Raw κ saturates near ~2 in our LWIR setup.

2. Scaled κ: jacobian_svd(J · diag(σ_T, σ_ε, ..., σ_ε)).condition_number.
   Column-scaled by prior 1σ parameter uncertainties (Tarantola 2005 §3.2,
   the J·C_M^{1/2} construction). Makes κ scale-invariant and a
   physically meaningful identifiability metric. With σ_T = 5 K,
   σ_ε = 0.05, scaled κ ranges roughly 10-60 — much wider variation
   than raw κ and the form thesis §1 actually defines as identifiability.

Both metrics use the helper `condition_to_error_correlation` already
defined in `src/utils/metrics.py:111`.

Pearson is run on (log1p(κ), |ΔT|); Spearman on raw ranks.

----------------------------------------------------------------------
"""
import os
import sys
import io
import numpy as np

# Force UTF-8 stdout so Greek letters in print() don't crash on Windows cp1252.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from _hybrid_common import (
    MATERIALS, N_BANDS, T_TRAIN_RANGE, generate_dataset, set_seed,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.inversion.jacobian import compute_jacobian, jacobian_svd, scale_jacobian
from src.inversion.tex_inversion import invert_tex
from src.utils.metrics import condition_to_error_correlation
from src.simulation.generator import generate_emissivity_curve
from src.forward_model import blackbody_environment
from _hybrid_common import WL_RANGE, L_ENV_T


N_SAMPLES = 500  # matches the §3.9 in-distribution test set size

# Tarantola priors — same as run_conditioning.py for direct comparability.
T_SCALE = 5.0   # K
E_SCALE = 0.05  # dimensionless

TABLES_DIR = os.path.join(ROOT, "results", "tables")
TABLE_PATHS = {
    "correlation": os.path.join(TABLES_DIR, "table_kappa_correlation.tex"),
    "distribution": os.path.join(TABLES_DIR, "table_kappa_distribution.tex"),
    "tail_bulk": os.path.join(TABLES_DIR, "table_kappa_tail_bulk.tex"),
}


def fig1b_equivalent_kappa(wl, n_bands):
    """
    Sanity check: compute scaled κ at T = 300 K with library ε (no variability),
    matching the run_conditioning.py / fig1b setup. Returns one κ per material.
    """
    L_env_fig1b = blackbody_environment(wl, L_ENV_T)
    out = {}
    for mat in MATERIALS:
        eps_lib = generate_emissivity_curve(
            n_bands, material=mat, wavelength_range=WL_RANGE,
            variability=0.0, seed=None,
        )
        J = compute_jacobian(wl, 300.0, eps_lib, L_env_fig1b)
        out[mat] = {
            "raw": jacobian_svd(J)["condition_number"],
            "scaled": jacobian_svd(scale_jacobian(J, T_SCALE, E_SCALE))["condition_number"],
            "eps_mean": float(np.mean(eps_lib)),
        }
    return out


def per_material_correlations(kappas, abs_T_err, mat_idx, label):
    """Print + collect the per-material correlation rows and percentile dict."""
    print(f"\n--- {label} κ ---")
    print(f"  {'material':>12}  {'n':>4}  {'<κ>':>8}"
          f"  {'p10':>7}  {'p50':>7}  {'p90':>7}  {'<|ΔT|>':>8}"
          f"  {'pearson_r':>9}  {'spearman_r':>10}")
    rows = []
    pct = {}
    for j, mat in enumerate(MATERIALS):
        mask = mat_idx == j
        if mask.sum() < 5:
            continue
        k_m = kappas[mask]
        c = condition_to_error_correlation(k_m, abs_T_err[mask])
        rows.append((
            mat, float(np.mean(k_m)), float(np.mean(abs_T_err[mask])),
            c["pearson_r"], c["spearman_r"], int(mask.sum()),
        ))
        pct[mat] = {
            "mean": float(np.mean(k_m)),
            "p10": float(np.percentile(k_m, 10)),
            "p50": float(np.percentile(k_m, 50)),
            "p90": float(np.percentile(k_m, 90)),
        }
        print(f"  {mat:>12}  {int(mask.sum()):>4}  {np.mean(k_m):>8.2f}"
              f"  {np.percentile(k_m, 10):>7.2f}"
              f"  {np.percentile(k_m, 50):>7.2f}"
              f"  {np.percentile(k_m, 90):>7.2f}"
              f"  {np.mean(abs_T_err[mask]):>8.2f}"
              f"  {c['pearson_r']:>9.3f}  {c['spearman_r']:>10.3f}")
    return rows, pct


def aggregate_correlations(kappas, abs_T_err, mat_idx, label):
    """Print + return the aggregate (all + no-metal) correlations."""
    c_all = condition_to_error_correlation(kappas, abs_T_err)
    keep = mat_idx != MATERIALS.index("metal")
    c_no_metal = condition_to_error_correlation(kappas[keep], abs_T_err[keep])
    print(f"\n  Aggregate ({label} κ, all 8): "
          f"pearson_r = {c_all['pearson_r']:+.3f} (p={c_all['pearson_p']:.3g})"
          f"   spearman_r = {c_all['spearman_r']:+.3f} (p={c_all['spearman_p']:.3g})")
    print(f"  Aggregate ({label} κ, no metal): "
          f"pearson_r = {c_no_metal['pearson_r']:+.3f} (p={c_no_metal['pearson_p']:.3g})"
          f"   spearman_r = {c_no_metal['spearman_r']:+.3f} (p={c_no_metal['spearman_p']:.3g})")
    return c_all, c_no_metal


def tail_vs_bulk_split(kappas, abs_T_err, label, top_pct=10):
    """
    Split samples by κ percentile and compare |ΔT| in tail vs bulk.

    Hypothesis: scaled κ's heavy upper tail is where the κ-as-amplification
    mechanism shows up. Bottom-90% samples are bias-floor-limited; top-10%
    are κ-limited. If true, |ΔT| should jump in the tail.
    """
    threshold = np.percentile(kappas, 100 - top_pct)
    tail_mask = kappas >= threshold
    bulk_mask = ~tail_mask
    t_kappa = kappas[tail_mask]
    t_err = abs_T_err[tail_mask]
    b_kappa = kappas[bulk_mask]
    b_err = abs_T_err[bulk_mask]
    c_tail = condition_to_error_correlation(t_kappa, t_err)
    c_bulk = condition_to_error_correlation(b_kappa, b_err)
    print(f"\n  {label} κ tail-vs-bulk split "
          f"(threshold = p{100-top_pct} = {threshold:.2f}):")
    print(f"    bulk (bottom {100-top_pct}%, n={int(bulk_mask.sum())}): "
          f"⟨κ⟩ = {np.mean(b_kappa):.2f}, ⟨|ΔT|⟩ = {np.mean(b_err):.2f} K, "
          f"pearson_r = {c_bulk['pearson_r']:+.3f}")
    print(f"    tail (top {top_pct}%,    n={int(tail_mask.sum())}): "
          f"⟨κ⟩ = {np.mean(t_kappa):.2f}, ⟨|ΔT|⟩ = {np.mean(t_err):.2f} K, "
          f"pearson_r = {c_tail['pearson_r']:+.3f}")
    print(f"    tail/bulk ⟨|ΔT|⟩ ratio: "
          f"{np.mean(t_err)/max(np.mean(b_err), 1e-12):.2f}×")
    return {
        "threshold": float(threshold),
        "bulk_mean_kappa": float(np.mean(b_kappa)),
        "bulk_mean_err": float(np.mean(b_err)),
        "bulk_pearson": c_bulk["pearson_r"],
        "tail_mean_kappa": float(np.mean(t_kappa)),
        "tail_mean_err": float(np.mean(t_err)),
        "tail_pearson": c_tail["pearson_r"],
        "ratio": float(np.mean(t_err) / max(np.mean(b_err), 1e-12)),
    }


def between_material_correlation(rows, label):
    """Correlation between per-material mean κ and per-material mean |ΔT|."""
    if len(rows) < 3:
        return None
    mean_kappas = np.array([r[1] for r in rows])
    mean_errors = np.array([r[2] for r in rows])
    c = condition_to_error_correlation(mean_kappas, mean_errors)
    print(f"  Between-material ({label} κ): "
          f"pearson_r on (mean κ, mean |ΔT|) = {c['pearson_r']:+.3f}"
          f"  (n={len(rows)} materials)")
    return c


def _write_lines(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def write_latex_table(raw_rows, scaled_rows, raw_agg, raw_no_metal,
                      scaled_agg, scaled_no_metal, raw_between, scaled_between,
                      raw_pct, scaled_pct, raw_tail, scaled_tail, fig1b,
                      paths):
    """
    Booktabs LaTeX tables summarising raw vs scaled κ correlations.

    Emits three .tex files (paths is a dict with keys
    "correlation", "distribution", "tail_bulk") so each can be \\input
    at the natural anchor in the Results chapter:

      correlation table  → §5.6.2 (κ-vs-|ΔT| diagnostic)
      distribution table → §5.4.1 (per-material κ summary)
      tail-bulk table    → §5.6.2 (headline 1.93×/0.26× split)
    """
    # --- Table 1: correlation ---
    lines = []
    lines.append(r"% Empirical kappa-vs-|dT| correlation, raw vs scaled Jacobian.")
    lines.append(r"% Generated by scripts/run_kappa_error_correlation.py.")
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \begin{tabular}{lrrrrrrrr}")
    lines.append(r"    \toprule")
    lines.append(r"    & \multicolumn{4}{c}{Raw $\kappa$} & \multicolumn{4}{c}{Scaled $\kappa$} \\")
    lines.append(r"    \cmidrule(lr){2-5} \cmidrule(lr){6-9}")
    lines.append(r"    Material & $\langle\kappa\rangle$ & $\langle|\Delta T|\rangle$ & "
                 r"Pears. & Spear. & $\langle\kappa\rangle$ & $\langle|\Delta T|\rangle$ & "
                 r"Pears. & Spear. \\")
    lines.append(r"    \midrule")
    raw_by_mat = {r[0]: r for r in raw_rows}
    sca_by_mat = {r[0]: r for r in scaled_rows}
    for mat in [r[0] for r in raw_rows]:
        r = raw_by_mat[mat]
        s = sca_by_mat[mat]
        lines.append(
            f"    {mat:<11} & {r[1]:.2f} & {r[2]:.2f} & "
            f"{r[3]:+.3f} & {r[4]:+.3f} & "
            f"{s[1]:.2f} & {s[2]:.2f} & "
            f"{s[3]:+.3f} & {s[4]:+.3f} \\\\"
        )
    lines.append(r"    \midrule")
    lines.append(
        f"    All 8 pooled       &      &       & "
        f"{raw_agg['pearson_r']:+.3f} & {raw_agg['spearman_r']:+.3f} &      &       & "
        f"{scaled_agg['pearson_r']:+.3f} & {scaled_agg['spearman_r']:+.3f} \\\\"
    )
    lines.append(
        f"    No-metal pooled    &      &       & "
        f"{raw_no_metal['pearson_r']:+.3f} & {raw_no_metal['spearman_r']:+.3f} &      &       & "
        f"{scaled_no_metal['pearson_r']:+.3f} & {scaled_no_metal['spearman_r']:+.3f} \\\\"
    )
    if raw_between is not None and scaled_between is not None:
        lines.append(
            f"    Between-material   &      &       & "
            f"{raw_between['pearson_r']:+.3f} & {raw_between['spearman_r']:+.3f} &      &       & "
            f"{scaled_between['pearson_r']:+.3f} & {scaled_between['spearman_r']:+.3f} \\\\"
        )
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"  \caption{Empirical correlation between Jacobian condition number")
    lines.append(r"    and inversion error $|\Delta T|$ on the in-distribution test")
    lines.append(r"    set (" + str(N_SAMPLES) +
                 r" samples, $T \in [250, 380]$~K, NEDT $= 0.1$~K). \emph{Raw}")
    lines.append(r"    $\kappa$ uses the unscaled Jacobian; \emph{Scaled} $\kappa$")
    lines.append(r"    uses the Tarantola (2005) §3.2 column-scaled form")
    lines.append(r"    $\kappa(J \cdot \operatorname{diag}(\sigma_T, \sigma_\varepsilon, \ldots))$")
    lines.append(r"    with $\sigma_T = " + f"{T_SCALE:g}" +
                 r"$~K, $\sigma_\varepsilon = " + f"{E_SCALE:g}" + r"$,")
    lines.append(r"    which makes $\kappa$ scale-invariant and is the form")
    lines.append(r"    used for identifiability in §1.7. Pearson is computed on")
    lines.append(r"    $(\log(1{+}\kappa), |\Delta T|)$; Spearman on raw ranks.}")
    lines.append(r"  \label{tab:kappa-error-correlation}")
    lines.append(r"\end{table}")
    _write_lines(paths["correlation"], lines)

    # --- Table 2: per-material distribution + fig1b sanity check ---
    lines = []
    lines.append(r"% Per-material kappa distribution (raw and scaled).")
    lines.append(r"% Generated by scripts/run_kappa_error_correlation.py.")
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \begin{tabular}{lrrrrrrrr}")
    lines.append(r"    \toprule")
    lines.append(r"    & \multicolumn{4}{c}{Raw $\kappa$ (T random, $\sigma_\varepsilon = 0.02$)} "
                 r"& \multicolumn{4}{c}{Scaled $\kappa$ (T random, $\sigma_\varepsilon = 0.02$)} \\")
    lines.append(r"    \cmidrule(lr){2-5} \cmidrule(lr){6-9}")
    lines.append(r"    Material & $\langle\kappa\rangle$ & p10 & p50 & p90 & "
                 r"$\langle\kappa\rangle$ & p50 & p90 & fig1b \\")
    lines.append(r"    \midrule")
    for mat in [r[0] for r in raw_rows]:
        rp = raw_pct[mat]
        sp = scaled_pct[mat]
        f1b = fig1b[mat]
        lines.append(
            f"    {mat:<11} & {rp['mean']:.2f} & {rp['p10']:.2f} & {rp['p50']:.2f} & {rp['p90']:.2f} & "
            f"{sp['mean']:.2f} & {sp['p50']:.2f} & {sp['p90']:.2f} & {f1b['scaled']:.2f} \\\\"
        )
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"  \caption{Per-material $\kappa$ distribution on the same 500-sample")
    lines.append(r"    test set. \emph{Means} are inflated by heavy upper tails for")
    lines.append(r"    moderately-sloped emissivity materials (concrete, tree); the")
    lines.append(r"    \emph{medians} (p50) give the typical-sample value. The fig1b")
    lines.append(r"    column shows the equivalent scaled $\kappa$ at the deterministic")
    lines.append(r"    fig 1b setpoint ($T = 300$~K, library $\varepsilon$, no per-sample")
    lines.append(r"    variability), extended to all eight materials. The")
    lines.append(r"    per-sample-perturbed medians (p50) in this table sit within")
    lines.append(r"    $\sim$1.5$\times$ of those setpoint values.}")
    lines.append(r"  \label{tab:kappa-distribution}")
    lines.append(r"\end{table}")
    _write_lines(paths["distribution"], lines)

    # --- Table 3: tail-vs-bulk split ---
    lines = []
    lines.append(r"% Tail-vs-bulk split at the kappa 90th percentile.")
    lines.append(r"% Generated by scripts/run_kappa_error_correlation.py.")
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \begin{tabular}{lrrrrr}")
    lines.append(r"    \toprule")
    lines.append(r"    & threshold (p90) & bulk $\langle\kappa\rangle$ & bulk $\langle|\Delta T|\rangle$ "
                 r"& tail $\langle\kappa\rangle$ & tail $\langle|\Delta T|\rangle$ \\")
    lines.append(r"    \midrule")
    lines.append(
        f"    Raw    & {raw_tail['threshold']:.2f} & "
        f"{raw_tail['bulk_mean_kappa']:.2f} & {raw_tail['bulk_mean_err']:.2f} & "
        f"{raw_tail['tail_mean_kappa']:.2f} & {raw_tail['tail_mean_err']:.2f} \\\\"
    )
    lines.append(
        f"    Scaled & {scaled_tail['threshold']:.2f} & "
        f"{scaled_tail['bulk_mean_kappa']:.2f} & {scaled_tail['bulk_mean_err']:.2f} & "
        f"{scaled_tail['tail_mean_kappa']:.2f} & {scaled_tail['tail_mean_err']:.2f} \\\\"
    )
    lines.append(r"    \midrule")
    lines.append(r"    \multicolumn{6}{l}{\emph{Tail / bulk ratios:} "
                 + f"raw = {raw_tail['ratio']:.2f}$\\times$ "
                 + f"\\ \\ scaled = {scaled_tail['ratio']:.2f}$\\times$}} \\\\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"  \caption{Tail-vs-bulk split at the $\kappa$ 90th percentile.")
    lines.append(r"    The ratio $\langle|\Delta T|\rangle_{\text{tail}} / \langle|\Delta T|\rangle_{\text{bulk}}$")
    lines.append(r"    is the headline diagnostic: $>1$ means high-$\kappa$ samples are")
    lines.append(r"    harder (amplification regime), $<1$ means high-$\kappa$ samples")
    lines.append(r"    are easier (dynamic-range / identifiability regime). Raw $\kappa$")
    lines.append(r"    tail is amplification-like; scaled $\kappa$ tail is identifiability-like,")
    lines.append(r"    confirming the two $\kappa$ flavours measure orthogonal properties (§3.3, §3.9.1b).}")
    lines.append(r"  \label{tab:kappa-tail-vs-bulk}")
    lines.append(r"\end{table}")
    _write_lines(paths["tail_bulk"], lines)


def main():
    set_seed()
    print("Empirical κ-vs-|ΔT| correlation on the §3.9 in-distribution test set")
    print(f"  N_BANDS={N_BANDS}  T in {T_TRAIN_RANGE}  samples={N_SAMPLES}")
    print(f"  Scaled κ priors: σ_T = {T_SCALE} K, σ_ε = {E_SCALE}\n")

    wl, L_env, T_true, eps_true, L_noisy, mat_idx = generate_dataset(
        N_SAMPLES, T_TRAIN_RANGE, seed=43
    )

    print("Sanity check — fig1b-equivalent κ "
          "(T = 300 K fixed, library ε, no variability):")
    fig1b = fig1b_equivalent_kappa(wl, N_BANDS)
    print(f"  {'material':>12}  {'eps_mean':>8}  {'raw κ':>8}  {'scaled κ':>10}")
    for mat in MATERIALS:
        f = fig1b[mat]
        print(f"  {mat:>12}  {f['eps_mean']:>8.3f}"
              f"  {f['raw']:>8.3f}  {f['scaled']:>10.3f}")

    print("\nComputing per-sample raw and scaled κ "
          "(T random, ε with variability=0.02)...")
    kappas_raw = np.empty(N_SAMPLES)
    kappas_scaled = np.empty(N_SAMPLES)
    for i in range(N_SAMPLES):
        J = compute_jacobian(wl, T_true[i], eps_true[i], L_env)
        kappas_raw[i] = jacobian_svd(J)["condition_number"]
        kappas_scaled[i] = jacobian_svd(
            scale_jacobian(J, T_SCALE, E_SCALE)
        )["condition_number"]

    print("Running physics inversion (serial)...")
    T_est = np.empty(N_SAMPLES)
    for i in range(N_SAMPLES):
        T_est[i] = invert_tex(wl, L_noisy[i], L_env)["T_est"]
    abs_T_err = np.abs(T_est - T_true)

    raw_rows, raw_pct = per_material_correlations(
        kappas_raw, abs_T_err, mat_idx, "Raw"
    )
    raw_agg, raw_no_metal = aggregate_correlations(
        kappas_raw, abs_T_err, mat_idx, "Raw"
    )
    raw_between = between_material_correlation(raw_rows, "Raw")
    raw_tail = tail_vs_bulk_split(kappas_raw, abs_T_err, "Raw")

    scaled_rows, scaled_pct = per_material_correlations(
        kappas_scaled, abs_T_err, mat_idx, "Scaled"
    )
    scaled_agg, scaled_no_metal = aggregate_correlations(
        kappas_scaled, abs_T_err, mat_idx, "Scaled"
    )
    scaled_between = between_material_correlation(scaled_rows, "Scaled")
    scaled_tail = tail_vs_bulk_split(kappas_scaled, abs_T_err, "Scaled")

    print("\nInterpretation guide:")
    print("  raw κ:    saturates near minimum (~2) for every material — no")
    print("            between-material signal; per-material correlations track")
    print("            T-distance-from-x0 (bias-floor mechanism, §3.8.1).")
    print("  scaled κ: Tarantola-rescaled. Restores the ε-dependence of the")
    print("            T-column relative weighting that raw SVD washes out.")
    print("            Should show stronger between-material correlation if")
    print("            κ-as-identifiability framing is the right diagnostic.")

    write_latex_table(
        raw_rows, scaled_rows, raw_agg, raw_no_metal,
        scaled_agg, scaled_no_metal, raw_between, scaled_between,
        raw_pct, scaled_pct, raw_tail, scaled_tail, fig1b,
        TABLE_PATHS,
    )
    print()
    for key, path in TABLE_PATHS.items():
        print(f"Wrote {key:>12}: {path}")


if __name__ == "__main__":
    main()
