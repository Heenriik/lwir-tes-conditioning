# Beyond Ill-Posedness: Hybrid Inversion for LWIR Temperature–Emissivity Separation

Code and reproducibility pipeline for the master's thesis
**"Beyond Ill-Posedness: Conditioning, Null-Space Ambiguity, and Hybrid Physics–ML Inversion for LWIR Temperature–Emissivity Separation"** (NTNU, 2026).

Long-wave infrared (LWIR, 8–14 μm) imaging measures surface temperature without external illumination, which makes it important for autonomous perception in fog, smoke, darkness and other low-visibility conditions. At its core sits the temperature–emissivity separation (TES) inverse problem, where $N$ spectral measurements must recover $N+1$ unknowns and the forward Jacobian carries a one-dimensional null space.

This repository implements the thesis's full diagnostic pipeline: forward model, Jacobian conditioning analysis, unconstrained Trust Region Reflective (TRF) inversion baseline, and four Lipschitz-bounded hybrid physics–ML residual variants.

---

## Central contribution

The thesis analyses LWIR TES by separating two distinct failure mechanisms:

1. **Image-side conditioning** of the forward Jacobian, measured by raw $\kappa(J)$ and the Tarantola prior-rescaled $\kappa(J \cdot C_M^{1/2})$. Raw $\kappa(J)$ saturates near 2 across all materials and band counts (formally well-conditioned); the scaled $\kappa$ separates materials by identifiability.
2. **Null-space ambiguity** from the underdetermined $N+1$-from-$N$ problem, which produces a **structural bias floor** in unconstrained TRF inversion. The bias is temperature-position dependent and emissivity-amplified, and persists even on noise-free radiance.

These are **orthogonal failure modes** and must be characterised independently. Four Lipschitz-bounded hybrid residual variants are then evaluated against the physics-only TRF baseline:

| Variant | Behaviour |
| --- | --- |
| **Parameter-space** (baseline) | Improves low-emissivity metal at the cost of degrading high-emissivity materials |
| **Physics-consistency loss** | Matches physics-only on average while limiting out-of-distribution drift |
| **Density-gated** | The only variant safer than physics-only at every tested OOD offset |
| **Forward-space** | Effectively inactive under an exact forward model |

The forward model uses the Kirchhoff opaque-surface radiance equation,

$$L(\lambda) = \varepsilon(\lambda) B(\lambda, T) + (1 - \varepsilon(\lambda)) L_{\mathrm{env}}(\lambda),$$

with downwelling radiance modelled as a blackbody at $T_{\mathrm{env}} = 280$ K. All hybrid residuals are spectrally-normalised so the network is globally 1-Lipschitz by construction.

---

## Project structure

```
.
├── src/
│   ├── forward_model/    # Planck radiance, Kirchhoff opaque-surface model
│   ├── inversion/        # Analytical Jacobian, TRF inversion, κ diagnostics
│   ├── simulation/       # Synthetic radiance pipeline, NEDT/calibration noise
│   ├── analysis/         # Band placement, conditioning sweeps
│   ├── hybrid/           # Spectrally-normalised ResidualMLP, four variants
│   └── utils/            # Plotting (matplotlib), metrics
├── configs/              # YAML defaults (simulation.yaml, noise.yaml)
├── scripts/              # One run_*.py per scenario; plot_all.py renders all figures
├── data/                 # ECOSTRESS emissivity spectra
├── results/              # Generated outputs (git-ignored)
│   ├── data/             # Tidy CSVs (one per figure)
│   ├── figures/          # PDFs rendered by plot_all.py
│   └── tables/           # LaTeX tables
├── environment.yaml
├── LICENSE
└── readme.md
```

Output directories (`results/`, `logs/`, `notebooks/`) are git-ignored; everything in them is regenerable from the scripts.

---

## Installation

```bash
conda env create -f environment.yaml
conda activate Mthesis
```

---

## Reproducing the thesis

Experiments and rendering are decoupled: each `run_*.py` writes tidy CSVs to `results/data/`; `scripts/plot_all.py` reads those CSVs and renders the PDFs in `results/figures/`. Bias-floor, κ-correlation and table scripts also write LaTeX tables to `results/tables/`.

```bash
# Forward model and conditioning
python scripts/run_sim.py                      # Forward-model validation
python scripts/run_conditioning.py             # κ vs band count, placement   → figs 1, 1b, 2, 3, 8, 8b

# Noise, NEDT and calibration sensitivity
python scripts/run_noise_study.py              # NEDT, δλ, joint perturbations → figs 4, 5, 6, 7, 9, 9b
python scripts/run_bias_floor.py               # Per-material structural bias floor table

# Hybrid variants (each writes the matching fig10/11/12 CSVs)
python scripts/run_hybrid.py                   # Parameter-space (baseline)
python scripts/run_hybrid_phys.py              # Physics-consistency loss
python scripts/run_hybrid_gated.py             # Density-gated residual
python scripts/run_hybrid_forward.py           # Forward-space residual
python scripts/run_lcurve_multiseed.py         # λ_phys sweep, multi-seed L-curve → fig 13

# Diagnostics and LaTeX tables
python scripts/run_kappa_error_correlation.py  # κ-vs-|ΔT| correlation table
python scripts/regenerate_tex_tables.py        # Regenerate all LaTeX tables

# Render every figure from the cached CSVs
python scripts/plot_all.py
```

To tweak titles, colours or font sizes, edit the `plot_*()` functions in `src/utils/plotting.py` and rerun only `scripts/plot_all.py`.

---

## Configuration

Experiment defaults are defined in two YAML files:

```
configs/simulation.yaml   # wavelength range, band counts, temperature defaults
configs/noise.yaml        # NEDT sweep, SNR range, δλ sweep
```

Example:

```yaml
simulation:
  temperature: 300
  band_counts: [3, 5, 10, 20, 30, 50]
  wavelength_range: [8e-6, 14e-6]
```

Per-experiment scripts override these defaults as needed.

---

## Outputs

```
results/
├── data/      # Tidy CSVs — durable experiment outputs
├── figures/   # PDFs — rendered from data/ by plot_all.py
└── tables/    # LaTeX tables
```

All three subdirectories are git-ignored. Every artefact in `results/` is regenerable from the scripts above; the CSVs in `results/data/` are the canonical output of each experiment.

---

## Citation

If you use this code or framework, please cite:

```
Gursli, H. (2026). Beyond Ill-Posedness: Conditioning, Null-Space Ambiguity,
and Hybrid Physics–ML Inversion for LWIR Temperature–Emissivity Separation.
Master's thesis, NTNU, Department of Engineering Cybernetics.
```

The thesis is available here:
[NTNU Open – Master's Thesis](https://hdl.handle.net/11250/5562621)


---

## License

[MIT License](LICENSE). See the `LICENSE` file for the full text.
