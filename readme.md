# HADAR TeX Thesis – Spectral Identifiability in LWIR

This repository contains the implementation and experimental framework for a master’s thesis on **spectral identifiability and numerical conditioning in hyperspectral LWIR TeX (temperature–emissivity–texture) decomposition**.

The project focuses on analyzing when thermal inverse problems are well-posed, how spectral design affects stability, and whether hybrid physics–machine learning models can improve robustness.

---

## Overview

Thermal hyperspectral imaging enables recovery of scene properties such as temperature and emissivity from measured radiance. However, this inversion problem is often **ill-conditioned and sensitive to noise, spectral resolution, and calibration errors**.

This project implements:

* A physics-based LWIR forward model
* A TeX inversion framework
* Jacobian-based conditioning analysis (raw and Tarantola-scaled κ)
* Spectral resolution, noise, and calibration sensitivity experiments
* Four Lipschitz-bounded hybrid physics–neural residual variants

---

## Project Structure

```
MASTERTHESIS/
│
├── src/
│   ├── forward_model/     # Physics-based radiative models
│   ├── inversion/         # Inverse problem formulation and solvers
│   ├── simulation/        # Synthetic data generation
│   ├── analysis/          # Experiments and conditioning studies
│   ├── hybrid/            # Neural residual models
│   └── utils/             # Plotting, metrics, helpers
│
├── configs/               # YAML experiment configurations
├── scripts/               # Runnable experiment scripts + plot_all.py renderer
├── notebooks/             # Exploratory analysis
├── results/               # Generated artifacts
│   ├── data/              # Tidy CSVs (one per figure, written by run_*.py)
│   ├── figures/           # PDFs (rendered by plot_all.py from data/)
│   └── tables/            # LaTeX tables (written by run_bias_floor.py etc.)
├── data/                  # Emissivity and simulation data
│
├── environment.yaml
└── README.md
```
| Layer        | Purpose               | 
| ------------ | --------------------- | 
| Forward mode | Physics               | 
| Inversion    | Solve inverse problem | 
| Simulation   | Generate data         | 
| Analysis     | Run experiments       | 
| Hybrid       | Neural residual       |
| Utils        | Support tools         | 

---

## Core Components

### Forward Model

Implements the LWIR radiative transfer model:

[
L(\lambda) = \epsilon(\lambda) B(\lambda, T) + (1 - \epsilon(\lambda)) L_{env}
]

Includes:

* Planck radiance computation
* Analytical and numerical derivatives for Jacobian analysis

---

### Inversion

Defines the TeX inverse problem and computes:

* Jacobian matrices
* Parameter sensitivities
* Conditioning metrics (raw and scaled κ via SVD)

---

### Simulation

Provides tools to generate synthetic experiments:

* Emissivity spectra (8 materials, HADAR-bundled ECOSTRESS)
* Spectral band configurations (uniform, random, targeted, greedy)
* Noise injection (NEDT-based)
* Calibration perturbations (wavelength shift)

---

### Analysis

Implements core experiments:

* Spectral resolution sweeps
* Condition number evaluation (raw and scaled)
* Stability analysis under noise and perturbations

---

### Hybrid Models

Four Lipschitz-bounded residual variants:

* Baseline — parameter-space additive residual
* Forward — radiance-space additive residual
* Gated — density-gated residual (GMM)
* Phys — physics-consistency loss term

---

## Installation

```
conda env create -f environment.yaml
conda activate Mthesis
```

---

## Running Experiments

Experiments and rendering are decoupled: each `run_*.py` writes tidy CSVs
to `results/data/`; [scripts/plot_all.py](scripts/plot_all.py) reads those
CSVs and produces the PDFs in `results/figures/`. Bias-floor and
κ-correlation scripts also write LaTeX tables to `results/tables/`.

```
python scripts/run_sim.py                       # Forward-model validation (.npy)
python scripts/run_conditioning.py              # κ vs band count, placement  → figs 1, 1b, 2, 3, 8, 8b CSVs
python scripts/run_noise_study.py               # NEDT, δλ, combined perturbations → figs 4, 5, 6, 7, 9, 9b CSVs
python scripts/run_hybrid.py                    # Baseline hybrid                 → fig 10/11/12 CSVs (_baseline)
python scripts/run_hybrid_forward.py            # Forward-space variant           → fig 10/11/12 CSVs (_forward)
python scripts/run_hybrid_gated.py              # Density-gated variant           → fig 10/11/12 CSVs (_gated)
python scripts/run_hybrid_phys.py               # Physics-consistency variant     → fig 10/11/12/13 CSVs (_phys)
python scripts/run_bias_floor.py                # Per-material bias floor table
python scripts/run_kappa_error_correlation.py   # κ-vs-error diagnostic table

python scripts/plot_all.py                      # Render every figure from results/data/
```

To tweak titles, colors, or font sizes, edit the `plot_*()` functions in
[src/utils/plotting.py](src/utils/plotting.py) and rerun only
`scripts/plot_all.py` — no need to recompute the experiments.

---

## Configuration

Experiment parameters are defined in YAML files:

```
configs/simulation.yaml
configs/noise.yaml
```

Example:

```yaml
simulation:
  temperature: 300
  band_counts: [3, 5, 10, 20, 50]
  wavelength_range: [8e-6, 14e-6]
```

## Outputs

Results are stored in:

```
results/
├── data/      # Tidy CSVs — durable experiment outputs
├── figures/   # PDFs — rendered from data/ by plot_all.py
└── tables/    # LaTeX tables
```
