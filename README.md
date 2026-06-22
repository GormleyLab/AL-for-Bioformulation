# Automation and Active Learning for Multi-Objective Optimization of Antibody Formulations



## Authors

**D. Christopher Radford**#, **Matthew Tamasi**#, **Elena Di Mare**, **Adam J. Gormley**\*

*Department of Biomedical Engineering, Rutgers, The State University of New Jersey, Piscataway, New Jersey 08854, USA*

## Accompanying Manuscript

[Read the full paper on ChemRxiv](https://chemrxiv.org/engage/chemrxiv/article-details/68f28d1edfd0d042d1228a4f)

## Abstract

Over the last forty years, biologics such as monoclonal antibodies have become an increasingly important therapeutic agent in the treatment of numerous diseases. Between 1986 and 2025, over 200 antibody-based treatments have been approved globally, most of which are manufactured as preformulated solutions for subsequent administration to patients. However, bioformulation of complex proteins is a difficult engineering challenge; formulations must be tailored to individual therapies, necessitating time- and material-intensive campaigns to select a combination of excipients to simultaneously optimize an array of design criteria. These many interacting additives complicate formulation design with unintuitive and non-linear relationships, thus creating a vast and multidimensional design space that is intrinsically difficult to navigate through and optimize using traditional techniques. To address this challenge, we investigated a high-throughput discovery pipeline using machine learning to model and predict formulation behavior of Generally Recognized as Safe (GRAS) excipients on a model antibody. This was supported by automation-assisted “on-demand” formulation to produce dozens of uniquely formulated antibody solutions with high reproducibility for downstream evaluation and biophysical characterization. This pipeline was then integrated into an iterative closed-loop cycle of automated Design-Build-Test-Learn (DBTL), where new rounds of experiments are designed by the ML model. The process yielded both optimized formulations as well as highly accurate predictive models of formulation behavior. This validates the utility of this technique to both map the underlying property-function landscape and effectively guide formulation development while balancing multiple competing design requirements.

---

This repository contains the code and data for the paper "Automation and Active Learning for the Multi-Objective Optimization of Antibody Formulations". It provides a machine learning pipeline for optimizing antibody formulations using Bayesian Optimization with BoTorch.

## Overview

The pipeline performs the following steps:

1.  **Data Loading & Preprocessing**: Loads formulation data from a JSON file, processes array-based measurements, and standardizes features.
2.  **Model Training**: Trains Gaussian Process (GP) models for each objective (Tm, Diffusion, Viscosity) using `SingleTaskGP`.
3.  **Model Validation**: Performs Group K-Fold Cross-Validation to ensure model robustness.
4.  **Optimization**:
    *   **Single-Objective**: Optimizes each target individually. The acquisition function is configurable (`single_acqf`): q-Expected Improvement (`qEI`, default), q-Upper Confidence Bound (`qUCB`), or a greedy constant-liar posterior-mean batch (`greedy_cl`).
    *   **Multi-Objective**: Optimizes all targets simultaneously. The acquisition function is configurable (`multi_acqf`): q-Expected Hypervolume Improvement (`qEHVI`, default) with a fixed reference point, or a greedy constant-liar batch (`greedy_cl`).
5.  **Explainability**:
    *   **SHAP Analysis**: Automatically generates SHAP values (as pickle files) to explain model predictions in real units for all objectives.

## Repository Structure

```
├── config/
│   └── config.py       # Configuration settings (paths, objectives, acquisition, bounds)
├── data/
│   └── Antibody_...    # Input JSON data
├── outputs/            # Generated models, logs, candidates, and SHAP results
├── scripts/
│   └── compare_models.py  # Standalone model-comparison benchmark (GP vs. linear/tree/etc.)
├── src/
│   ├── pipeline.py     # Core optimization logic (BoTorchPipeline with integrated SHAP)
│   ├── utils.py        # Utility functions (data loading, logging)
│   └── shap_utils.py   # SHAP analysis and visualization utilities
├── main.py             # Entry point script (runs full pipeline including SHAP)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Installation

We recommend using `uv` for fast and reliable dependency management.

1.  Clone the repository.
2.  Install `uv` (if not already installed):
3. 
    ```bash
    pip install uv
    ```
4.  Create a virtual environment and install dependencies:

    ```bash
    # Create virtual environment
    uv venv

    # Install dependencies
    uv pip install -r requirements.txt
    ```

Alternatively, you can use standard pip:

```bash
pip install -r requirements.txt
```

## Usage

To run the complete optimization pipeline (including automatic SHAP analysis):

**Windows:**
```bash
.venv\Scripts\python main.py
```

**macOS/Linux:**
```bash
.venv/bin/python main.py
```

The pipeline automatically generates SHAP analysis results after model training, saving them to `outputs/shap_results/`.

### Customization

You can customize the execution using command-line arguments:

```bash
.venv\Scripts\python main.py --config config/custom_config.yaml --output_dir outputs/experiment_1
```

*   `--config`: Path to a YAML file that overrides **top-level scalar** config fields (e.g. `single_acqf`, `batch_size`, `random_state`, `data_file`). Nested settings such as the `objectives` dictionary are **not** overridable via YAML — change those directly in `config/config.py` (see [Configuration Reference](#configuration-reference) below).
*   `--output_dir`: Directory to save results (models, logs, candidates).
*   `--debug`: Enable debug logging.

## Results

The pipeline outputs:

*   **optimization_candidates.xlsx**: Suggested formulation candidates for experimental validation.
*   **models/**: Saved PyTorch/GPyTorch models (`.pkl` files).
*   **optimization.log**: Detailed execution log.
*   **shap_results/**: SHAP analysis results including:
    *   Pickle files for each objective (e.g., `tm_shap_results.pkl`, `tm_shap_real_units.pkl`)

## Model Comparison Utility

`scripts/compare_models.py` benchmarks the production Gaussian-process (GP) surrogate against a panel of common regression models, under the **same** 10-fold GroupKFold cross-validation (grouped by `Formulation ID`) used by the main pipeline. This reproduces the expanded model comparison reported in the revised manuscript.

Models compared (per objective): ordinary linear regression, linear regression with pairwise interactions, a degree-2 ridge polynomial, RBF-kernel SVR, random forest, extremely randomized trees, gradient boosting, a small regularized MLP, and the production BoTorch GP.

```bash
python scripts/compare_models.py
```

Options:

*   `--config`: YAML config override (same semantics as `main.py`).
*   `--output-dir`: Output directory (default: `outputs/benchmark/`).
*   `--seed`: Override `random_state`.
*   `--verbose`: Enable INFO logging.

Outputs (in `outputs/benchmark/`):

*   `model_comparison_metrics.csv` — per model × objective metrics (pooled and per-fold R², MAE in scaled and real units).
*   `oof_predictions_scaled.csv` — per-row out-of-fold predictions in scaled space.
*   `model_comparison_summary.md` — a human-readable pooled-R² table.

Because the utility reads its objectives and features from `config/config.py`, it automatically benchmarks whatever objectives are defined there — no edits needed to point it at a different dataset (subject to the data-format requirements in [Adapting the Pipeline to Your Own Data](#adapting-the-pipeline-to-your-own-data)).

## Configuration Reference

All settings live in `config/config.py` as a `Config` dataclass. Top-level **scalar** fields can also be overridden via a YAML file passed with `--config`; nested fields (notably `objectives`) must be edited directly in `config.py`.

| Field | Type | Default | Description |
|---|---|---|---|
| `random_state` | int | `42` | Global random seed (model fitting, CV fold shuffling). |
| `device` | str | `"auto"` | Compute device: `"auto"`, `"cpu"`, or `"cuda"`. |
| `data_file` | str | `data/Antibody_...JSON.json` | Path to the input JSON dataset. |
| `output_dir` | str | `"outputs"` | Base directory for results. |
| `feature_columns` | list[str] | 7 columns | Input features used by the models, **after** array expansion. Includes `Concentration (mg/mL)`. |
| `objectives` | dict | 3 objectives | Per-objective configuration (see below). |
| `excipient_columns` | list[str] | 6 columns | Excipient/formulation columns present in the raw JSON. |
| `metadata_columns` | list[str] | 3 columns | Non-feature metadata (`Formulation ID`, `Objective`, `Generation`). |
| `cv_folds` | int | `10` | Number of GroupKFold splits (grouped by `Formulation ID`). |
| `min_samples` | int | `15` | Minimum expanded rows required to train an objective's model. |
| `min_r2` | float | `0.2` | Minimum train-set R² for a fitted GP to be accepted. |
| `batch_size` | int | `6` | Number of candidates (`q`) returned per acquisition. |
| `mc_samples` | int | `128` | Monte Carlo samples for the qMC acquisition sampler. |
| `num_restarts` | int | `10` | Restarts for acquisition-function optimization. |
| `reference_point` | list[float] | `[0,0,0]` | Reference point for `qEHVI` (one entry per objective, in scaled/direction-adjusted space). |
| `single_acqf` | str | `"qEI"` | Single-objective acquisition: `"qEI"`, `"qUCB"`, or `"greedy_cl"`. |
| `multi_acqf` | str | `"qEHVI"` | Multi-objective acquisition: `"qEHVI"` or `"greedy_cl"`. |
| `ucb_beta` | float | `0.1` | Exploration weight for `qUCB` (only used when `single_acqf="qUCB"`). |
| `valid_buffer_pH_pairs` | dict | — | Allowed pH values per buffer system (reference metadata). |
| `pred_conc` | dict | `{tm:15, diffusion:15, viscosity:120}` | Antibody concentration (mg/mL) at which candidate predictions are reported per objective. |
| `save_models` | bool | `True` | Save trained GP models to `outputs/models/`. |
| `save_candidates` | bool | `True` | Save candidate formulations to Excel. |
| `run_shap_analysis` | bool | `True` | Run SHAP explainability after training. |
| `shap_verbose` | bool | `False` | Print SHAP progress. |

**Each entry in `objectives`** is keyed by a short objective name (`tm`, `diffusion`, `viscosity`) and contains:

| Key | Description |
|---|---|
| `target_column` | Name of the expanded mean column the model is trained on (e.g. `"Tm (C) Mean"`). |
| `std_column` | Name of the expanded standard-deviation column (e.g. `"Tm (C) Std"`). |
| `direction` | `1.0` to **maximize**, `-1.0` to **minimize** the objective. |
| `concentration` | Raw JSON key holding the per-measurement concentration array (e.g. `"concentration_tm"`). |
| `value` | Raw JSON key holding the per-measurement value array (e.g. `"tm"`). |
| `std` | Raw JSON key holding the per-measurement std array (e.g. `"tm_std"`). |

### Acquisition functions

The defaults (`single_acqf="qEI"`, `multi_acqf="qEHVI"`) reproduce the published pipeline exactly. The alternatives are more exploitative and were added to test how acquisition policy affects candidate selection:

*   **`qUCB`** (single-objective) — q-Upper Confidence Bound with exploration weight `ucb_beta` (small β ⇒ exploitative).
*   **`greedy_cl`** (single- and multi-objective) — greedy posterior-mean selection with **constant-liar (CL-min)** batching (Ginsbourger et al., 2010): after each pick the surrogate is conditioned on the worst-case scaled outcome so the next pick moves elsewhere. This is the pure-exploitation extreme.

## Adapting the Pipeline to Your Own Data

### Input data format

Data is a single JSON file with the structure:

```json
{
  "sheets": {
    "Consolidated": [
      {
        "Formulation ID": 1,
        "Molarity": 90.1, "NaCl": 46.9, "Sucrose": 1.56, "Arginine": 49.5, "pH": 6.0, "Buffer": 2,
        "Objective": "Seed", "Generation": "Seed",
        "concentration_tm":  [2.5, 5.0, 10.0, 15.0],
        "tm":                [67.3, 66.3, 65.6, 65.5],
        "tm_std":            [0.22, 0.15, 0.24, 0.16],
        "concentration_diff": [...], "diff": [...], "diff_std": [...],
        "concentration_visc": [...], "visc": [...], "visc_std": [...]
      }
    ]
  }
}
```

Key points:

*   The loader uses the **first** sheet under `"sheets"`.
*   Each object is **one formulation**. Excipient/feature values are scalars; each objective's measurements are **parallel arrays** of `concentration`, `value`, and `std`.
*   The pipeline **expands** these arrays into one row per (formulation, concentration) measurement, and groups rows by `Formulation ID` during cross-validation so replicates of the same formulation never straddle train/test folds.

### Steps to retarget the pipeline

1.  **Point at your data**: set `data_file` in `config.py` (or via YAML).
2.  **Reuse the three existing objectives as-is** if your JSON uses the same array keys (`tm`/`diff`/`visc`, `concentration_*`, `*_std`). Just supply your own values — no code changes needed.
3.  **Change features**: edit `feature_columns` (model inputs) and `excipient_columns` (raw scalar columns). Keep `Concentration (mg/mL)` in `feature_columns` if your objectives are concentration-dependent.
4.  **Change objective directions / reporting**: edit each objective's `direction` (maximize vs. minimize), and `pred_conc` (the concentration at which candidate predictions are reported). Set `reference_point` to have one entry per objective.
5.  **Add or rename an objective (requires a small code edit)**: the array-expansion step in `src/utils.py` (`expand_array_data`) maps the raw array keys `"tm"`/`"diff"`/`"visc"` to the standardized `target_column`/`std_column` names. To add a genuinely new measurement type (new `value` key and new `target_column`), add a corresponding branch in `expand_array_data` that writes your `target_column`/`std_column`, then add the matching entry to `objectives`.
6.  **Validate**: run `python scripts/compare_models.py` to confirm your data loads and models train, then run `python main.py` for the full pipeline.

### Choosing an acquisition strategy

Set `single_acqf` / `multi_acqf` in the config. Use the `qEI`/`qEHVI` defaults for balanced exploration/exploitation; use `qUCB` (tune `ucb_beta`) or `greedy_cl` when you want more exploitative batches.