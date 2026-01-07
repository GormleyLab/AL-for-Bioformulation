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
    *   **Single-Objective**: Optimizes each target individually using q-Expected Improvement (qEI).
    *   **Multi-Objective**: Optimizes all targets simultaneously using q-Expected Hypervolume Improvement (qEHVI) with a fixed reference point.
5.  **Explainability**:
    *   **SHAP Analysis**: Automatically generates SHAP values (as pickle files) to explain model predictions in real units for all objectives.

## Repository Structure

```
├── config/
│   └── config.py       # Configuration settings (paths, objectives, bounds)
├── data/
│   └── Antibody_...    # Input JSON data
├── outputs/            # Generated models, logs, candidates, and SHAP results
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
    ```bash
    pip install uv
    ```
3.  Create a virtual environment and install dependencies:
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

*   `--config`: Path to a YAML configuration file to override defaults.
*   `--output_dir`: Directory to save results (models, logs, candidates).
*   `--debug`: Enable debug logging.

## Results

The pipeline outputs:
*   **optimization_candidates.xlsx**: Suggested formulation candidates for experimental validation.
*   **models/**: Saved PyTorch/GPyTorch models (`.pkl` files).
*   **optimization.log**: Detailed execution log.
*   **shap_results/**: SHAP analysis results including:
    *   Pickle files for each objective (e.g., `tm_shap_results.pkl`, `tm_shap_real_units.pkl`)