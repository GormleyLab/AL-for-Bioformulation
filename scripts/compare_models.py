#!/usr/bin/env python
"""
Standalone model-comparison utility.

Benchmarks the production Gaussian-process (GP) surrogate against a panel of
common regression models (ordinary linear, linear + pairwise interactions,
degree-2 ridge polynomial, RBF-kernel SVR, random forest, extra trees, gradient
boosting, and a small regularized MLP) under the SAME 10-fold GroupKFold
cross-validation by Formulation ID used by the main pipeline. This reproduces the
expanded model benchmark presented in Figure S9 of the Supplementary Information document.

The utility reuses the main pipeline's ``DataProcessor`` (for scaling) and
``ModelTrainer`` (for the production GP), so it automatically tracks whatever
objectives / features are defined in ``config/config.py`` -- point it at a new
dataset and it benchmarks that dataset's objectives with no code changes.

Outputs (written to ``outputs/benchmark/`` by default):
  * ``model_comparison_metrics.csv``    -- per model x objective metrics (pooled/mean R2, MAE)
  * ``oof_predictions_scaled.csv``      -- per-row out-of-fold predictions (scaled space)
  * ``model_comparison_summary.md``     -- a short human-readable summary table

Run as::

    python scripts/compare_models.py
    python scripts/compare_models.py --config config/custom_config.yaml --output-dir outputs/my_benchmark
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR

# Make the repo root importable when run as `python scripts/compare_models.py`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  

from config.config import load_config  
from src.pipeline import DataProcessor, ModelTrainer  


MODEL_ORDER = [
    "Linear",
    "Linear + 2-way interactions",
    "Polynomial (deg 2, Ridge)",
    "RBF-SVR",
    "Random forest",
    "Extra trees",
    "Gradient boosting",
    "MLP (small, regularized)",
    "Gaussian process (production)",
]


def expanded_model_factories(seed: int) -> Dict[str, object]:
    """Return deterministic sklearn estimators for the benchmark."""
    return {
        "Linear": LinearRegression(),
        "Linear + 2-way interactions": make_pipeline(
            PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
            LinearRegression(),
        ),
        "Polynomial (deg 2, Ridge)": make_pipeline(
            PolynomialFeatures(degree=2, interaction_only=False, include_bias=False),
            Ridge(alpha=1.0, random_state=seed),
        ),
        # RBF-SVR is a compact nonlinear kernel comparator for small datasets.
        # C/epsilon are intentionally fixed rather than tuned on the test folds.
        "RBF-SVR": SVR(kernel="rbf", C=10.0, epsilon=0.03, gamma="scale"),
        "Random forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=1,
        ),
        "Extra trees": ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=1,
        ),
        "Gradient boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.035, max_depth=2,
            min_samples_leaf=3, random_state=seed,
        ),
        # Deliberately small and regularized for n ~= 72.
        "MLP (small, regularized)": MLPRegressor(
            hidden_layer_sizes=(16, 8), activation="relu", solver="adam",
            alpha=1.0e-2, learning_rate_init=2.0e-3, max_iter=3000,
            n_iter_no_change=80, random_state=seed,
        ),
    }


# ---------------------------------------------------------------------------
# Cross-validation helpers (mirrors ModelTrainer.cross_validate's fold splits).
# ---------------------------------------------------------------------------
def grouped_kfold_indices(
    groups: np.ndarray, n_splits: int, random_state: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return (train_idx, test_idx) splits matching the main pipeline's CV.

    This matches the production pipeline which shuffles the unique group labels with a NumPy
    ``default_rng(random_state)`` permutation before handing them to GroupKFold,
    so concentration replicates from the same formulation stay in the same fold.
    """
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(unique_groups)
    mapping = {old: new for new, old in enumerate(shuffled)}
    mapped = np.array([mapping[g] for g in groups])
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(np.zeros((len(groups), 1)), np.zeros(len(groups)), mapped))


def posterior_mean_std(model, X: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) numpy arrays from a fitted SingleTaskGP posterior at X."""
    model.eval()
    with torch.no_grad():
        post = model.posterior(X)
        mean = post.mean.detach().cpu().numpy().reshape(-1)
        std = post.variance.detach().cpu().numpy().reshape(-1) ** 0.5
    return mean, std


def _real_unit_mae(data_processor, obj_name, y_true_scaled, y_pred_scaled) -> float:
    scaler = data_processor.y_scalers[obj_name]
    yt = scaler.inverse_transform(np.asarray(y_true_scaled).reshape(-1, 1)).reshape(-1)
    yp = scaler.inverse_transform(np.asarray(y_pred_scaled).reshape(-1, 1)).reshape(-1)
    return float(mean_absolute_error(yt, yp))


def _all_cv_metrics(y_true_folds, y_pred_folds, data_processor, obj_name) -> dict:
    y_true = np.concatenate(y_true_folds)
    y_pred = np.concatenate(y_pred_folds)
    fold_r2 = [r2_score(t, p) for t, p in zip(y_true_folds, y_pred_folds)]
    fold_mae_scaled = [mean_absolute_error(t, p) for t, p in zip(y_true_folds, y_pred_folds)]
    fold_mae_real = [
        _real_unit_mae(data_processor, obj_name, t, p)
        for t, p in zip(y_true_folds, y_pred_folds)
    ]
    return {
        "pooled_r2": float(r2_score(y_true, y_pred)),
        "mean_fold_r2": float(np.mean(fold_r2)),
        "std_fold_r2": float(np.std(fold_r2, ddof=1)) if len(fold_r2) > 1 else 0.0,
        "mean_mae_scaled": float(np.mean(fold_mae_scaled)),
        "mean_mae_real": float(np.mean(fold_mae_real)),
        "std_mae_real": float(np.std(fold_mae_real, ddof=1)) if len(fold_mae_real) > 1 else 0.0,
        "n_test_points": int(len(y_true)),
        "n_folds_used": int(len(y_true_folds)),
    }


def run_sklearn_cv(X_scaled, y_scaled, splits, estimator, data_processor, obj_name):
    """Fit one fresh estimator per fold; return (metrics, predictions_df)."""
    y_true_folds, y_pred_folds, prediction_rows = [], [], []
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        est = clone(estimator)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            est.fit(X_scaled[train_idx], y_scaled[train_idx].ravel())
        y_pred = np.asarray(est.predict(X_scaled[test_idx])).reshape(-1)
        y_true = y_scaled[test_idx].ravel()
        y_true_folds.append(y_true)
        y_pred_folds.append(y_pred)
        for row_idx, yt, yp in zip(test_idx, y_true, y_pred):
            prediction_rows.append(
                {"fold": fold_idx, "row_index": int(row_idx),
                 "y_true_scaled": float(yt), "y_pred_scaled": float(yp)}
            )
    return _all_cv_metrics(y_true_folds, y_pred_folds, data_processor, obj_name), pd.DataFrame(prediction_rows)


def run_gp_cv(config, obj_name, datasets, splits, data_processor):
    """Same CV protocol as run_sklearn_cv but for the production BoTorch GP."""
    trainer = ModelTrainer(config)
    X = datasets[obj_name]["X"]
    y = datasets[obj_name]["y"]
    y_true_folds, y_pred_folds, prediction_rows = [], [], []
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = trainer._train_single_model(X[train_idx], y[train_idx], f"{obj_name}_bench_fold{fold_idx}")
        if model is None:
            continue
        mean_scaled, std_scaled = posterior_mean_std(model, X[test_idx])
        y_true = y[test_idx].detach().cpu().numpy().reshape(-1)
        y_true_folds.append(y_true)
        y_pred_folds.append(mean_scaled)
        for row_idx, yt, yp, sig in zip(test_idx, y_true, mean_scaled, std_scaled):
            prediction_rows.append(
                {"fold": fold_idx, "row_index": int(row_idx),
                 "y_true_scaled": float(yt), "y_pred_scaled": float(yp),
                 "y_pred_sigma_scaled": float(sig)}
            )
    return _all_cv_metrics(y_true_folds, y_pred_folds, data_processor, obj_name), pd.DataFrame(prediction_rows)


def compute_all(config, data_processor, datasets, formulation_ids):
    """Run every architecture x objective x fold; return (metrics_df, predictions_df)."""
    metrics_rows: List[dict] = []
    prediction_frames: List[pd.DataFrame] = []

    for obj_name in config.objective_names:
        if obj_name not in datasets:
            continue
        X_scaled = datasets[obj_name]["X"].detach().cpu().numpy()
        y_scaled = datasets[obj_name]["y"].detach().cpu().numpy()
        groups = formulation_ids[obj_name]
        splits = grouped_kfold_indices(groups, config.cv_folds, config.random_state)

        for model_name, estimator in expanded_model_factories(config.random_state).items():
            print(f"  [cv] {obj_name:12s}  {model_name}")
            metrics, preds = run_sklearn_cv(X_scaled, y_scaled, splits, estimator, data_processor, obj_name)
            metrics_rows.append({"model": model_name, "objective": obj_name, **metrics})
            preds["model"] = model_name
            preds["objective"] = obj_name
            prediction_frames.append(preds)

        print(f"  [cv] {obj_name:12s}  Gaussian process (production)")
        gp_metrics, gp_preds = run_gp_cv(config, obj_name, datasets, splits, data_processor)
        metrics_rows.append({"model": "Gaussian process (production)", "objective": obj_name, **gp_metrics})
        gp_preds["model"] = "Gaussian process (production)"
        gp_preds["objective"] = obj_name
        prediction_frames.append(gp_preds)

    metrics_df = pd.DataFrame(metrics_rows)
    predictions_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    return metrics_df, predictions_df


def write_summary(metrics_df: pd.DataFrame, objective_names: List[str], output_path: Path) -> None:
    """Write a short markdown summary with a pooled-R2 table."""
    present_models = [m for m in MODEL_ORDER if m in set(metrics_df["model"])]
    wide = metrics_df.pivot(index="model", columns="objective", values="pooled_r2").reindex(present_models)

    lines = [
        "# Model comparison summary",
        "",
        "Pooled out-of-fold R^2 under GroupKFold CV by Formulation ID "
        "(same fold splits as the main pipeline).",
        "",
        "| Model | " + " | ".join(objective_names) + " |",
        "|---" * (len(objective_names) + 1) + "|",
    ]
    for model in present_models:
        vals = [wide.loc[model].get(obj, np.nan) for obj in objective_names]
        cells = ["NA" if pd.isna(v) else f"{v:.3f}" for v in vals]
        lines.append("| " + model + " | " + " | ".join(cells) + " |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare regression models against the production GP.")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config override.")
    parser.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "outputs" / "benchmark"),
                        help="Output directory (default: outputs/benchmark/).")
    parser.add_argument("--seed", type=int, default=None, help="Override config.random_state.")
    parser.add_argument("--verbose", action="store_true", help="Enable INFO logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )

    config = load_config(args.config)
    if args.seed is not None:
        config.random_state = args.seed

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_processor = DataProcessor(config)
    datasets, formulation_ids = data_processor.load_and_process_data()

    print("=" * 80)
    print("Model comparison benchmark")
    print("=" * 80)
    print(f"\nRunning {config.cv_folds}-fold CV for {len(MODEL_ORDER)} models "
          f"x {len(config.objective_names)} objectives...\n")

    metrics_df, predictions_df = compute_all(config, data_processor, datasets, formulation_ids)

    metrics_df.to_csv(out_dir / "model_comparison_metrics.csv", index=False)
    if not predictions_df.empty:
        predictions_df.to_csv(out_dir / "oof_predictions_scaled.csv", index=False)
    write_summary(metrics_df, config.objective_names, out_dir / "model_comparison_summary.md")

    print("\n" + "=" * 80)
    print("Pooled out-of-fold R^2 by objective")
    print("=" * 80)
    present = [m for m in MODEL_ORDER if m in set(metrics_df["model"])]
    display = metrics_df.pivot(index="model", columns="objective", values="pooled_r2").reindex(present)
    print(display.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\nOutputs written to: {out_dir}")


if __name__ == "__main__":
    main()
