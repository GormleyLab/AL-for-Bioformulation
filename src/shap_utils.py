#!/usr/bin/env python
"""
Custom functions for SHAP analysis with real-unit conversions.
These functions enhance SHAP visualizations by converting normalized SHAP values
to real units of measurement for better interpretability.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import torch
import os
import sys
import pickle
import joblib
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Any
from contextlib import contextmanager, redirect_stdout, redirect_stderr


@contextmanager
def suppress_shap_output(verbose=False):
    """
    Context manager to suppress SHAP's verbose output including logging messages.
    
    Args:
        verbose (bool): If True, allows SHAP output. If False, suppresses it.
    """
    if not verbose:
        # Temporarily increase logging level to suppress INFO messages
        shap_logger = logging.getLogger('shap')
        original_level = shap_logger.level
        # Set to WARNING to suppress function info messages in INFO AND DEBUG
        shap_logger.setLevel(logging.WARNING)
        
        try:
            with open(os.devnull, 'w') as devnull:
                with redirect_stdout(devnull), redirect_stderr(devnull):
                    yield
        finally:
            # Restore original logging level
            shap_logger.setLevel(original_level)
    else:
        yield

def prepare_data_for_shap(model, X, feature_names):
    """
    Prepare model and data for SHAP analysis.
    
    Args:
        model: Trained BoTorch SingleTaskGP model
        X: Input features (tensor)
        feature_names: List of feature names
        
    Returns:
        shap_explainer: SHAP explainer object
        X_background: Background dataset for SHAP analysis (numpy array)
    """
    # Set model to eval mode
    model.eval()
    
    # Convert X tensor to numpy array for SHAP
    X_background = X.detach().cpu().numpy()

    # Define prediction function with squeeze to remove trailing singleton dimension
    def predict_fn(X_np):
        X_torch = torch.tensor(X_np, dtype=torch.float64)
        with torch.no_grad():
            posterior = model.posterior(X_torch)
            return posterior.mean.squeeze(-1).cpu().numpy()  # Shape: (n_samples,)
    
    # Create SHAP explainer (suppress output by default)
    with suppress_shap_output(verbose=False):
        explainer = shap.KernelExplainer(predict_fn, X_background)
    # Note: feature_names are passed later in plotting or if explicitly creating specific explainers
    
    return explainer, X_background

def convert_shap_to_real_units(model_name, shap_values, base_value, data_processor, feature_names=None):
    """
    Convert SHAP values from normalized space to real target units.
    
    Args:
        model_name (str): Name of the model (e.g., 'tm', 'diffusion', 'viscosity')
        shap_values (shap.Explanation or np.ndarray): SHAP values
        base_value (float): Base value from SHAP explainer (in normalized units)
        data_processor (DataProcessor): DataProcessor object with fitted scalers
        feature_names (List[str], optional): List of feature names. If None, will use data_processor.config.feature_columns
        
    Returns:
        Tuple containing:
            - shap_values_real (np.ndarray): SHAP values in real units
            - base_value_real (float): Base value in real units
            - feature_contribution_real (dict): Dictionary mapping feature names to their average absolute 
              contribution in real units
    """
    # Get the y_scaler for this model
    y_scaler = data_processor.y_scalers.get(model_name)
    
    if y_scaler is None:
        raise ValueError(f"No y_scaler found for model {model_name}")
    
    # Check if shap_values is an Explanation object
    if hasattr(shap_values, 'values'):
        shap_values_normalized = shap_values.values
    else:
        shap_values_normalized = np.array(shap_values)
    
    # Get the scale factor from normalized to real units
    # MinMaxScaler uses y_real = y_norm * (max - min) + min
    scale_range = y_scaler.data_max_ - y_scaler.data_min_
    
    # Convert SHAP values to real units by multiplying by the scale range
    shap_values_real = shap_values_normalized * scale_range
    
    # Convert base value to real units
    base_value_real = y_scaler.inverse_transform([[base_value]])[0][0]
    
    # Use feature names from function parameter or get from data_processor's config
    if feature_names is None:
        feature_names = data_processor.config.feature_columns
    
    # Calculate feature contributions in real units (average absolute impact)
    feature_contribution_real = {}
    
    for i, feature in enumerate(feature_names):
        # Get absolute values for this feature across all samples
        abs_impact = np.abs(shap_values_real[:, i])
        # Calculate average absolute impact
        avg_impact = np.mean(abs_impact)
        feature_contribution_real[feature] = avg_impact
    
    return shap_values_real, base_value_real, feature_contribution_real


def run_shap_analysis(models, datasets, data_processor, formulation_ids, config):
    """
    Run SHAP analysis, rescale values, and save results to pickle files.
    
    Args:
        models (dict): Trained models.
        datasets (dict): Processed datasets.
        data_processor (DataProcessor): DataProcessor instance with scalers.
        formulation_ids (dict): Formulation IDs for each dataset.
        config (Config): Configuration object.
        
    Returns:
        dict: All SHAP results and real unit conversions
    """
    # Create a directory for SHAP results
    shap_dir = Path(config.output_dir) / 'shap_results'
    shap_dir.mkdir(exist_ok=True, parents=True)
    
    feature_names = config.feature_columns
    all_shap_results = {}
    all_real_unit_results = {}

    # Get verbosity setting from config
    verbose = getattr(config, 'shap_verbose', False)
    
    # Analyze each model
    for model_name, model in models.items():
        print(f"\nAnalyzing {model_name} model with SHAP...")
        
        X = datasets[model_name]['X']
        
        # Prepare data for SHAP analysis
        explainer, X_background = prepare_data_for_shap(model, X, feature_names)
        
        # Generate SHAP values with optional output suppression
        print(f"Calculating SHAP values for {model_name}...")
        
        with suppress_shap_output(verbose=verbose):
            shap_values_scaled = explainer.shap_values(X_background)

        # Unwrap if SHAP returns a list (common in some SHAP versions for regression)
        if isinstance(shap_values_scaled, list):
            shap_values_scaled = shap_values_scaled[0]
            
        # Calculate expected value
        expected_value_scaled = explainer.expected_value
        # unwrap if it's an array
        if isinstance(expected_value_scaled, np.ndarray):
            expected_value_scaled = expected_value_scaled[0]

        # Convert to real units using helper
        shap_values_real, base_value_real, feature_contrib = convert_shap_to_real_units(
            model_name, 
            shap_values_scaled, 
            expected_value_scaled, 
            data_processor, 
            feature_names
        )

        # Get raw features for context
        X_raw = datasets[model_name].get('X_raw', X_background)
        if isinstance(X_raw, torch.Tensor):
            X_raw = X_raw.detach().cpu().numpy()

        # --- Save Scaled Results ---
        result_summary = {
            'shap_values': shap_values_scaled,
            'expected_value': expected_value_scaled,
            'X_background': X_background,
            'features_raw': X_raw,
            'feature_names': feature_names,
            'formulation_ids': formulation_ids.get(model_name)
        }
        all_shap_results[model_name] = result_summary
        
        # Save results object to a pickle file
        pickle_path = shap_dir / f"{model_name}_shap_results.pkl"
        with open(pickle_path, 'wb') as f:
            joblib.dump(result_summary, f)
        print(f"Saved SHAP results object to {pickle_path}")

        # --- Save Real Unit Results ---
        real_unit_summary = {
            'shap_values_real': shap_values_real,
            'base_value_real': base_value_real,
            'feature_contribution': feature_contrib
        }
        all_real_unit_results[model_name] = real_unit_summary

        # Save Real Unit Results to pickle
        real_pickle_path = shap_dir / f"{model_name}_shap_real_units.pkl"
        with open(real_pickle_path, 'wb') as f:
             joblib.dump(real_unit_summary, f)
        print(f"Saved Real Unit SHAP results to {real_pickle_path}")

    return all_shap_results, all_real_unit_results


def plot_real_unit_shap_summary_for_all_models(results, real_unit_results, feature_names, 
                                              figsize=(12, 8), save_dir=None):
    """
    Create and display SHAP summary plots for all models using real unit SHAP values.
    
    Args:
        results (dict): Dictionary with original SHAP results
        real_unit_results (dict): Dictionary with real unit SHAP results
        feature_names (list): List of feature names
        figsize (tuple, optional): Figure size for each plot. Default is (12, 8).
        save_dir (str, optional): Directory to save figures. If None, figures are not saved.
    
    Returns:
        dict: Dictionary of saved file paths, keyed by model name (if save_dir is provided)
    """
    # Get list of models
    model_names = list(results.keys())
    
    # Create save directory if specified and doesn't exist
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        
    # Dictionary to store file paths of saved figures
    saved_files = {}
    
    # For each model, create and display the plot
    for model_name in model_names:
        if model_name not in results or model_name not in real_unit_results:
            print(f"Warning: Model '{model_name}' not found in results. Skipping.")
            continue
        
        print(f"\n{'-'*50}\n{model_name.upper()} SHAP Summary Plot (Real Units)\n{'-'*50}")
        
        # Get real unit SHAP values and background data
        shap_values_real = real_unit_results[model_name]['shap_values_real']
        X_background = results[model_name]['X_background']
        
        # Create a new figure with exact size before SHAP plotting
        plt.figure(figsize=figsize)
        
        # Create SHAP explanation object
        explanation = shap.Explanation(
            values=shap_values_real,
            data=X_background,
            feature_names=feature_names
        )
        
        # Create summary plot directly with matplotlib managing the figure
        shap.plots.beeswarm(explanation, show=False)
        
        # Set title and apply tight layout
        plt.title(f"SHAP Summary Plot – {model_name}", fontsize=16)

        # Change x axis label to SHAP Value, which specifying the actual units
        # Specify real units for the label
        real_unit_label = {
            'tm': '°C',
            'diffusion': 'cm²/s',
            'viscosity': 'cP'
        }
        x_label = f"SHAP Value ({real_unit_label.get(model_name, 'units')})"
        plt.xlabel(x_label, fontsize=14)

        plt.tight_layout()
        
        # Force size immediately before display to override any SHAP adjustments
        plt.gcf().set_size_inches(figsize)
        
        # Save the figure if a save directory was provided
        if save_dir is not None:
            # Hardcoded file format and DPI
            file_format = 'png'
            dpi = 300
            
            filename = f"{model_name}_shap_summary.{file_format}"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
            saved_files[model_name] = filepath
            print(f"Saved figure to: {filepath}")
        
        # Show the plot (optional, can comment out if running headless)
        # plt.show()
        plt.close() # Close to free memory
        
        # Print feature importance summary
        print("\nFeature Importance Summary (average absolute SHAP value):")
        
        # Calculate average absolute SHAP value for each feature
        feature_importance = []
        for i, feature in enumerate(feature_names):
            avg_impact = np.mean(np.abs(shap_values_real[:, i]))
            feature_importance.append((feature, avg_impact))
        
        # Sort by importance
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        # Print top features
        for feature, importance in feature_importance:
            print(f"  - {feature}: {importance:.6f}")
        
        print(f"\nTotal samples analyzed: {len(shap_values_real)}")
    
    return saved_files if save_dir is not None else None


