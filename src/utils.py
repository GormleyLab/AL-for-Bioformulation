#!/usr/bin/env python
"""
Essential utilities for BoTorch antibody formulation optimization.
"""

import logging
import argparse
import json
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from sklearn.preprocessing import MinMaxScaler
import joblib
import sys


def setup_logging(output_dir: str, level: str = "INFO") -> None:
    """Setup basic logging configuration."""
    log_level = getattr(logging, level.upper())
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_path = Path(output_dir) / "optimization.log"
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='BoTorch Antibody Optimization')
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    return parser.parse_args()


def setup_output_directory(output_dir: str) -> None:
    """Create output directory structure."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "models").mkdir(exist_ok=True)
    (output_path / "candidates").mkdir(exist_ok=True)


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from JSON file format."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find file: {path}")
    
    if path.suffix != '.json':
        raise ValueError(f"Only JSON files are supported. Got: {path.suffix}")
    
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Expect nested structure with 'sheets' key
        if not isinstance(data, dict) or 'sheets' not in data:
            raise ValueError(f"JSON must have nested 'sheets' structure. Got keys: {data.keys() if isinstance(data, dict) else 'not a dict'}")
        
        # Use first sheet
        sheet_data = next(iter(data['sheets'].values()))
        return pd.DataFrame(sheet_data)
        
    except Exception as e:
        raise ValueError(f"Error loading JSON file: {e}")


def convert_to_tensors(
    data: pd.DataFrame, 
    feature_start_idx: int = 2,
    dtype: torch.dtype = torch.float64
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert DataFrame to PyTorch tensors."""
    if data.empty:
        n_features = len(data.columns) - feature_start_idx
        return torch.empty(0, n_features, dtype=dtype), torch.empty(0, feature_start_idx, dtype=dtype)
    
    # Features (from feature_start_idx onwards)
    X = torch.tensor(data.iloc[:, feature_start_idx:].values, dtype=dtype)
    
    # Targets (first feature_start_idx columns)
    y = torch.tensor(data.iloc[:, :feature_start_idx].values, dtype=dtype)
    
    return X, y


def scale_data(
    X: torch.Tensor, 
    y: torch.Tensor, 
    X_scaler: Optional[MinMaxScaler] = None,
    y_scaler: Optional[MinMaxScaler] = None,
    fit_scalers: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, MinMaxScaler, MinMaxScaler]:
    """Scale input and output tensors."""
    # Scale features
    if X_scaler is None:
        X_scaler = MinMaxScaler()
    
    if fit_scalers:
        X_scaled = torch.tensor(X_scaler.fit_transform(X.numpy()), dtype=X.dtype)
    else:
        X_scaled = torch.tensor(X_scaler.transform(X.numpy()), dtype=X.dtype)
    
    # Scale targets
    if y_scaler is None:
        y_scaler = MinMaxScaler()
    
    if fit_scalers:
        y_scaled = torch.tensor(y_scaler.fit_transform(y.numpy()), dtype=y.dtype)
    else:
        y_scaled = torch.tensor(y_scaler.transform(y.numpy()), dtype=y.dtype)
    
    return X_scaled, y_scaled, X_scaler, y_scaler


def save_scalers(X_scaler: MinMaxScaler, y_scalers: Dict[str, MinMaxScaler], output_dir: str) -> None:
    """Save scalers to disk."""
    scaler_dir = Path(output_dir) / "scalers"
    scaler_dir.mkdir(exist_ok=True)
    
    joblib.dump(X_scaler, scaler_dir / "X_scaler.pkl")
    
    for name, scaler in y_scalers.items():
        joblib.dump(scaler, scaler_dir / f"{name}_y_scaler.pkl")


def log_dataset_stats(data: pd.DataFrame, name: str = "Dataset") -> None:
    """Log basic dataset statistics."""
    logging.info(f"{name}: {data.shape[0]} samples, {data.shape[1]} features")
    
    # Log target statistics if present
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        target_col = numeric_cols[0]  # Assume first numeric column is target
        logging.info(f"  {target_col}: {data[target_col].mean():.3f} ± {data[target_col].std():.3f}")


def parse_array_string(array_str: str) -> list:
    """Parse string representation of array into list of floats. Also handles lists directly."""
    # Handle None or empty string
    if array_str is None or array_str == '':
        return []
    
    # Handle if it's already a list (from JSON)
    if isinstance(array_str, list):
        return [float(x) for x in array_str]
    
    # Handle pandas NA values
    if pd.isna(array_str):
        return []
    
    # Parse string representation
    try:
        import ast
        array_list = ast.literal_eval(array_str)
        return [float(x) for x in array_list]
    except:
        array_str = str(array_str).strip('[]')
        values = array_str.split(',') if ',' in array_str else array_str.split()
        return [float(x.strip()) for x in values if x.strip()]


def expand_array_data(df: pd.DataFrame, target_col: str, std_col: str, concentration_col: str) -> pd.DataFrame:
    """
    Expand array data from the main DataFrame into individual rows.
    
    Args:
        df: Main DataFrame containing array data
        target_col: Column name for target values
        std_col: Column name for standard deviation values  
        concentration_col: Column name for concentration values
        
    Returns:
        Expanded DataFrame with individual measurement rows
    """
    try:
        # For the actual data structure, the target_col directly contains the arrays
        # No need to search for array-like column names
        
        # Check if the target column exists directly
        if target_col not in df.columns:
            logging.warning(f"Target column {target_col} not found in DataFrame")
            return pd.DataFrame()
            
        # Check if std and concentration columns exist
        if std_col not in df.columns:
            logging.warning(f"Std column {std_col} not found in DataFrame")
            return pd.DataFrame()
            
        # For concentration, we'll create a "Concentration (mg/mL)" column from the concentration data
        if concentration_col not in df.columns:
            logging.warning(f"Concentration column {concentration_col} not found in DataFrame")
            return pd.DataFrame()
        
        logging.info(f"Expanding arrays from columns: {target_col}, {std_col}, {concentration_col}")
        
        # Create expanded data
        expanded_rows = []
        
        for idx, row in df.iterrows():
            try:
                # Parse target values
                target_values = parse_array_string(row[target_col])
                if not target_values:
                    continue
                    
                # Parse std values - handle both None/NA and actual values
                std_val = row[std_col]
                if std_val is None or (isinstance(std_val, float) and pd.isna(std_val)):
                    std_values = []
                else:
                    std_values = parse_array_string(std_val)
                
                # Parse concentration values - handle both None/NA and actual values
                conc_val = row[concentration_col]
                if conc_val is None or (isinstance(conc_val, float) and pd.isna(conc_val)):
                    conc_values = []
                else:
                    conc_values = parse_array_string(conc_val)
                
                # Create individual rows for each measurement
                for i, target_val in enumerate(target_values):
                    new_row = row.copy()
                    
                    # Create standardized column names for the pipeline
                    if target_col == "tm":
                        new_row["Tm (C) Mean"] = target_val
                        new_row["Tm (C) Std"] = std_values[i] if i < len(std_values) else 0.0
                    elif target_col == "diff":
                        new_row["Diffusion Coefficient Mean (cm²/s)"] = target_val
                        new_row["Diffusion Coefficient Std (cm²/s)"] = std_values[i] if i < len(std_values) else 0.0
                    elif target_col == "visc":
                        new_row["Viscosity (cP) Mean"] = target_val
                        new_row["Viscosity (cP) Std"] = std_values[i] if i < len(std_values) else 0.0
                    
                    # Add concentration value
                    if conc_values and i < len(conc_values):
                        new_row["Concentration (mg/mL)"] = conc_values[i]
                    else:
                        # Set default concentrations based on measurement type
                        # Note: These defaults should match the actual experimental conditions
                        if target_col == "tm" or target_col == "diff":
                            new_row["Concentration (mg/mL)"] = 15.0  # Default for Tm and Diffusion
                        elif target_col == "visc":
                            new_row["Concentration (mg/mL)"] = 120.0  # Default for Viscosity
                        else:
                            new_row["Concentration (mg/mL)"] = 15.0  # Default
                    
                    expanded_rows.append(new_row)
                    
            except Exception as e:
                logging.warning(f"Error processing row {idx}: {e}")
                continue
        
        if expanded_rows:
            expanded_df = pd.DataFrame(expanded_rows)
            logging.info(f"Expanded from {len(df)} formulations to {len(expanded_df)} measurement points")
            return expanded_df
        else:
            logging.warning("No valid data found during expansion")
            return pd.DataFrame()
            
    except Exception as e:
        logging.error(f"Error in expand_array_data: {e}")
        return pd.DataFrame() 