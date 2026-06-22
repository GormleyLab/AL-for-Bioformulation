#!/usr/bin/env python
"""
BoTorch configuration - provides settings for the optimization pipeline.
"""

import torch
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import yaml


# Valid acquisition-function choices for the single- and multi-objective arms.
VALID_SINGLE_ACQF = {"qEI", "qUCB", "greedy_cl"}
VALID_MULTI_ACQF = {"qEHVI", "greedy_cl"}


@dataclass
class Config:
    """Configuration for BoTorch optimization pipeline."""
    
    # Core settings
    random_state: int = 42
    device: str = "auto"
    data_file: str = "data/Antibody_Formulation_Data_Complete_Seed_to_Batch_2_JSON.json"
    output_dir: str = "outputs"
    
    # Feature columns (consistent across all datasets)
    feature_columns: List[str] = field(default_factory=lambda: [
        "Molarity", "NaCl", "Sucrose", "Arginine", "pH", "Buffer", "Concentration (mg/mL)"
    ])
    
    # Target datasets and objectives (1.0 = maximize, -1.0 = minimize)
    objectives: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "tm": {
            "target_column": "Tm (C) Mean",
            "std_column": "Tm (C) Std", 
            "direction": 1.0,  # maximize
            "concentration": "concentration_tm",
            "value": "tm",
            "std": "tm_std"
        },
        "diffusion": {
            "target_column": "Diffusion Coefficient Mean (cm²/s)",
            "std_column": "Diffusion Coefficient Std (cm²/s)",
            "direction": 1.0,  # maximize
            "concentration": "concentration_diff", 
            "value": "diff",
            "std": "diff_std"
        },
        "viscosity": {
            "target_column": "Viscosity (cP) Mean",
            "std_column": "Viscosity (cP) Std",
            "direction": -1.0,  # minimize
            "concentration": "concentration_visc",
            "value": "visc", 
            "std": "visc_std"
        }
    })
    
    # Source columns for data loading
    excipient_columns: List[str] = field(default_factory=lambda: [
        "Molarity", "NaCl", "Sucrose", "Arginine", "pH", "Buffer"
    ])
    
    metadata_columns: List[str] = field(default_factory=lambda: [
        "Formulation ID", "Objective", "Generation"
    ])
    
    # Model settings
    cv_folds: int = 10
    min_samples: int = 15
    min_r2: float = 0.2
    
    # Optimization settings
    batch_size: int = 6
    mc_samples: int = 128
    num_restarts: int = 10
    reference_point: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    # Acquisition function selection
    #
    # single_acqf controls the single-objective optimization arm:
    #   "qEI"       q-Expected Improvement (default; reproduces the published pipeline)
    #   "qUCB"      q-Upper Confidence Bound (exploitative; uses ucb_beta below)
    #   "greedy_cl" greedy posterior-mean with constant-liar (CL-min) batching (pure exploitation)
    #
    # multi_acqf controls the multi-objective arm:
    #   "qEHVI"     q-Expected Hypervolume Improvement (default; reproduces the published pipeline)
    #   "greedy_cl" greedy geometric-mean posterior with constant-liar batching (pure exploitation)
    #
    # NOTE: the "qEI"/"qEHVI" defaults preserve the exact published behavior. 
    single_acqf: str = "qEI"
    multi_acqf: str = "qEHVI"
    ucb_beta: float = 0.1  # exploration weight for qUCB (only used when single_acqf == "qUCB")
    


    valid_buffer_pH_pairs = {
        1: [4.0, 4.5, 5.0, 5.5],  # Acetate
        2: [4.0, 5.0, 6.0, 7.0]   # Citrate-Phosphate
    }


    # Prediction settings
    # concentrations used for each objective when calculting objectivce predictions on candidates
    # returned by the optimization loop
    pred_conc = {
        "tm": 15.0,
        "diffusion": 15.0,
        "viscosity": 120.0
    }

    # Output flags
    save_models: bool = True
    save_candidates: bool = True

    # SHAP flags
    run_shap_analysis: bool = True   # flag to run SHAP analysis
    shap_verbose: bool = False        # flag to print real-time progress of SHAP analysis
    
    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate enumerated config fields, raising ValueError on bad input."""
        if self.single_acqf not in VALID_SINGLE_ACQF:
            raise ValueError(
                f"single_acqf must be one of {sorted(VALID_SINGLE_ACQF)}, got {self.single_acqf!r}"
            )
        if self.multi_acqf not in VALID_MULTI_ACQF:
            raise ValueError(
                f"multi_acqf must be one of {sorted(VALID_MULTI_ACQF)}, got {self.multi_acqf!r}"
            )

    @property
    def torch_device(self) -> torch.device:
        """Get torch device."""
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)
    
    @property
    def objective_names(self) -> List[str]:
        """Get ordered list of objective names."""
        return list(self.objectives.keys())
    
    @property
    def objective_directions(self) -> Dict[str, float]:
        """Get objective directions."""
        return {name: obj["direction"] for name, obj in self.objectives.items()}


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration with optional YAML override.
    
    Args:
        config_path: Optional path to YAML config file
        
    Returns:
        Config instance
    """
    config = Config()
    
    # Override with YAML if provided
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        # Simple override of top-level attributes
        for key, value in yaml_config.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # Re-validate after applying overrides (e.g. single_acqf / multi_acqf).
        config.validate()

    return config
