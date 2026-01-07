#!/usr/bin/env python
"""
BoTorch main script for antibody formulation optimization.
"""

import sys
import logging
from pathlib import Path

from config.config import load_config
from src.utils import parse_arguments, setup_output_directory, setup_logging
from src.pipeline import BoTorchPipeline


def main():
    """Main orchestration function for the optimization pipeline."""
    
    # Parse arguments and load configuration
    args = parse_arguments()
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.debug:
        config.logging_level = "DEBUG"
    
    # Setup environment
    setup_output_directory(config.output_dir)
    setup_logging(config.output_dir, "DEBUG" if args.debug else "INFO")
    
    try:
        logging.info("=" * 60)
        logging.info("ANTIBODY FORMULATION OPTIMIZATION PIPELINE")
        logging.info("=" * 60)
        
        
        # Initialize and run pipeline
        pipeline = BoTorchPipeline(config)
        results = pipeline.run_full_pipeline()
        
        if results["success"]:
            # Display summary
            summary = pipeline.get_summary()
            
            logging.info("\n" + "=" * 50)
            logging.info("OPTIMIZATION COMPLETED SUCCESSFULLY")
            logging.info("=" * 50)
            
            logging.info(f"Datasets processed: {summary['datasets']}")
            logging.info(f"Models trained: {summary['models_trained']}")
            
            total_candidates = sum(summary['optimization_candidates'].values())
            logging.info(f"Total candidates: {total_candidates}")
            logging.info(f"Results saved to: {config.output_dir}")
            
            # Cross-validation summary
            if 'cv_performance' in summary:
                logging.info("\nCross-validation results (Mean R² across folds):")
                for obj_name, performance in summary['cv_performance'].items():
                    logging.info(f"  {obj_name}: {performance}")
            
            # Candidate breakdown
            if summary['optimization_candidates']:
                logging.info("\nCandidate breakdown:")
                for opt_type, count in summary['optimization_candidates'].items():
                    logging.info(f"  {opt_type}: {count} candidates")
            
            logging.info("\nPipeline completed successfully!")
            
            return results
        else:
            error_msg = results.get("error", "Unknown error")
            logging.error(f"Pipeline failed: {error_msg}")
            sys.exit(1)
        
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main() 