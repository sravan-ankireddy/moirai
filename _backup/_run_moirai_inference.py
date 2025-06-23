#!/usr/bin/env python3
"""
Run inference script for Moirai model evaluation using refactored modules.
"""

import argparse
import os
import sys
from pathlib import Path
import torch

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from moirai_perplexity_core import MoiraiPerplexityEvaluator
from moirai_perplexity_utils import (
    setup_logging, load_data, validate_data, plot_time_series,
    plot_forecasts, plot_metrics_comparison, save_results,
    print_summary_report
)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Moirai model perplexity evaluation and forecasting"
    )
    
    # Model arguments
    parser.add_argument(
        "--model", 
        type=str, 
        default="moirai",
        choices=["moirai"],
        help="Model type to use (default: moirai)"
    )
    
    parser.add_argument(
        "--size", 
        type=str, 
        default="small",
        choices=["small", "base", "large"],
        help="Model size (default: small)"
    )
    
    # Data arguments
    parser.add_argument(
        "--csv-path", 
        type=str, 
        required=True,
        help="Path to the CSV file containing time series data"
    )
    
    parser.add_argument(
        "--target-column",
        type=int,
        default=1,
        help="Index of the target column in CSV (0-based, default: 1)"
    )
    
    parser.add_argument(
        "--timestamp-column",
        type=int,
        default=0,
        help="Index of the timestamp column in CSV (0-based, default: 0)"
    )
    
    # Test/Window arguments
    parser.add_argument(
        "--num-windows",
        type=int,
        default=96,
        help="Number of test windows for evaluation (default: 96)"
    )
    
    # Prediction arguments
    parser.add_argument(
        "--pdt", 
        type=int, 
        default=8,
        help="Prediction length (default: 8)"
    )
    
    parser.add_argument(
        "--ctx", 
        type=int, 
        default=128,
        help="Context length (default: 128)"
    )
    
    parser.add_argument(
        "--num-samples", 
        type=int, 
        default=1000,
        help="Number of samples for probabilistic forecasting (default: 1000)"
    )
    
    # Hardware arguments
    parser.add_argument(
        "--gpu", 
        type=int, 
        default=0,
        help="GPU device ID (default: 0)"
    )
    
    # Analysis arguments
    parser.add_argument(
        "--analysis-freq", 
        type=int, 
        default=10,
        help="Frequency for analysis updates (default: 10)"
    )
    
    parser.add_argument(
        "--num-forecast-plots",
        type=int,
        default=3,
        help="Number of forecast plots to generate (default: 3)"
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Directory to save results (default: ./results)"
    )
    
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save plots to files instead of displaying"
    )
    
    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots"
    )
    
    return parser.parse_args()

def setup_device(gpu_id: int):
    """Setup compute device."""
    if torch.cuda.is_available() and gpu_id >= 0:
        device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)
        print(f"Using GPU: {device}")
    else:
        device = "cpu"
        print("Using CPU")
    
    return device

def get_column_name(df, column_index, column_type="column"):
    """Get column name from index with validation."""
    if column_index >= len(df.columns):
        raise ValueError(f"{column_type} index {column_index} is out of range. DataFrame has {len(df.columns)} columns.")
    return df.columns[column_index]

def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Setup device
    device = setup_device(args.gpu)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print("="*60)
        print("MOIRAI MODEL EVALUATION")
        print("="*60)
        print(f"Model: {args.model}")
        print(f"Size: {args.size}")
        print(f"CSV Path: {args.csv_path}")
        print(f"Target Column Index: {args.target_column}")
        print(f"Timestamp Column Index: {args.timestamp_column}")
        print(f"Prediction Length: {args.pdt}")
        print(f"Context Length: {args.ctx}")
        print(f"Number of Samples: {args.num_samples}")
        print(f"Number of Windows: {args.num_windows}")
        print(f"Device: {device}")
        print("="*60)
        
        # Load and validate data
        print("\n1. Loading and validating data...")
        df = load_data(args.csv_path)
        
        # Get actual column names from indices
        target_column_name = get_column_name(df, args.target_column, "Target")
        timestamp_column_name = get_column_name(df, args.timestamp_column, "Timestamp")
        
        print(f"Target column: {target_column_name} (index {args.target_column})")
        print(f"Timestamp column: {timestamp_column_name} (index {args.timestamp_column})")
        print(f"Available columns: {list(df.columns)}")
        
        validate_data(df, target_column_name, timestamp_column_name)
        
        print(f"Data shape: {df.shape}")
        print(f"Date range: {df[timestamp_column_name].min()} to {df[timestamp_column_name].max()}")
        
        # Plot original time series if requested
        if not args.no_plots:
            print("\n2. Plotting original time series...")
            plot_time_series(
                df, 
                target_column_name, 
                timestamp_column_name,
                title=f"Original Time Series - {Path(args.csv_path).stem}"
            )
        
        # Initialize evaluator with custom parameters
        print("\n3. Initializing Moirai evaluator...")
        evaluator = MoiraiPerplexityEvaluator(
            model_size=args.size,
            prediction_length=args.pdt,
            num_windows=args.num_windows
        )
        
        # Update predictor parameters to match arguments
        print("\n4. Loading model...")
        evaluator.load_model()
        
        # Update predictor with custom parameters
        if evaluator.predictor is not None:
            evaluator.predictor.context_length = args.ctx
            evaluator.predictor.num_samples = args.num_samples
        
        # Prepare dataset with windows
        print("\n5. Preparing dataset with windows...")
        train_ds, test_ds = evaluator.prepare_dataset_with_windows(
            df, 
            target_column_name, 
            timestamp_column_name
        )
        
        # Calculate perplexity
        print("\n6. Calculating perplexity...")
        perplexity_metrics = evaluator.calculate_perplexity(test_ds)
        
        # Evaluate forecast accuracy
        print("\n7. Evaluating forecast accuracy...")
        accuracy_metrics = evaluator.evaluate_forecast_accuracy(test_ds)
        
        # Generate sample forecasts
        print("\n8. Generating sample forecasts...")
        forecast_results = evaluator.generate_forecasts(
            train_ds, 
            num_series=args.num_forecast_plots
        )
        
        # Plot forecasts if requested
        if not args.no_plots:
            print("\n9. Plotting forecasts...")
            plot_forecasts(
                forecast_results["forecasts"],
                forecast_results["inputs"],
                num_plots=args.num_forecast_plots
            )
        
        # Create metrics comparison
        all_metrics = {
            "perplexity_metrics": perplexity_metrics,
            "accuracy_metrics": accuracy_metrics
        }
        
        if not args.no_plots:
            print("\n10. Plotting metrics comparison...")
            plot_metrics_comparison(all_metrics)
        
        # Save results
        print("\n11. Saving results...")
        results = {
            "arguments": vars(args),
            "column_mapping": {
                "target_column_name": target_column_name,
                "timestamp_column_name": timestamp_column_name,
                "target_column_index": args.target_column,
                "timestamp_column_index": args.timestamp_column
            },
            "window_info": {
                "num_windows": args.num_windows,
                "prediction_length": args.pdt
            },
            "perplexity_metrics": perplexity_metrics,
            "accuracy_metrics": accuracy_metrics,
            "data_info": {
                "shape": df.shape,
                "date_range": {
                    "start": str(df[timestamp_column_name].min()),
                    "end": str(df[timestamp_column_name].max())
                },
                "columns": list(df.columns)
            }
        }
        
        results_file = output_dir / f"results_{args.size}_{Path(args.csv_path).stem}.json"
        save_results(results, results_file)
        
        # Print summary report
        print("\n12. Summary Report:")
        print_summary_report(perplexity_metrics, accuracy_metrics)
        
        print(f"\nResults saved to: {results_file}")
        print("Evaluation completed successfully!")
        
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()