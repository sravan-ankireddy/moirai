import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
import json

def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def load_data(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Load data from various file formats.
    
    Args:
        file_path: Path to the data file
        **kwargs: Additional arguments for pandas read functions
        
    Returns:
        Loaded DataFrame
    """
    try:
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path, **kwargs)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path, **kwargs)
        elif file_path.suffix.lower() == '.parquet':
            df = pd.read_parquet(file_path, **kwargs)
        elif file_path.suffix.lower() == '.json':
            df = pd.read_json(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        logging.info(f"Successfully loaded data from {file_path}")
        logging.info(f"Data shape: {df.shape}")
        return df
        
    except Exception as e:
        logging.error(f"Failed to load data from {file_path}: {e}")
        raise

def validate_data(df: pd.DataFrame, target_column: str, 
                 timestamp_column: str = "timestamp") -> bool:
    """
    Validate the input data for time series forecasting.
    
    Args:
        df: Input DataFrame
        target_column: Name of the target column
        timestamp_column: Name of the timestamp column
        
    Returns:
        True if data is valid, raises exception otherwise
    """
    try:
        # Check if required columns exist
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame. Available columns: {list(df.columns)}")
        
        if timestamp_column not in df.columns:
            raise ValueError(f"Timestamp column '{timestamp_column}' not found in DataFrame. Available columns: {list(df.columns)}")
        
        # Check for missing values
        if df[target_column].isnull().any():
            logging.warning(f"Found {df[target_column].isnull().sum()} missing values in target column")
        
        if df[timestamp_column].isnull().any():
            raise ValueError("Found missing values in timestamp column")
        
        # Check data types
        if not pd.api.types.is_numeric_dtype(df[target_column]):
            logging.warning(f"Target column '{target_column}' is not numeric")
        
        # Check minimum data length
        if len(df) < 100:
            logging.warning(f"Dataset is quite small ({len(df)} rows). Consider using more data.")
        
        logging.info("Data validation completed successfully")
        return True
        
    except Exception as e:
        logging.error(f"Data validation failed: {e}")
        raise

def plot_time_series(df: pd.DataFrame, target_column: str, 
                    timestamp_column: str = "timestamp", 
                    title: str = "Time Series Data",
                    figsize: tuple = (12, 6)) -> None:
    """
    Plot the time series data.
    
    Args:
        df: Input DataFrame
        target_column: Name of the target column
        timestamp_column: Name of the timestamp column
        title: Plot title
        figsize: Figure size
    """
    try:
        plt.figure(figsize=figsize)
        plt.plot(pd.to_datetime(df[timestamp_column]), df[target_column])
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel(target_column)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        logging.error(f"Failed to plot time series: {e}")

def plot_forecasts(forecasts: List[Any], inputs: List[Any], 
                  num_plots: int = 3, figsize: tuple = (15, 10)) -> None:
    """
    Plot forecast results.
    
    Args:
        forecasts: List of forecast objects
        inputs: List of input time series
        num_plots: Number of plots to display
        figsize: Figure size
    """
    try:
        num_plots = min(num_plots, len(forecasts))
        fig, axes = plt.subplots(num_plots, 1, figsize=figsize)
        
        if num_plots == 1:
            axes = [axes]
        
        for i in range(num_plots):
            ts = inputs[i]
            forecast = forecasts[i]
            
            # Plot historical data
            axes[i].plot(ts["target"], label="Historical", color="blue")
            
            # Plot forecast
            forecast_start = len(ts["target"])
            forecast_range = range(forecast_start, forecast_start + len(forecast.mean))
            
            axes[i].plot(forecast_range, forecast.mean, label="Forecast", color="red")
            
            # Plot confidence intervals
            if hasattr(forecast, 'samples') and forecast.samples is not None:
                quantiles = np.percentile(forecast.samples, [10, 90], axis=0)
                axes[i].fill_between(forecast_range, quantiles[0], quantiles[1], 
                                   alpha=0.3, color="red", label="80% CI")
            
            axes[i].set_title(f"Forecast {i+1}")
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        logging.error(f"Failed to plot forecasts: {e}")

def plot_metrics_comparison(metrics_dict: Dict[str, Dict[str, float]], 
                          figsize: tuple = (12, 8)) -> None:
    """
    Plot comparison of different metrics.
    
    Args:
        metrics_dict: Dictionary of metrics for different models/experiments
        figsize: Figure size
    """
    try:
        if not metrics_dict:
            logging.warning("No metrics to plot")
            return
        
        # Extract metric names (excluding non-numeric metrics)
        numeric_metrics = set()
        for metrics in metrics_dict.values():
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not key.endswith('_count'):
                    numeric_metrics.add(key)
        
        numeric_metrics = sorted(list(numeric_metrics))
        
        if not numeric_metrics:
            logging.warning("No numeric metrics found to plot")
            return
        
        # Create subplots
        n_metrics = len(numeric_metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, metric in enumerate(numeric_metrics):
            if i < len(axes):
                values = [metrics_dict[exp].get(metric, 0) for exp in metrics_dict.keys()]
                labels = list(metrics_dict.keys())
                
                axes[i].bar(labels, values)
                axes[i].set_title(f"{metric.upper()}")
                axes[i].tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for i in range(len(numeric_metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        logging.error(f"Failed to plot metrics comparison: {e}")

def save_results(results: Dict[str, Any], output_path: str) -> None:
    """
    Save results to file.
    
    Args:
        results: Dictionary containing results
        output_path: Path to save the results
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        results_serializable = convert_numpy_types(results)
        
        with open(output_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        logging.info(f"Results saved to {output_path}")
        
    except Exception as e:
        logging.error(f"Failed to save results: {e}")
        raise

def generate_summary_report(perplexity_metrics: Dict[str, float], 
                          accuracy_metrics: Dict[str, float]) -> str:
    """
    Generate a summary report of the evaluation results.
    
    Args:
        perplexity_metrics: Perplexity evaluation metrics
        accuracy_metrics: Accuracy evaluation metrics
        
    Returns:
        Formatted summary report as string
    """
    try:
        report = []
        report.append("="*60)
        report.append("MOIRAI MODEL EVALUATION SUMMARY")
        report.append("="*60)
        report.append()
        
        # Perplexity section
        report.append("PERPLEXITY METRICS:")
        report.append("-" * 20)
        for key, value in perplexity_metrics.items():
            if isinstance(value, float):
                report.append(f"{key.upper()}: {value:.4f}")
            else:
                report.append(f"{key.upper()}: {value}")
        report.append()
        
        # Accuracy section
        report.append("ACCURACY METRICS:")
        report.append("-" * 20)
        for key, value in accuracy_metrics.items():
            if isinstance(value, float):
                report.append(f"{key.upper()}: {value:.4f}")
            else:
                report.append(f"{key.upper()}: {value}")
        report.append()
        
        # Interpretation
        report.append("INTERPRETATION:")
        report.append("-" * 15)
        
        perplexity = perplexity_metrics.get('perplexity', float('inf'))
        if perplexity < 10:
            report.append("• Excellent model fit (perplexity < 10)")
        elif perplexity < 50:
            report.append("• Good model fit (perplexity < 50)")
        elif perplexity < 100:
            report.append("• Moderate model fit (perplexity < 100)")
        else:
            report.append("• Poor model fit (perplexity >= 100)")
        
        mape = accuracy_metrics.get('mape', float('inf'))
        if mape < 10:
            report.append("• Excellent forecast accuracy (MAPE < 10%)")
        elif mape < 20:
            report.append("• Good forecast accuracy (MAPE < 20%)")
        elif mape < 50:
            report.append("• Moderate forecast accuracy (MAPE < 50%)")
        else:
            report.append("• Poor forecast accuracy (MAPE >= 50%)")
        
        report.append("="*60)
        
        return "\n".join(report)
        
    except Exception as e:
        logging.error(f"Failed to generate summary report: {e}")
        return f"Error generating report: {e}"

def print_summary_report(perplexity_metrics: Dict[str, float], 
                        accuracy_metrics: Dict[str, float]) -> None:
    """
    Print the summary report to console.
    
    Args:
        perplexity_metrics: Perplexity evaluation metrics
        accuracy_metrics: Accuracy evaluation metrics
    """
    report = generate_summary_report(perplexity_metrics, accuracy_metrics)
    print(report)