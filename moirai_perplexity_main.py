import torch
import matplotlib.pyplot as plt
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from huggingface_hub import hf_hub_download
import numpy as np
from tqdm import tqdm
import os
import seaborn as sns
from typing import Dict, Any, Optional, Tuple
import warnings
import argparse
from scipy import interpolate
warnings.filterwarnings('ignore')

from uni2ts.eval_util.plot import plot_single
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Moirai Uncertainty Analysis v3')
    
    parser.add_argument('--gpu', type=int, default=0, 
                        help='GPU device ID (default: 0)')
    parser.add_argument('--model', type=str, default='moirai', choices=['moirai', 'moirai-moe'],
                        help='Model name (default: moirai)')
    parser.add_argument('--size', type=str, default='large', choices=['small', 'base', 'large'],
                        help='Model size (default: large)')
    parser.add_argument('--pdt', type=int, default=8,
                        help='Prediction length (default: 8)')
    parser.add_argument('--ctx', type=int, default=64,
                        help='Context length for forecasting (default: 64)')
    parser.add_argument('--input-ctx', type=int, default=None,
                        help='Context length for input importance measurement (default: same as --ctx)')
    parser.add_argument('--psz', type=str, default='auto',
                        help='Patch size (default: auto)')
    parser.add_argument('--bsz', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--test', type=int, default=None,
                        help='Test set length (default: 100*PDT)')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of samples for uncertainty estimation (default: 100)')
    parser.add_argument('--csv-path', type=str, 
                        default="/home/sa53869/time-series/moirai/time-moe-eval/ETT-small/ETTm2.csv",
                        help='Path to CSV data file')
    parser.add_argument('--column', type=int, default=1,
                        help='Column number to analyze (0-indexed, default: 1 for 2nd column)')
    parser.add_argument('--analysis-freq', type=int, default=None,
                        help='Generate plots after every N windows (default: None, plot only at end)')
    parser.add_argument('--context-keep-ratio', type=float, default=0.75,
                        help='Ratio of context to keep in reduced methods (default: 0.75 for 75%%)')
    
    return parser.parse_args()

# Parse command line arguments.
args = parse_args()

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
print(f"Using GPU: {args.gpu}")

# Set parameters from arguments
MODEL = args.model
SIZE = args.size
PDT = args.pdt
CTX = args.ctx
INPUT_CTX = args.input_ctx if args.input_ctx is not None else args.ctx
PSZ = 16#args.psz
BSZ = args.bsz
TEST = args.test if args.test is not None else int(1000 * PDT)
NUM_SAMPLES = args.num_samples
COLUMN_NUM = args.column
ANALYSIS_FREQ = args.analysis_freq
CONTEXT_KEEP_RATIO = args.context_keep_ratio

print(f"Configuration:")
print(f"  Model: {MODEL}-{SIZE}")
print(f"  Context Length (Forecasting): {CTX}")
print(f"  Input Context Length (Importance): {INPUT_CTX}")
print(f"  Prediction Length: {PDT}")
print(f"  Patch Size: {PSZ}")
print(f"  Batch Size: {BSZ}")
print(f"  Test Length: {TEST}")
print(f"  Num Samples: {NUM_SAMPLES}")
print(f"  CSV Path: {args.csv_path}")
print(f"  Column: {COLUMN_NUM} (0-indexed)")
print(f"  Analysis Frequency: {ANALYSIS_FREQ if ANALYSIS_FREQ else 'End only'}")
print(f"  Context Keep Ratio: {CONTEXT_KEEP_RATIO} ({CONTEXT_KEEP_RATIO*100:.0f}%)")

# Read data into pandas DataFrame
csv_path = args.csv_path
df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

# Extract dataset name from CSV path
dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
print(f"Dataset name: {dataset_name}")

# Select the specified column
available_columns = df.columns.tolist()
print(f"Available columns: {available_columns}")

if args.column >= len(available_columns):
    raise ValueError(f"Column {args.column} not available. Dataset has {len(available_columns)} columns (0-{len(available_columns)-1})")

selected_column = available_columns[args.column]
SELECTED_COLUMN = selected_column  # Make it global for plotting functions
print(f"Selected column {args.column}: '{selected_column}'")

# Create results directory structure with dataset name and compression ratio
results_dir = f"results_v3_patch16/{dataset_name}/{MODEL}-{SIZE}/ctx{CTX}/comp{CONTEXT_KEEP_RATIO:.2f}"
os.makedirs(results_dir, exist_ok=True)
print(f"Results will be saved to: {results_dir}")

# Focus on selected column only
df_selected = df[[selected_column]].copy()
print(f"Focusing on column '{selected_column}' only. Data shape: {df_selected.shape}")

def interpolate_missing_values(values, missing_indices):
    """
    Interpolate missing values using linear interpolation based on neighboring values
    """
    if len(missing_indices) == 0:
        return values.copy()
    
    # Create a copy of the values
    interpolated_values = values.copy()
    
    # Create position indices
    positions = np.arange(len(values))
    available_positions = np.setdiff1d(positions, missing_indices)
    
    if len(available_positions) < 2:
        # If we have less than 2 available points, use simple filling
        if len(available_positions) == 1:
            interpolated_values[missing_indices] = values[available_positions[0]]
        else:
            # No available points, use mean of original series
            interpolated_values[missing_indices] = np.mean(values)
    else:
        # Use numpy-based linear interpolation
        # Fit a linear function to available points
        if len(available_positions) >= 2:
            # Use numpy polyfit for linear interpolation
            p = np.polyfit(available_positions, values[available_positions], 1)
            interpolated_values[missing_indices] = np.polyval(p, missing_indices)
    
    return interpolated_values

# Convert into GluonTS dataset
ds = PandasDataset(dict(df_selected))

# Get dataset information dynamically
num_series = len(list(ds))
print(f"Dataset info:")
print(f"Number of time series: {num_series}")
print(f"Available columns in CSV: {df.columns.tolist()}")
print(f"Target column being forecasted: {list(ds)[0]['target'][:5]}")

# Split into train/test set
train, test_template = split(ds, offset=-TEST)

# Generate test data 
test_data_full = test_template.generate_instances(
    prediction_length=PDT,
    windows=TEST // PDT,
    distance=PDT,
)

# Calculate actual number of windows
actual_windows = len(list(test_data_full.input))
print(f"Windows per series: {TEST // PDT}")
print(f"Expected total windows ({num_series} series × {TEST // PDT}): {num_series * (TEST // PDT)}")
print(f"Actual total windows: {actual_windows}")

# Load the base module once
print("Loading base model module...")
if MODEL == "moirai":
    base_module = MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-{SIZE}")
elif MODEL == "moirai-moe":
    base_module = MoiraiMoEModule.from_pretrained(f"Salesforce/moirai-moe-1.0-R-{SIZE}")

def create_model_with_context_length(context_length):
    """Create a new model with specific context length"""
    if MODEL == "moirai":
        model = MoiraiForecast(
            module=base_module,
            prediction_length=1,
            context_length=context_length,
            patch_size=PSZ,
            num_samples=NUM_SAMPLES,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
    elif MODEL == "moirai-moe":
        model = MoiraiMoEForecast(
            module=base_module,
            prediction_length=1,
            context_length=context_length,
            patch_size=16,
            num_samples=NUM_SAMPLES,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
    return model.create_predictor(batch_size=BSZ)

def calculate_forecast_uncertainty(predictor, input_data, prediction_length) -> Dict[str, np.ndarray]:
    """
    Calculate forecast uncertainty using Moirai's built-in sampling capability
    """
    print(f"Generating forecast with {NUM_SAMPLES} samples for uncertainty estimation...")
    
    # Create forecast input
    forecast_input = [input_data]
    
    # Get forecast with samples (Moirai automatically generates multiple samples)
    forecast = next(iter(predictor.predict(forecast_input)))
    
    # Extract samples - shape should be [num_samples, prediction_length]
    forecast_samples = forecast.samples
    print(f"Forecast samples shape: {forecast_samples.shape}")
    
    # Ensure we have the right dimensions
    if forecast_samples.ndim == 3:
        forecast_samples = forecast_samples.squeeze(0)  # Remove batch dimension if present
    
    # Calculate statistics across samples (axis=0 since samples are in first dimension)
    mean_forecast = np.mean(forecast_samples, axis=0)
    std_forecast = np.std(forecast_samples, axis=0)
    median_forecast = np.median(forecast_samples, axis=0)
    
    # Calculate different uncertainty measures
    cv_uncertainty = std_forecast / (np.abs(mean_forecast) + 1e-6)
    q25 = np.percentile(forecast_samples, 25, axis=0)
    q75 = np.percentile(forecast_samples, 75, axis=0)
    iqr_uncertainty = (q75 - q25) / (np.abs(median_forecast) + 1e-6)
    entropy_uncertainty = np.log(std_forecast + 1e-6)
    min_forecast = np.min(forecast_samples, axis=0)
    max_forecast = np.max(forecast_samples, axis=0)
    range_uncertainty = (max_forecast - min_forecast) / (np.abs(mean_forecast) + 1e-6)
    
    print(f"Forecast uncertainty statistics:")
    print(f"  Mean CV uncertainty: {np.mean(cv_uncertainty):.4f}")
    print(f"  Mean IQR uncertainty: {np.mean(iqr_uncertainty):.4f}")
    print(f"  Mean entropy uncertainty: {np.mean(entropy_uncertainty):.4f}")
    
    return {
        'samples': forecast_samples,
        'mean': mean_forecast,
        'std': std_forecast,
        'median': median_forecast,
        'cv_uncertainty': cv_uncertainty,
        'iqr_uncertainty': iqr_uncertainty,
        'entropy_uncertainty': entropy_uncertainty,
        'range_uncertainty': range_uncertainty,
        'q25': q25,
        'q75': q75,
        'min': min_forecast,
        'max': max_forecast
    }

def calculate_autoregressive_input_uncertainty(input_data, window_id, full_dataset=None, window_start_idx=None) -> Dict[str, Any]:
    """
    Calculate uncertainty autoregressively for input context reconstruction using Moirai
    In v3: Always use the most recent INPUT_CTX samples as context for importance measurement
    """
    print(f"Starting autoregressive input uncertainty estimation for window {window_id} (v3 logic)...")
    print(f"Using INPUT_CTX={INPUT_CTX} for importance measurement, CTX={CTX} for forecasting")
    
    if full_dataset is None or window_start_idx is None:
        # Fallback to v2 logic if full dataset not provided
        print("Warning: Full dataset not provided, falling back to v2 logic")
        full_sequence = input_data["target"]
    else:
        # v3 logic: Use full dataset to always have INPUT_CTX context
        full_sequence = full_dataset
        print(f"Window {window_id}: Using full dataset with {len(full_sequence)} samples, window starts at index {window_start_idx}")
    
    # Get the current window's data for reference
    current_window_sequence = input_data["target"]
    if len(current_window_sequence) > CTX:
        current_window_sequence = current_window_sequence[-CTX:]
    
    # For v3: Calculate the positions we want to predict in the current window
    window_length = len(current_window_sequence)
    
    # Storage for results at each autoregressive step
    ar_uncertainties = []
    ar_predictions = []
    ar_samples_all = []
    ar_errors = []
    ar_cv_uncertainties = []
    ar_iqr_uncertainties = []
    ar_entropy_uncertainties = []
    
    print(f"Window {window_id}: Computing v3 autoregressive uncertainty for {window_length} samples...")
    print(f"  V3 Logic: Always using most recent {INPUT_CTX} samples as context for importance measurement")
    
    # In v3: For each position in the current window, use the most recent INPUT_CTX samples as context
    for pos in tqdm(range(window_length), desc=f"Window {window_id} - AR uncertainty (v3)"):
        # Calculate the absolute position in the full dataset
        if full_dataset is not None and window_start_idx is not None:
            absolute_pos = window_start_idx + pos
            true_value = current_window_sequence[pos]
            
            # Get the most recent INPUT_CTX samples as context (ending just before the current position)
            context_start = absolute_pos - INPUT_CTX
            context_end = absolute_pos
            context_data = full_sequence[context_start:context_end]
            context_length = INPUT_CTX
            
        else:
            # Fallback to v2 logic (shouldn't happen in normal operation)
            true_value = current_window_sequence[pos]
            if pos == 0:
                context_data = []
                context_length = 0
            else:
                context_data = current_window_sequence[:pos]
                context_length = len(context_data)
        
        if context_length == 0:
            # No context available - use a simple fallback (only for edge cases)
            print(f"  Position {pos}: No context available, using fallback")
            pred_mean = 0.0
            pred_samples = np.random.normal(0.0, 1.0, NUM_SAMPLES)
            pred_std = np.std(pred_samples)
            pred_cv = 1.0  # High uncertainty when no context
            pred_iqr = np.percentile(pred_samples, 75) - np.percentile(pred_samples, 25)
            pred_iqr_uncertainty = 1.0
            pred_entropy = np.log(pred_std + 1e-6)
            prediction_error = abs(pred_mean - true_value)
        else:
            # Create model with appropriate context length
            predictor = create_model_with_context_length(context_length)
            
            # Create input for prediction
            step_input_data = input_data.copy()
            step_input_data["target"] = context_data
            
            try:
                # Get forecast with uncertainty (Moirai generates samples automatically)
                step_forecast = next(iter(predictor.predict([step_input_data])))
                step_samples = step_forecast.samples  # Shape: [num_samples, 1]
                
                # Flatten samples if needed
                if step_samples.ndim > 1:
                    step_samples = step_samples.flatten()
                
                # Calculate statistics
                pred_mean = np.mean(step_samples)
                pred_std = np.std(step_samples)
                pred_cv = pred_std / (abs(pred_mean) + 1e-6)
                pred_iqr = np.percentile(step_samples, 75) - np.percentile(step_samples, 25)
                pred_iqr_uncertainty = pred_iqr / (abs(pred_mean) + 1e-6)
                pred_entropy = np.log(pred_std + 1e-6)
                prediction_error = abs(pred_mean - true_value)
                
            except Exception as e:
                print(f"  Error at position {pos}: {e}, using fallback")
                # Fallback: use simple prediction with artificial uncertainty
                pred_mean = context_data[-1] if len(context_data) > 0 else 0.0
                pred_samples = np.random.normal(pred_mean, abs(pred_mean) * 0.1, NUM_SAMPLES)
                pred_std = np.std(pred_samples)
                pred_cv = pred_std / (abs(pred_mean) + 1e-6)
                pred_iqr = np.percentile(pred_samples, 75) - np.percentile(pred_samples, 25)
                pred_iqr_uncertainty = pred_iqr / (abs(pred_mean) + 1e-6)
                pred_entropy = np.log(pred_std + 1e-6)
                prediction_error = abs(pred_mean - true_value)
        
        # Store results
        ar_predictions.append(pred_mean)
        ar_uncertainties.append(prediction_error)  # Use prediction error as importance
        ar_cv_uncertainties.append(pred_cv)
        ar_iqr_uncertainties.append(pred_iqr_uncertainty)
        ar_entropy_uncertainties.append(pred_entropy)
        ar_samples_all.append(pred_samples if 'pred_samples' in locals() else step_samples if 'step_samples' in locals() else np.array([pred_mean]))
        ar_errors.append(prediction_error)
        
        if pos % 25 == 0:
            print(f"  Position {pos}: ctx_len={context_length}, error={prediction_error:.4f}, CV_unc={pred_cv:.4f}")
    
    print(f"Completed v3 autoregressive uncertainty estimation for window {window_id} - analyzed {window_length} positions")
    
    return {
        'ar_predictions': np.array(ar_predictions),
        'ar_uncertainties': np.array(ar_uncertainties),  # This is prediction errors (importance)
        'ar_cv_uncertainties': np.array(ar_cv_uncertainties),
        'ar_iqr_uncertainties': np.array(ar_iqr_uncertainties),
        'ar_entropy_uncertainties': np.array(ar_entropy_uncertainties),
        'ar_samples': ar_samples_all,
        'ar_errors': np.array(ar_errors),
        'true_context': current_window_sequence,
        'context_length': window_length,
        'window_id': window_id
    }

def perform_forecasting_comparison_with_uncertainty(input_data, label_data, importance_scores, window_id):
    """
    Perform forecasting with uncertainty analysis for TEN different context selection strategies:
    1. Full context (all CTX samples)
    2. Random subset of context (configurable %)
    3. Most important subset of context (configurable %)
    4. Random subset with interpolation (maintaining original context length)
    5. Most important subset with interpolation (maintaining original context length)
    6. Least important subset of context (configurable %)
    7. Least important subset with interpolation (maintaining original context length)
    8. Uniform sampling by compression ratio (drop every Nth sample)
    9. Uniform sampling with resampling (drop every Nth then resample back to original length)
    10. Recent context only (take most recent compression_ratio * CTX samples)
    """
    # Get the target data and ensure it's exactly CTX length
    target_data = input_data['target']
    
    # Ensure we use the same sequence as importance analysis
    if len(target_data) > CTX:
        context_target = target_data[-CTX:]
    else:
        context_target = target_data
    
    actual_context_length = len(context_target)
    reduced_ctx = max(1, int(CONTEXT_KEEP_RATIO * actual_context_length))  # configurable % of actual context
    
    print(f"Window {window_id} Forecasting with Uncertainty (10 Methods):")
    print(f"  Full context length: {actual_context_length}")
    print(f"  Reduced context length ({CONTEXT_KEEP_RATIO*100:.0f}%): {reduced_ctx}")
    
    # 1. Full context forecasting with uncertainty
    print(f"  1. Full context forecasting with uncertainty...")
    forecast_input_data_full = {
        'target': context_target,
        'start': input_data['start'],
        'item_id': input_data.get('item_id', 0)
    }
    
    # Create model for full PDT prediction
    if MODEL == "moirai":
        model_full = MoiraiForecast(
            module=base_module,
            prediction_length=PDT,
            context_length=actual_context_length,
            patch_size=PSZ,
            num_samples=NUM_SAMPLES,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
    elif MODEL == "moirai-moe":
        model_full = MoiraiMoEForecast(
            module=base_module,
            prediction_length=PDT,
            context_length=actual_context_length,
            patch_size=16,
            num_samples=NUM_SAMPLES,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
    predictor_full = model_full.create_predictor(batch_size=BSZ)
    
    forecast_uncertainty_full = calculate_forecast_uncertainty(predictor_full, forecast_input_data_full, PDT)
    
    # 2. Random ${CONTEXT_KEEP_RATIO*100:.0f}% context forecasting with uncertainty
    print(f"  2. Random ${CONTEXT_KEEP_RATIO*100:.0f}% context forecasting with uncertainty...")
    np.random.seed(42 + window_id)  # Reproducible random selection
    random_indices = np.sort(np.random.choice(actual_context_length, reduced_ctx, replace=False))
    context_target_random = context_target[random_indices]
    
    forecast_input_data_random = {
        'target': context_target_random,
        'start': input_data['start'],
        'item_id': input_data.get('item_id', 0)
    }
    
    if MODEL == "moirai":
        model_random = MoiraiForecast(
            module=base_module,
            prediction_length=PDT,
            context_length=reduced_ctx,
            patch_size=PSZ,
            num_samples=NUM_SAMPLES,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
    elif MODEL == "moirai-moe":
        model_random = MoiraiMoEForecast(
            module=base_module,
            prediction_length=PDT,
            context_length=reduced_ctx,
            patch_size=16,
            num_samples=NUM_SAMPLES,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
    predictor_random = model_random.create_predictor(batch_size=BSZ)
    
    forecast_uncertainty_random = calculate_forecast_uncertainty(predictor_random, forecast_input_data_random, PDT)
    
    # 3. Most important {CONTEXT_KEEP_RATIO*100:.0f}% context forecasting with uncertainty
    print(f"  3. Most important {CONTEXT_KEEP_RATIO*100:.0f}% context forecasting with uncertainty...")
    
    # Ensure importance scores match the context length
    if len(importance_scores) == actual_context_length:
        context_importance = importance_scores
    elif len(importance_scores) > actual_context_length:
        context_importance = importance_scores[-actual_context_length:]
    else:
        mean_importance = np.mean(importance_scores) if len(importance_scores) > 0 else 1.0
        context_importance = np.concatenate([
            np.full(actual_context_length - len(importance_scores), mean_importance),
            importance_scores
        ])
    
    # Select the most important samples
    most_important_indices = np.argsort(context_importance)[-reduced_ctx:]
    most_important_indices = np.sort(most_important_indices)
    context_target_most_important = context_target[most_important_indices]
    
    forecast_input_data_most_important = {
        'target': context_target_most_important,
        'start': input_data['start'],
        'item_id': input_data.get('item_id', 0)
    }
    
    if MODEL == "moirai":
        model_most_important = MoiraiForecast(
            module=base_module,
            prediction_length=PDT,
            context_length=reduced_ctx,
            patch_size=PSZ,
            num_samples=NUM_SAMPLES,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
    elif MODEL == "moirai-moe":
        model_most_important = MoiraiMoEForecast(
            module=base_module,
            prediction_length=PDT,
            context_length=reduced_ctx,
            patch_size=16,
            num_samples=NUM_SAMPLES,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
    predictor_most_important = model_most_important.create_predictor(batch_size=BSZ)
    
    forecast_uncertainty_most_important = calculate_forecast_uncertainty(predictor_most_important, forecast_input_data_most_important, PDT)
    
    # 4. Random ${CONTEXT_KEEP_RATIO*100:.0f}% with interpolation (maintaining original context length)
    print(f"  4. Random ${CONTEXT_KEEP_RATIO*100:.0f}% with interpolation forecasting...")
    
    # Get indices to remove (25% least important random samples)
    missing_random_indices = np.setdiff1d(np.arange(actual_context_length), random_indices)
    
    # Interpolate missing values
    context_target_random_interp = interpolate_missing_values(context_target, missing_random_indices)
    
    forecast_input_data_random_interp = {
        'target': context_target_random_interp,
        'start': input_data['start'],
        'item_id': input_data.get('item_id', 0)
    }
    
    # Use full context length model since we maintain original length
    forecast_uncertainty_random_interp = calculate_forecast_uncertainty(predictor_full, forecast_input_data_random_interp, PDT)
    
    # 5. Most important {CONTEXT_KEEP_RATIO*100:.0f}% with interpolation (maintaining original context length)
    print(f"  5. Most important {CONTEXT_KEEP_RATIO*100:.0f}% with interpolation forecasting...")
    
    # Get indices to remove (25% least important samples)
    missing_important_indices = np.setdiff1d(np.arange(actual_context_length), most_important_indices)
    
    # Interpolate missing values
    context_target_important_interp = interpolate_missing_values(context_target, missing_important_indices)
    
    forecast_input_data_important_interp = {
        'target': context_target_important_interp,
        'start': input_data['start'],
        'item_id': input_data.get('item_id', 0)
    }
    
    # Use full context length model since we maintain original length
    forecast_uncertainty_important_interp = calculate_forecast_uncertainty(predictor_full, forecast_input_data_important_interp, PDT)
    
    # 6. Least important {CONTEXT_KEEP_RATIO*100:.0f}% (keep least important samples)
    print(f"  6. Least important {CONTEXT_KEEP_RATIO*100:.0f}% context forecasting...")
    
    # Select the least important samples (opposite of most important)
    least_important_indices = np.argsort(context_importance)[:reduced_ctx]
    least_important_indices = np.sort(least_important_indices)
    context_target_least_important = context_target[least_important_indices]
    
    forecast_input_data_least_important = {
        'target': context_target_least_important,
        'start': input_data['start'],
        'item_id': input_data.get('item_id', 0)
    }
    
    if MODEL == "moirai":
        model_least_important = MoiraiForecast(
            module=base_module,
            prediction_length=PDT,
            context_length=reduced_ctx,
            patch_size=PSZ,
            num_samples=NUM_SAMPLES,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
    elif MODEL == "moirai-moe":
        model_least_important = MoiraiMoEForecast(
            module=base_module,
            prediction_length=PDT,
            context_length=reduced_ctx,
            patch_size=16,
            num_samples=NUM_SAMPLES,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
    predictor_least_important = model_least_important.create_predictor(batch_size=BSZ)
    
    forecast_uncertainty_least_important = calculate_forecast_uncertainty(predictor_least_important, forecast_input_data_least_important, PDT)
    
    # 7. Least important {CONTEXT_KEEP_RATIO*100:.0f}% with interpolation (maintaining original context length)
    print(f"  7. Least important {CONTEXT_KEEP_RATIO*100:.0f}% with interpolation forecasting...")
    
    # Get indices to remove (25% most important samples)
    missing_least_important_indices = np.setdiff1d(np.arange(actual_context_length), least_important_indices)
    
    # Interpolate missing values
    context_target_least_important_interp = interpolate_missing_values(context_target, missing_least_important_indices)
    
    forecast_input_data_least_important_interp = {
        'target': context_target_least_important_interp,
        'start': input_data['start'],
        'item_id': input_data.get('item_id', 0)
    }
    
    # Use full context length model since we maintain original length
    forecast_uncertainty_least_important_interp = calculate_forecast_uncertainty(predictor_full, forecast_input_data_least_important_interp, PDT)
    
    # 8. Uniform sampling by compression ratio (drop every Nth sample)
    print(f"  8. Uniform sampling ({CONTEXT_KEEP_RATIO*100:.0f}%) forecasting...")
    
    # Calculate sampling step to achieve target compression
    step_size = max(1, int(1 / CONTEXT_KEEP_RATIO))
    uniform_indices = np.arange(0, actual_context_length, step_size)[:reduced_ctx]
    context_target_uniform = context_target[uniform_indices]
    
    forecast_input_data_uniform = {
        'target': context_target_uniform,
        'start': input_data['start'],
        'item_id': input_data.get('item_id', 0)
    }
    
    if MODEL == "moirai":
        model_uniform = MoiraiForecast(
            module=base_module,
            prediction_length=PDT,
            context_length=len(context_target_uniform),
            patch_size=PSZ,
            num_samples=NUM_SAMPLES,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
    elif MODEL == "moirai-moe":
        model_uniform = MoiraiMoEForecast(
            module=base_module,
            prediction_length=PDT,
            context_length=len(context_target_uniform),
            patch_size=PSZ,
            num_samples=NUM_SAMPLES,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
    predictor_uniform = model_uniform.create_predictor(batch_size=BSZ)
    
    forecast_uncertainty_uniform = calculate_forecast_uncertainty(predictor_uniform, forecast_input_data_uniform, PDT)
    
    # 9. Uniform sampling with resampling (drop every Nth then resample back to original length)
    print(f"  9. Uniform sampling with resampling ({CONTEXT_KEEP_RATIO*100:.0f}%) forecasting...")
    
    # Get indices that would be missing in uniform sampling
    all_indices = np.arange(actual_context_length)
    missing_uniform_indices = np.setdiff1d(all_indices, uniform_indices)
    
    # Interpolate missing values to maintain original context length
    context_target_uniform_resampled = interpolate_missing_values(context_target, missing_uniform_indices)
    
    forecast_input_data_uniform_resampled = {
        'target': context_target_uniform_resampled,
        'start': input_data['start'],
        'item_id': input_data.get('item_id', 0)
    }
    
    # Use full context length model since we maintain original length
    forecast_uncertainty_uniform_resampled = calculate_forecast_uncertainty(predictor_full, forecast_input_data_uniform_resampled, PDT)
    
    # 10. Recent context only (take most recent compression_ratio * CTX samples)
    print(f"  10. Recent context only ({CONTEXT_KEEP_RATIO*100:.0f}%) forecasting...")
    
    # Take the most recent samples based on compression ratio
    recent_context_length = max(1, int(CONTEXT_KEEP_RATIO * actual_context_length))
    context_target_recent = context_target[-recent_context_length:]
    
    forecast_input_data_recent = {
        'target': context_target_recent,
        'start': input_data['start'],
        'item_id': input_data.get('item_id', 0)
    }
    
    if MODEL == "moirai":
        model_recent = MoiraiForecast(
            module=base_module,
            prediction_length=PDT,
            context_length=len(context_target_recent),
            patch_size=PSZ,
            num_samples=NUM_SAMPLES,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
    elif MODEL == "moirai-moe":
        model_recent = MoiraiMoEForecast(
            module=base_module,
            prediction_length=PDT,
            context_length=len(context_target_recent),
            patch_size=PSZ,
            num_samples=NUM_SAMPLES,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
    predictor_recent = model_recent.create_predictor(batch_size=BSZ)
    
    forecast_uncertainty_recent = calculate_forecast_uncertainty(predictor_recent, forecast_input_data_recent, PDT)
    
    # Get true values for comparison
    true_values = label_data["target"][:PDT]
    
    # Calculate forecast metrics for all ten methods
    def calculate_metrics(forecast_mean, true_vals):
        forecast_errors = np.abs(forecast_mean - true_vals)
        mae = np.mean(forecast_errors)
        mse = np.mean((forecast_mean - true_vals) ** 2)
        rmse = np.sqrt(mse)
        return mae, rmse, forecast_errors
    
    # Full context
    mae_full, rmse_full, forecast_errors_full = calculate_metrics(forecast_uncertainty_full['mean'], true_values)
    
    # Random context
    mae_random, rmse_random, forecast_errors_random = calculate_metrics(forecast_uncertainty_random['mean'], true_values)
    
    # Most important context
    mae_most_important, rmse_most_important, forecast_errors_most_important = calculate_metrics(forecast_uncertainty_most_important['mean'], true_values)
    
    # Random with interpolation
    mae_random_interp, rmse_random_interp, forecast_errors_random_interp = calculate_metrics(forecast_uncertainty_random_interp['mean'], true_values)
    
    # Most important with interpolation
    mae_important_interp, rmse_important_interp, forecast_errors_important_interp = calculate_metrics(forecast_uncertainty_important_interp['mean'], true_values)
    
    # Least important
    mae_least_important, rmse_least_important, forecast_errors_least_important = calculate_metrics(forecast_uncertainty_least_important['mean'], true_values)
    
    # Least important with interpolation
    mae_least_important_interp, rmse_least_important_interp, forecast_errors_least_important_interp = calculate_metrics(forecast_uncertainty_least_important_interp['mean'], true_values)
    
    # Uniform sampling
    mae_uniform, rmse_uniform, forecast_errors_uniform = calculate_metrics(forecast_uncertainty_uniform['mean'], true_values)
    
    # Uniform resampling
    mae_uniform_resampled, rmse_uniform_resampled, forecast_errors_uniform_resampled = calculate_metrics(forecast_uncertainty_uniform_resampled['mean'], true_values)
    
    # Recent context
    mae_recent, rmse_recent, forecast_errors_recent = calculate_metrics(forecast_uncertainty_recent['mean'], true_values)
    
    print(f"  Results:")
    print(f"    Full Context (len={actual_context_length})        - MAE: {mae_full:.4f}, RMSE: {rmse_full:.4f}, CV_unc: {np.mean(forecast_uncertainty_full['cv_uncertainty']):.4f}")
    print(f"    Random ${CONTEXT_KEEP_RATIO*100:.0f}% (len={reduced_ctx})                   - MAE: {mae_random:.4f}, RMSE: {rmse_random:.4f}, CV_unc: {np.mean(forecast_uncertainty_random['cv_uncertainty']):.4f}")
    print(f"    Most Imp {CONTEXT_KEEP_RATIO*100:.0f}% (len={reduced_ctx})                 - MAE: {mae_most_important:.4f}, RMSE: {rmse_most_important:.4f}, CV_unc: {np.mean(forecast_uncertainty_most_important['cv_uncertainty']):.4f}")
    print(f"    Random ${CONTEXT_KEEP_RATIO*100:.0f}% + Interp (len={actual_context_length}) - MAE: {mae_random_interp:.4f}, RMSE: {rmse_random_interp:.4f}, CV_unc: {np.mean(forecast_uncertainty_random_interp['cv_uncertainty']):.4f}")
    print(f"    Most Imp {CONTEXT_KEEP_RATIO*100:.0f}% + Interp (len={actual_context_length}) - MAE: {mae_important_interp:.4f}, RMSE: {rmse_important_interp:.4f}, CV_unc: {np.mean(forecast_uncertainty_important_interp['cv_uncertainty']):.4f}")
    print(f"    Least Imp {CONTEXT_KEEP_RATIO*100:.0f}% (len={reduced_ctx})                - MAE: {mae_least_important:.4f}, RMSE: {rmse_least_important:.4f}, CV_unc: {np.mean(forecast_uncertainty_least_important['cv_uncertainty']):.4f}")
    print(f"    Least Imp {CONTEXT_KEEP_RATIO*100:.0f}% + Interp (len={actual_context_length}) - MAE: {mae_least_important_interp:.4f}, RMSE: {rmse_least_important_interp:.4f}, CV_unc: {np.mean(forecast_uncertainty_least_important_interp['cv_uncertainty']):.4f}")
    print(f"    Uniform Sampling (len={reduced_ctx})                   - MAE: {mae_uniform:.4f}, RMSE: {rmse_uniform:.4f}, CV_unc: {np.mean(forecast_uncertainty_uniform['cv_uncertainty']):.4f}")
    print(f"    Uniform Resampling (len={actual_context_length}) - MAE: {mae_uniform_resampled:.4f}, RMSE: {rmse_uniform_resampled:.4f}, CV_unc: {np.mean(forecast_uncertainty_uniform_resampled['cv_uncertainty']):.4f}")
    print(f"    Recent Context (len={recent_context_length})                   - MAE: {mae_recent:.4f}, RMSE: {rmse_recent:.4f}, CV_unc: {np.mean(forecast_uncertainty_recent['cv_uncertainty']):.4f}")
    
    return {
        'window_id': window_id,
        'actual_context_length': actual_context_length,
        'reduced_ctx': reduced_ctx,
        # Full context results
        'mae_full': mae_full,
        'rmse_full': rmse_full,
        'forecast_errors_full': forecast_errors_full,
        'forecast_uncertainty_full': forecast_uncertainty_full,
        # Random context results
        'mae_random': mae_random,
        'rmse_random': rmse_random,
        'forecast_errors_random': forecast_errors_random,
        'forecast_uncertainty_random': forecast_uncertainty_random,
        # Most important context results
        'mae_most_important': mae_most_important,
        'rmse_most_important': rmse_most_important,
        'forecast_errors_most_important': forecast_errors_most_important,
        'forecast_uncertainty_most_important': forecast_uncertainty_most_important,
        # Random with interpolation results
        'mae_random_interp': mae_random_interp,
        'rmse_random_interp': rmse_random_interp,
        'forecast_errors_random_interp': forecast_errors_random_interp,
        'forecast_uncertainty_random_interp': forecast_uncertainty_random_interp,
        # Most important with interpolation results
        'mae_important_interp': mae_important_interp,
        'rmse_important_interp': rmse_important_interp,
        'forecast_errors_important_interp': forecast_errors_important_interp,
        'forecast_uncertainty_important_interp': forecast_uncertainty_important_interp,
        # Least important results
        'mae_least_important': mae_least_important,
        'rmse_least_important': rmse_least_important,
        'forecast_errors_least_important': forecast_errors_least_important,
        'forecast_uncertainty_least_important': forecast_uncertainty_least_important,
        # Least important with interpolation results
        'mae_least_important_interp': mae_least_important_interp,
        'rmse_least_important_interp': rmse_least_important_interp,
        'forecast_errors_least_important_interp': forecast_errors_least_important_interp,
        'forecast_uncertainty_least_important_interp': forecast_uncertainty_least_important_interp,
        # Uniform sampling results
        'mae_uniform': mae_uniform,
        'rmse_uniform': rmse_uniform,
        'forecast_errors_uniform': forecast_errors_uniform,
        'forecast_uncertainty_uniform': forecast_uncertainty_uniform,
        # Uniform resampling results
        'mae_uniform_resampled': mae_uniform_resampled,
        'rmse_uniform_resampled': rmse_uniform_resampled,
        'forecast_errors_uniform_resampled': forecast_errors_uniform_resampled,
        'forecast_uncertainty_uniform_resampled': forecast_uncertainty_uniform_resampled,
        # Recent context results
        'mae_recent': mae_recent,
        'rmse_recent': rmse_recent,
        'forecast_errors_recent': forecast_errors_recent,
        'forecast_uncertainty_recent': forecast_uncertainty_recent,
        # Common data
        'true_values': true_values,
        'random_indices': random_indices,
        'most_important_indices': most_important_indices,
        'least_important_indices': least_important_indices,
        'missing_random_indices': missing_random_indices,
        'missing_important_indices': missing_important_indices,
        'missing_least_important_indices': missing_least_important_indices,
        'context_target_random_interp': context_target_random_interp,
        'context_target_important_interp': context_target_important_interp,
        'context_target_least_important_interp': context_target_least_important_interp
    }
    """
    Perform forecasting with uncertainty analysis for three different context selection strategies
    """
    # Get the target data and ensure it's exactly CTX length
    target_data = input_data['target']
    
    # Ensure we use the same sequence as importance analysis
    if len(target_data) > CTX:
        context_target = target_data[-CTX:]
    else:
        context_target = target_data
    
    actual_context_length = len(context_target)
    reduced_ctx = max(1, int(CONTEXT_KEEP_RATIO * actual_context_length))  # configurable % of actual context
    
    print(f"Window {window_id} Forecasting with Uncertainty:")
    print(f"  Full context length: {actual_context_length}")
    print(f"  Reduced context length ({CONTEXT_KEEP_RATIO*100:.0f}%): {reduced_ctx}")
    
    # 1. Full context forecasting with uncertainty
    print(f"  1. Full context forecasting with uncertainty...")
    forecast_input_data_full = {
        'target': context_target,
        'start': input_data['start'],
        'item_id': input_data.get('item_id', 0)
    }
    
    # Create model for full PDT prediction
    if MODEL == "moirai":
        model_full = MoiraiForecast(
            module=base_module,
            prediction_length=PDT,
            context_length=actual_context_length,
            patch_size=PSZ,
            num_samples=NUM_SAMPLES,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
    elif MODEL == "moirai-moe":
        model_full = MoiraiMoEForecast(
            module=base_module,
            prediction_length=PDT,
            context_length=actual_context_length,
            patch_size=16,
            num_samples=NUM_SAMPLES,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
    predictor_full = model_full.create_predictor(batch_size=BSZ)
    
    forecast_uncertainty_full = calculate_forecast_uncertainty(predictor_full, forecast_input_data_full, PDT)
    
    # 2. Random ${CONTEXT_KEEP_RATIO*100:.0f}% context forecasting with uncertainty
    print(f"  2. Random ${CONTEXT_KEEP_RATIO*100:.0f}% context forecasting with uncertainty...")
    np.random.seed(42 + window_id)  # Reproducible random selection
    random_indices = np.sort(np.random.choice(actual_context_length, reduced_ctx, replace=False))
    context_target_random = context_target[random_indices]
    
    forecast_input_data_random = {
        'target': context_target_random,
        'start': input_data['start'],
        'item_id': input_data.get('item_id', 0)
    }
    
    if MODEL == "moirai":
        model_random = MoiraiForecast(
            module=base_module,
            prediction_length=PDT,
            context_length=reduced_ctx,
            patch_size=PSZ,
            num_samples=NUM_SAMPLES,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
    elif MODEL == "moirai-moe":
        model_random = MoiraiMoEForecast(
            module=base_module,
            prediction_length=PDT,
            context_length=reduced_ctx,
            patch_size=16,
            num_samples=NUM_SAMPLES,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
    predictor_random = model_random.create_predictor(batch_size=BSZ)
    
    forecast_uncertainty_random = calculate_forecast_uncertainty(predictor_random, forecast_input_data_random, PDT)
    
    # 3. Most important {CONTEXT_KEEP_RATIO*100:.0f}% context forecasting with uncertainty
    print(f"  3. Most important {CONTEXT_KEEP_RATIO*100:.0f}% context forecasting with uncertainty...")
    
    # Ensure importance scores match the context length
    if len(importance_scores) == actual_context_length:
        context_importance = importance_scores
    elif len(importance_scores) > actual_context_length:
        context_importance = importance_scores[-actual_context_length:]
    else:
        mean_importance = np.mean(importance_scores) if len(importance_scores) > 0 else 1.0
        context_importance = np.concatenate([
            np.full(actual_context_length - len(importance_scores), mean_importance),
            importance_scores
        ])
    
    # Select the most important samples
    most_important_indices = np.argsort(context_importance)[-reduced_ctx:]
    most_important_indices = np.sort(most_important_indices)
    context_target_most_important = context_target[most_important_indices]
    
    forecast_input_data_most_important = {
        'target': context_target_most_important,
        'start': input_data['start'],
        'item_id': input_data.get('item_id', 0)
    }
    
    if MODEL == "moirai":
        model_most_important = MoiraiForecast(
            module=base_module,
            prediction_length=PDT,
            context_length=reduced_ctx,
            patch_size=PSZ,
            num_samples=NUM_SAMPLES,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
    elif MODEL == "moirai-moe":
        model_most_important = MoiraiMoEForecast(
            module=base_module,
            prediction_length=PDT,
            context_length=reduced_ctx,
            patch_size=16,
            num_samples=NUM_SAMPLES,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
    predictor_most_important = model_most_important.create_predictor(batch_size=BSZ)
    
    forecast_uncertainty_most_important = calculate_forecast_uncertainty(predictor_most_important, forecast_input_data_most_important, PDT)
    
    # 4. Random ${CONTEXT_KEEP_RATIO*100:.0f}% with interpolation (maintaining original context length)
    print(f"  4. Random ${CONTEXT_KEEP_RATIO*100:.0f}% with interpolation forecasting...")
    
    # Get indices to remove (25% least important random samples)
    missing_random_indices = np.setdiff1d(np.arange(actual_context_length), random_indices)
    
    # Interpolate missing values
    context_target_random_interp = interpolate_missing_values(context_target, missing_random_indices)
    
    forecast_input_data_random_interp = {
        'target': context_target_random_interp,
        'start': input_data['start'],
        'item_id': input_data.get('item_id', 0)
    }
    
    # Use full context length model since we maintain original length
    forecast_uncertainty_random_interp = calculate_forecast_uncertainty(predictor_full, forecast_input_data_random_interp, PDT)
    
    # 5. Most important {CONTEXT_KEEP_RATIO*100:.0f}% with interpolation (maintaining original context length)
    print(f"  5. Most important {CONTEXT_KEEP_RATIO*100:.0f}% with interpolation forecasting...")
    
    # Get indices to remove (25% least important samples)
    missing_important_indices = np.setdiff1d(np.arange(actual_context_length), most_important_indices)
    
    # Interpolate missing values
    context_target_important_interp = interpolate_missing_values(context_target, missing_important_indices)
    
    forecast_input_data_important_interp = {
        'target': context_target_important_interp,
        'start': input_data['start'],
        'item_id': input_data.get('item_id', 0)
    }
    
    # Use full context length model since we maintain original length
    forecast_uncertainty_important_interp = calculate_forecast_uncertainty(predictor_full, forecast_input_data_important_interp, PDT)
    
    # 6. Least important {CONTEXT_KEEP_RATIO*100:.0f}% (keep least important samples)
    print(f"  6. Least important {CONTEXT_KEEP_RATIO*100:.0f}% context forecasting...")
    
    # Select the least important samples (opposite of most important)
    least_important_indices = np.argsort(context_importance)[:reduced_ctx]
    least_important_indices = np.sort(least_important_indices)
    context_target_least_important = context_target[least_important_indices]
    
    forecast_input_data_least_important = {
        'target': context_target_least_important,
        'start': input_data['start'],
        'item_id': input_data.get('item_id', 0)
    }
    
    if MODEL == "moirai":
        model_least_important = MoiraiForecast(
            module=base_module,
            prediction_length=PDT,
            context_length=reduced_ctx,
            patch_size=PSZ,
            num_samples=NUM_SAMPLES,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
    elif MODEL == "moirai-moe":
        model_least_important = MoiraiMoEForecast(
            module=base_module,
            prediction_length=PDT,
            context_length=reduced_ctx,
            patch_size=16,
            num_samples=NUM_SAMPLES,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
    predictor_least_important = model_least_important.create_predictor(batch_size=BSZ)
    
    forecast_uncertainty_least_important = calculate_forecast_uncertainty(predictor_least_important, forecast_input_data_least_important, PDT)
    
    # 7. Least important {CONTEXT_KEEP_RATIO*100:.0f}% with interpolation (maintaining original context length)
    print(f"  7. Least important {CONTEXT_KEEP_RATIO*100:.0f}% with interpolation forecasting...")
    
    # Get indices to remove (25% most important samples)
    missing_least_important_indices = np.setdiff1d(np.arange(actual_context_length), least_important_indices)
    
    # Interpolate missing values
    context_target_least_important_interp = interpolate_missing_values(context_target, missing_least_important_indices)
    
    forecast_input_data_least_important_interp = {
        'target': context_target_least_important_interp,
        'start': input_data['start'],
        'item_id': input_data.get('item_id', 0)
    }
    
    # Use full context length model since we maintain original length
    forecast_uncertainty_least_important_interp = calculate_forecast_uncertainty(predictor_full, forecast_input_data_least_important_interp, PDT)
    
    # 8. Uniform sampling by compression ratio (drop every Nth sample)
    print(f"  8. Uniform sampling ({CONTEXT_KEEP_RATIO*100:.0f}%) forecasting...")
    
    # Calculate sampling step to achieve target compression
    step_size = max(1, int(1 / CONTEXT_KEEP_RATIO))
    uniform_indices = np.arange(0, actual_context_length, step_size)[:reduced_ctx]
    context_target_uniform = context_target[uniform_indices]
    
    forecast_input_data_uniform = {
        'target': context_target_uniform,
        'start': input_data['start'],
        'item_id': input_data.get('item_id', 0)
    }
    
    if MODEL == "moirai":
        model_uniform = MoiraiForecast(
            module=base_module,
            prediction_length=PDT,
            context_length=len(context_target_uniform),
            patch_size=PSZ,
            num_samples=NUM_SAMPLES,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
    elif MODEL == "moirai-moe":
        model_uniform = MoiraiMoEForecast(
            module=base_module,
            prediction_length=PDT,
            context_length=len(context_target_uniform),
            patch_size=PSZ,
            num_samples=NUM_SAMPLES,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
    predictor_uniform = model_uniform.create_predictor(batch_size=BSZ)
    
    forecast_uncertainty_uniform = calculate_forecast_uncertainty(predictor_uniform, forecast_input_data_uniform, PDT)
    
    # 9. Uniform sampling with resampling (drop every Nth then resample back to original length)
    print(f"  9. Uniform sampling with resampling ({CONTEXT_KEEP_RATIO*100:.0f}%) forecasting...")
    
    # Get indices that would be missing in uniform sampling
    all_indices = np.arange(actual_context_length)
    missing_uniform_indices = np.setdiff1d(all_indices, uniform_indices)
    
    # Interpolate missing values to maintain original context length
    context_target_uniform_resampled = interpolate_missing_values(context_target, missing_uniform_indices)
    
    forecast_input_data_uniform_resampled = {
        'target': context_target_uniform_resampled,
        'start': input_data['start'],
        'item_id': input_data.get('item_id', 0)
    }
    
    # Use full context length model since we maintain original length
    forecast_uncertainty_uniform_resampled = calculate_forecast_uncertainty(predictor_full, forecast_input_data_uniform_resampled, PDT)
    
    # 10. Recent context only (take most recent compression_ratio * CTX samples)
    print(f"  10. Recent context only ({CONTEXT_KEEP_RATIO*100:.0f}%) forecasting...")
    
    # Take the most recent samples based on compression ratio
    recent_context_length = max(1, int(CONTEXT_KEEP_RATIO * actual_context_length))
    context_target_recent = context_target[-recent_context_length:]
    
    forecast_input_data_recent = {
        'target': context_target_recent,
        'start': input_data['start'],
        'item_id': input_data.get('item_id', 0)
    }
    
    if MODEL == "moirai":
        model_recent = MoiraiForecast(
            module=base_module,
            prediction_length=PDT,
            context_length=len(context_target_recent),
            patch_size=PSZ,
            num_samples=NUM_SAMPLES,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
    elif MODEL == "moirai-moe":
        model_recent = MoiraiMoEForecast(
            module=base_module,
            prediction_length=PDT,
            context_length=len(context_target_recent),
            patch_size=PSZ,
            num_samples=NUM_SAMPLES,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
    predictor_recent = model_recent.create_predictor(batch_size=BSZ)
    
    forecast_uncertainty_recent = calculate_forecast_uncertainty(predictor_recent, forecast_input_data_recent, PDT)
    
    # Get true values for comparison
    true_values = label_data["target"][:PDT]
    
    # Calculate forecast metrics for all ten methods
    def calculate_metrics(forecast_mean, true_vals):
        forecast_errors = np.abs(forecast_mean - true_vals)
        mae = np.mean(forecast_errors)
        mse = np.mean((forecast_mean - true_vals) ** 2)
        rmse = np.sqrt(mse)
        return mae, rmse, forecast_errors
    
    # Full context
    mae_full, rmse_full, forecast_errors_full = calculate_metrics(forecast_uncertainty_full['mean'], true_values)
    
    # Random context
    mae_random, rmse_random, forecast_errors_random = calculate_metrics(forecast_uncertainty_random['mean'], true_values)
    
    # Most important context
    mae_most_important, rmse_most_important, forecast_errors_most_important = calculate_metrics(forecast_uncertainty_most_important['mean'], true_values)
    
    # Random with interpolation
    mae_random_interp, rmse_random_interp, forecast_errors_random_interp = calculate_metrics(forecast_uncertainty_random_interp['mean'], true_values)
    
    # Most important with interpolation
    mae_important_interp, rmse_important_interp, forecast_errors_important_interp = calculate_metrics(forecast_uncertainty_important_interp['mean'], true_values)
    
    # Least important
    mae_least_important, rmse_least_important, forecast_errors_least_important = calculate_metrics(forecast_uncertainty_least_important['mean'], true_values)
    
    # Least important with interpolation
    mae_least_important_interp, rmse_least_important_interp, forecast_errors_least_important_interp = calculate_metrics(forecast_uncertainty_least_important_interp['mean'], true_values)
    
    # Uniform sampling
    mae_uniform, rmse_uniform, forecast_errors_uniform = calculate_metrics(forecast_uncertainty_uniform['mean'], true_values)
    
    # Uniform resampling
    mae_uniform_resampled, rmse_uniform_resampled, forecast_errors_uniform_resampled = calculate_metrics(forecast_uncertainty_uniform_resampled['mean'], true_values)
    
    # Recent context
    mae_recent, rmse_recent, forecast_errors_recent = calculate_metrics(forecast_uncertainty_recent['mean'], true_values)
    
    print(f"  Results:")
    print(f"    Full Context (len={actual_context_length})        - MAE: {mae_full:.4f}, RMSE: {rmse_full:.4f}, CV_unc: {np.mean(forecast_uncertainty_full['cv_uncertainty']):.4f}")
    print(f"    Random ${CONTEXT_KEEP_RATIO*100:.0f}% (len={reduced_ctx})                   - MAE: {mae_random:.4f}, RMSE: {rmse_random:.4f}, CV_unc: {np.mean(forecast_uncertainty_random['cv_uncertainty']):.4f}")
    print(f"    Most Imp {CONTEXT_KEEP_RATIO*100:.0f}% (len={reduced_ctx})                 - MAE: {mae_most_important:.4f}, RMSE: {rmse_most_important:.4f}, CV_unc: {np.mean(forecast_uncertainty_most_important['cv_uncertainty']):.4f}")
    print(f"    Random ${CONTEXT_KEEP_RATIO*100:.0f}% + Interp (len={actual_context_length}) - MAE: {mae_random_interp:.4f}, RMSE: {rmse_random_interp:.4f}, CV_unc: {np.mean(forecast_uncertainty_random_interp['cv_uncertainty']):.4f}")
    print(f"    Most Imp {CONTEXT_KEEP_RATIO*100:.0f}% + Interp (len={actual_context_length}) - MAE: {mae_important_interp:.4f}, RMSE: {rmse_important_interp:.4f}, CV_unc: {np.mean(forecast_uncertainty_important_interp['cv_uncertainty']):.4f}")
    print(f"    Least Imp {CONTEXT_KEEP_RATIO*100:.0f}% (len={reduced_ctx})                - MAE: {mae_least_important:.4f}, RMSE: {rmse_least_important:.4f}, CV_unc: {np.mean(forecast_uncertainty_least_important['cv_uncertainty']):.4f}")
    print(f"    Least Imp {CONTEXT_KEEP_RATIO*100:.0f}% + Interp (len={actual_context_length}) - MAE: {mae_least_important_interp:.4f}, RMSE: {rmse_least_important_interp:.4f}, CV_unc: {np.mean(forecast_uncertainty_least_important_interp['cv_uncertainty']):.4f}")
    print(f"    Uniform Sampling (len={reduced_ctx})                   - MAE: {mae_uniform:.4f}, RMSE: {rmse_uniform:.4f}, CV_unc: {np.mean(forecast_uncertainty_uniform['cv_uncertainty']):.4f}")
    print(f"    Uniform Resampling (len={actual_context_length}) - MAE: {mae_uniform_resampled:.4f}, RMSE: {rmse_uniform_resampled:.4f}, CV_unc: {np.mean(forecast_uncertainty_uniform_resampled['cv_uncertainty']):.4f}")
    print(f"    Recent Context (len={recent_context_length})                   - MAE: {mae_recent:.4f}, RMSE: {rmse_recent:.4f}, CV_unc: {np.mean(forecast_uncertainty_recent['cv_uncertainty']):.4f}")
    
    return {
        'window_id': window_id,
        'actual_context_length': actual_context_length,
        'reduced_ctx': reduced_ctx,
        # Full context results
        'mae_full': mae_full,
        'rmse_full': rmse_full,
        'forecast_errors_full': forecast_errors_full,
        'forecast_uncertainty_full': forecast_uncertainty_full,
        # Random context results
        'mae_random': mae_random,
        'rmse_random': rmse_random,
        'forecast_errors_random': forecast_errors_random,
        'forecast_uncertainty_random': forecast_uncertainty_random,
        # Most important context results
        'mae_most_important': mae_most_important,
        'rmse_most_important': rmse_most_important,
        'forecast_errors_most_important': forecast_errors_most_important,
        'forecast_uncertainty_most_important': forecast_uncertainty_most_important,
        # Random with interpolation results
        'mae_random_interp': mae_random_interp,
        'rmse_random_interp': rmse_random_interp,
        'forecast_errors_random_interp': forecast_errors_random_interp,
        'forecast_uncertainty_random_interp': forecast_uncertainty_random_interp,
        # Most important with interpolation results
        'mae_important_interp': mae_important_interp,
        'rmse_important_interp': rmse_important_interp,
        'forecast_errors_important_interp': forecast_errors_important_interp,
        'forecast_uncertainty_important_interp': forecast_uncertainty_important_interp,
        # Least important results
        'mae_least_important': mae_least_important,
        'rmse_least_important': rmse_least_important,
        'forecast_errors_least_important': forecast_errors_least_important,
        'forecast_uncertainty_least_important': forecast_uncertainty_least_important,
        # Least important with interpolation results
        'mae_least_important_interp': mae_least_important_interp,
        'rmse_least_important_interp': rmse_least_important_interp,
        'forecast_errors_least_important_interp': forecast_errors_least_important_interp,
        'forecast_uncertainty_least_important_interp': forecast_uncertainty_least_important_interp,
        # Uniform sampling results
        'mae_uniform': mae_uniform,
        'rmse_uniform': rmse_uniform,
        'forecast_errors_uniform': forecast_errors_uniform,
        'forecast_uncertainty_uniform': forecast_uncertainty_uniform,
        # Uniform resampling results
        'mae_uniform_resampled': mae_uniform_resampled,
        'rmse_uniform_resampled': rmse_uniform_resampled,
        'forecast_errors_uniform_resampled': forecast_errors_uniform_resampled,
        'forecast_uncertainty_uniform_resampled': forecast_uncertainty_uniform_resampled,
        # Recent context results
        'mae_recent': mae_recent,
        'rmse_recent': rmse_recent,
        'forecast_errors_recent': forecast_errors_recent,
        'forecast_uncertainty_recent': forecast_uncertainty_recent,
        # Common data
        'true_values': true_values,
        'random_indices': random_indices,
        'most_important_indices': most_important_indices,
        'least_important_indices': least_important_indices,
        'missing_random_indices': missing_random_indices,
        'missing_important_indices': missing_important_indices,
        'missing_least_important_indices': missing_least_important_indices,
        'context_target_random_interp': context_target_random_interp,
        'context_target_important_interp': context_target_important_interp,
        'context_target_least_important_interp': context_target_least_important_interp
    }

def create_individual_window_plots(all_importance_results, save_path=None):
    """
    Create 10 individual plots showing uncertainty (left y-axis) and time series values (right y-axis) 
    for each window for detailed inspection of uncertainty patterns (V3 Logic)
    """
    # Select 10 windows for detailed analysis
    n_windows = min(10, len(all_importance_results))
    window_indices = np.linspace(0, len(all_importance_results)-1, n_windows, dtype=int)
    
    # Create a 2x5 subplot grid for 10 individual plots
    fig, axes = plt.subplots(2, 5, figsize=(25, 12))
    fig.suptitle(f'Individual Window Analysis: Uncertainty vs Time Series Values (V3)\n'
                f'Model: {MODEL}-{SIZE} | Import_CTX: {INPUT_CTX} | Forecast_CTX: {CTX} | 10 Representative Windows\n'
                f'V3: Always using most recent {INPUT_CTX} samples as context for importance measurement', 
                fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for i, window_idx in enumerate(window_indices):
        result = all_importance_results[window_idx]
        ax = axes[i]
        
        # Create twin axis for time series values
        ax_twin = ax.twinx()
        
        # Plot uncertainty on left axis (primary) - red color scheme
        ar_positions = np.arange(2, len(result['ar_cv_uncertainties']) + 2)
        line1 = ax.plot(ar_positions, result['ar_cv_uncertainties'], 
                       color='red', linewidth=2.5, marker='o', markersize=4,
                       label='AR Uncertainty', alpha=0.8)
        
        # Plot time series values on right axis (secondary) - blue color scheme
        context_positions = np.arange(1, len(result['true_context']) + 1)
        line2 = ax_twin.plot(context_positions, result['true_context'], 
                            color='blue', linewidth=2, marker='s', markersize=3,
                            label='Time Series Values', alpha=0.7, linestyle='--')
        
        # Configure axes
        ax.set_title(f'Window {result["window_id"]}', fontweight='bold', fontsize=12)
        ax.set_xlabel('Context Position', fontsize=10)
        ax.set_ylabel('AR CV Uncertainty', color='red', fontsize=10)
        ax_twin.set_ylabel('Time Series Value', color='blue', fontsize=10)
        
        # Color the y-axis ticks to match the data
        ax.tick_params(axis='y', labelcolor='red', labelsize=9)
        ax_twin.tick_params(axis='y', labelcolor='blue', labelsize=9)
        ax.tick_params(axis='x', labelsize=9)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Create legend combining both lines
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left', fontsize=8)
        
        # Add statistics text box
        mean_uncertainty = np.mean(result['ar_cv_uncertainties'])
        max_uncertainty = np.max(result['ar_cv_uncertainties'])
        uncertainty_trend = np.polyfit(ar_positions, result['ar_cv_uncertainties'], 1)[0]
        
        stats_text = f'μ={mean_uncertainty:.3f}\nmax={max_uncertainty:.3f}\ntrend={uncertainty_trend:.4f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                fontsize=8)
        
        # Highlight potential spikes in uncertainty
        uncertainty_threshold = mean_uncertainty + 2 * np.std(result['ar_cv_uncertainties'])
        spike_positions = ar_positions[result['ar_cv_uncertainties'] > uncertainty_threshold]
        if len(spike_positions) > 0:
            ax.scatter(spike_positions, 
                      result['ar_cv_uncertainties'][result['ar_cv_uncertainties'] > uncertainty_threshold],
                      color='orange', s=60, marker='*', zorder=5, 
                      label='Uncertainty Spikes')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Individual window plots saved to: {save_path}")
    
    plt.show()

def create_uncertainty_comparison_plots(all_importance_results, all_forecast_results, save_path=None):
    """
    Create enhanced uncertainty analysis plots with time series values overlay and 10 forecasting methods (V3)
    """
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    fig.suptitle(f'Moirai Input Context Uncertainty Analysis v3 (Column: {SELECTED_COLUMN})\n'
                f'Model: {MODEL}-{SIZE} | Import_CTX: {INPUT_CTX} | Forecast_CTX: {CTX} | Prediction: {PDT} | Samples: {NUM_SAMPLES}\n'
                f'V3: Using {INPUT_CTX} samples for importance, {CTX} samples for forecasting', 
                fontsize=16, fontweight='bold')
    
    # Collect all data
    all_ar_uncertainties = np.concatenate([result['ar_cv_uncertainties'] for result in all_importance_results])
    all_ar_errors = np.concatenate([result['ar_errors'] for result in all_importance_results])
    all_importance_scores = np.concatenate([result['ar_uncertainties'] for result in all_importance_results])
    
    # 1. Position-wise uncertainty evolution with time series values (5 samples)
    ax1 = axes[0, 0]
    max_length = max(len(result['ar_cv_uncertainties']) for result in all_importance_results)
    
    # Select 5 representative windows
    n_samples = min(5, len(all_importance_results))
    sample_indices = np.linspace(0, len(all_importance_results)-1, n_samples, dtype=int)
    
    # Plot for selected samples
    colors = plt.cm.tab10(np.linspace(0, 1, n_samples))
    ax1_twin = ax1.twinx()
    
    for i, idx in enumerate(sample_indices):
        result = all_importance_results[idx]
        
        # Uncertainty on left axis (primary)
        ar_positions = np.arange(2, len(result['ar_cv_uncertainties']) + 2)
        ax1.plot(ar_positions, result['ar_cv_uncertainties'], 
                alpha=0.8, color=colors[i], linewidth=2, linestyle='-',
                marker='o', markersize=3, label=f'W{result["window_id"]} Uncertainty')
        
        # Time series values on right axis (secondary)
        context_positions = np.arange(1, len(result['true_context']) + 1)
        ax1_twin.plot(context_positions, result['true_context'], 
                     alpha=0.8, color=colors[i], linewidth=2, linestyle='--',
                     marker='s', markersize=2)
    
    ax1.set_title('AR Uncertainty (Left) + Time Series Values (Right)\n(5 Sample Windows)', fontweight='bold')
    ax1.set_xlabel('Context Position')
    ax1.set_ylabel('AR CV Uncertainty', color='red')
    ax1_twin.set_ylabel('Time Series Value', color='blue')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Color the y-axis labels to match the data
    ax1.tick_params(axis='y', labelcolor='red')
    ax1_twin.tick_params(axis='y', labelcolor='blue')
    
    # 2. Ten-method forecasting performance comparison
    ax2 = axes[0, 1]
    
    methods = ['Full', f'Random\n{CONTEXT_KEEP_RATIO*100:.0f}%', f'Most Imp\n{CONTEXT_KEEP_RATIO*100:.0f}%', 'Random+\nInterp', 'MostImp+\nInterp', f'Least Imp\n{CONTEXT_KEEP_RATIO*100:.0f}%', 'LeastImp+\nInterp', 'Uniform\nSampling', 'Uniform\nResampling', 'Recent\nContext']
    mae_means = [
        np.mean([r['mae_full'] for r in all_forecast_results]),
        np.mean([r['mae_random'] for r in all_forecast_results]),
        np.mean([r['mae_most_important'] for r in all_forecast_results]),
        np.mean([r['mae_random_interp'] for r in all_forecast_results]),
        np.mean([r['mae_important_interp'] for r in all_forecast_results]),
        np.mean([r['mae_least_important'] for r in all_forecast_results]),
        np.mean([r['mae_least_important_interp'] for r in all_forecast_results]),
        np.mean([r['mae_uniform'] for r in all_forecast_results]),
        np.mean([r['mae_uniform_resampled'] for r in all_forecast_results]),
        np.mean([r['mae_recent'] for r in all_forecast_results])
    ]
    mae_stds = [
        np.std([r['mae_full'] for r in all_forecast_results]),
        np.std([r['mae_random'] for r in all_forecast_results]),
        np.std([r['mae_most_important'] for r in all_forecast_results]),
        np.std([r['mae_random_interp'] for r in all_forecast_results]),
        np.std([r['mae_important_interp'] for r in all_forecast_results]),
        np.std([r['mae_least_important'] for r in all_forecast_results]),
        np.std([r['mae_least_important_interp'] for r in all_forecast_results]),
        np.std([r['mae_uniform'] for r in all_forecast_results]),
        np.std([r['mae_uniform_resampled'] for r in all_forecast_results]),
        np.std([r['mae_recent'] for r in all_forecast_results])
    ]
    
    colors_bar = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'cyan', 'magenta', 'yellow']
    bars = ax2.bar(methods, mae_means, yerr=mae_stds, capsize=5, 
                   color=colors_bar, alpha=0.7)
    ax2.set_title('Mean MAE by Method (10 Methods)', fontweight='bold')
    ax2.set_ylabel('MAE')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add values on bars
    for bar, mean_val in zip(bars, mae_means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{mean_val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Method comparison over windows (all 10 methods)
    ax3 = axes[0, 2]
    
    window_numbers = range(1, len(all_forecast_results) + 1)
    ax3.plot(window_numbers, [r['mae_full'] for r in all_forecast_results], 
            'o-', label='Full Context', color='blue', linewidth=2, markersize=4)
    ax3.plot(window_numbers, [r['mae_random'] for r in all_forecast_results], 
            's-', label=f'Random {CONTEXT_KEEP_RATIO*100:.0f}%', color='green', linewidth=2, markersize=4)
    ax3.plot(window_numbers, [r['mae_most_important'] for r in all_forecast_results], 
            '^-', label=f'Most Imp {CONTEXT_KEEP_RATIO*100:.0f}%', color='red', linewidth=2, markersize=4)
    ax3.plot(window_numbers, [r['mae_random_interp'] for r in all_forecast_results], 
            'd-', label='Random+Interp', color='orange', linewidth=2, markersize=4)
    ax3.plot(window_numbers, [r['mae_important_interp'] for r in all_forecast_results], 
            'v-', label='MostImp+Interp', color='purple', linewidth=2, markersize=4)
    ax3.plot(window_numbers, [r['mae_least_important'] for r in all_forecast_results], 
            'p-', label=f'Least Imp {CONTEXT_KEEP_RATIO*100:.0f}%', color='brown', linewidth=2, markersize=4)
    ax3.plot(window_numbers, [r['mae_least_important_interp'] for r in all_forecast_results], 
            'h-', label='LeastImp+Interp', color='pink', linewidth=2, markersize=4)
    ax3.plot(window_numbers, [r['mae_uniform'] for r in all_forecast_results], 
            '*-', label=f'Uniform Drop {CONTEXT_KEEP_RATIO*100:.0f}%', color='cyan', linewidth=2, markersize=4)
    ax3.plot(window_numbers, [r['mae_uniform_resampled'] for r in all_forecast_results], 
            'x-', label=f'Uniform+Resample {CONTEXT_KEEP_RATIO*100:.0f}%', color='magenta', linewidth=2, markersize=4)
    ax3.plot(window_numbers, [r['mae_recent'] for r in all_forecast_results], 
            '+-', label=f'Recent {CONTEXT_KEEP_RATIO*100:.0f}%', color='yellow', linewidth=2, markersize=4)
    
    ax3.set_title('MAE by Window (10 Methods)', fontweight='bold')
    ax3.set_xlabel('Window')
    ax3.set_ylabel('MAE')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. Detailed sample analysis with interpolation visualization
    ax4 = axes[1, 0]
    
    # Select window with most interesting pattern
    window_variances = [np.var(result['ar_cv_uncertainties']) for result in all_importance_results]
    selected_window_idx = np.argmax(window_variances)
    selected_result = all_importance_results[selected_window_idx]
    selected_forecast = all_forecast_results[selected_window_idx]
    
    # Plot original time series
    context_positions = np.arange(len(selected_result['true_context']))
    ax4.plot(context_positions, selected_result['true_context'], 'b-', 
            linewidth=3, marker='o', markersize=4, label='Original Values')
    
    # Plot interpolated series
    ax4.plot(context_positions, selected_forecast['context_target_random_interp'], 'g--', 
            linewidth=2, marker='^', markersize=3, label='Random + Interp')
    ax4.plot(context_positions, selected_forecast['context_target_important_interp'], 'r--', 
            linewidth=2, marker='s', markersize=3, label='Most Imp + Interp')
    
    # Highlight missing points
    missing_random = selected_forecast['missing_random_indices']
    missing_important = selected_forecast['missing_important_indices']
    
    if len(missing_random) > 0:
        ax4.scatter(missing_random, selected_result['true_context'][missing_random], 
                   c='green', s=50, marker='x', label='Random Missing')
    if len(missing_important) > 0:
        ax4.scatter(missing_important, selected_result['true_context'][missing_important], 
                   c='red', s=50, marker='+', label='Important Missing')
    
    ax4.set_title(f'Interpolation Visualization\nWindow {selected_result["window_id"]}', fontweight='bold')
    ax4.set_xlabel('Context Position')
    ax4.set_ylabel('Value')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Performance improvement heatmap
    ax5 = axes[1, 1]
    
    # Calculate improvement matrices
    improvements = np.zeros((len(all_forecast_results), 4))  # 4 methods compared to full
    
    for i, result in enumerate(all_forecast_results):
        mae_full = result['mae_full']
        improvements[i, 0] = ((mae_full - result['mae_random']) / mae_full) * 100
        improvements[i, 1] = ((mae_full - result['mae_most_important']) / mae_full) * 100
        improvements[i, 2] = ((mae_full - result['mae_random_interp']) / mae_full) * 100
        improvements[i, 3] = ((mae_full - result['mae_important_interp']) / mae_full) * 100
    
    im = ax5.imshow(improvements.T, aspect='auto', cmap='RdYlBu_r', 
                   vmin=-20, vmax=20, interpolation='nearest')
    
    ax5.set_title('Performance Improvement vs Full Context\n(% MAE Change)', fontweight='bold')
    ax5.set_xlabel('Window Index')
    ax5.set_ylabel('Method')
    ax5.set_yticks(range(4))
    ax5.set_yticklabels(['Random ${CONTEXT_KEEP_RATIO*100:.0f}%', 'Most Imp {CONTEXT_KEEP_RATIO*100:.0f}%', 'Random+Interp', 'MostImp+Interp'])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax5, shrink=0.8)
    cbar.set_label('% Improvement')
    
    # 6-9. Additional plots similar to original but adapted for 5 methods
    
    # 6. Position importance ranking
    ax6 = axes[1, 2]
    
    # Calculate average importance by position
    position_importance = np.zeros(max_length)
    position_counts = np.zeros(max_length)
    
    for result in all_importance_results:
        seq_len = len(result['ar_uncertainties'])
        for pos in range(seq_len):
            position_importance[pos] += result['ar_uncertainties'][pos]
            position_counts[pos] += 1
    
    mean_position_importance = np.divide(position_importance, position_counts, 
                                       out=np.zeros_like(position_importance), where=position_counts!=0)
    
    # Rank positions by importance
    valid_indices = np.where(position_counts > 0)[0]
    importance_ranking = np.argsort(mean_position_importance[valid_indices])[::-1]
    
    # Show top positions
    n_top = min(15, len(importance_ranking))
    top_positions = valid_indices[importance_ranking[:n_top]] + 2  # +2 for position offset
    top_importance = mean_position_importance[valid_indices[importance_ranking[:n_top]]]
    
    bars = ax6.bar(range(n_top), top_importance, color='orange', alpha=0.7)
    ax6.set_title(f'Top {n_top} Most Important Positions', fontweight='bold')
    ax6.set_xlabel('Rank')
    ax6.set_ylabel('Average Importance')
    ax6.set_xticks(range(n_top))
    ax6.set_xticklabels([f'P{pos}' for pos in top_positions], rotation=45)
    ax6.grid(True, alpha=0.3)
    
    # 7. Aggregate uncertainty by position
    ax7 = axes[2, 0]
    
    # Calculate average uncertainty by position
    position_uncertainties = np.zeros(max_length)
    position_std = np.zeros(max_length)
    
    for result in all_importance_results:
        seq_len = len(result['ar_cv_uncertainties'])
        for pos in range(seq_len):
            position_uncertainties[pos] += result['ar_cv_uncertainties'][pos]
    
    # Calculate mean and std for each position
    mean_position_uncertainty = np.divide(position_uncertainties, position_counts, 
                                        out=np.zeros_like(position_uncertainties), where=position_counts!=0)
    
    valid_positions = position_counts > 0
    pos_indices = np.arange(2, max_length + 2)[valid_positions]
    mean_vals = mean_position_uncertainty[valid_positions]
    
    ax7.plot(pos_indices, mean_vals, 'purple', linewidth=3, marker='o', 
            markersize=6, label='Mean AR Uncertainty')
    
    # Add trend line
    if len(pos_indices) > 2:
        z = np.polyfit(pos_indices, mean_vals, 1)
        p = np.poly1d(z)
        ax7.plot(pos_indices, p(pos_indices), "r--", linewidth=2, 
                label=f'Trend (slope: {z[0]:.4f})')
    
    ax7.set_title('Average AR Uncertainty by Position', fontweight='bold')
    ax7.set_xlabel('Context Position')
    ax7.set_ylabel('AR CV Uncertainty')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Uncertainty distribution heatmap
    ax8 = axes[2, 1]
    
    # Create matrix for heatmap
    uncertainty_matrix = np.full((len(all_importance_results), max_length), np.nan)
    for i, result in enumerate(all_importance_results):
        uncertainty_matrix[i, :len(result['ar_cv_uncertainties'])] = result['ar_cv_uncertainties']
    
    im = ax8.imshow(uncertainty_matrix, aspect='auto', cmap='viridis', 
                   interpolation='nearest')
    
    ax8.set_title('AR Uncertainty Heatmap\n(Rows: Windows, Cols: Positions)', fontweight='bold')
    ax8.set_xlabel('Context Position')
    ax8.set_ylabel('Window Index')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax8, shrink=0.8)
    cbar.set_label('AR CV Uncertainty')
    
    # 9. Summary statistics
    ax9 = axes[2, 2]
    ax9.axis('off')
    
    # Calculate comprehensive summary statistics
    summary_text = f"""ENHANCED UNCERTAINTY ANALYSIS SUMMARY

DATASET & MODEL:
• Windows Processed: {len(all_importance_results)}
• Context Length: {CTX}
• Prediction Length: {PDT}
• Model: {MODEL}-{SIZE}

FORECASTING PERFORMANCE (MAE):
• Full Context: {np.mean([r['mae_full'] for r in all_forecast_results]):.4f}
• Random {CONTEXT_KEEP_RATIO*100:.0f}%: {np.mean([r['mae_random'] for r in all_forecast_results]):.4f}
• Most Important {CONTEXT_KEEP_RATIO*100:.0f}%: {np.mean([r['mae_most_important'] for r in all_forecast_results]):.4f}
• Random + Interpolation: {np.mean([r['mae_random_interp'] for r in all_forecast_results]):.4f}
• Most Imp + Interpolation: {np.mean([r['mae_important_interp'] for r in all_forecast_results]):.4f}
• Least Important {CONTEXT_KEEP_RATIO*100:.0f}%: {np.mean([r['mae_least_important'] for r in all_forecast_results]):.4f}
• Least Imp + Interpolation: {np.mean([r['mae_least_important_interp'] for r in all_forecast_results]):.4f}

IMPROVEMENT vs FULL CONTEXT:
• Random {CONTEXT_KEEP_RATIO*100:.0f}%: {((np.mean([r['mae_random'] for r in all_forecast_results]) / np.mean([r['mae_full'] for r in all_forecast_results])) - 1) * 100:+.2f}%
• Most Important {CONTEXT_KEEP_RATIO*100:.0f}%: {((np.mean([r['mae_most_important'] for r in all_forecast_results]) / np.mean([r['mae_full'] for r in all_forecast_results])) - 1) * 100:+.2f}%
• Random + Interp: {((np.mean([r['mae_random_interp'] for r in all_forecast_results]) / np.mean([r['mae_full'] for r in all_forecast_results])) - 1) * 100:+.2f}%
• Most Imp + Interp: {((np.mean([r['mae_important_interp'] for r in all_forecast_results]) / np.mean([r['mae_full'] for r in all_forecast_results])) - 1) * 100:+.2f}%
• Least Important {CONTEXT_KEEP_RATIO*100:.0f}%: {((np.mean([r['mae_least_important'] for r in all_forecast_results]) / np.mean([r['mae_full'] for r in all_forecast_results])) - 1) * 100:+.2f}%
• Least Imp + Interp: {((np.mean([r['mae_least_important_interp'] for r in all_forecast_results]) / np.mean([r['mae_full'] for r in all_forecast_results])) - 1) * 100:+.2f}%

UNCERTAINTY STATISTICS:
• Mean AR CV Uncertainty: {np.mean(all_ar_uncertainties):.4f}
• Mean AR Error: {np.mean(all_ar_errors):.4f}
• Uncertainty-Error Correlation: {np.corrcoef(all_ar_uncertainties, all_ar_errors)[0, 1]:.3f}

BEST PERFORMING METHOD:
• {methods[np.argmin(mae_means)]}
• Performance: {min(mae_means):.4f} MAE
    """
    
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightcyan', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Enhanced uncertainty analysis plots saved to: {save_path}")
    
    plt.show()
    """
    Create simplified uncertainty analysis plots focusing on position-wise analysis
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Moirai Input Context Uncertainty Analysis (Column: {SELECTED_COLUMN})\n'
                f'Model: {MODEL}-{SIZE} | Import_CTX: {INPUT_CTX} | Forecast_CTX: {CTX} | Prediction: {PDT} | Samples: {NUM_SAMPLES}', 
                fontsize=16, fontweight='bold')
    
    # Collect all data
    all_ar_uncertainties = np.concatenate([result['ar_cv_uncertainties'] for result in all_importance_results])
    all_ar_errors = np.concatenate([result['ar_errors'] for result in all_importance_results])
    all_importance_scores = np.concatenate([result['ar_uncertainties'] for result in all_importance_results])
    
    # 1. Position-wise uncertainty evolution (individual windows) with time series values
    ax1 = axes[0, 0]
    max_length = max(len(result['ar_cv_uncertainties']) for result in all_importance_results)
    
    # Select 5 representative windows instead of all windows
    n_samples = min(5, len(all_importance_results))
    sample_indices = np.linspace(0, len(all_importance_results)-1, n_samples, dtype=int)
    
    # Create twin axis for time series values
    ax1_twin = ax1.twinx()
    
    # Plot individual windows with different colors (5 windows only)
    colors = plt.cm.tab10(np.linspace(0, 1, n_samples))
   
    for i, idx in enumerate(sample_indices):
        result = all_importance_results[idx]
        
        # Plot uncertainty on left axis (primary)
        positions = np.arange(2, len(result['ar_cv_uncertainties']) + 2)
        ax1.plot(positions, result['ar_cv_uncertainties'], 
                alpha=0.8, color=colors[i], linewidth=2, linestyle='-',
                label=f'W{result["window_id"]} Uncertainty')
        
        # Plot time series values on right axis (secondary)
        context_positions = np.arange(1, len(result['true_context']) + 1)
        ax1_twin.plot(context_positions, result['true_context'], 
                     alpha=0.6, color=colors[i], linewidth=1.5, linestyle='--',
                     marker='s', markersize=2)
    
    ax1.set_title('AR Uncertainty (Left) + Time Series Values (Right)\n(5 Individual Windows)', fontweight='bold')
    ax1.set_xlabel('Context Position')
    ax1.set_ylabel('AR CV Uncertainty', color='red')
    ax1_twin.set_ylabel('Time Series Value', color='blue')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Color the y-axis labels to match the data
    ax1.tick_params(axis='y', labelcolor='red')
    ax1_twin.tick_params(axis='y', labelcolor='blue')
    
    # 2. Position-wise uncertainty evolution (aggregate)
    ax2 = axes[0, 1]
    
    # Calculate average uncertainty by position
    position_uncertainties = np.zeros(max_length)
    position_counts = np.zeros(max_length)
    position_std = np.zeros(max_length)
    
    for result in all_importance_results:
        seq_len = len(result['ar_cv_uncertainties'])
        for pos in range(seq_len):
            position_uncertainties[pos] += result['ar_cv_uncertainties'][pos]
            position_counts[pos] += 1
    
    # Calculate mean and std for each position
    mean_position_uncertainty = np.divide(position_uncertainties, position_counts, 
                                        out=np.zeros_like(position_uncertainties), where=position_counts!=0)
    
    # Calculate std for each position
    for pos in range(max_length):
        if position_counts[pos] > 1:
            values_at_pos = []
            for result in all_importance_results:
                if pos < len(result['ar_cv_uncertainties']):
                    values_at_pos.append(result['ar_cv_uncertainties'][pos])
            position_std[pos] = np.std(values_at_pos) if len(values_at_pos) > 1 else 0
    
    valid_positions = position_counts > 0
    pos_indices = np.arange(2, max_length + 2)[valid_positions]  # Start from position 2
    mean_vals = mean_position_uncertainty[valid_positions]
    std_vals = position_std[valid_positions]
    
    ax2.plot(pos_indices, mean_vals, 'purple', linewidth=3, marker='o', 
            markersize=6, label='Mean AR Uncertainty')
    ax2.fill_between(pos_indices, mean_vals - std_vals, mean_vals + std_vals, 
                    alpha=0.3, color='purple', label='±1 Std Dev')
    
    # Add trend line
    if len(pos_indices) > 2:
        z = np.polyfit(pos_indices, mean_vals, 1)
        p = np.poly1d(z)
        ax2.plot(pos_indices, p(pos_indices), "r--", linewidth=2, 
                label=f'Trend (slope: {z[0]:.4f})')
    
    ax2.set_title('Average AR Uncertainty by Position\n(With Confidence Band)', fontweight='bold')
    ax2.set_xlabel('Context Position')
    ax2.set_ylabel('AR CV Uncertainty')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Position-wise importance/error evolution (aggregate)
    ax3 = axes[0, 2]
    
    # Calculate average importance (error) by position
    position_importance = np.zeros(max_length)
    position_imp_std = np.zeros(max_length)
    
    for result in all_importance_results:
        seq_len = len(result['ar_uncertainties'])
        for pos in range(seq_len):
            position_importance[pos] += result['ar_uncertainties'][pos]
    
    mean_position_importance = np.divide(position_importance, position_counts, 
                                       out=np.zeros_like(position_importance), where=position_counts!=0)
    
    # Calculate std for importance at each position
    for pos in range(max_length):
        if position_counts[pos] > 1:
            values_at_pos = []
            for result in all_importance_results:
                if pos < len(result['ar_uncertainties']):
                    values_at_pos.append(result['ar_uncertainties'][pos])
            position_imp_std[pos] = np.std(values_at_pos) if len(values_at_pos) > 1 else 0
    
    mean_imp_vals = mean_position_importance[valid_positions]
    std_imp_vals = position_imp_std[valid_positions]
    
    ax3.plot(pos_indices, mean_imp_vals, 'orange', linewidth=3, marker='s', 
            markersize=6, label='Mean Sample Importance (Error)')
    ax3.fill_between(pos_indices, mean_imp_vals - std_imp_vals, mean_imp_vals + std_imp_vals, 
                    alpha=0.3, color='orange', label='±1 Std Dev')
    
    # Add trend line
    if len(pos_indices) > 2:
        z_imp = np.polyfit(pos_indices, mean_imp_vals, 1)
        p_imp = np.poly1d(z_imp)
        ax3.plot(pos_indices, p_imp(pos_indices), "r--", linewidth=2, 
                label=f'Trend (slope: {z_imp[0]:.4f})')
    
    ax3.set_title('Average Sample Importance by Position\n(Prediction Error)', fontweight='bold')
    ax3.set_xlabel('Context Position')
    ax3.set_ylabel('Sample Importance (Absolute Error)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Detailed sample-level analysis for selected window
    ax4 = axes[1, 0]
    
    # Select window with most interesting uncertainty pattern (highest variance)
    window_variances = [np.var(result['ar_cv_uncertainties']) for result in all_importance_results]
    selected_window_idx = np.argmax(window_variances)
    selected_result = all_importance_results[selected_window_idx]
    
    # Full context positions
    context_positions = np.arange(1, len(selected_result['true_context']) + 1)
    # AR predictions positions (start from position 2 since we skip position 0)
    ar_positions = np.arange(2, len(selected_result['ar_predictions']) + 2)  
    
    # Plot true values, predictions, and uncertainty
    ax4_twin = ax4.twinx()
    
    # True values on left axis (full context)
    ax4.plot(context_positions, selected_result['true_context'], 'b-', 
            linewidth=3, marker='o', markersize=4, label='True Values')
    # AR predictions on left axis (offset by 1 position)
    ax4.plot(ar_positions, selected_result['ar_predictions'], 'g--', 
            linewidth=2, marker='^', markersize=4, label='AR Predictions')
    
    # Uncertainty on right axis (same positions as AR predictions)
    ax4_twin.plot(ar_positions, selected_result['ar_cv_uncertainties'], 'purple', 
                 linewidth=2, marker='s', markersize=5, label='AR CV Uncertainty')
    ax4_twin.plot(ar_positions, selected_result['ar_uncertainties'], 'orange', 
                 linewidth=2, marker='d', markersize=5, label='Sample Importance (Error)')
    
    ax4.set_title(f'Sample-Level Analysis\nWindow {selected_result["window_id"]} (Highest Uncertainty Variance)', 
                 fontweight='bold')
    ax4.set_xlabel('Context Position')
    ax4.set_ylabel('Value', color='blue')
    ax4_twin.set_ylabel('Uncertainty / Importance', color='purple')
    
    # Combine legends
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    # 5. Uncertainty distribution by position (heatmap style)
    ax5 = axes[1, 1]
    
    # Create matrix for heatmap
    uncertainty_matrix = np.full((len(all_importance_results), max_length), np.nan)
    for i, result in enumerate(all_importance_results):
        uncertainty_matrix[i, :len(result['ar_cv_uncertainties'])] = result['ar_cv_uncertainties']
    
    im = ax5.imshow(uncertainty_matrix, aspect='auto', cmap='viridis', 
                   interpolation='nearest')
    
    ax5.set_title('AR Uncertainty Heatmap\n(Rows: Windows, Cols: Positions)', fontweight='bold')
    ax5.set_xlabel('Context Position')
    ax5.set_ylabel('Window Index')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax5, shrink=0.8)
    cbar.set_label('AR CV Uncertainty')
    
    # 6. Statistics summary
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Calculate comprehensive summary statistics
    summary_text = f"""POSITION-WISE UNCERTAINTY STATISTICS

Total Windows: {len(all_importance_results)}
Context Length: {max_length} positions
Total AR Steps: {len(all_ar_uncertainties)}

UNCERTAINTY BY POSITION:
Early Positions (1-5):
• Mean Uncertainty: {np.nanmean(mean_position_uncertainty[:5]):.4f}
• Mean Importance: {np.nanmean(mean_position_importance[:5]):.4f}

Middle Positions ({max_length//2-2}-{max_length//2+2}):
• Mean Uncertainty: {np.nanmean(mean_position_uncertainty[max_length//2-2:max_length//2+3]):.4f}
• Mean Importance: {np.nanmean(mean_position_importance[max_length//2-2:max_length//2+3]):.4f}

Late Positions (last 5):
• Mean Uncertainty: {np.nanmean(mean_position_uncertainty[-5:]):.4f}
• Mean Importance: {np.nanmean(mean_position_importance[-5:]):.4f}

OVERALL TRENDS:
• Uncertainty trend: {"Increasing" if z[0] > 0 else "Decreasing"} (slope: {z[0]:.4f})
• Importance trend: {"Increasing" if z_imp[0] > 0 else "Decreasing"} (slope: {z_imp[0]:.4f})

CORRELATIONS:
• Position vs Uncertainty: {np.corrcoef(pos_indices, mean_vals)[0, 1]:.3f}
• Position vs Importance: {np.corrcoef(pos_indices, mean_imp_vals)[0, 1]:.3f}
• Uncertainty vs Importance: {np.corrcoef(all_ar_uncertainties, all_importance_scores)[0, 1]:.3f}

FORECAST PERFORMANCE:
• Mean MAE (Full): {np.mean([r['mae_full'] for r in all_forecast_results]):.4f}
• Mean MAE (Random {CONTEXT_KEEP_RATIO*100:.0f}%): {np.mean([r['mae_random'] for r in all_forecast_results]):.4f}
• Mean MAE (Most Imp {CONTEXT_KEEP_RATIO*100:.0f}%): {np.mean([r['mae_most_important'] for r in all_forecast_results]):.4f}
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightcyan', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Uncertainty analysis plots saved to: {save_path}")
    
    plt.show()

def mean_absolute_scaled_error(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
    """Calculate Mean Absolute Scaled Error (MASE)"""
    naive_mae = np.mean(np.abs(np.diff(y_train)))
    if naive_mae == 0:
        naive_mae = 1e-10
    forecast_mae = np.mean(np.abs(y_true - y_pred))
    mase = forecast_mae / naive_mae
    return mase

def create_position_focused_plots(all_importance_results, save_path=None):
    """
    Create additional plots specifically focused on position-wise uncertainty patterns
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Position-Focused Uncertainty Analysis (Column: {SELECTED_COLUMN})\n'
                f'Model: {MODEL}-{SIZE} | Import_CTX: {INPUT_CTX} | Forecast_CTX: {CTX}', 
                fontsize=16, fontweight='bold')
    
    max_length = max(len(result['ar_cv_uncertainties']) for result in all_importance_results)
    
    # 1. Box plots of uncertainty by position
    ax1 = axes[0, 0]
    
    # Collect data for each position
    position_data = []
    positions_with_data = []
    
    for pos in range(max_length):
        values_at_pos = []
        for result in all_importance_results:
            if pos < len(result['ar_cv_uncertainties']):
                values_at_pos.append(result['ar_cv_uncertainties'][pos])
        
        if len(values_at_pos) > 1:  # Only include positions with multiple samples
            position_data.append(values_at_pos)
            positions_with_data.append(pos + 2)  # +2 since we skip position 0
    
    if position_data:
        bp = ax1.boxplot(position_data, positions=positions_with_data, widths=0.6, 
                        patch_artist=True, showfliers=True, notch=True)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax1.set_title('Distribution of AR Uncertainty by Position\n(Box Plots)', fontweight='bold')
    ax1.set_xlabel('Context Position')
    ax1.set_ylabel('AR CV Uncertainty')
    ax1.grid(True, alpha=0.3)
    
    # 2. Line plot showing individual window trajectories with emphasis on patterns
    ax2 = axes[0, 1]
    
    # Calculate trajectory characteristics
    trajectory_stats = []
    for result in all_importance_results:
        uncertainties = result['ar_cv_uncertainties']
        if len(uncertainties) > 2:
            # Calculate trend
            positions = np.arange(len(uncertainties))
            slope = np.polyfit(positions, uncertainties, 1)[0]
            variance = np.var(uncertainties)
            trajectory_stats.append({'slope': slope, 'variance': variance, 'result': result})
    
    # Sort by interesting patterns
    trajectory_stats.sort(key=lambda x: abs(x['slope']), reverse=True)
    
    # Plot most interesting trajectories
    n_trajectories = min(10, len(trajectory_stats))
    colors = plt.cm.tab10(np.linspace(0, 1, n_trajectories))
    
    for i in range(n_trajectories):
        result = trajectory_stats[i]['result']
        slope = trajectory_stats[i]['slope']
        positions = np.arange(2, len(result['ar_cv_uncertainties']) + 2)  # Start from position 2
        
        line_style = '-' if slope > 0 else '--'
        ax2.plot(positions, result['ar_cv_uncertainties'], 
                color=colors[i], linewidth=2, linestyle=line_style,
                label=f'W{result["window_id"]} (slope: {slope:.3f})')
    
    ax2.set_title('Individual Window Uncertainty Trajectories\n(Top 10 by Trend Magnitude)', fontweight='bold')
    ax2.set_xlabel('Context Position')
    ax2.set_ylabel('AR CV Uncertainty')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Correlation matrix between different positions
    ax3 = axes[1, 0]
    
    # Create correlation matrix
    n_positions = min(20, max_length)  # Limit to first 20 positions for readability
    correlation_matrix = np.full((n_positions, n_positions), np.nan)
    
    for i in range(n_positions):
        for j in range(n_positions):
            values_i = []
            values_j = []
            
            for result in all_importance_results:
                if i < len(result['ar_cv_uncertainties']) and j < len(result['ar_cv_uncertainties']):
                    values_i.append(result['ar_cv_uncertainties'][i])
                    values_j.append(result['ar_cv_uncertainties'][j])
            
            if len(values_i) > 2:
                correlation_matrix[i, j] = np.corrcoef(values_i, values_j)[0, 1]
    
    im = ax3.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax3.set_title('Position-to-Position Uncertainty Correlations\n(First 20 Positions)', fontweight='bold')
    ax3.set_xlabel('Context Position')
    ax3.set_ylabel('Context Position')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
    cbar.set_label('Correlation Coefficient')
    
    # Set ticks
    tick_positions = np.arange(0, n_positions, 5)
    ax3.set_xticks(tick_positions)
    ax3.set_yticks(tick_positions)
    ax3.set_xticklabels(tick_positions + 2)  # +2 since we skip position 0
    ax3.set_yticklabels(tick_positions + 2)  # +2 since we skip position 0
    
    # 4. Position importance ranking
    ax4 = axes[1, 1]
    
    # Calculate average importance and uncertainty for each position
    position_avg_importance = np.zeros(max_length)
    position_avg_uncertainty = np.zeros(max_length)
    position_counts = np.zeros(max_length);
    
    for result in all_importance_results:
        for pos in range(len(result['ar_uncertainties'])):
            position_avg_importance[pos] += result['ar_uncertainties'][pos]
            position_avg_uncertainty[pos] += result['ar_cv_uncertainties'][pos]
            position_counts[pos] += 1
    
    valid_positions = position_counts > 0
    position_avg_importance = np.divide(position_avg_importance, position_counts, 
                                       out=np.zeros_like(position_avg_importance), where=valid_positions)
    position_avg_uncertainty = np.divide(position_avg_uncertainty, position_counts, 
                                        out=np.zeros_like(position_avg_uncertainty), where=valid_positions)
    
    # Rank positions by importance
    valid_indices = np.where(valid_positions)[0]
    importance_ranking = np.argsort(position_avg_importance[valid_indices])[::-1]  # Descending order
    
    # Show top 15 most important positions
    n_top = min(15, len(importance_ranking))
    top_positions = valid_indices[importance_ranking[:n_top]] + 2  # +2 for position offset (skip 0)
    top_importance = position_avg_importance[valid_indices[importance_ranking[:n_top]]]
    top_uncertainty = position_avg_uncertainty[valid_indices[importance_ranking[:n_top]]]
    
    # Create combined bar plot
    x = np.arange(n_top)
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, top_importance, width, label='Sample Importance (Error)', 
                   color='orange', alpha=0.7)
    
    # Scale uncertainty to make it visible on same plot
    uncertainty_scaled = top_uncertainty * (max(top_importance) / max(top_uncertainty))
    bars2 = ax4.bar(x + width/2, uncertainty_scaled, width, label='AR Uncertainty (scaled)', 
                   color='purple', alpha=0.7)
    
    ax4.set_title(f'Top {n_top} Most Important Context Positions\n(Ranked by Average Prediction Error)', 
                 fontweight='bold')
    ax4.set_xlabel('Rank')
    ax4.set_ylabel('Average Values')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'Pos {pos}' for pos in top_positions], rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add values on bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        ax4.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.001, 
                f'{top_importance[i]:.3f}', ha='center', va='bottom', fontsize=8)
        ax4.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.001, 
                f'{top_uncertainty[i]:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Position-focused plots saved to: {save_path}")
    
    plt.show()

def create_forecasting_results_plot(all_forecast_results, save_path=None):
    """
    Create a dedicated plot for forecasting performance comparison of all 10 methods (V3)
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f'Forecasting Performance Comparison (10 Methods) - V3\n'
                f'Model: {MODEL}-{SIZE} | Import_CTX: {INPUT_CTX} | Forecast_CTX: {CTX} | Prediction: {PDT}\n'
                f'V3: Using {INPUT_CTX} for importance, {CTX} for forecasting', 
                fontsize=16, fontweight='bold')
    
    # Method names and colors for all 10 methods
    methods = ['Full', f'Random {CONTEXT_KEEP_RATIO*100:.0f}%', f'Most Imp {CONTEXT_KEEP_RATIO*100:.0f}%', 'Random+Interp', 
               'MostImp+Interp', f'Least Imp {CONTEXT_KEEP_RATIO*100:.0f}%', 'LeastImp+Interp',
               f'Uniform Drop {CONTEXT_KEEP_RATIO*100:.0f}%', f'Uniform+Resample {CONTEXT_KEEP_RATIO*100:.0f}%', f'Recent {CONTEXT_KEEP_RATIO*100:.0f}%']
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
    
    # Extract MAE values for all 10 methods
    mae_data = {
        'Full': [r['mae_full'] for r in all_forecast_results],
        f'Random {CONTEXT_KEEP_RATIO*100:.0f}%': [r['mae_random'] for r in all_forecast_results],
        f'Most Imp {CONTEXT_KEEP_RATIO*100:.0f}%': [r['mae_most_important'] for r in all_forecast_results],
        'Random+Interp': [r['mae_random_interp'] for r in all_forecast_results],
        'MostImp+Interp': [r['mae_important_interp'] for r in all_forecast_results],
        f'Least Imp {CONTEXT_KEEP_RATIO*100:.0f}%': [r['mae_least_important'] for r in all_forecast_results],
        'LeastImp+Interp': [r['mae_least_important_interp'] for r in all_forecast_results],
        f'Uniform Drop {CONTEXT_KEEP_RATIO*100:.0f}%': [r['mae_uniform'] for r in all_forecast_results],
        f'Uniform+Resample {CONTEXT_KEEP_RATIO*100:.0f}%': [r['mae_uniform_resampled'] for r in all_forecast_results],
        f'Recent {CONTEXT_KEEP_RATIO*100:.0f}%': [r['mae_recent'] for r in all_forecast_results]
    }
    
    # 1. Bar plot of mean MAE with error bars
    ax1 = axes[0, 0]
    mae_means = [np.mean(mae_data[method]) for method in methods]
    mae_stds = [np.std(mae_data[method]) for method in methods]
    
    bars = ax1.bar(range(len(methods)), mae_means, yerr=mae_stds, 
                   capsize=5, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for i, (bar, mean_val, std_val) in enumerate(zip(bars, mae_means, mae_stds)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_val + 0.02, 
                f'{mean_val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_title('Mean MAE Comparison (±1 Std Dev)', fontweight='bold')
    ax1.set_ylabel('Mean Absolute Error (MAE)')
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Box plot showing distribution of MAE values
    ax2 = axes[0, 1]
    mae_values = [mae_data[method] for method in methods]
    
    bp = ax2.boxplot(mae_values, labels=methods, patch_artist=True, 
                     showfliers=True, notch=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_title('MAE Distribution (Box Plots)', fontweight='bold')
    ax2.set_ylabel('Mean Absolute Error (MAE)')
    ax2.tick_params(axis='x', rotation=45, labelsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Line plot showing MAE across windows
    ax3 = axes[1, 0]
    window_numbers = range(1, len(all_forecast_results) + 1)
    
    for i, (method, color) in enumerate(zip(methods, colors)):
        ax3.plot(window_numbers, mae_data[method], 
                marker='o', linewidth=2, color=color, label=method, 
                markersize=4, alpha=0.8)
    
    ax3.set_title('MAE Across Windows (All Methods)', fontweight='bold')
    ax3.set_xlabel('Window Number')
    ax3.set_ylabel('Mean Absolute Error (MAE)')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. Performance ranking and improvement analysis
    ax4 = axes[1, 1]
    
    # Calculate improvement relative to full context
    baseline_mae = np.mean(mae_data['Full'])
    improvements = [(baseline_mae - np.mean(mae_data[method])) / baseline_mae * 100 
                   for method in methods[1:]]  # Skip 'Full' context
    
    # Create horizontal bar chart for improvements
    y_pos = np.arange(len(methods[1:]))
    colors_imp = colors[1:]  # Skip blue for full context
    
    bars_imp = ax4.barh(y_pos, improvements, color=colors_imp, alpha=0.7, 
                       edgecolor='black', linewidth=1)
    
    # Add value labels
    for i, (bar, imp_val) in enumerate(zip(bars_imp, improvements)):
        x_pos = bar.get_width() + (0.5 if imp_val >= 0 else -0.5)
        ax4.text(x_pos, bar.get_y() + bar.get_height()/2, 
                f'{imp_val:+.1f}%', ha='left' if imp_val >= 0 else 'right', 
                va='center', fontsize=10, fontweight='bold')
    
    ax4.set_title('Improvement vs Full Context (%)', fontweight='bold')
    ax4.set_xlabel('Improvement (%)')
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(methods[1:])
    ax4.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax4.grid(True, alpha=0.3, axis='x')
    
    # Add text summary
    best_method_idx = np.argmin(mae_means)
    best_method = methods[best_method_idx]
    best_mae = mae_means[best_method_idx]
    
    summary_text = f"""PERFORMANCE SUMMARY:
    
Best Method: {best_method}
Best MAE: {best_mae:.4f}

Baseline (Full): {baseline_mae:.4f}
Worst Method: {methods[np.argmax(mae_means)]}
Worst MAE: {max(mae_means):.4f}

Windows Analyzed: {len(all_forecast_results)}
    """
    
    fig.text(0.02, 0.02, summary_text, fontsize=10, fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Forecasting results plot saved to: {save_path}")
    
    plt.show()

def create_intermediate_plots(importance_results, forecast_results, results_dir, window_count):
    """
    Create intermediate plots after processing a certain number of windows
    """
    print(f"\n{'='*60}")
    print(f"CREATING INTERMEDIATE PLOTS (After {window_count} windows)")
    print(f"{'='*60}")
    
    # Create intermediate plots subfolder
    intermediate_dir = os.path.join(results_dir, 'intermediate-plots')
    os.makedirs(intermediate_dir, exist_ok=True)
    
    # Generate timestamped filenames for intermediate results
    import time
    timestamp = int(time.time())
    
    # Create intermediate uncertainty analysis plots
    intermediate_uncertainty_file = os.path.join(intermediate_dir, f'intermediate_uncertainty_w{window_count}_{timestamp}.png')
    create_uncertainty_comparison_plots(importance_results, forecast_results, intermediate_uncertainty_file)
    print(f"Created intermediate uncertainty analysis: {intermediate_uncertainty_file}")
    
    # Create intermediate individual window plots (showing available windows)
    intermediate_individual_file = os.path.join(intermediate_dir, f'intermediate_individual_w{window_count}_{timestamp}.png')
    create_individual_window_plots(importance_results, intermediate_individual_file)
    print(f"Created intermediate individual window plots: {intermediate_individual_file}")
    
    # Create intermediate forecasting results plot
    intermediate_forecast_file = os.path.join(intermediate_dir, f'intermediate_forecasting_w{window_count}_{timestamp}.png')
    create_forecasting_results_plot(forecast_results, intermediate_forecast_file)
    print(f"Created intermediate forecasting comparison: {intermediate_forecast_file}")
    
    print(f"Intermediate analysis completed for {window_count} windows")
    print(f"Intermediate plots stored in: {intermediate_dir}")

# Create results directory structure
results_base_dir = "results_v3_patch16"  # Changed to v3
dataset_dir = dataset_name
model_dir = f"{MODEL}-{SIZE}"
context_dir = f"ctx{CTX}"
comp_dir = f"comp{CONTEXT_KEEP_RATIO:.2f}"
results_dir = os.path.join(results_base_dir, dataset_dir, model_dir, context_dir, comp_dir)

# Create directories if they don't exist
os.makedirs(results_dir, exist_ok=True)
print(f"Results will be saved to: {results_dir}")

# Get the full dataset for v3 logic
print("\nExtracting full dataset for v3 analysis...")
# full_data_values = df_hufl['HUFL'].values
full_data_values = df_selected[selected_column].values
print(f"Full dataset length: {len(full_data_values)} samples")

# Main analysis loop with enhanced uncertainty (7 methods) - V3 Logic
print("\nAnalyzing sample importance and forecasting with enhanced uncertainty (7 methods) - V3 Logic...")
print("V3 Change: Always using most recent CTX samples as context, even across window boundaries")

all_importance_results = []
all_forecast_results = []
input_it = iter(test_data_full.input)
label_it = iter(test_data_full.label)

# Calculate window start positions for v3 logic
window_step = PDT  # Distance between window starts
total_data_length = len(full_data_values)

for window_idx, (input_data, label_data) in enumerate(zip(input_it, label_it)):
    if window_idx >= actual_windows:
        break
    
    print(f"\n{'='*80}")
    print(f"Processing Window {window_idx + 1}/{actual_windows} (V3 Logic)")
    print(f"{'='*80}")
    
    # Calculate the start position of this window in the full dataset
    # In the test split, windows are offset by -TEST from the end
    # Each window starts TEST - (window_idx + 1) * PDT from the end
    window_start_in_test = window_idx * window_step
    window_start_absolute = total_data_length - TEST + window_start_in_test
    
    print(f"V3 Info: Window starts at absolute position {window_start_absolute} in full dataset")
    print(f"         Window data length: {len(input_data['target'])}")
    
    # 1. Autoregressive input uncertainty analysis (V3)
    print(f"1. Computing autoregressive input uncertainty (V3 logic)...")
    importance_result = calculate_autoregressive_input_uncertainty(
        input_data, window_idx + 1, 
        full_dataset=full_data_values, 
        window_start_idx=window_start_absolute
    )
    all_importance_results.append(importance_result)
    
    # 2. Enhanced forecasting comparison with 10 methods
    print(f"2. Performing forecasting comparison with 10 methods...")
    forecast_result = perform_forecasting_comparison_with_uncertainty(
        input_data, label_data, importance_result['ar_uncertainties'], window_idx + 1
    )
    all_forecast_results.append(forecast_result)
    
    # Print window summary
    print(f"\nWindow {window_idx + 1} Summary:")
    print(f"  Autoregressive Input Analysis:")
    print(f"    Samples analyzed: {importance_result['context_length']}")
    print(f"    Mean AR CV uncertainty: {np.mean(importance_result['ar_cv_uncertainties']):.4f}")
    print(f"    Mean AR error: {np.mean(importance_result['ar_errors']):.4f}")
    print(f"  Enhanced Forecasting Performance (7 Methods):")
    print(f"    Full: {forecast_result['mae_full']:.4f}")
    print(f"    Random {CONTEXT_KEEP_RATIO*100:.0f}%: {forecast_result['mae_random']:.4f}")
    print(f"    Most Imp {CONTEXT_KEEP_RATIO*100:.0f}%: {forecast_result['mae_most_important']:.4f}")
    print(f"    Random + Interp: {forecast_result['mae_random_interp']:.4f}")
    print(f"    Most Imp + Interp: {forecast_result['mae_important_interp']:.4f}")
    print(f"    Least Imp {CONTEXT_KEEP_RATIO*100:.0f}%: {forecast_result['mae_least_important']:.4f}")
    print(f"    Least Imp + Interp: {forecast_result['mae_least_important_interp']:.4f}")
    print(f"    Uniform Sampling: {forecast_result['mae_uniform']:.4f}")
    print(f"    Uniform Resampling: {forecast_result['mae_uniform_resampled']:.4f}")
    print(f"    Recent Context: {forecast_result['mae_recent']:.4f}")
    
    # Check if we should create intermediate plots
    if ANALYSIS_FREQ is not None and (window_idx + 1) % ANALYSIS_FREQ == 0:
        create_intermediate_plots(all_importance_results, all_forecast_results, results_dir, window_idx + 1)

# Create enhanced uncertainty analysis plots
print(f"\n{'='*80}")
print("CREATING FINAL ENHANCED UNCERTAINTY ANALYSIS VISUALIZATIONS")
print(f"{'='*80}")

create_uncertainty_comparison_plots(all_importance_results, all_forecast_results, 
                                      os.path.join(results_dir, 'uncertainty_analysis.png'))

# Create 10 individual window plots for detailed analysis
print(f"\nCreating 10 individual window plots...")
create_individual_window_plots(all_importance_results, 
                               os.path.join(results_dir, 'individual_window_plots.png'))

# Create dedicated forecasting results comparison plot
print(f"\nCreating forecasting results comparison plot...")
create_forecasting_results_plot(all_forecast_results, 
                                os.path.join(results_dir, 'forecasting_results_comparison.png'))

# Enhanced aggregate analysis
print(f"\n{'='*80}")
print("ENHANCED AGGREGATE UNCERTAINTY ANALYSIS (V3)")
print(f"{'='*80}")
print("V3 Logic: Always using most recent CTX samples as context, even across window boundaries")

# Combine all data for 7 methods
all_importance_scores = np.concatenate([result['ar_uncertainties'] for result in all_importance_results])
all_ar_cv_uncertainties = np.concatenate([result['ar_cv_uncertainties'] for result in all_importance_results])
all_ar_errors = np.concatenate([result['ar_errors'] for result in all_importance_results])

# Extract results for all 10 methods
all_mae_full = np.array([result['mae_full'] for result in all_forecast_results])
all_mae_random = np.array([result['mae_random'] for result in all_forecast_results])
all_mae_most_important = np.array([result['mae_most_important'] for result in all_forecast_results])
all_mae_random_interp = np.array([result['mae_random_interp'] for result in all_forecast_results])
all_mae_important_interp = np.array([result['mae_important_interp'] for result in all_forecast_results])
all_mae_least_important = np.array([result['mae_least_important'] for result in all_forecast_results])
all_mae_least_important_interp = np.array([result['mae_least_important_interp'] for result in all_forecast_results])
all_mae_uniform = np.array([result['mae_uniform'] for result in all_forecast_results])
all_mae_uniform_resampled = np.array([result['mae_uniform_resampled'] for result in all_forecast_results])
all_mae_recent = np.array([result['mae_recent'] for result in all_forecast_results])

print(f"Total samples analyzed (AR uncertainty): {len(all_ar_cv_uncertainties)}")
print(f"Total windows processed: {len(all_importance_results)}")

# Enhanced performance statistics
print(f"\nEnhanced Forecasting Statistics (10 Methods):")
print(f"Full Context:")
print(f"  Mean MAE: {np.mean(all_mae_full):.4f} ± {np.std(all_mae_full):.4f}")

print(f"Random {CONTEXT_KEEP_RATIO*100:.0f}% Context:")
print(f"  Mean MAE: {np.mean(all_mae_random):.4f} ± {np.std(all_mae_random):.4f}")
print(f"  Improvement vs Full: {((np.mean(all_mae_full) - np.mean(all_mae_random)) / np.mean(all_mae_full)) * 100:+.2f}%")

print(f"Most Important {CONTEXT_KEEP_RATIO*100:.0f}% Context:")
print(f"  Mean MAE: {np.mean(all_mae_most_important):.4f} ± {np.std(all_mae_most_important):.4f}")
print(f"  Improvement vs Full: {((np.mean(all_mae_full) - np.mean(all_mae_most_important)) / np.mean(all_mae_full)) * 100:+.2f}%")

print(f"Random {CONTEXT_KEEP_RATIO*100:.0f}% + Interpolation:")
print(f"  Mean MAE: {np.mean(all_mae_random_interp):.4f} ± {np.std(all_mae_random_interp):.4f}")
print(f"  Improvement vs Full: {((np.mean(all_mae_full) - np.mean(all_mae_random_interp)) / np.mean(all_mae_full)) * 100:+.2f}%")

print(f"Most Important {CONTEXT_KEEP_RATIO*100:.0f}% + Interpolation:")
print(f"  Mean MAE: {np.mean(all_mae_important_interp):.4f} ± {np.std(all_mae_important_interp):.4f}")
print(f"  Improvement vs Full: {((np.mean(all_mae_full) - np.mean(all_mae_important_interp)) / np.mean(all_mae_full)) * 100:+.2f}%")

print(f"Least Important {CONTEXT_KEEP_RATIO*100:.0f}% Context:")
print(f"  Mean MAE: {np.mean(all_mae_least_important):.4f} ± {np.std(all_mae_least_important):.4f}")
print(f"  Improvement vs Full: {((np.mean(all_mae_full) - np.mean(all_mae_least_important)) / np.mean(all_mae_full)) * 100:+.2f}%")

print(f"Least Important {CONTEXT_KEEP_RATIO*100:.0f}% + Interpolation:")
print(f"  Mean MAE: {np.mean(all_mae_least_important_interp):.4f} ± {np.std(all_mae_least_important_interp):.4f}")
print(f"  Improvement vs Full: {((np.mean(all_mae_full) - np.mean(all_mae_least_important_interp)) / np.mean(all_mae_full)) * 100:+.2f}%")

print(f"Uniform Sampling {CONTEXT_KEEP_RATIO*100:.0f}%:")
print(f"  Mean MAE: {np.mean(all_mae_uniform):.4f} ± {np.std(all_mae_uniform):.4f}")
print(f"  Improvement vs Full: {((np.mean(all_mae_full) - np.mean(all_mae_uniform)) / np.mean(all_mae_full)) * 100:+.2f}%")

print(f"Uniform Resampling {CONTEXT_KEEP_RATIO*100:.0f}%:")
print(f"  Mean MAE: {np.mean(all_mae_uniform_resampled):.4f} ± {np.std(all_mae_uniform_resampled):.4f}")
print(f"  Improvement vs Full: {((np.mean(all_mae_full) - np.mean(all_mae_uniform_resampled)) / np.mean(all_mae_full)) * 100:+.2f}%")

print(f"Recent Context {CONTEXT_KEEP_RATIO*100:.0f}%:")
print(f"  Mean MAE: {np.mean(all_mae_recent):.4f} ± {np.std(all_mae_recent):.4f}")
print(f"  Improvement vs Full: {((np.mean(all_mae_full) - np.mean(all_mae_recent)) / np.mean(all_mae_full)) * 100:+.2f}%")

# Method ranking
method_names = ['Full', f'Random {CONTEXT_KEEP_RATIO*100:.0f}%', f'Most Imp {CONTEXT_KEEP_RATIO*100:.0f}%', 'Random+Interp', 'MostImp+Interp', f'Least Imp {CONTEXT_KEEP_RATIO*100:.0f}%', 'LeastImp+Interp', 'Uniform Sampling', 'Uniform Resampling', 'Recent Context']
method_maes = [np.mean(all_mae_full), np.mean(all_mae_random), np.mean(all_mae_most_important), 
               np.mean(all_mae_random_interp), np.mean(all_mae_important_interp),
               np.mean(all_mae_least_important), np.mean(all_mae_least_important_interp),
               np.mean(all_mae_uniform), np.mean(all_mae_uniform_resampled), np.mean(all_mae_recent)]

ranking_indices = np.argsort(method_maes)
print(f"\nMethod Ranking (Best to Worst):")
for i, idx in enumerate(ranking_indices):
    print(f"  {i+1}. {method_names[idx]}: {method_maes[idx]:.4f} MAE")

# Save enhanced results
print(f"\nSaving enhanced results...")
results_summary = {
    # Original data
    'all_importance_scores': all_importance_scores,
    'all_ar_cv_uncertainties': all_ar_cv_uncertainties,
    'all_ar_errors': all_ar_errors,
    
    # All 10 methods MAE
    'all_mae_full': all_mae_full,
    'all_mae_random': all_mae_random,
    'all_mae_most_important': all_mae_most_important,
    'all_mae_random_interp': all_mae_random_interp,
    'all_mae_important_interp': all_mae_important_interp,
    'all_mae_least_important': all_mae_least_important,
    'all_mae_least_important_interp': all_mae_least_important_interp,
    'all_mae_uniform': all_mae_uniform,
    'all_mae_uniform_resampled': all_mae_uniform_resampled,
    'all_mae_recent': all_mae_recent,
    'all_mae_least_important_interp': all_mae_least_important_interp,
    
    # Method ranking
    'method_names': method_names,
    'method_maes': method_maes,
    'method_ranking': ranking_indices,
    
    # Metadata
    'per_window_importance_results': all_importance_results,
    'per_window_forecast_results': all_forecast_results,
    'total_windows': len(all_importance_results),
    'total_samples': len(all_importance_scores),
    
    # Configuration
    'model': MODEL,
    'size': SIZE,
    'ctx': CTX,
    'pdt': PDT,
    'num_samples': NUM_SAMPLES,
}

np.savez(os.path.join(results_dir, 'uncertainty_analysis_results.npz'), **results_summary)
print(f"Enhanced results saved to '{os.path.join(results_dir, 'uncertainty_analysis_results.npz')}'")

print(f"\n{'='*80}")
print("ENHANCED ANALYSIS COMPLETE! (V3)")
print(f"{'='*80}")
print(f"Enhanced V3 results saved to: {results_dir}")
print(f"  • uncertainty_analysis.png - Enhanced uncertainty plots (V3)")
print(f"  • individual_window_plots.png - 10 individual window analyses (V3)")
print(f"  • forecasting_results_comparison.png - 7-method performance comparison (V3)")
print(f"  • uncertainty_analysis_results.npz - Enhanced numerical results (V3)")
print(f"\nEnhanced Processing Summary (V3):")
print(f"  • V3 Logic: Always using most recent {INPUT_CTX} samples as context for importance measurement")
print(f"  • Forecasting using context length: {CTX}")
print(f"  • Analysis Frequency: {ANALYSIS_FREQ if ANALYSIS_FREQ else 'Final only'}")
print(f"  • Processed {len(all_importance_results)} windows with 10 forecasting methods")
print(f"  • Analyzed {len(all_ar_cv_uncertainties)} AR steps for uncertainty")
print(f"  • Created detailed plots: uncertainty_analysis.png + individual_window_plots.png + forecasting_results_comparison.png")
if ANALYSIS_FREQ is not None:
    intermediate_count = len(all_importance_results) // ANALYSIS_FREQ
    print(f"  • Generated {intermediate_count} sets of intermediate plots during processing")
print(f"  • Best performing method: {method_names[ranking_indices[0]]} (MAE: {method_maes[ranking_indices[0]]:.4f})")
print(f"  • Mean AR CV uncertainty: {np.mean(all_ar_cv_uncertainties):.4f}")