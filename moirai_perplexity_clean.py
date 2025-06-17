"""
Moirai Uncertainty Analysis - Main Script
Core functions and logic for uncertainty analysis
"""

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

# Import all plotting functions
from plot_utils import (
    create_individual_window_plots,
    create_uncertainty_comparison_plots,
    create_forecasting_results_plot,
    create_intermediate_plots,
    create_position_focused_plots
)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Moirai Uncertainty Analysis - Enhanced V3')
    
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
    
    return parser.parse_args()

# Parse command line arguments
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
PSZ = args.psz
BSZ = args.bsz
TEST = args.test if args.test is not None else int(100 * PDT)
NUM_SAMPLES = args.num_samples
COLUMN_NUM = args.column
ANALYSIS_FREQ = args.analysis_freq

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

# Create results directory structure with dataset name
results_dir = f"results_v3_v2/{dataset_name}/{MODEL}-{SIZE}/ctx{CTX}"
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
    
    # Create forecast using the predictor
    forecast_list = list(predictor.predict(input_data))
    forecast = forecast_list[0]  # Get the first (and only) forecast
    
    # Extract samples from the forecast
    forecast_samples = forecast.samples  # Shape: (num_samples, prediction_length)
    
    # Compute statistics
    forecast_mean = np.mean(forecast_samples, axis=0)
    forecast_std = np.std(forecast_samples, axis=0)
    forecast_cv = forecast_std / (np.abs(forecast_mean) + 1e-8)  # Add small epsilon to avoid division by zero
    
    # Get quantiles
    p05 = np.percentile(forecast_samples, 5, axis=0)
    p25 = np.percentile(forecast_samples, 25, axis=0)
    p50 = np.percentile(forecast_samples, 50, axis=0)
    p75 = np.percentile(forecast_samples, 75, axis=0)
    p95 = np.percentile(forecast_samples, 95, axis=0)
    
    return {
        'samples': forecast_samples,
        'mean': forecast_mean,
        'std': forecast_std,
        'cv': forecast_cv,
        'p05': p05,
        'p25': p25,
        'p50': p50,
        'p75': p75,
        'p95': p95
    }

def calculate_perplexity_importance(perplexity_changes):
    """Convert perplexity changes to importance scores (higher = more important)"""
    # If perplexity increases when input is removed, that input was important
    importance_scores = perplexity_changes
    return importance_scores

def autoregressive_input_uncertainty(values, input_context_length=INPUT_CTX):
    """
    Autoregressive uncertainty analysis using the most recent input_context_length samples
    """
    total_length = len(values)
    if total_length < input_context_length:
        print(f"Warning: Time series length ({total_length}) is less than input context length ({input_context_length})")
        input_context_length = total_length
    
    # Always use the most recent samples for context
    start_idx = total_length - input_context_length
    context_values = values[start_idx:]
    
    print(f"AR uncertainty: Using most recent {input_context_length} samples as context (indices {start_idx} to {total_length-1})")
    
    # Create predictor once for this context
    predictor = create_model_with_context_length(input_context_length)
    
    uncertainties = []
    
    for i in tqdm(range(input_context_length), desc="AR uncertainty analysis"):
        # Use all context values up to position i as input
        input_values = context_values[:i+1]
        
        if len(input_values) == 0:
            continue
            
        # Create input data
        input_data = [{"target": input_values, "start": pd.Timestamp("2000-01-01")}]
        
        # Calculate uncertainty for the next step
        uncertainty_results = calculate_forecast_uncertainty(predictor, input_data, 1)
        uncertainties.append(uncertainty_results['cv'][0])  # CV for the single prediction step
    
    return np.array(uncertainties)

def calculate_perplexity_score(model, input_data, target_values):
    """Calculate perplexity score for given input data and target values"""
    # For Moirai, we use the negative log-likelihood as a proxy for perplexity
    predictor = model
    
    # Create forecast
    forecast_list = list(predictor.predict(input_data))
    forecast = forecast_list[0]
    
    # Get the samples
    samples = forecast.samples  # Shape: (num_samples, prediction_length)
    
    # Calculate log-likelihood
    # We'll use the negative log of the probability density
    log_likelihood = 0
    for i, target_val in enumerate(target_values):
        # Estimate probability density at target value using samples
        sample_values = samples[:, i] if i < samples.shape[1] else samples[:, -1]
        
        # Use kernel density estimation to estimate probability
        from scipy import stats
        kde = stats.gaussian_kde(sample_values)
        prob_density = kde(target_val)[0]
        
        # Add small epsilon to avoid log(0)
        prob_density = max(prob_density, 1e-10)
        log_likelihood += np.log(prob_density)
    
    # Return negative log-likelihood (higher = worse fit = higher perplexity)
    return -log_likelihood

def importance_measurement_v3(window_input, window_target):
    """
    Enhanced importance measurement using most recent INPUT_CTX samples as context
    """
    values = window_input['target']
    total_length = len(values)
    
    if total_length < INPUT_CTX:
        context_length = total_length
        print(f"Warning: Window length ({total_length}) < INPUT_CTX ({INPUT_CTX}), using {context_length}")
    else:
        context_length = INPUT_CTX
    
    # Always use the most recent samples as context
    start_idx = total_length - context_length
    context_values = values[start_idx:]
    
    print(f"Importance: Using most recent {context_length} samples as context (indices {start_idx} to {total_length-1})")
    
    # Create predictor
    predictor = create_model_with_context_length(context_length)
    
    # Calculate baseline perplexity with full context
    input_data_full = [{"target": context_values, "start": window_input["start"]}]
    baseline_perplexity = calculate_perplexity_score(predictor, input_data_full, window_target[:1])
    
    importance_scores = []
    
    for pos in tqdm(range(context_length), desc="Calculating importance"):
        # Create modified context with this position removed (set to interpolated value)
        modified_values = context_values.copy()
        
        # Interpolate the missing value
        missing_indices = [pos]
        modified_values = interpolate_missing_values(modified_values, missing_indices)
        
        # Calculate perplexity with modified input
        input_data_modified = [{"target": modified_values, "start": window_input["start"]}]
        modified_perplexity = calculate_perplexity_score(predictor, input_data_modified, window_target[:1])
        
        # Importance = increase in perplexity when this input is modified
        importance = modified_perplexity - baseline_perplexity
        importance_scores.append(importance)
    
    return np.array(importance_scores)

def enhanced_forecasting_strategies_v3(window_input, window_target, importance_scores):
    """
    Enhanced forecasting with 7 different strategies using flexible context lengths
    """
    values = window_input['target']
    total_length = len(values)
    
    if total_length < CTX:
        context_length = total_length
        print(f"Warning: Window length ({total_length}) < CTX ({CTX}), using {context_length}")
    else:
        context_length = CTX
    
    # Always use the most recent samples as forecasting context
    start_idx = total_length - context_length
    forecasting_context = values[start_idx:]
    
    print(f"Forecasting: Using most recent {context_length} samples as context (indices {start_idx} to {total_length-1})")
    
    results = {}
    
    # 1. Full context (baseline)
    predictor_full = create_model_with_context_length(context_length)
    input_data_full = [{"target": forecasting_context, "start": window_input["start"]}]
    forecast_full = list(predictor_full.predict(input_data_full))[0]
    mae_full = np.mean(np.abs(forecast_full.mean - window_target))
    results['mae_full'] = mae_full
    
    # 2. Random subset (same size as most important)
    if len(importance_scores) > 0:
        # Map importance scores to forecasting context
        if len(importance_scores) != len(forecasting_context):
            # If importance was measured on different context length, we need to map it
            importance_context_length = len(importance_scores)
            if importance_context_length <= len(forecasting_context):
                # Pad importance scores with zeros for older values
                mapped_importance = np.zeros(len(forecasting_context))
                mapped_importance[-importance_context_length:] = importance_scores
            else:
                # Truncate importance scores to match forecasting context
                mapped_importance = importance_scores[-len(forecasting_context):]
        else:
            mapped_importance = importance_scores
        
        num_important = max(1, len(mapped_importance) // 4)  # Use 1/4 of context
        
        random_indices = np.random.choice(len(forecasting_context), size=num_important, replace=False)
        random_subset = forecasting_context[random_indices]
        
        # Create modified predictor for random subset
        predictor_random = create_model_with_context_length(len(random_subset))
        input_data_random = [{"target": random_subset, "start": window_input["start"]}]
        forecast_random = list(predictor_random.predict(input_data_random))[0]
        mae_random = np.mean(np.abs(forecast_random.mean - window_target))
        results['mae_random'] = mae_random
        
        # 3. Most important subset
        most_important_indices = np.argsort(mapped_importance)[-num_important:]
        most_important_subset = forecasting_context[most_important_indices]
        
        predictor_important = create_model_with_context_length(len(most_important_subset))
        input_data_important = [{"target": most_important_subset, "start": window_input["start"]}]
        forecast_important = list(predictor_important.predict(input_data_important))[0]
        mae_most_important = np.mean(np.abs(forecast_important.mean - window_target))
        results['mae_most_important'] = mae_most_important
        
        # 4. Random with interpolation
        random_values_interp = forecasting_context.copy()
        non_random_indices = np.setdiff1d(np.arange(len(forecasting_context)), random_indices)
        random_values_interp = interpolate_missing_values(random_values_interp, non_random_indices)
        
        input_data_random_interp = [{"target": random_values_interp, "start": window_input["start"]}]
        forecast_random_interp = list(predictor_full.predict(input_data_random_interp))[0]
        mae_random_interp = np.mean(np.abs(forecast_random_interp.mean - window_target))
        results['mae_random_interp'] = mae_random_interp
        
        # 5. Important with interpolation
        important_values_interp = forecasting_context.copy()
        non_important_indices = np.setdiff1d(np.arange(len(forecasting_context)), most_important_indices)
        important_values_interp = interpolate_missing_values(important_values_interp, non_important_indices)
        
        input_data_important_interp = [{"target": important_values_interp, "start": window_input["start"]}]
        forecast_important_interp = list(predictor_full.predict(input_data_important_interp))[0]
        mae_important_interp = np.mean(np.abs(forecast_important_interp.mean - window_target))
        results['mae_important_interp'] = mae_important_interp
        
        # 6. Least important subset
        least_important_indices = np.argsort(mapped_importance)[:num_important]
        least_important_subset = forecasting_context[least_important_indices]
        
        predictor_least = create_model_with_context_length(len(least_important_subset))
        input_data_least = [{"target": least_important_subset, "start": window_input["start"]}]
        forecast_least = list(predictor_least.predict(input_data_least))[0]
        mae_least_important = np.mean(np.abs(forecast_least.mean - window_target))
        results['mae_least_important'] = mae_least_important
        
        # 7. Least important with interpolation
        least_important_values_interp = forecasting_context.copy()
        non_least_important_indices = np.setdiff1d(np.arange(len(forecasting_context)), least_important_indices)
        least_important_values_interp = interpolate_missing_values(least_important_values_interp, non_least_important_indices)
        
        input_data_least_interp = [{"target": least_important_values_interp, "start": window_input["start"]}]
        forecast_least_interp = list(predictor_full.predict(input_data_least_interp))[0]
        mae_least_important_interp = np.mean(np.abs(forecast_least_interp.mean - window_target))
        results['mae_least_important_interp'] = mae_least_important_interp
    else:
        # If no importance scores, set other methods to same as full
        results.update({
            'mae_random': mae_full,
            'mae_most_important': mae_full,
            'mae_random_interp': mae_full,
            'mae_important_interp': mae_full,
            'mae_least_important': mae_full,
            'mae_least_important_interp': mae_full
        })
    
    return results

# Main analysis loop
print(f"\\n{'='*80}")
print("STARTING ENHANCED UNCERTAINTY ANALYSIS (V3)")
print(f"{'='*80}")

all_importance_scores = []
all_ar_cv_uncertainties = []
all_ar_errors = []
all_mae_full = []
all_mae_random = []
all_mae_most_important = []
all_mae_random_interp = []
all_mae_important_interp = []
all_mae_least_important = []
all_mae_least_important_interp = []

# For storing detailed results
all_importance_results = []
all_forecast_results = []

window_count = 0
for window_input, window_target in tqdm(test_data_full, desc="Processing windows"):
    window_count += 1
    
    print(f"\\n--- Processing Window {window_count} ---")
    print(f"Input length: {len(window_input['target'])}, Target: {window_target[:3]}...")
    
    try:
        # 1. Calculate importance scores using INPUT_CTX context
        importance_scores = importance_measurement_v3(window_input, window_target)
        all_importance_scores.extend(importance_scores)
        
        # 2. Calculate AR uncertainty 
        ar_uncertainties = autoregressive_input_uncertainty(window_input['target'])
        all_ar_cv_uncertainties.extend(ar_uncertainties)
        
        # 3. Enhanced forecasting with 7 strategies using CTX context
        forecast_results = enhanced_forecasting_strategies_v3(window_input, window_target, importance_scores)
        
        # Store individual results
        all_mae_full.append(forecast_results['mae_full'])
        all_mae_random.append(forecast_results['mae_random'])
        all_mae_most_important.append(forecast_results['mae_most_important'])
        all_mae_random_interp.append(forecast_results['mae_random_interp'])
        all_mae_important_interp.append(forecast_results['mae_important_interp'])
        all_mae_least_important.append(forecast_results['mae_least_important'])
        all_mae_least_important_interp.append(forecast_results['mae_least_important_interp'])
        
        # Store detailed per-window results
        importance_result = {
            'window': window_count,
            'importance_scores': importance_scores,
            'ar_uncertainties': ar_uncertainties,
            'input_length': len(window_input['target']),
            'target_preview': window_target[:3],
            'importance_stats': {
                'mean': np.mean(importance_scores) if len(importance_scores) > 0 else 0,
                'std': np.std(importance_scores) if len(importance_scores) > 0 else 0,
                'max': np.max(importance_scores) if len(importance_scores) > 0 else 0,
                'min': np.min(importance_scores) if len(importance_scores) > 0 else 0
            }
        }
        all_importance_results.append(importance_result)
        
        forecast_result = {
            'window': window_count,
            'mae_full': forecast_results['mae_full'],
            'mae_random': forecast_results['mae_random'],
            'mae_most_important': forecast_results['mae_most_important'],
            'mae_random_interp': forecast_results['mae_random_interp'],
            'mae_important_interp': forecast_results['mae_important_interp'],
            'mae_least_important': forecast_results['mae_least_important'],
            'mae_least_important_interp': forecast_results['mae_least_important_interp']
        }
        all_forecast_results.append(forecast_result)
        
        print(f"Window {window_count} Results:")
        print(f"  Importance scores: mean={np.mean(importance_scores):.4f}, std={np.std(importance_scores):.4f}")
        print(f"  AR uncertainties: mean={np.mean(ar_uncertainties):.4f}")
        print(f"  MAE Full: {forecast_results['mae_full']:.4f}")
        print(f"  MAE Most Important: {forecast_results['mae_most_important']:.4f}")
        print(f"  MAE Random: {forecast_results['mae_random']:.4f}")
        
        # Generate intermediate plots if requested
        if ANALYSIS_FREQ is not None and window_count % ANALYSIS_FREQ == 0:
            print(f"\\nGenerating intermediate plots after {window_count} windows...")
            
            # Create intermediate plots directory
            intermediate_dir = os.path.join(results_dir, "intermediate-plots")
            os.makedirs(intermediate_dir, exist_ok=True)
            
            # Create model config for plotting
            model_config = {
                'MODEL': MODEL,
                'SIZE': SIZE, 
                'CTX': CTX,
                'INPUT_CTX': INPUT_CTX,
                'PDT': PDT,
                'NUM_SAMPLES': args.num_samples,
                'SELECTED_COLUMN': SELECTED_COLUMN,
                'dataset_name': dataset_name
            }
            
            create_intermediate_plots(
                all_importance_scores[:len(all_importance_scores)],
                all_ar_cv_uncertainties[:len(all_ar_cv_uncertainties)],
                intermediate_dir,
                window_count,
                model_config
            )
            
            print(f"Intermediate plots saved to {intermediate_dir}/")
        
    except Exception as e:
        print(f"Error processing window {window_count}: {e}")
        continue

print(f"\\n{'='*80}")
print("ENHANCED ANALYSIS RESULTS (V3)")
print(f"{'='*80}")

if len(all_importance_scores) == 0:
    print("No data processed successfully!")
else:
    # Print enhanced summary statistics
    print(f"\\nEnhanced V3 Processing Summary:")
    print(f"  • V3 Logic: Always using most recent {INPUT_CTX} samples as context for importance measurement")
    print(f"  • Forecasting using context length: {CTX}")
    print(f"  • Processed {len(all_importance_results)} windows successfully")
    print(f"  • Collected {len(all_importance_scores)} importance scores")
    print(f"  • Analyzed {len(all_ar_cv_uncertainties)} AR uncertainty steps")
    
    print(f"\\nImportance Scores Statistics:")
    print(f"  Mean: {np.mean(all_importance_scores):.4f}")
    print(f"  Std: {np.std(all_importance_scores):.4f}")
    print(f"  Min: {np.min(all_importance_scores):.4f}")
    print(f"  Max: {np.max(all_importance_scores):.4f}")
    
    print(f"\\nAR CV Uncertainty Statistics:")
    print(f"  Mean: {np.mean(all_ar_cv_uncertainties):.4f}")
    print(f"  Std: {np.std(all_ar_cv_uncertainties):.4f}")
    print(f"  Min: {np.min(all_ar_cv_uncertainties):.4f}")
    print(f"  Max: {np.max(all_ar_cv_uncertainties):.4f}")
    
    # Enhanced forecasting results comparison
    method_names = [
        'Full Context',
        'Random Subset',
        'Most Important Subset',
        'Random + Interpolation',
        'Important + Interpolation',
        'Least Important Subset',
        'Least Important + Interpolation'
    ]
    
    method_maes = [
        np.mean(all_mae_full),
        np.mean(all_mae_random),
        np.mean(all_mae_most_important),
        np.mean(all_mae_random_interp),
        np.mean(all_mae_important_interp),
        np.mean(all_mae_least_important),
        np.mean(all_mae_least_important_interp)
    ]
    
    print(f"\\nEnhanced 7-Method Forecasting Results:")
    for i, (name, mae) in enumerate(zip(method_names, method_maes)):
        print(f"  {i+1}. {name}: MAE = {mae:.4f}")
    
    # Calculate performance ranking
    ranking_indices = np.argsort(method_maes)
    print(f"\\nMethod Ranking (Best to Worst):")
    for i, idx in enumerate(ranking_indices):
        print(f"  {i+1}. {method_names[idx]}: MAE = {method_maes[idx]:.4f}")

    # Create final enhanced visualizations
    print(f"\\nGenerating enhanced visualizations...")
    
    # Create model config for final plotting
    model_config = {
        'MODEL': MODEL,
        'SIZE': SIZE, 
        'CTX': CTX,
        'INPUT_CTX': INPUT_CTX,
        'PDT': PDT,
        'NUM_SAMPLES': args.num_samples,
        'SELECTED_COLUMN': SELECTED_COLUMN,
        'dataset_name': dataset_name
    }
    
    # Main uncertainty analysis plots
    create_uncertainty_comparison_plots(
        all_importance_results,
        all_forecast_results,
        results_dir,
        model_config
    )
    
    # Individual window plots (10 windows)
    create_individual_window_plots(
        all_importance_results,
        results_dir,
        model_config
    )
    
    # Enhanced forecasting comparison plot
    create_forecasting_results_plot(
        all_forecast_results,
        results_dir,
        model_config
    )

    # Save enhanced results
    print(f"\\nSaving enhanced results...")
    results_summary = {
        'all_importance_scores': all_importance_scores,
        'all_ar_cv_uncertainties': all_ar_cv_uncertainties,
        'all_ar_errors': all_ar_errors,
        'all_mae_full': all_mae_full,
        'all_mae_random': all_mae_random,
        'all_mae_most_important': all_mae_most_important,
        'all_mae_random_interp': all_mae_random_interp,
        'all_mae_important_interp': all_mae_important_interp,
        'all_mae_least_important': all_mae_least_important,
        'all_mae_least_important_interp': all_mae_least_important_interp,
        'method_names': method_names,
        'method_maes': method_maes,
        'method_ranking': ranking_indices,
        'per_window_importance_results': all_importance_results,
        'per_window_forecast_results': all_forecast_results,
        'total_windows': len(all_importance_results),
        'total_samples': len(all_importance_scores),
        'model': MODEL,
        'size': SIZE,
        'ctx': CTX,
        'input_ctx': INPUT_CTX,
        'pdt': PDT,
        'num_samples': NUM_SAMPLES,
        'selected_column': SELECTED_COLUMN,
        'dataset_name': dataset_name
    }

    np.savez(os.path.join(results_dir, 'uncertainty_analysis_results.npz'), **results_summary)
    print(f"Enhanced results saved to '{os.path.join(results_dir, 'uncertainty_analysis_results.npz')}'")

    print(f"\\n{'='*80}")
    print("ENHANCED ANALYSIS COMPLETE! (V3)")
    print(f"{'='*80}")
    print(f"Enhanced V3 results saved to: {results_dir}")
    print(f"  • uncertainty_analysis.png - Enhanced uncertainty plots (V3)")
    print(f"  • individual_window_plots.png - 10 individual window analyses (V3)")
    print(f"  • forecasting_results_comparison.png - 7-method performance comparison (V3)")
    print(f"  • uncertainty_analysis_results.npz - Enhanced numerical results (V3)")
    print(f"\\nEnhanced Processing Summary (V3):")
    print(f"  • V3 Logic: Always using most recent {INPUT_CTX} samples as context for importance measurement")
    print(f"  • Forecasting using context length: {CTX}")
    print(f"  • Analysis Frequency: {ANALYSIS_FREQ if ANALYSIS_FREQ else 'Final only'}")
    print(f"  • Processed {len(all_importance_results)} windows with 7 forecasting methods")
    print(f"  • Analyzed {len(all_ar_cv_uncertainties)} AR steps for uncertainty")
    print(f"  • Created detailed plots: uncertainty_analysis.png + individual_window_plots.png + forecasting_results_comparison.png")
    if ANALYSIS_FREQ is not None:
        intermediate_count = len(all_importance_results) // ANALYSIS_FREQ
        print(f"  • Generated {intermediate_count} sets of intermediate plots during processing")
    print(f"  • Best performing method: {method_names[ranking_indices[0]]} (MAE: {method_maes[ranking_indices[0]]:.4f})")
    print(f"  • Mean AR CV uncertainty: {np.mean(all_ar_cv_uncertainties):.4f}")
