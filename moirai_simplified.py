import torch
import matplotlib.pyplot as plt
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
import numpy as np
from tqdm import tqdm
import os
import seaborn as sns
from typing import Dict, Any, Optional, Tuple
import warnings
import argparse
from scipy import interpolate
warnings.filterwarnings('ignore')

from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Moirai Uncertainty Analysis - Simplified')
    
    parser.add_argument('--gpu', type=int, default=0, 
                        help='GPU device ID (default: 0)')
    parser.add_argument('--model', type=str, default='moirai', choices=['moirai', 'moirai-moe'],
                        help='Model name (default: moirai)')
    parser.add_argument('--size', type=str, default='large', choices=['small', 'base', 'large'],
                        help='Model size (default: large)')
    parser.add_argument('--pdt', type=int, default=8,
                        help='Prediction length (default: 8)')
    parser.add_argument('--ctx', type=int, default=64,
                        help='Context length (default: 64)')
    parser.add_argument('--psz', type=str, default='auto',
                        help='Patch size (default: auto)')
    parser.add_argument('--bsz', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--num-windows', type=int, default=100,
                        help='Number of windows to analyze (default: 100)')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of samples for uncertainty estimation (default: 100)')
    parser.add_argument('--csv-path', type=str, 
                        default="/home/sa53869/time-series/moirai/time-moe-eval/synthetic_sinusoidal.csv",
                        help='Path to CSV data file')
    parser.add_argument('--column', type=int, default=1,
                        help='Column number to analyze (0-indexed, default: 1 for 2nd column)')
    
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
PSZ = args.psz
BSZ = args.bsz
NUM_WINDOWS = args.num_windows
NUM_SAMPLES = args.num_samples
COLUMN_NUM = args.column

print(f"Configuration:")
print(f"  Model: {MODEL}-{SIZE}")
print(f"  Context Length: {CTX}")
print(f"  Prediction Length: {PDT}")
print(f"  Patch Size: {PSZ}")
print(f"  Batch Size: {BSZ}")
print(f"  Number of Windows: {NUM_WINDOWS}")
print(f"  Num Samples: {NUM_SAMPLES}")
print(f"  CSV Path: {args.csv_path}")
print(f"  Column: {COLUMN_NUM} (0-indexed)")

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
results_dir = f"results/{dataset_name}/{MODEL}-{SIZE}/ctx{CTX}"
os.makedirs(results_dir, exist_ok=True)
print(f"Results will be saved to: {results_dir}")

# Focus on selected column only
df_selected = df[[selected_column]].copy()
print(f"Focusing on column '{selected_column}' only. Data shape: {df_selected.shape}")

# Get the full time series data
full_data = df_selected[selected_column].values
total_length = len(full_data)
print(f"Total data length: {total_length}")

# Calculate required length per window
window_length = 2 * CTX + PDT
print(f"Required samples per window: {window_length} (2*CTX + PDT = 2*{CTX} + {PDT})")

# Check if we have enough data
if total_length < NUM_WINDOWS * window_length:
    print(f"Warning: Not enough data for {NUM_WINDOWS} non-overlapping windows")
    print(f"Available: {total_length}, Required: {NUM_WINDOWS * window_length}")
    # Use overlapping windows instead
    max_start_idx = total_length - window_length
    if max_start_idx < 0:
        raise ValueError(f"Data too short. Need at least {window_length} samples, have {total_length}")
    
    window_starts = np.linspace(0, max_start_idx, NUM_WINDOWS, dtype=int)
    print(f"Using overlapping windows with starts: {window_starts[:5]}...")
else:
    # Use non-overlapping windows
    window_starts = np.arange(0, NUM_WINDOWS * window_length, window_length)
    print(f"Using non-overlapping windows")

print(f"Will analyze {len(window_starts)} windows")

# Load the base module once
print("Loading base model module...")
if MODEL == "moirai":
    base_module = MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-{SIZE}")
elif MODEL == "moirai-moe":
    base_module = MoiraiMoEModule.from_pretrained(f"Salesforce/moirai-moe-1.0-R-{SIZE}")

def create_model_with_context_length(context_length, prediction_length=1):
    """Create a new model with specific context length"""
    if MODEL == "moirai":
        model = MoiraiForecast(
            module=base_module,
            prediction_length=prediction_length,
            context_length=context_length,
            patch_size=PSZ,
            num_samples=NUM_SAMPLES,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
    elif MODEL == "moirai-moe":
        model = MoiraiMoEForecast(
            module=base_module,
            prediction_length=prediction_length,
            context_length=context_length,
            patch_size=16,
            num_samples=NUM_SAMPLES,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
    return model.create_predictor(batch_size=BSZ)

def calculate_uncertainty_for_prediction(predictor, context_data, start_time):
    """
    Calculate uncertainty for a single prediction
    """
    # Create input data structure
    input_data = {
        'target': context_data,
        'start': start_time,
        'item_id': 0
    }
    
    # Generate forecast with samples
    forecast = next(iter(predictor.predict([input_data])))
    
    # Extract samples
    forecast_samples = forecast.samples
    if forecast_samples.ndim == 3:
        forecast_samples = forecast_samples[0]  # Remove batch dimension
    
    # Calculate statistics
    mean_forecast = np.mean(forecast_samples, axis=0)
    std_forecast = np.std(forecast_samples, axis=0)
    
    # Calculate uncertainty measures
    cv_uncertainty = std_forecast / (np.abs(mean_forecast) + 1e-6)
    q25 = np.percentile(forecast_samples, 25, axis=0)
    q75 = np.percentile(forecast_samples, 75, axis=0)
    iqr_uncertainty = (q75 - q25) / (np.abs(mean_forecast) + 1e-6)
    
    return {
        'samples': forecast_samples,
        'mean': mean_forecast[0],  # Single prediction
        'std': std_forecast[0],
        'cv_uncertainty': cv_uncertainty[0],
        'iqr_uncertainty': iqr_uncertainty[0],
        'q25': q25[0],
        'q75': q75[0]
    }

def analyze_single_window(window_data, window_idx, start_time):
    """
    Analyze a single window using the sliding context approach
    """
    print(f"\\nAnalyzing window {window_idx + 1}/{len(window_starts)}...")
    
    # Window has 2*CTX + PDT samples
    # We'll do CTX + PDT predictions (from position CTX to position 2*CTX + PDT - 1)
    num_predictions = CTX + PDT
    
    predictions = []
    uncertainties = []
    true_values = []
    contexts_used = []
    
    # Create predictor once for this window
    predictor = create_model_with_context_length(CTX, prediction_length=1)
    
    for step in tqdm(range(num_predictions), desc=f"Window {window_idx + 1} predictions"):
        # Context starts at position 'step' and has CTX samples
        context_start = step
        context_end = step + CTX
        target_position = step + CTX  # The position we want to predict
        
        if target_position >= len(window_data):
            break
            
        # Extract context and target
        context = window_data[context_start:context_end]
        true_value = window_data[target_position]
        
        # Calculate uncertainty for this prediction
        uncertainty_result = calculate_uncertainty_for_prediction(
            predictor, context, start_time
        )
        
        predictions.append(uncertainty_result['mean'])
        uncertainties.append(uncertainty_result['cv_uncertainty'])
        true_values.append(true_value)
        contexts_used.append(context)
    
    return {
        'window_idx': window_idx,
        'predictions': np.array(predictions),
        'uncertainties': np.array(uncertainties),
        'true_values': np.array(true_values),
        'contexts_used': contexts_used,
        'window_data': window_data,
        'num_predictions': len(predictions)
    }

def create_visualization_plots(all_results, save_path=None):
    """
    Create comprehensive visualization plots
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Moirai Uncertainty Analysis - {dataset_name}\\n'
                f'Model: {MODEL}-{SIZE} | CTX: {CTX} | PDT: {PDT} | Windows: {len(all_results)}', 
                fontsize=16, fontweight='bold')
    
    # 1. Prediction errors across all windows
    ax1 = axes[0, 0]
    all_errors = []
    all_uncertainties = []
    
    for result in all_results:
        errors = np.abs(result['predictions'] - result['true_values'])
        all_errors.extend(errors)
        all_uncertainties.extend(result['uncertainties'])
    
    ax1.scatter(all_uncertainties, all_errors, alpha=0.6, s=20)
    ax1.set_xlabel('CV Uncertainty')
    ax1.set_ylabel('Absolute Error')
    ax1.set_title('Uncertainty vs Error')
    ax1.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    corr = np.corrcoef(all_uncertainties, all_errors)[0, 1]
    ax1.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax1.transAxes, 
             bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
    
    # 2. Uncertainty distribution
    ax2 = axes[0, 1]
    ax2.hist(all_uncertainties, bins=50, alpha=0.7, density=True)
    ax2.set_xlabel('CV Uncertainty')
    ax2.set_ylabel('Density')
    ax2.set_title('Uncertainty Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    mean_unc = np.mean(all_uncertainties)
    std_unc = np.std(all_uncertainties)
    ax2.axvline(mean_unc, color='red', linestyle='--', label=f'Mean: {mean_unc:.3f}')
    ax2.legend()
    
    # 3. Prediction accuracy by window
    ax3 = axes[0, 2]
    window_maes = []
    window_uncertainties = []
    
    for result in all_results:
        mae = np.mean(np.abs(result['predictions'] - result['true_values']))
        mean_uncertainty = np.mean(result['uncertainties'])
        window_maes.append(mae)
        window_uncertainties.append(mean_uncertainty)
    
    ax3.plot(range(1, len(window_maes) + 1), window_maes, 'bo-', label='MAE')
    ax3_twin = ax3.twinx()
    ax3_twin.plot(range(1, len(window_uncertainties) + 1), window_uncertainties, 'ro-', label='Mean Uncertainty')
    
    ax3.set_xlabel('Window Index')
    ax3.set_ylabel('MAE', color='blue')
    ax3_twin.set_ylabel('Mean Uncertainty', color='red')
    ax3.set_title('Performance by Window')
    ax3.grid(True, alpha=0.3)
    
    # 4. Example window analysis
    ax4 = axes[1, 0]
    if len(all_results) > 0:
        # Select middle window for example
        example_idx = len(all_results) // 2
        example_result = all_results[example_idx]
        
        positions = range(len(example_result['predictions']))
        ax4.plot(positions, example_result['true_values'], 'b-', label='True Values', linewidth=2)
        ax4.plot(positions, example_result['predictions'], 'r--', label='Predictions', linewidth=2)
        
        # Add uncertainty bands
        uncertainties = example_result['uncertainties']
        predictions = example_result['predictions']
        ax4.fill_between(positions, 
                        predictions - uncertainties * predictions,
                        predictions + uncertainties * predictions,
                        alpha=0.3, color='red', label='Uncertainty Band')
        
        ax4.set_xlabel('Prediction Step')
        ax4.set_ylabel('Value')
        ax4.set_title(f'Example Window {example_idx + 1}')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # 5. Position-wise uncertainty pattern
    ax5 = axes[1, 1]
    max_steps = max(len(result['uncertainties']) for result in all_results)
    position_uncertainties = [[] for _ in range(max_steps)]
    
    for result in all_results:
        for pos, unc in enumerate(result['uncertainties']):
            if pos < max_steps:
                position_uncertainties[pos].append(unc)
    
    # Calculate mean and std for each position
    mean_uncertainties = [np.mean(pos_unc) if pos_unc else 0 for pos_unc in position_uncertainties]
    std_uncertainties = [np.std(pos_unc) if pos_unc else 0 for pos_unc in position_uncertainties]
    
    positions = range(len(mean_uncertainties))
    ax5.plot(positions, mean_uncertainties, 'g-', linewidth=2, label='Mean Uncertainty')
    ax5.fill_between(positions,
                    np.array(mean_uncertainties) - np.array(std_uncertainties),
                    np.array(mean_uncertainties) + np.array(std_uncertainties),
                    alpha=0.3, color='green')
    
    ax5.set_xlabel('Prediction Step within Window')
    ax5.set_ylabel('CV Uncertainty')
    ax5.set_title('Position-wise Uncertainty Pattern')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Summary statistics
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Calculate overall statistics
    overall_mae = np.mean(all_errors)
    overall_uncertainty = np.mean(all_uncertainties)
    
    summary_text = f'''Summary Statistics:
    
Total Predictions: {len(all_errors):,}
Total Windows: {len(all_results)}
Predictions per Window: {CTX + PDT}

Overall Performance:
• Mean Absolute Error: {overall_mae:.4f}
• Mean CV Uncertainty: {overall_uncertainty:.4f}
• Error-Uncertainty Correlation: {corr:.3f}

Window Statistics:
• Mean Window MAE: {np.mean(window_maes):.4f}
• Std Window MAE: {np.std(window_maes):.4f}
• Best Window MAE: {np.min(window_maes):.4f}
• Worst Window MAE: {np.max(window_maes):.4f}

Model Configuration:
• Context Length: {CTX}
• Prediction Length: {PDT}
• Model: {MODEL}-{SIZE}
• Samples: {NUM_SAMPLES}'''
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()

# Main analysis loop
print(f"\\n{'='*80}")
print("STARTING UNCERTAINTY ANALYSIS")
print(f"{'='*80}")

all_results = []

for i, start_idx in enumerate(window_starts):
    # Extract window data
    end_idx = start_idx + window_length
    if end_idx > total_length:
        print(f"Skipping window {i+1} - not enough data")
        continue
        
    window_data = full_data[start_idx:end_idx]
    
    # Create a dummy start time (for GluonTS compatibility)
    start_time = pd.Timestamp('2024-01-01') + pd.Timedelta(hours=start_idx)
    
    # Analyze this window
    result = analyze_single_window(window_data, i, start_time)
    all_results.append(result)

print(f"\\n{'='*80}")
print("ANALYSIS COMPLETE - CREATING VISUALIZATIONS")
print(f"{'='*80}")

# Create comprehensive visualization
create_visualization_plots(all_results, os.path.join(results_dir, 'uncertainty_analysis.png'))

# Save results
print(f"\\nSaving results...")
results_summary = {
    'all_results': all_results,
    'total_windows': len(all_results),
    'model': MODEL,
    'size': SIZE,
    'ctx': CTX,
    'pdt': PDT,
    'num_samples': NUM_SAMPLES,
    'dataset_name': dataset_name,
    'selected_column': selected_column
}

np.savez(os.path.join(results_dir, 'uncertainty_analysis_results.npz'), **results_summary)
print(f"Results saved to '{os.path.join(results_dir, 'uncertainty_analysis_results.npz')}'")

print(f"\\n{'='*80}")
print("ANALYSIS COMPLETE!")
print(f"{'='*80}")
print(f"Results saved to: {results_dir}")
print(f"  • uncertainty_analysis.png - Visualization plots")
print(f"  • uncertainty_analysis_results.npz - Numerical results")
print(f"\\nProcessing Summary:")
print(f"  • Analyzed {len(all_results)} windows")
print(f"  • Each window: {window_length} samples (2*{CTX} + {PDT})")
print(f"  • Predictions per window: {CTX + PDT}")
print(f"  • Total predictions: {sum(len(r['predictions']) for r in all_results):,}")

if len(all_results) > 0:
    all_errors = []
    all_uncertainties = []
    for result in all_results:
        errors = np.abs(result['predictions'] - result['true_values'])
        all_errors.extend(errors)
        all_uncertainties.extend(result['uncertainties'])
    
    print(f"  • Overall MAE: {np.mean(all_errors):.4f}")
    print(f"  • Overall mean uncertainty: {np.mean(all_uncertainties):.4f}")
