import torch
import matplotlib.pyplot as plt
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from huggingface_hub import hf_hub_download
import numpy as np
from tqdm import tqdm
import os

# Set CUDA device to GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from uni2ts.eval_util.plot import plot_single
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule

MODEL = "moirai"  # model name: choose from {'moirai', 'moirai-moe'}
SIZE = "small"  # model size: choose from {'small', 'base', 'large'}
PDT = 8  # prediction length: any positive integer
CTX = 64  # context length: any positive integer
PSZ = "auto"  # patch size: choose from {"auto", 8, 16, 32, 64, 128}
BSZ = 32  # batch size
TEST = int(1*PDT)  # test set length: any positive integer

# Read data into pandas DataFrame
csv_path = "/home/sa53869/time-series/moirai/time-moe-eval/ETT-small/ETTm2.csv"
df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

# Convert into GluonTS dataset
ds = PandasDataset(dict(df))

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
            num_samples=100,
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
            num_samples=100,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
    return model.create_predictor(batch_size=1)

# Sample importance analysis function
def compute_sample_importance(input_data, window_id, min_context=PDT, max_analysis=CTX):
    """
    Compute importance of each sample by predicting it from preceding context
    Context length changes for each iteration: 8, 9, 10, etc.
    """
    full_sequence = input_data["target"]
    
    # Determine analysis length
    if max_analysis is None:
        analysis_length = min(len(full_sequence) - min_context, CTX - min_context)
    else:
        analysis_length = min(max_analysis, len(full_sequence) - min_context, CTX - min_context)
    
    sample_importance = np.zeros(analysis_length)
    sample_uncertainties = np.zeros(analysis_length)
    sample_predictions = np.zeros(analysis_length)
    sample_true_values = np.zeros(analysis_length)
    context_lengths = np.zeros(analysis_length, dtype=int)
    
    print(f"Window {window_id}: Analyzing importance of {analysis_length} samples (starting from position {min_context})...")
    
    for pos in tqdm(range(analysis_length), desc=f"Window {window_id} - Computing sample importance"):
        # Current position to predict (min_context + pos)
        current_pos = min_context + pos
        
        # Use preceding context to predict current position
        context_data = full_sequence[:current_pos]
        true_value = full_sequence[current_pos]
        
        # Context length for this iteration
        current_context_length = len(context_data)
        context_lengths[pos] = current_context_length
        
        # Create model with appropriate context length
        predictor = create_model_with_context_length(current_context_length)
        
        # Create input for prediction
        step_input_data = input_data.copy()
        step_input_data["target"] = context_data
        step_input = [step_input_data]
        
        # Predict the next value
        forecast = next(iter(predictor.predict(step_input)))
        pred_samples = forecast.samples[:, 0]
        pred_mean = np.median(pred_samples)
        pred_uncertainty = np.std(pred_samples)
        
        # Calculate prediction error (importance)
        prediction_error = abs(pred_mean - true_value)
        
        # Store results
        sample_importance[pos] = prediction_error
        sample_uncertainties[pos] = pred_uncertainty
        sample_predictions[pos] = pred_mean
        sample_true_values[pos] = true_value
        
        if pos < 5 or pos % 5 == 0:  # Print every 5th iteration to reduce clutter
            print(f"  Position {current_pos}: ctx_len={current_context_length}, error={prediction_error:.4f}, uncertainty={pred_uncertainty:.4f}")
    
    return {
        'importance': sample_importance,
        'uncertainties': sample_uncertainties,
        'predictions': sample_predictions,
        'true_values': sample_true_values,
        'context_lengths': context_lengths,
        'start_position': min_context,
        'analyzed_length': analysis_length,
        'window_id': window_id
    }

# NEW: Analyze sample importance AND forecasting for ALL windows
print("\nAnalyzing sample importance and forecasting across all windows...")

all_importance_results = []
all_forecast_results = []
input_it = iter(test_data_full.input)
label_it = iter(test_data_full.label)

for window_idx, (input_data, label_data) in enumerate(zip(input_it, label_it)):
    if window_idx >= actual_windows:
        break
    
    print(f"\n{'='*60}")
    print(f"Processing Window {window_idx + 1}/{actual_windows}")
    print(f"{'='*60}")
    
    # 1. Sample importance analysis
    importance_result = compute_sample_importance(input_data, window_idx + 1, min_context=PDT, max_analysis=CTX)
    all_importance_results.append(importance_result)
    
    # 2. Forecasting using full context
    print(f"Performing forecasting for Window {window_idx + 1}...")
    
    # Create predictor with the specified context length
    predictor = create_model_with_context_length(CTX)
    
    # Prepare input for forecasting
    forecast_input = [input_data]
    
    # Generate forecast
    forecast = next(iter(predictor.predict(forecast_input)))
    
    # Extract forecast results
    forecast_samples = forecast.samples  # Shape: (num_samples, prediction_length)
    forecast_mean = np.median(forecast_samples, axis=0)
    forecast_uncertainty = np.std(forecast_samples, axis=0)
    forecast_quantiles = np.percentile(forecast_samples, [10, 25, 50, 75, 90], axis=0)
    
    # Get true values for comparison
    true_values = label_data["target"][:PDT]  # First PDT values
    
    # Calculate forecast metrics
    forecast_errors = np.abs(forecast_mean - true_values)
    mae = np.mean(forecast_errors)
    mse = np.mean((forecast_mean - true_values) ** 2)
    rmse = np.sqrt(mse)
    
    # Store forecast results
    forecast_result = {
        'window_id': window_idx + 1,
        'forecast_mean': forecast_mean,
        'forecast_uncertainty': forecast_uncertainty,
        'forecast_quantiles': forecast_quantiles,
        'true_values': true_values,
        'forecast_errors': forecast_errors,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'prediction_length': PDT,
        'context_length': CTX
    }
    all_forecast_results.append(forecast_result)
    
    # Print summaries for this window
    print(f"\nWindow {window_idx + 1} Summary:")
    print(f"  Sample Importance:")
    print(f"    Samples analyzed: {importance_result['analyzed_length']}")
    print(f"    Mean importance: {np.mean(importance_result['importance']):.4f}")
    print(f"    Max importance: {np.max(importance_result['importance']):.4f}")
    print(f"    Min importance: {np.min(importance_result['importance']):.4f}")
    print(f"  Forecasting:")
    print(f"    MAE: {mae:.4f}")
    print(f"    RMSE: {rmse:.4f}")
    print(f"    Mean uncertainty: {np.mean(forecast_uncertainty):.4f}")
    print(f"    Max forecast error: {np.max(forecast_errors):.4f}")

# NEW: Aggregate analysis across all windows
print(f"\n{'='*60}")
print("AGGREGATE ANALYSIS ACROSS ALL WINDOWS")
print(f"{'='*60}")

# Combine all importance scores
all_importance_scores = np.concatenate([result['importance'] for result in all_importance_results])
all_uncertainty_scores = np.concatenate([result['uncertainties'] for result in all_importance_results])
all_context_lengths = np.concatenate([result['context_lengths'] for result in all_importance_results])

# Combine all forecasting scores
all_forecast_mae = np.array([result['mae'] for result in all_forecast_results])
all_forecast_rmse = np.array([result['rmse'] for result in all_forecast_results])
all_forecast_errors = np.concatenate([result['forecast_errors'] for result in all_forecast_results])
all_forecast_uncertainties = np.concatenate([result['forecast_uncertainty'] for result in all_forecast_results])

print(f"Total samples analyzed (importance): {len(all_importance_scores)}")
print(f"Total windows processed: {len(all_importance_results)}")
print(f"Total forecast points: {len(all_forecast_errors)}")

# Overall statistics
print(f"\nOverall Importance Statistics:")
print(f"Mean importance: {np.mean(all_importance_scores):.4f}")
print(f"Std importance: {np.std(all_importance_scores):.4f}")
print(f"Max importance: {np.max(all_importance_scores):.4f}")
print(f"Min importance: {np.min(all_importance_scores):.4f}")

print(f"\nOverall Forecasting Statistics:")
print(f"Mean MAE: {np.mean(all_forecast_mae):.4f}")
print(f"Mean RMSE: {np.mean(all_forecast_rmse):.4f}")
print(f"Overall MAE: {np.mean(all_forecast_errors):.4f}")
print(f"Overall forecast uncertainty: {np.mean(all_forecast_uncertainties):.4f}")
print(f"Max forecast error: {np.max(all_forecast_errors):.4f}")
print(f"Min forecast error: {np.min(all_forecast_errors):.4f}")

# Per-window statistics
print(f"\nPer-Window Statistics:")
print(f"{'Window':<8} {'Imp_Mean':<10} {'Imp_Std':<10} {'Imp_Max':<10} {'MAE':<8} {'RMSE':<8} {'Samples':<8}")
print(f"{'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
for i, (imp_result, forecast_result) in enumerate(zip(all_importance_results, all_forecast_results)):
    mean_imp = np.mean(imp_result['importance'])
    std_imp = np.std(imp_result['importance'])
    max_imp = np.max(imp_result['importance'])
    mae = forecast_result['mae']
    rmse = forecast_result['rmse']
    samples = imp_result['analyzed_length']
    print(f"W{i+1:<7d} {mean_imp:<10.4f} {std_imp:<10.4f} {max_imp:<10.4f} {mae:<8.4f} {rmse:<8.4f} {samples:<8d}")

# Context length analysis
print(f"\nContext Length Analysis:")
unique_context_lengths = sorted(set(all_context_lengths))
for ctx_len in unique_context_lengths:
    mask = all_context_lengths == ctx_len
    if np.sum(mask) > 0:
        avg_error = np.mean(all_importance_scores[mask])
        avg_uncertainty = np.mean(all_uncertainty_scores[mask])
        count = np.sum(mask)
        print(f"Context Length {ctx_len:2d}: Avg Error={avg_error:.4f}, Avg Uncertainty={avg_uncertainty:.4f}, Count={count}")

# Compression analysis
thresholds = [10, 20, 30]
print(f"\nCompression Potential Analysis:")
for threshold in thresholds:
    threshold_value = np.percentile(all_importance_scores, threshold)
    compressible_samples = np.sum(all_importance_scores <= threshold_value)
    compression_ratio = compressible_samples / len(all_importance_scores) * 100
    print(f"Bottom {threshold:2d}% threshold: {compressible_samples:4d}/{len(all_importance_scores)} samples ({compression_ratio:.1f}% compression potential)")

# Prepare data for plotting
unique_context_lengths = sorted(set(all_context_lengths))

# NEW: Visualize aggregate results including forecasting
fig, axes = plt.subplots(3, 3, figsize=(18, 18))
fig.suptitle('Sample Importance Analysis and Forecasting Results Across All Windows', fontsize=16)

# Plot 1: Distribution of importance scores
ax1 = axes[0, 0]
ax1.hist(all_importance_scores, bins=50, alpha=0.7, color='red', edgecolor='black')
ax1.set_title('Distribution of Importance Scores')
ax1.set_xlabel('Importance Score')
ax1.set_ylabel('Frequency')
ax1.grid(True, alpha=0.3)

# Plot 2: Distribution of forecast errors
ax2 = axes[0, 1]
ax2.hist(all_forecast_errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
ax2.set_title('Distribution of Forecast Errors')
ax2.set_xlabel('Forecast Error (MAE)')
ax2.set_ylabel('Frequency')
ax2.grid(True, alpha=0.3)

# Plot 3: Importance vs Forecast Uncertainty
ax3 = axes[0, 2]
# Sample subset for readability if too many points
if len(all_importance_scores) > 1000:
    sample_idx = np.random.choice(len(all_importance_scores), 1000, replace=False)
    plot_importance = all_importance_scores[sample_idx]
    plot_uncertainty = all_uncertainty_scores[sample_idx]
else:
    plot_importance = all_importance_scores
    plot_uncertainty = all_uncertainty_scores
ax3.scatter(plot_importance, plot_uncertainty, alpha=0.6, s=20)
ax3.set_title('Sample Importance vs Uncertainty')
ax3.set_xlabel('Importance Score')
ax3.set_ylabel('Sample Uncertainty')
ax3.grid(True, alpha=0.3)

# Plot 4: Importance vs Context Length
ax4 = axes[1, 0]
scatter = ax4.scatter(all_context_lengths, all_importance_scores, alpha=0.6, s=20)
ax4.set_title('Importance vs Context Length')
ax4.set_xlabel('Context Length')
ax4.set_ylabel('Importance Score')
ax4.grid(True, alpha=0.3)

# Plot 5: Box plot of MAE by window
ax5 = axes[1, 1]
ax5.boxplot([result['forecast_errors'] for result in all_forecast_results], 
           tick_labels=[f'W{i+1}' for i in range(len(all_forecast_results))])
ax5.set_title('Forecast Errors by Window')
ax5.set_xlabel('Window')
ax5.set_ylabel('Forecast Error (MAE)')
ax5.tick_params(axis='x', rotation=45)
ax5.grid(True, alpha=0.3)

# Plot 6: Forecast Error vs Uncertainty
ax6 = axes[1, 2]
# Debug: Check array sizes
print(f"Debug - all_forecast_errors shape: {all_forecast_errors.shape}")
print(f"Debug - all_forecast_uncertainties shape: {all_forecast_uncertainties.shape}")

# Ensure arrays are the same size
min_size = min(len(all_forecast_errors), len(all_forecast_uncertainties))
forecast_errors_plot = all_forecast_errors[:min_size]
forecast_uncertainties_plot = all_forecast_uncertainties[:min_size]

ax6.scatter(forecast_errors_plot, forecast_uncertainties_plot, alpha=0.6, s=20)
ax6.set_title('Forecast Error vs Uncertainty')
ax6.set_xlabel('Forecast Error')
ax6.set_ylabel('Forecast Uncertainty')
ax6.grid(True, alpha=0.3)

# Plot 7: Box plot of importance by window
ax7 = axes[2, 0]
window_importance_data = [result['importance'] for result in all_importance_results]
ax7.boxplot(window_importance_data, tick_labels=[f'W{i+1}' for i in range(len(all_importance_results))])
ax7.set_title('Importance Distribution by Window')
ax7.set_xlabel('Window')
ax7.set_ylabel('Importance Score')
ax7.tick_params(axis='x', rotation=45)
ax7.grid(True, alpha=0.3)

# Plot 8: Mean importance by context length
ax8 = axes[2, 1]
ctx_means = []
ctx_stds = []
for ctx_len in unique_context_lengths:
    mask = all_context_lengths == ctx_len
    if np.sum(mask) > 0:
        ctx_means.append(np.mean(all_importance_scores[mask]))
        ctx_stds.append(np.std(all_importance_scores[mask]))
    else:
        ctx_means.append(0)
        ctx_stds.append(0)

ax8.errorbar(unique_context_lengths, ctx_means, yerr=ctx_stds, marker='o', capsize=5)
ax8.set_title('Mean Importance by Context Length')
ax8.set_xlabel('Context Length')
ax8.set_ylabel('Mean Importance Score')
ax8.grid(True, alpha=0.3)

# Plot 9: Cumulative distribution of importance and forecast errors
ax9 = axes[2, 2]
sorted_importance = np.sort(all_importance_scores)
sorted_forecast_errors = np.sort(all_forecast_errors)
cumulative_imp = np.arange(1, len(sorted_importance) + 1) / len(sorted_importance) * 100
cumulative_forecast = np.arange(1, len(sorted_forecast_errors) + 1) / len(sorted_forecast_errors) * 100

ax9.plot(sorted_importance, cumulative_imp, linewidth=2, label='Importance', color='red')
ax9_twin = ax9.twinx()
ax9_twin.plot(sorted_forecast_errors, cumulative_forecast, linewidth=2, label='Forecast Error', color='blue')
ax9.set_title('Cumulative Distribution')
ax9.set_xlabel('Score')
ax9.set_ylabel('Cumulative % (Importance)', color='red')
ax9_twin.set_ylabel('Cumulative % (Forecast)', color='blue')
ax9.grid(True, alpha=0.3)
ax9.legend(loc='upper left')
ax9_twin.legend(loc='upper right')

plt.tight_layout()
plt.savefig('sample_importance_all_windows.png', dpi=300, bbox_inches='tight')
plt.show()

# NEW: Export results for further analysis including forecasting
print(f"\nSaving results...")
results_summary = {
    'all_importance_scores': all_importance_scores,
    'all_uncertainty_scores': all_uncertainty_scores,
    'all_context_lengths': all_context_lengths,
    'all_forecast_mae': all_forecast_mae,
    'all_forecast_rmse': all_forecast_rmse,
    'all_forecast_errors': all_forecast_errors,
    'all_forecast_uncertainties': all_forecast_uncertainties,
    'per_window_importance_results': all_importance_results,
    'per_window_forecast_results': all_forecast_results,
    'total_windows': len(all_importance_results),
    'total_samples': len(all_importance_scores),
    'total_forecast_points': len(all_forecast_errors)
}

# Save as numpy arrays for easy loading
np.savez('sample_importance_results.npz', **results_summary)
print(f"Results saved to 'sample_importance_results.npz'")

print(f"\nAnalysis complete!")
print(f"  Processed {len(all_importance_results)} windows")
print(f"  Analyzed {len(all_importance_scores)} samples for importance")
print(f"  Generated {len(all_forecast_errors)} forecast points")
print(f"  Mean MAE across all windows: {np.mean(all_forecast_mae):.4f}")
print(f"  Mean importance score: {np.mean(all_importance_scores):.4f}")