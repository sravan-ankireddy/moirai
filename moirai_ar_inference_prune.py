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
SIZE = "large"  # model size: choose from {'small', 'base', 'large'}
PDT = 8  # prediction length: any positive integer
CTX = 64  # context length: any positive integer
PSZ = "auto"  # patch size: choose from {"auto", 8, 16, 32, 64, 128}
BSZ = 128  # batch size
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
    return model.create_predictor(batch_size=BSZ)

# Sample importance analysis function
def compute_sample_importance(input_data, window_id, min_context=0, max_analysis=CTX):
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
    importance_result = compute_sample_importance(input_data, window_idx + 1, min_context=0, max_analysis=CTX)
    all_importance_results.append(importance_result)
    
    # 2. Forecasting using full context - ONLY for current window
    print(f"Performing forecasting for Window {window_idx + 1}...")
    
    # Get the target data and ensure it's exactly CTX length
    target_data = input_data['target']
    actual_input_length = len(target_data)
    print(f"  Original input data length: {actual_input_length}")
    print(f"  Required context length (CTX): {CTX}")
    
    # Take only the most recent CTX samples
    if actual_input_length >= CTX:
        # Use the last CTX samples
        context_target = target_data[-CTX:]
        print(f"  Using last {CTX} samples from {actual_input_length} available")
    else:
        # If we have fewer than CTX samples, use all available (pad if necessary)
        context_target = target_data
        print(f"  WARNING: Only {actual_input_length} samples available, less than CTX={CTX}")
    
    # Calculate reduced context length (75% of original)
    reduced_ctx = int(0.75 * CTX)  # 75% of CTX = 48 for CTX=64
    print(f"  Reduced context length: {reduced_ctx}")
    
    # 2a. Full context forecasting
    print(f"  2a. Full context forecasting (CTX={CTX})...")
    forecast_input_data_full = {
        'target': context_target,
        'start': input_data['start'],
        'item_id': input_data.get('item_id', 0)
    }
    
    predictor_full = create_model_with_context_length(CTX)
    forecast_it_full = predictor_full.predict([forecast_input_data_full])
    forecast_full = next(forecast_it_full)
    
    forecast_samples_full = forecast_full.samples
    forecast_mean_full = np.median(forecast_samples_full, axis=0)
    forecast_uncertainty_full = np.std(forecast_samples_full, axis=0)
    forecast_quantiles_full = np.percentile(forecast_samples_full, [10, 25, 50, 75, 90], axis=0)
    
    # 2b. Random 75% context forecasting
    print(f"  2b. Random 75% context forecasting (CTX={reduced_ctx})...")
    np.random.seed(42 + window_idx)  # Reproducible random selection
    random_indices = np.sort(np.random.choice(len(context_target), reduced_ctx, replace=False))
    context_target_random = context_target[random_indices]
    
    forecast_input_data_random = {
        'target': context_target_random,
        'start': input_data['start'],
        'item_id': input_data.get('item_id', 0)
    }
    
    predictor_random = create_model_with_context_length(reduced_ctx)
    forecast_it_random = predictor_random.predict([forecast_input_data_random])
    forecast_random = next(forecast_it_random)
    
    forecast_samples_random = forecast_random.samples
    forecast_mean_random = np.median(forecast_samples_random, axis=0)
    forecast_uncertainty_random = np.std(forecast_samples_random, axis=0)
    forecast_quantiles_random = np.percentile(forecast_samples_random, [10, 25, 50, 75, 90], axis=0)
    
    # 2c. Most important 75% context forecasting (drop least important 25%)
    print(f"  2c. Most important 75% context forecasting (CTX={reduced_ctx})...")
    
    # Get importance scores for the current context
    # We need to map the importance results to the current context
    importance_scores = importance_result['importance']
    importance_context_lengths = importance_result['context_lengths']
    
    # Find importance scores that correspond to the current context length
    ctx_mask = importance_context_lengths == CTX
    if np.any(ctx_mask):
        # Use importance scores from CTX length analysis
        ctx_importance = importance_scores[ctx_mask]
        if len(ctx_importance) >= CTX:
            # Take the first CTX importance scores
            context_importance = ctx_importance[:CTX]
        else:
            # If not enough, use the last available importance scores
            context_importance = importance_scores[-CTX:]
    else:
        # Fallback: use the last CTX importance scores
        context_importance = importance_scores[-CTX:]
    breakpoint()
    # Select the 75% MOST important samples (top 75% by importance)
    most_important_indices = np.argsort(context_importance)[-reduced_ctx:]  # Take the top 75%
    most_important_indices = np.sort(most_important_indices)  # Keep temporal order
    context_target_most_important = context_target[most_important_indices]
    
    forecast_input_data_most_important = {
        'target': context_target_most_important,
        'start': input_data['start'],
        'item_id': input_data.get('item_id', 0)
    }
    
    predictor_most_important = create_model_with_context_length(reduced_ctx)
    forecast_it_most_important = predictor_most_important.predict([forecast_input_data_most_important])
    forecast_most_important = next(forecast_it_most_important)
    
    forecast_samples_most_important = forecast_most_important.samples
    forecast_mean_most_important = np.median(forecast_samples_most_important, axis=0)
    forecast_uncertainty_most_important = np.std(forecast_samples_most_important, axis=0)
    forecast_quantiles_most_important = np.percentile(forecast_samples_most_important, [10, 25, 50, 75, 90], axis=0)
    
    # Get true values for comparison - ONLY for this window
    true_values = label_data["target"][:PDT]
    
    # Calculate forecast metrics for all three methods
    # Full context
    forecast_errors_full = np.abs(forecast_mean_full - true_values)
    mae_full = np.mean(forecast_errors_full)
    mse_full = np.mean((forecast_mean_full - true_values) ** 2)
    rmse_full = np.sqrt(mse_full)
    
    # Random context
    forecast_errors_random = np.abs(forecast_mean_random - true_values)
    mae_random = np.mean(forecast_errors_random)
    mse_random = np.mean((forecast_mean_random - true_values) ** 2)
    rmse_random = np.sqrt(mse_random)
    
    # Most important context
    forecast_errors_most_important = np.abs(forecast_mean_most_important - true_values)
    mae_most_important = np.mean(forecast_errors_most_important)
    mse_most_important = np.mean((forecast_mean_most_important - true_values) ** 2)
    rmse_most_important = np.sqrt(mse_most_important)
    
    print(f"  Window {window_idx + 1} Results:")
    print(f"    Full Context (CTX={CTX})      - MAE: {mae_full:.4f}, RMSE: {rmse_full:.4f}")
    print(f"    Random 75% (CTX={reduced_ctx})     - MAE: {mae_random:.4f}, RMSE: {rmse_random:.4f}")
    print(f"    Most Imp 75% (CTX={reduced_ctx})   - MAE: {mae_most_important:.4f}, RMSE: {rmse_most_important:.4f}")
    
    # Store forecast results for this window (updated structure)
    forecast_result = {
        'window_id': window_idx + 1,
        # Full context results
        'mae_full': mae_full,
        'rmse_full': rmse_full,
        'forecast_errors_full': forecast_errors_full,
        'forecast_uncertainty_full': forecast_uncertainty_full,
        'forecast_mean_full': forecast_mean_full,
        'forecast_quantiles_full': forecast_quantiles_full,
        # Random context results
        'mae_random': mae_random,
        'rmse_random': rmse_random,
        'forecast_errors_random': forecast_errors_random,
        'forecast_uncertainty_random': forecast_uncertainty_random,
        'forecast_mean_random': forecast_mean_random,
        'forecast_quantiles_random': forecast_quantiles_random,
        # Most important context results
        'mae_most_important': mae_most_important,
        'rmse_most_important': rmse_most_important,
        'forecast_errors_most_important': forecast_errors_most_important,
        'forecast_uncertainty_most_important': forecast_uncertainty_most_important,
        'forecast_mean_most_important': forecast_mean_most_important,
        'forecast_quantiles_most_important': forecast_quantiles_most_important,
        # Common data
        'true_values': true_values,
        'reduced_ctx': reduced_ctx,
        'random_indices': random_indices,
        'most_important_indices': most_important_indices
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
    # print(f"    MAE: {mae:.4f}")
    # print(f"    RMSE: {rmse:.4f}")
    # print(f"    Mean uncertainty: {np.mean(forecast_uncertainty):.4f}")
    # print(f"    Max forecast error: {np.max(forecast_errors):.4f}")

# NEW: Aggregate analysis across all windows
print(f"\n{'='*60}")
print("AGGREGATE ANALYSIS ACROSS ALL WINDOWS")
print(f"{'='*60}")

# Combine all importance scores
all_importance_scores = np.concatenate([result['importance'] for result in all_importance_results])
all_uncertainty_scores = np.concatenate([result['uncertainties'] for result in all_importance_results])
all_context_lengths = np.concatenate([result['context_lengths'] for result in all_importance_results])

# Combine all forecasting scores
all_forecast_mae_full = np.array([result['mae_full'] for result in all_forecast_results])
all_forecast_rmse_full = np.array([result['rmse_full'] for result in all_forecast_results])
all_forecast_errors_full = np.concatenate([result['forecast_errors_full'] for result in all_forecast_results])
all_forecast_uncertainties_full = np.concatenate([result['forecast_uncertainty_full'] for result in all_forecast_results])

all_forecast_mae_random = np.array([result['mae_random'] for result in all_forecast_results])
all_forecast_rmse_random = np.array([result['rmse_random'] for result in all_forecast_results])
all_forecast_errors_random = np.concatenate([result['forecast_errors_random'] for result in all_forecast_results])
all_forecast_uncertainties_random = np.concatenate([result['forecast_uncertainty_random'] for result in all_forecast_results])

all_forecast_mae_most_important = np.array([result['mae_most_important'] for result in all_forecast_results])
all_forecast_rmse_most_important = np.array([result['rmse_most_important'] for result in all_forecast_results])
all_forecast_errors_most_important = np.concatenate([result['forecast_errors_most_important'] for result in all_forecast_results])
all_forecast_uncertainties_most_important = np.concatenate([result['forecast_uncertainty_most_important'] for result in all_forecast_results])

print(f"Total samples analyzed (importance): {len(all_importance_scores)}")
print(f"Total windows processed: {len(all_importance_results)}")
print(f"Total forecast points per method: {len(all_forecast_errors_full)}")


# Overall statistics
print(f"\nOverall Forecasting Statistics:")
print(f"Full Context (CTX={CTX}):")
print(f"  Mean MAE: {np.mean(all_forecast_mae_full):.4f}")
print(f"  Mean RMSE: {np.mean(all_forecast_rmse_full):.4f}")
print(f"  Overall MAE: {np.mean(all_forecast_errors_full):.4f}")

print(f"Random 75% Context (CTX={int(0.75*CTX)}):")
print(f"  Mean MAE: {np.mean(all_forecast_mae_random):.4f}")
print(f"  Mean RMSE: {np.mean(all_forecast_rmse_random):.4f}")
print(f"  Overall MAE: {np.mean(all_forecast_errors_random):.4f}")

print(f"Most Important 75% Context (CTX={int(0.75*CTX)}):")
print(f"  Mean MAE: {np.mean(all_forecast_mae_most_important):.4f}")
print(f"  Mean RMSE: {np.mean(all_forecast_rmse_most_important):.4f}")
print(f"  Overall MAE: {np.mean(all_forecast_errors_most_important):.4f}")

# Performance comparison
print(f"\nPerformance Comparison:")
print(f"Random vs Full: {((np.mean(all_forecast_mae_random) / np.mean(all_forecast_mae_full)) - 1) * 100:+.2f}% MAE change")
print(f"Most Imp vs Full: {((np.mean(all_forecast_mae_most_important) / np.mean(all_forecast_mae_full)) - 1) * 100:+.2f}% MAE change")
print(f"Most Imp vs Random: {((np.mean(all_forecast_mae_most_important) / np.mean(all_forecast_mae_random)) - 1) * 100:+.2f}% MAE change")

# Per-window statistics
print(f"\nPer-Window Statistics:")
print(f"{'Window':<8} {'Imp_Mean':<10} {'Imp_Std':<10} {'Imp_Max':<10} {'MAE_Full':<10} {'MAE_Rand':<10} {'MAE_Imp':<10} {'Samples':<8}")
print(f"{'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
for i, (imp_result, forecast_result) in enumerate(zip(all_importance_results, all_forecast_results)):
    mean_imp = np.mean(imp_result['importance'])
    std_imp = np.std(imp_result['importance'])
    max_imp = np.max(imp_result['importance'])
    mae_full = forecast_result['mae_full']
    mae_random = forecast_result['mae_random']
    mae_most_important = forecast_result['mae_most_important']
    samples = imp_result['analyzed_length']
    print(f"W{i+1:<7d} {mean_imp:<10.4f} {std_imp:<10.4f} {max_imp:<10.4f} {mae_full:<10.4f} {mae_random:<10.4f} {mae_most_important:<10.4f} {samples:<8d}")


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

# Plot 2: Distribution of forecast errors (all methods)
ax2 = axes[0, 1]
ax2.hist(all_forecast_errors_full, bins=50, alpha=0.5, color='blue', edgecolor='black', label='Full Context')
ax2.hist(all_forecast_errors_random, bins=50, alpha=0.5, color='green', edgecolor='black', label='Random 75%')
ax2.hist(all_forecast_errors_most_important, bins=50, alpha=0.5, color='red', edgecolor='black', label='Most Imp 75%')
ax2.set_title('Distribution of Forecast Errors (All Methods)')
ax2.set_xlabel('Forecast Error (MAE)')
ax2.set_ylabel('Frequency')
ax2.legend()
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
ax5.boxplot([result['forecast_errors_full'] for result in all_forecast_results], 
           tick_labels=[f'W{i+1}' for i in range(len(all_forecast_results))])
ax5.set_title('Forecast Errors by Window (Full Context)')
ax5.set_xlabel('Window')
ax5.set_ylabel('Forecast Error (MAE)')
ax5.tick_params(axis='x', rotation=45)
ax5.grid(True, alpha=0.3)

# Plot 6: Forecast Error vs Uncertainty
ax6 = axes[1, 2]
# Debug: Check array sizes
print(f"Debug - all_forecast_errors_full shape: {all_forecast_errors_full.shape}")
print(f"Debug - all_forecast_uncertainties_full shape: {all_forecast_uncertainties_full.shape}")

# Ensure arrays are the same size
min_size = min(len(all_forecast_errors_full), len(all_forecast_uncertainties_full))
forecast_errors_plot = all_forecast_errors_full[:min_size]
forecast_uncertainties_plot = all_forecast_uncertainties_full[:min_size]

ax6.scatter(forecast_errors_plot, forecast_uncertainties_plot, alpha=0.6, s=20)
ax6.set_title('Forecast Error vs Uncertainty (Full Context)')
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
sorted_forecast_errors = np.sort(all_forecast_errors_full)
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
    'all_forecast_mae_full': all_forecast_mae_full,
    'all_forecast_rmse_full': all_forecast_rmse_full,
    'all_forecast_mae_random': all_forecast_mae_random,
    'all_forecast_rmse_random': all_forecast_rmse_random,
    'all_forecast_mae_most_important': all_forecast_mae_most_important,
    'all_forecast_rmse_most_important': all_forecast_rmse_most_important,
    'all_forecast_errors_full': all_forecast_errors_full,
    'all_forecast_errors_random': all_forecast_errors_random,
    'all_forecast_errors_most_important': all_forecast_errors_most_important,
    'all_forecast_uncertainties_full': all_forecast_uncertainties_full,
    'all_forecast_uncertainties_random': all_forecast_uncertainties_random,
    'all_forecast_uncertainties_most_important': all_forecast_uncertainties_most_important,
    'per_window_importance_results': all_importance_results,
    'per_window_forecast_results': all_forecast_results,
    'total_windows': len(all_importance_results),
    'total_samples': len(all_importance_scores),
    'total_forecast_points_per_method': len(all_forecast_errors_full)
}

# Save as numpy arrays for easy loading
np.savez('sample_importance_results.npz', **results_summary)
print(f"Results saved to 'sample_importance_results.npz'")

print(f"\nAnalysis complete!")
print(f"  Processed {len(all_importance_results)} windows")
print(f"  Analyzed {len(all_importance_scores)} samples for importance")
print(f"  Generated {len(all_forecast_errors_full)} forecast points per method")
print(f"  Mean MAE (Full) across all windows: {np.mean(all_forecast_mae_full):.4f}")
print(f"  Mean MAE (Random) across all windows: {np.mean(all_forecast_mae_random):.4f}")
print(f"  Mean MAE (Most Imp) across all windows: {np.mean(all_forecast_mae_most_important):.4f}")
print(f"  Mean importance score: {np.mean(all_importance_scores):.4f}")