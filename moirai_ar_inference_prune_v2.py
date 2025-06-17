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

def compute_sample_importance(input_data, window_id):
    """
    Compute importance of each sample in the entire context by predicting it 
    from all preceding context (leave-one-out style analysis)
    """
    full_sequence = input_data["target"]
    
    # Ensure we have exactly CTX samples for analysis
    if len(full_sequence) > CTX:
        # Take the last CTX samples
        analysis_sequence = full_sequence[-CTX:]
        print(f"Window {window_id}: Using last {CTX} samples from {len(full_sequence)} available")
    else:
        analysis_sequence = full_sequence
        print(f"Window {window_id}: Using all {len(analysis_sequence)} samples (less than CTX={CTX})")
    
    sequence_length = len(analysis_sequence)
    
    # Initialize arrays for results
    sample_importance = np.zeros(sequence_length)
    sample_uncertainties = np.zeros(sequence_length)
    sample_predictions = np.zeros(sequence_length)
    sample_true_values = np.zeros(sequence_length)
    
    print(f"Window {window_id}: Computing importance for {sequence_length} samples in context...")
    
    for pos in tqdm(range(sequence_length), desc=f"Window {window_id} - Computing sample importance"):
        # Position to predict
        true_value = analysis_sequence[pos]
        sample_true_values[pos] = true_value
        
        if pos == 0:
            # For the first position, we can't predict from preceding context
            # Use a simple baseline (mean of available data or zero)
            prediction_error = abs(true_value)  # Assume prediction is 0
            pred_uncertainty = 1.0  # High uncertainty
            pred_mean = 0.0
        else:
            # Use all preceding context to predict current position
            context_data = analysis_sequence[:pos]
            current_context_length = len(context_data)
            
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
        
        if pos < 5 or pos % 10 == 0:  # Print progress
            print(f"  Position {pos}: error={prediction_error:.4f}, uncertainty={pred_uncertainty:.4f}")
    
    return {
        'importance': sample_importance,
        'uncertainties': sample_uncertainties,
        'predictions': sample_predictions,
        'true_values': sample_true_values,
        'sequence_length': sequence_length,
        'window_id': window_id
    }

def perform_forecasting_comparison(input_data, label_data, importance_scores, window_id):
    """
    Perform forecasting with three different context selection strategies:
    1. Full context (all CTX samples)
    2. Random 75% of context
    3. Most important 75% of context (based on importance scores)
    """
    # Get the target data and ensure it's exactly CTX length
    target_data = input_data['target']
    
    # Ensure we use the same sequence as importance analysis
    if len(target_data) > CTX:
        context_target = target_data[-CTX:]
    else:
        context_target = target_data
    
    actual_context_length = len(context_target)
    reduced_ctx = max(1, int(0.75 * actual_context_length))  # 75% of actual context
    
    print(f"Window {window_id} Forecasting:")
    print(f"  Full context length: {actual_context_length}")
    print(f"  Reduced context length (75%): {reduced_ctx}")
    
    # 1. Full context forecasting
    print(f"  1. Full context forecasting...")
    forecast_input_data_full = {
        'target': context_target,
        'start': input_data['start'],
        'item_id': input_data.get('item_id', 0)
    }
    
    predictor_full = create_model_with_context_length(actual_context_length)
    forecast_it_full = predictor_full.predict([forecast_input_data_full])
    forecast_full = next(forecast_it_full)
    
    forecast_samples_full = forecast_full.samples
    forecast_mean_full = np.median(forecast_samples_full, axis=0)
    forecast_uncertainty_full = np.std(forecast_samples_full, axis=0)
    forecast_quantiles_full = np.percentile(forecast_samples_full, [10, 25, 50, 75, 90], axis=0)
    
    # 2. Random 75% context forecasting
    print(f"  2. Random 75% context forecasting...")
    np.random.seed(42 + window_id)  # Reproducible random selection
    random_indices = np.sort(np.random.choice(actual_context_length, reduced_ctx, replace=False))
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
    
    # 3. Most important 75% context forecasting
    print(f"  3. Most important 75% context forecasting...")
    
    # Ensure importance scores match the context length
    if len(importance_scores) == actual_context_length:
        context_importance = importance_scores
    elif len(importance_scores) > actual_context_length:
        # Take the last actual_context_length importance scores
        context_importance = importance_scores[-actual_context_length:]
    else:
        # Pad with mean importance if needed
        mean_importance = np.mean(importance_scores) if len(importance_scores) > 0 else 1.0
        context_importance = np.concatenate([
            np.full(actual_context_length - len(importance_scores), mean_importance),
            importance_scores
        ])
    
    # Select the most important samples (top 75% by importance score)
    most_important_indices = np.argsort(context_importance)[-reduced_ctx:]
    most_important_indices = np.sort(most_important_indices)  # Maintain temporal order
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
    
    # Get true values for comparison
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
    
    print(f"  Results:")
    print(f"    Full Context (len={actual_context_length})  - MAE: {mae_full:.4f}, RMSE: {rmse_full:.4f}")
    print(f"    Random 75% (len={reduced_ctx})             - MAE: {mae_random:.4f}, RMSE: {rmse_random:.4f}")
    print(f"    Most Imp 75% (len={reduced_ctx})           - MAE: {mae_most_important:.4f}, RMSE: {rmse_most_important:.4f}")
    
    return {
        'window_id': window_id,
        'actual_context_length': actual_context_length,
        'reduced_ctx': reduced_ctx,
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
        'random_indices': random_indices,
        'most_important_indices': most_important_indices
    }

# Main analysis loop
print("\nAnalyzing sample importance and forecasting across all windows...")

all_importance_results = []
all_forecast_results = []
input_it = iter(test_data_full.input)
label_it = iter(test_data_full.label)

for window_idx, (input_data, label_data) in enumerate(zip(input_it, label_it)):
    if window_idx >= actual_windows:
        break
    
    print(f"\n{'='*80}")
    print(f"Processing Window {window_idx + 1}/{actual_windows}")
    print(f"{'='*80}")
    
    # 1. Sample importance analysis
    print(f"1. Computing sample importance for entire context...")
    importance_result = compute_sample_importance(input_data, window_idx + 1)
    all_importance_results.append(importance_result)
    
    # 2. Forecasting comparison using different context selection strategies
    print(f"2. Performing forecasting comparison...")
    forecast_result = perform_forecasting_comparison(
        input_data, label_data, importance_result['importance'], window_idx + 1
    )
    all_forecast_results.append(forecast_result)
    
    # Print window summary
    print(f"\nWindow {window_idx + 1} Summary:")
    print(f"  Sample Importance:")
    print(f"    Samples analyzed: {importance_result['sequence_length']}")
    print(f"    Mean importance: {np.mean(importance_result['importance']):.4f}")
    print(f"    Max importance: {np.max(importance_result['importance']):.4f}")
    print(f"    Min importance: {np.min(importance_result['importance']):.4f}")
    print(f"  Forecasting Performance:")
    print(f"    Full vs Random: {((forecast_result['mae_random'] / forecast_result['mae_full']) - 1) * 100:+.2f}% MAE change")
    print(f"    Full vs Most Imp: {((forecast_result['mae_most_important'] / forecast_result['mae_full']) - 1) * 100:+.2f}% MAE change")
    print(f"    Random vs Most Imp: {((forecast_result['mae_most_important'] / forecast_result['mae_random']) - 1) * 100:+.2f}% MAE change")

# Aggregate analysis across all windows
print(f"\n{'='*80}")
print("AGGREGATE ANALYSIS ACROSS ALL WINDOWS")
print(f"{'='*80}")

# Combine all importance scores
all_importance_scores = np.concatenate([result['importance'] for result in all_importance_results])
all_uncertainty_scores = np.concatenate([result['uncertainties'] for result in all_importance_results])

# Combine all forecasting scores
all_forecast_mae_full = np.array([result['mae_full'] for result in all_forecast_results])
all_forecast_rmse_full = np.array([result['rmse_full'] for result in all_forecast_results])
all_forecast_errors_full = np.concatenate([result['forecast_errors_full'] for result in all_forecast_results])

all_forecast_mae_random = np.array([result['mae_random'] for result in all_forecast_results])
all_forecast_rmse_random = np.array([result['rmse_random'] for result in all_forecast_results])
all_forecast_errors_random = np.concatenate([result['forecast_errors_random'] for result in all_forecast_results])

all_forecast_mae_most_important = np.array([result['mae_most_important'] for result in all_forecast_results])
all_forecast_rmse_most_important = np.array([result['rmse_most_important'] for result in all_forecast_results])
all_forecast_errors_most_important = np.concatenate([result['forecast_errors_most_important'] for result in all_forecast_results])

print(f"Total samples analyzed (importance): {len(all_importance_scores)}")
print(f"Total windows processed: {len(all_importance_results)}")
print(f"Total forecast points per method: {len(all_forecast_errors_full)}")

# Overall statistics
print(f"\nOverall Forecasting Statistics:")
print(f"Full Context:")
print(f"  Mean MAE: {np.mean(all_forecast_mae_full):.4f}")
print(f"  Mean RMSE: {np.mean(all_forecast_rmse_full):.4f}")

print(f"Random 75% Context:")
print(f"  Mean MAE: {np.mean(all_forecast_mae_random):.4f}")
print(f"  Mean RMSE: {np.mean(all_forecast_rmse_random):.4f}")

print(f"Most Important 75% Context:")
print(f"  Mean MAE: {np.mean(all_forecast_mae_most_important):.4f}")
print(f"  Mean RMSE: {np.mean(all_forecast_rmse_most_important):.4f}")

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
    samples = imp_result['sequence_length']
    print(f"W{i+1:<7d} {mean_imp:<10.4f} {std_imp:<10.4f} {max_imp:<10.4f} {mae_full:<10.4f} {mae_random:<10.4f} {mae_most_important:<10.4f} {samples:<8d}")

# Visualization
fig, axes = plt.subplots(3, 3, figsize=(18, 18))
fig.suptitle('Sample Importance Analysis and Forecasting Results (Full Context)', fontsize=16)

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
ax2.set_title('Distribution of Forecast Errors')
ax2.set_xlabel('Forecast Error (MAE)')
ax2.set_ylabel('Frequency')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Importance vs Uncertainty
ax3 = axes[0, 2]
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

# Plot 4: MAE comparison by method
ax4 = axes[1, 0]
methods = ['Full', 'Random 75%', 'Most Imp 75%']
mae_means = [np.mean(all_forecast_mae_full), np.mean(all_forecast_mae_random), np.mean(all_forecast_mae_most_important)]
mae_stds = [np.std(all_forecast_mae_full), np.std(all_forecast_mae_random), np.std(all_forecast_mae_most_important)]
bars = ax4.bar(methods, mae_means, yerr=mae_stds, capsize=5, color=['blue', 'green', 'red'], alpha=0.7)
ax4.set_title('Mean MAE by Method')
ax4.set_ylabel('MAE')
ax4.grid(True, alpha=0.3)

# Plot 5: Box plot of importance by window
ax5 = axes[1, 1]
window_importance_data = [result['importance'] for result in all_importance_results]
ax5.boxplot(window_importance_data, tick_labels=[f'W{i+1}' for i in range(len(all_importance_results))])
ax5.set_title('Importance Distribution by Window')
ax5.set_xlabel('Window')
ax5.set_ylabel('Importance Score')
ax5.tick_params(axis='x', rotation=45)
ax5.grid(True, alpha=0.3)

# Plot 6: MAE by window for each method
ax6 = axes[1, 2]
window_numbers = range(1, len(all_forecast_results) + 1)
ax6.plot(window_numbers, all_forecast_mae_full, 'o-', label='Full Context', color='blue')
ax6.plot(window_numbers, all_forecast_mae_random, 's-', label='Random 75%', color='green')
ax6.plot(window_numbers, all_forecast_mae_most_important, '^-', label='Most Imp 75%', color='red')
ax6.set_title('MAE by Window')
ax6.set_xlabel('Window')
ax6.set_ylabel('MAE')
ax6.legend()
ax6.grid(True, alpha=0.3)

# Plot 7: Cumulative distribution of importance
ax7 = axes[2, 0]
sorted_importance = np.sort(all_importance_scores)
cumulative_imp = np.arange(1, len(sorted_importance) + 1) / len(sorted_importance) * 100
ax7.plot(sorted_importance, cumulative_imp, linewidth=2, color='red')
ax7.set_title('Cumulative Distribution of Importance')
ax7.set_xlabel('Importance Score')
ax7.set_ylabel('Cumulative Percentage')
ax7.grid(True, alpha=0.3)

# Plot 8: Importance score position analysis
ax8 = axes[2, 1]
# Calculate mean importance by position across all windows
max_length = max(result['sequence_length'] for result in all_importance_results)
position_importance = np.zeros(max_length)
position_counts = np.zeros(max_length)

for result in all_importance_results:
    seq_len = result['sequence_length']
    for pos in range(seq_len):
        position_importance[pos] += result['importance'][pos]
        position_counts[pos] += 1

# Calculate mean importance per position
mean_position_importance = np.divide(position_importance, position_counts, 
                                   out=np.zeros_like(position_importance), where=position_counts!=0)

valid_positions = position_counts > 0
ax8.plot(np.arange(max_length)[valid_positions], mean_position_importance[valid_positions], 'o-')
ax8.set_title('Mean Importance by Position in Context')
ax8.set_xlabel('Position in Context')
ax8.set_ylabel('Mean Importance Score')
ax8.grid(True, alpha=0.3)

# Plot 9: Performance improvement analysis
ax9 = axes[2, 2]
improvement_random = ((all_forecast_mae_full - all_forecast_mae_random) / all_forecast_mae_full) * 100
improvement_most_important = ((all_forecast_mae_full - all_forecast_mae_most_important) / all_forecast_mae_full) * 100

ax9.scatter(improvement_random, improvement_most_important, alpha=0.7, s=50)
ax9.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax9.axvline(x=0, color='black', linestyle='--', alpha=0.5)
ax9.set_title('Pruning Performance Comparison')
ax9.set_xlabel('Random 75% Improvement (%)')
ax9.set_ylabel('Most Important 75% Improvement (%)')
ax9.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sample_importance_full_context_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Save results
print(f"\nSaving results...")
results_summary = {
    'all_importance_scores': all_importance_scores,
    'all_uncertainty_scores': all_uncertainty_scores,
    'all_forecast_mae_full': all_forecast_mae_full,
    'all_forecast_rmse_full': all_forecast_rmse_full,
    'all_forecast_mae_random': all_forecast_mae_random,
    'all_forecast_rmse_random': all_forecast_rmse_random,
    'all_forecast_mae_most_important': all_forecast_mae_most_important,
    'all_forecast_rmse_most_important': all_forecast_rmse_most_important,
    'all_forecast_errors_full': all_forecast_errors_full,
    'all_forecast_errors_random': all_forecast_errors_random,
    'all_forecast_errors_most_important': all_forecast_errors_most_important,
    'per_window_importance_results': all_importance_results,
    'per_window_forecast_results': all_forecast_results,
    'total_windows': len(all_importance_results),
    'total_samples': len(all_importance_scores)
}

np.savez('sample_importance_full_context_results.npz', **results_summary)
print(f"Results saved to 'sample_importance_full_context_results.npz'")

print(f"\nAnalysis complete!")
print(f"  Processed {len(all_importance_results)} windows")
print(f"  Analyzed {len(all_importance_scores)} samples for importance")
print(f"  Mean MAE (Full): {np.mean(all_forecast_mae_full):.4f}")
print(f"  Mean MAE (Random 75%): {np.mean(all_forecast_mae_random):.4f}")
print(f"  Mean MAE (Most Important 75%): {np.mean(all_forecast_mae_most_important):.4f}")
print(f"  Mean importance score: {np.mean(all_importance_scores):.4f}")