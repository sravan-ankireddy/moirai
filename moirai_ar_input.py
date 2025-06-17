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
BSZ = 32  # batch size: use 1 for autoregressive
TEST = int(2*PDT)  # test set length: any positive integer

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

# NEW: Analyze sample importance for ALL windows
print("\nAnalyzing sample importance across all windows...")

all_importance_results = []
input_it = iter(test_data_full.input)

for window_idx, input_data in enumerate(input_it):
    if window_idx >= actual_windows:
        break
    
    print(f"\n{'='*60}")
    print(f"Processing Window {window_idx + 1}/{actual_windows}")
    print(f"{'='*60}")
    
    # Analyze up to 15 samples per window to keep it manageable
    importance_result = compute_sample_importance(input_data, window_idx + 1, min_context=PDT, max_analysis=CTX)
    all_importance_results.append(importance_result)
    
    # Print summary for this window
    print(f"\nWindow {window_idx + 1} Summary:")
    print(f"  Samples analyzed: {importance_result['analyzed_length']}")
    print(f"  Mean importance: {np.mean(importance_result['importance']):.4f}")
    print(f"  Max importance: {np.max(importance_result['importance']):.4f}")
    print(f"  Min importance: {np.min(importance_result['importance']):.4f}")

# NEW: Aggregate analysis across all windows
print(f"\n{'='*60}")
print("AGGREGATE ANALYSIS ACROSS ALL WINDOWS")
print(f"{'='*60}")

# Combine all importance scores
all_importance_scores = np.concatenate([result['importance'] for result in all_importance_results])
all_uncertainty_scores = np.concatenate([result['uncertainties'] for result in all_importance_results])
all_context_lengths = np.concatenate([result['context_lengths'] for result in all_importance_results])

print(f"Total samples analyzed: {len(all_importance_scores)}")
print(f"Total windows processed: {len(all_importance_results)}")

# Overall statistics
print(f"\nOverall Importance Statistics:")
print(f"Mean importance: {np.mean(all_importance_scores):.4f}")
print(f"Std importance: {np.std(all_importance_scores):.4f}")
print(f"Max importance: {np.max(all_importance_scores):.4f}")
print(f"Min importance: {np.min(all_importance_scores):.4f}")

# Per-window statistics
print(f"\nPer-Window Statistics:")
for i, result in enumerate(all_importance_results):
    mean_imp = np.mean(result['importance'])
    std_imp = np.std(result['importance'])
    max_imp = np.max(result['importance'])
    print(f"Window {i+1:2d}: Mean={mean_imp:.4f}, Std={std_imp:.4f}, Max={max_imp:.4f}, Samples={result['analyzed_length']}")

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

# NEW: Visualize aggregate results
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Sample Importance Analysis Across All Windows', fontsize=16)

# Plot 1: Distribution of importance scores
ax1 = axes[0, 0]
ax1.hist(all_importance_scores, bins=50, alpha=0.7, color='red', edgecolor='black')
ax1.set_title('Distribution of Importance Scores')
ax1.set_xlabel('Importance Score')
ax1.set_ylabel('Frequency')
ax1.grid(True, alpha=0.3)

# Plot 2: Importance vs Context Length
ax2 = axes[0, 1]
scatter = ax2.scatter(all_context_lengths, all_importance_scores, alpha=0.6, s=20)
ax2.set_title('Importance vs Context Length')
ax2.set_xlabel('Context Length')
ax2.set_ylabel('Importance Score')
ax2.grid(True, alpha=0.3)

# Plot 3: Uncertainty vs Importance
ax3 = axes[0, 2]
ax3.scatter(all_importance_scores, all_uncertainty_scores, alpha=0.6, s=20)
ax3.set_title('Uncertainty vs Importance (All Windows)')
ax3.set_xlabel('Importance Score')
ax3.set_ylabel('Uncertainty')
ax3.grid(True, alpha=0.3)

# Plot 4: Box plot of importance by window
ax4 = axes[1, 0]
window_importance_data = [result['importance'] for result in all_importance_results]
ax4.boxplot(window_importance_data, labels=[f'W{i+1}' for i in range(len(all_importance_results))])
ax4.set_title('Importance Distribution by Window')
ax4.set_xlabel('Window')
ax4.set_ylabel('Importance Score')
ax4.tick_params(axis='x', rotation=45)
ax4.grid(True, alpha=0.3)

# Plot 5: Mean importance by context length
ax5 = axes[1, 1]
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

ax5.errorbar(unique_context_lengths, ctx_means, yerr=ctx_stds, marker='o', capsize=5)
ax5.set_title('Mean Importance by Context Length')
ax5.set_xlabel('Context Length')
ax5.set_ylabel('Mean Importance Score')
ax5.grid(True, alpha=0.3)

# Plot 6: Cumulative distribution
ax6 = axes[1, 2]
sorted_scores = np.sort(all_importance_scores)
cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores) * 100
ax6.plot(sorted_scores, cumulative, linewidth=2)
ax6.set_title('Cumulative Distribution of Importance')
ax6.set_xlabel('Importance Score')
ax6.set_ylabel('Cumulative Percentage')
ax6.grid(True, alpha=0.3)

# Add compression thresholds
for threshold in [20, 50, 80]:
    threshold_value = np.percentile(all_importance_scores, threshold)
    ax6.axvline(threshold_value, color='red', linestyle='--', alpha=0.7, 
               label=f'{threshold}th percentile')
ax6.legend()

plt.tight_layout()
plt.savefig('sample_importance_all_windows.png', dpi=300, bbox_inches='tight')
plt.show()

# NEW: Export results for further analysis
print(f"\nSaving results...")
results_summary = {
    'all_importance_scores': all_importance_scores,
    'all_uncertainty_scores': all_uncertainty_scores,
    'all_context_lengths': all_context_lengths,
    'per_window_results': all_importance_results,
    'total_windows': len(all_importance_results),
    'total_samples': len(all_importance_scores)
}

# Save as numpy arrays for easy loading
np.savez('sample_importance_results.npz', **results_summary)
print(f"Results saved to 'sample_importance_results.npz'")

print(f"\nAnalysis complete! Processed {len(all_importance_results)} windows with {len(all_importance_scores)} total samples.")