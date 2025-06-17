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
TEST = int(10*PDT)  # test set length: any positive integer

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

# Generate test data with FULL prediction length to get true labels
test_data_full = test_template.generate_instances(
    prediction_length=PDT,
    windows=TEST // PDT,
    distance=PDT,
)

# Generate test data for single-step predictions (for autoregressive)
test_data_single = test_template.generate_instances(
    prediction_length=1,
    windows=TEST // PDT,
    distance=PDT,
)

# Calculate actual number of windows
actual_windows = len(list(test_data_full.input))
print(f"Windows per series: {TEST // PDT}")
print(f"Expected total windows ({num_series} series × {TEST // PDT}): {num_series * (TEST // PDT)}")
print(f"Actual total windows: {actual_windows}")

# Prepare pre-trained model for single-step predictions
print("Loading model...")
if MODEL == "moirai":
    model = MoiraiForecast(
        module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-{SIZE}"),
        prediction_length=1,  # Single step for autoregressive
        context_length=CTX,
        patch_size=PSZ,
        num_samples=100,
        target_dim=1,
        feat_dynamic_real_dim=ds.num_feat_dynamic_real,
        past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
    )
elif MODEL == "moirai-moe":
    model = MoiraiMoEForecast(
        module=MoiraiMoEModule.from_pretrained(f"Salesforce/moirai-moe-1.0-R-{SIZE}"),
        prediction_length=1,
        context_length=CTX,
        patch_size=16,
        num_samples=100,
        target_dim=1,
        feat_dynamic_real_dim=ds.num_feat_dynamic_real,
        past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
    )

predictor = model.create_predictor(batch_size=1)

# Autoregressive prediction function using actual test data structure
def predict_autoregressive(predictor, base_input, prediction_length, num_samples=100):
    """
    Perform autoregressive prediction step by step using proper GluonTS data structure
    """
    # Start with the base input context
    current_target = base_input["target"].copy()
    
    predictions = []
    prediction_samples = []
    
    for step in range(prediction_length):
        # Ensure context doesn't exceed CTX length
        if len(current_target) > CTX:
            current_target = current_target[-CTX:]
        
        # Create input data structure similar to base_input but with updated target
        step_input_data = base_input.copy()
        step_input_data["target"] = current_target
        
        # Create iterable input for predictor
        step_input = [step_input_data]
        
        # Get one-step prediction
        forecast = next(iter(predictor.predict(step_input)))
        
        # Get the prediction samples and mean
        step_samples = forecast.samples[:, 0]  # Shape: [100,] - samples for this single step
        step_mean = np.median(step_samples)
        
        # Store prediction
        predictions.append(step_mean)
        prediction_samples.append(step_samples)
        
        # Append prediction mean to context for next step
        current_target = np.append(current_target, step_mean)
    
    # Stack samples to get shape [100, PDT]
    all_samples = np.column_stack(prediction_samples)
    
    return np.array(predictions), all_samples

# Generate autoregressive forecasts
print("Generating autoregressive forecasts...")
forecasts = []
autoregressive_predictions = []

# Use the full test data for getting proper inputs and labels
input_it = iter(test_data_full.input)
label_it = iter(test_data_full.label)

for inp, label in tqdm(zip(input_it, label_it), total=actual_windows, desc="Autoregressive Forecasting"):
    # Use the actual input structure from test_data_full
    # This contains proper 'target', 'start', and any other fields
    true_values = label["target"]  # Full PDT length
    
    # Perform autoregressive prediction using the actual input structure
    ar_predictions, ar_samples = predict_autoregressive(predictor, inp, PDT, num_samples=100)
    
    # Create a mock forecast object to maintain compatibility
    class MockForecast:
        def __init__(self, samples):
            self.samples = samples
    
    mock_forecast = MockForecast(ar_samples)
    forecasts.append(mock_forecast)
    autoregressive_predictions.append(ar_predictions)

print(f"Generated {len(forecasts)} autoregressive forecasts")

# Calculate evaluation metrics manually
print("Calculating evaluation metrics...")

mae_scores = []
mase_scores = []
rmse_scores = []

# Track metrics per series
series_metrics = {f'series_{i}': {'mae': [], 'mase': [], 'rmse': []} for i in range(num_series)}
window_count = 0

# Reset iterators for evaluation
input_it = iter(test_data_full.input)
label_it = iter(test_data_full.label)

for inp, label, forecast in tqdm(zip(input_it, label_it, forecasts), total=len(forecasts), desc="Evaluating"):
    # Get forecast mean (median of samples)
    forecast_mean = np.median(forecast.samples, axis=0)
    
    # Get true values (full PDT length)
    true_values = label["target"]
    
    # Get historical values for MASE calculation
    hist_values = inp["target"]
    
    # Determine which series this window belongs to
    series_idx = window_count % num_series

    # Calculate MAE
    mae = np.mean(np.abs(forecast_mean - true_values))
    mae_scores.append(mae)
    series_metrics[f'series_{series_idx}']['mae'].append(mae)
    
    # Calculate MASE (Mean Absolute Scaled Error)
    # Naive forecast: repeat the last observed value for all prediction steps
    if len(hist_values) > 0:
        last_observed_value = hist_values[-1]  # Last value in context
        naive_forecast = np.full(PDT, last_observed_value)  # Repeat for prediction_length steps
        mae_naive = np.mean(np.abs(true_values - naive_forecast))
        
        if mae_naive > 0:
            mase = mae / mae_naive
        else:
            mase = float('inf')
    else:
        mase = float('inf')
    
    mase_scores.append(mase)
    series_metrics[f'series_{series_idx}']['mase'].append(mase)
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((forecast_mean - true_values) ** 2))
    rmse_scores.append(rmse)
    series_metrics[f'series_{series_idx}']['rmse'].append(rmse)
    
    window_count += 1

# Print overall results
print(f"\nOverall Autoregressive Evaluation Results:")
print(f"Model: {MODEL}-{SIZE} (Autoregressive)")
print(f"Average MAE: {np.mean(mae_scores):.4f}")
print(f"Average MASE: {np.mean(mase_scores):.4f}")
print(f"Average RMSE: {np.mean(rmse_scores):.4f}")
print(f"Number of evaluation windows: {len(mae_scores)}")
print(f"Number of time series: {num_series}")
print(f"Windows per series: {len(mae_scores) // num_series}")
print(f"Prediction length: {PDT}")
print(f"Context length: {CTX}")

# Print per-series results
print(f"\nPer-Series Results:")
for series_name, metrics in series_metrics.items():
    if metrics['mae']:
        print(f"{series_name}: MAE={np.mean(metrics['mae']):.4f}, "
              f"MASE={np.mean(metrics['mase']):.4f}, "
              f"RMSE={np.mean(metrics['rmse']):.4f}, "
              f"Windows={len(metrics['mae'])}")

# Uncertainty Analysis
print("Analyzing autoregressive prediction uncertainty...")

# Reset iterators for uncertainty analysis
input_it = iter(test_data_full.input)
label_it = iter(test_data_full.label)

timestep_uncertainties = []
timestep_coverage = []
timestep_mae = []

# Take first few forecasts for detailed analysis
analysis_forecasts = 5
forecast_data = []

for i, (inp, label, forecast) in enumerate(zip(input_it, label_it, forecasts)):
    if i >= analysis_forecasts:
        break
    
    # Get 100 samples for this forecast (shape: [100, PDT])
    samples = forecast.samples
    true_values = label["target"]
    
    # Calculate uncertainty metrics for each time step
    timestep_std = np.std(samples, axis=0)
    timestep_iqr = np.percentile(samples, 75, axis=0) - np.percentile(samples, 25, axis=0)
    timestep_range = np.max(samples, axis=0) - np.min(samples, axis=0)
    
    # Calculate prediction intervals
    lower_bound = np.percentile(samples, 5, axis=0)
    upper_bound = np.percentile(samples, 95, axis=0)
    
    # Check coverage
    coverage = (true_values >= lower_bound) & (true_values <= upper_bound)
    
    # Calculate point-wise MAE
    forecast_mean = np.median(samples, axis=0)
    pointwise_mae = np.abs(forecast_mean - true_values)
    
    # Store data
    forecast_data.append({
        'samples': samples,
        'true_values': true_values,
        'timestep_std': timestep_std,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'coverage': coverage,
        'pointwise_mae': pointwise_mae,
        'forecast_mean': forecast_mean
    })
    
    # Aggregate
    timestep_uncertainties.append(timestep_std)
    timestep_coverage.append(coverage.astype(float))
    timestep_mae.append(pointwise_mae)

# Calculate averages
avg_uncertainty = np.mean(timestep_uncertainties, axis=0)
avg_coverage = np.mean(timestep_coverage, axis=0)
avg_mae = np.mean(timestep_mae, axis=0)

# Create visualization
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
fig.suptitle('Autoregressive Prediction Uncertainty Analysis', fontsize=16)

# Plot 1: Individual forecast samples
ax1 = axes[0, 0]
data = forecast_data[0]
time_steps = range(1, PDT + 1)

for i in range(min(20, data['samples'].shape[0])):
    ax1.plot(time_steps, data['samples'][i], color='lightgray', alpha=0.3, linewidth=0.5)

ax1.plot(time_steps, data['forecast_mean'], 'b-', label='Forecast Median', linewidth=2)
ax1.fill_between(time_steps, data['lower_bound'], data['upper_bound'], 
                alpha=0.3, color='blue', label='90% Prediction Interval')
ax1.plot(time_steps, data['true_values'], 'r-', label='True Values', linewidth=2)
ax1.set_title('Autoregressive Forecast Samples')
ax1.set_xlabel('Time Step')
ax1.set_ylabel('Value')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Uncertainty by time step
ax2 = axes[0, 1]
ax2.plot(time_steps, avg_uncertainty, 'g-', linewidth=2, marker='o')
ax2.set_title('Uncertainty Growth in Autoregressive Prediction')
ax2.set_xlabel('Time Step')
ax2.set_ylabel('Standard Deviation')
ax2.grid(True, alpha=0.3)

most_reliable_step = np.argmin(avg_uncertainty) + 1
least_reliable_step = np.argmax(avg_uncertainty) + 1
ax2.axvline(most_reliable_step, color='green', linestyle='--', alpha=0.7, 
           label=f'Most Reliable (Step {most_reliable_step})')
ax2.axvline(least_reliable_step, color='red', linestyle='--', alpha=0.7, 
           label=f'Least Reliable (Step {least_reliable_step})')
ax2.legend()

# Plot 3: Coverage rate
ax3 = axes[1, 0]
ax3.plot(time_steps, avg_coverage * 100, 'purple', linewidth=2, marker='s')
ax3.axhline(90, color='red', linestyle='--', alpha=0.7, label='Target Coverage (90%)')
ax3.set_title('Coverage Rate by Time Step')
ax3.set_xlabel('Time Step')
ax3.set_ylabel('Coverage Rate (%)')
ax3.set_ylim([0, 100])
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: MAE by time step
ax4 = axes[1, 1]
ax4.plot(time_steps, avg_mae, 'orange', linewidth=2, marker='^')
ax4.set_title('Error Growth in Autoregressive Prediction')
ax4.set_xlabel('Time Step')
ax4.set_ylabel('MAE')
ax4.grid(True, alpha=0.3)

# Plot 5: Uncertainty vs MAE
ax5 = axes[2, 0]
scatter = ax5.scatter(avg_uncertainty, avg_mae, c=time_steps, cmap='viridis', s=50)
ax5.set_xlabel('Uncertainty (Std Dev)')
ax5.set_ylabel('MAE')
ax5.set_title('Uncertainty vs Error (Autoregressive)')
plt.colorbar(scatter, ax=ax5, label='Time Step')
ax5.grid(True, alpha=0.3)

# Plot 6: Reliability ranking
ax6 = axes[2, 1]
reliability_score = avg_uncertainty + avg_mae
sorted_indices = np.argsort(reliability_score)

colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(sorted_indices)))
ax6.bar(range(len(sorted_indices)), reliability_score[sorted_indices], color=colors)
ax6.set_title('Time Step Reliability Ranking')
ax6.set_xlabel('Rank (1=Most Reliable)')
ax6.set_ylabel('Reliability Score')
ax6.set_xticks(range(0, len(sorted_indices), 5))
ax6.set_xticklabels([f'Step {sorted_indices[i]+1}' for i in range(0, len(sorted_indices), 5)], rotation=45)
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('uncertainty_autoregressive.png', dpi=300, bbox_inches='tight')
plt.show()

# Print results
print(f"\nAutoregressive Uncertainty Analysis:")
print(f"Most reliable step: {most_reliable_step} (uncertainty: {avg_uncertainty[most_reliable_step-1]:.4f})")
print(f"Least reliable step: {least_reliable_step} (uncertainty: {avg_uncertainty[least_reliable_step-1]:.4f})")
print(f"Uncertainty growth: {avg_uncertainty[-1]/avg_uncertainty[0]:.2f}x from step 1 to {PDT}")
print(f"Average coverage: {np.mean(avg_coverage)*100:.1f}%")

top_reliable = sorted_indices[:5] + 1
top_unreliable = sorted_indices[-5:] + 1
print(f"Top 5 most reliable steps: {top_reliable}")
print(f"Top 5 least reliable steps: {top_unreliable}")