# Simple Moirai Inference and Visualization
# Load model, run inference on full context, and visualize results
# This script demonstrates basic forecasting with the Moirai model

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from huggingface_hub import hf_hub_download
import warnings
warnings.filterwarnings('ignore')

from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from scipy import stats

# Configuration
MODEL = "moirai"  # or "moirai-moe"
SIZE = "large"    # small, base, large
CTX = 128          # Context length
PDT = 8           # Prediction length
BSZ = 128         # Batch size
GPU = 1           # GPU device
PSZ = "auto"

# Data configuration
CSV_PATH = "/home/sa53869/time-series/moirai/time-moe-eval/synthetic_sinusoidal.csv"
COLUMN = 0        # Column to analyze (0-indexed)
TEST_LENGTH = 10  # Test set length

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)

print(f"Configuration:")
print(f"  Model: {MODEL}-{SIZE}")
print(f"  Context Length: {CTX}")
print(f"  Prediction Length: {PDT}")
print(f"  Test Length: {TEST_LENGTH}")
print(f"  Using GPU: {GPU}")

# Load and prepare data
print("Loading data...")
df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
dataset_name = os.path.splitext(os.path.basename(CSV_PATH))[0]

# Select column
available_columns = df.columns.tolist()
selected_column = available_columns[COLUMN]
df_selected = df[[selected_column]].copy()

print(f"Dataset: {dataset_name}")
print(f"Selected column: {selected_column}")
# print(f"Data shape: {df_selected.shape}")
# print(f"Data preview:")
# print(df_selected.head())

# Create results directory
results_dir = f"results_inf_ar/{dataset_name}/{MODEL}-{SIZE}/{CTX}-{PDT}"
os.makedirs(results_dir, exist_ok=True)
print(f"Results will be saved to: {results_dir}")

# Create GluonTS dataset
ds = PandasDataset(dict(df_selected))
train, test_template = split(ds, offset=-TEST_LENGTH*PDT)

# Generate test instances
test_data = test_template.generate_instances(
    prediction_length=PDT,
    windows=TEST_LENGTH,
    distance=PDT,
)

print(f"Number of test windows: {TEST_LENGTH}")

# Load Moirai model
print("Loading Moirai model...")
base_module = MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-{SIZE}")

# Create model with specific configuration
model = MoiraiForecast(
    module=base_module,
    prediction_length=PDT,
    context_length=CTX,
    patch_size=PSZ,
    num_samples=100,  # Number of samples for probabilistic forecasting
    target_dim=1,
    feat_dynamic_real_dim=ds.num_feat_dynamic_real,
    past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
)

# Create model for autoregressive inference on input data
model_ar = MoiraiForecast(
    module=base_module,
    prediction_length=1,
    context_length=CTX,
    patch_size=PSZ,
    num_samples=100,  # Number of samples for probabilistic forecasting
    target_dim=1,
    feat_dynamic_real_dim=ds.num_feat_dynamic_real,
    past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
)

# Create predictor
predictor = model.create_predictor(batch_size=BSZ)
predictor_ar = model_ar.create_predictor(batch_size=BSZ)
print("Model loaded successfully!")

# Run inference on test data
print("Running inference...")
input_data = list(test_data.input)
label_data = list(test_data.label)

# Run predictions
print("Running main predictions...")
forecasts = list(tqdm(predictor.predict(input_data), 
                     total=len(input_data), 
                     desc="Main forecasts"))

print(f"Generated {len(forecasts)} forecasts")

# Prepare data for visualization
sample_results = []
full_data_values = df_selected[selected_column].values

print("Processing samples...")
for i, (input_item, label_item, forecast) in enumerate(tqdm(zip(input_data, label_data, forecasts), 
                                                           total=len(input_data), 
                                                           desc="Processing samples")):
    # Get context data
    context = input_item['target']

    ## Fix me: not changing the start time index
    input_data_context = []
    context_range = range(len(context) - 2*CTX, len(context) - CTX)
    for j in tqdm(context_range, desc=f"AR windows (sample {i+1})", leave=False):
        # Create a new input item for autoregressive prediction
        ar_item = input_item.copy()

        # Extract CTX-length window starting from position j
        ar_item['target'] = context[j:j + CTX]
        input_data_context.append(ar_item)

    # Run autoregressive predictions with progress tracking
    forecasts_ar = list(tqdm(predictor_ar.predict(input_data_context), 
                            total=len(input_data_context),
                            desc=f"AR forecasts (sample {i+1})", 
                            leave=False))
    
    # keep the last `CTX` values for context, which will be used for predicting forecast
    if len(context) > CTX:
        context = context[-CTX:]

    # compute the uncertainity of input context using the ratio of std to mean in the forecasts_ar
    context_samples = np.array([f.samples for f in forecasts_ar])
    context_mean = np.mean(context_samples, axis=1)
    context_std = np.std(context_samples, axis=1)
    context_uncertainty = context_std / (np.abs(context_mean) + 1e-8)

    # # Basic statistics
    # context_mean = np.mean(context_samples, axis=1)
    # context_std = np.std(context_samples, axis=1, ddof=1)  # Use sample std
    
    # # Improved coefficient of variation (your current approach, but better)
    # # Handle near-zero means more robustly
    # relative_uncertainty = np.where(
    #     np.abs(context_mean) > 1e-6,
    #     context_std / np.abs(context_mean),
    #     context_std  # For near-zero means, just use absolute uncertainty
    # )
    
    # # Prediction interval width (95% confidence)
    # pred_intervals_95 = np.percentile(context_samples, [2.5, 97.5], axis=1)
    # interval_width = pred_intervals_95[1] - pred_intervals_95[0]
    
    # # Entropy-based uncertainty
    # def sample_entropy(samples):
    #     # Use KDE-based entropy estimate for better accuracy
    #     try:
    #         kde = stats.gaussian_kde(samples)
    #         # Sample points for entropy calculation
    #         x_range = np.linspace(samples.min(), samples.max(), 100)
    #         density = kde(x_range)
    #         # Avoid log(0)
    #         density = np.maximum(density, 1e-12)
    #         entropy = -np.trapz(density * np.log(density), x_range)
    #         return entropy
    #     except:
    #         # Fallback to histogram method
    #         hist, bin_edges = np.histogram(samples, bins='auto', density=True)
    #         hist = hist[hist > 0]
    #         if len(hist) == 0:
    #             return 0
    #         dx = bin_edges[1] - bin_edges[0]
    #         return -np.sum(hist * np.log(hist)) * dx
    
    # entropy = np.array([sample_entropy(context_samples[i]) for i in range(context_samples.shape[0])])

    # # context_uncertainty = relative_uncertainty

    # Get ground truth
    ground_truth = label_item['target'][:PDT]
    
    # Get prediction (mean of samples)
    prediction = np.mean(forecast.samples, axis=0)
    
    # Store results
    sample_results.append({
        'window_id': i,
        'context': context,
        'ground_truth': ground_truth,
        'prediction': prediction,
        'mae': np.mean(np.abs(prediction - ground_truth)),
        'context_uncertainty': context_uncertainty,
        'context_ar_predictions': context_mean.flatten()  # Autoregressive predictions of context
    })

print(f"Processed {len(sample_results)} samples")


# plot the results. for 3 random samples, plot the context, ground truth and prediction
num_samples = 3
sample_indices = np.random.choice(len(sample_results), num_samples, replace=False)
for plot_idx, idx in enumerate(sample_indices, 1):  # Start from 1 for filenames
    result = sample_results[idx]
    
    plt.figure(figsize=(15, 6))
    
    # Full context
    # Create proper time indices
    context_len = len(result['context'])
    context_indices = np.arange(-context_len, 0)  # Context before 0
    forecast_indices = np.arange(0, PDT)  # Forecast starts at 0
    
    plt.plot(context_indices, result['context'], label='Context (Actual)', color='blue', linewidth=2, linestyle='-.')
    
    # Plot autoregressive predictions of context
    ar_pred_indices = np.arange(-len(result['context_ar_predictions']), 0)
    plt.plot(ar_pred_indices, result['context_ar_predictions'], label='Context (AR Predicted)', 
             color='orange', linewidth=2, linestyle='-.', marker='x', markersize=3, alpha=0.8)
    
    plt.plot(forecast_indices, result['ground_truth'], label='Ground Truth', color='green', 
             linewidth=3, marker='o', markersize=4, linestyle='-.')
    plt.plot(forecast_indices, result['prediction'], label='Prediction', color='red', 
             linewidth=2, linestyle='-.', marker='s', markersize=4)
    plt.axvline(x=0, color='black', linestyle=':', alpha=0.7, label='Forecast Start')
    plt.title(f"Full Context (len={context_len}) - Sample {result['window_id']} - MAE: {result['mae']:.4f}", loc='left')
    plt.title(f"Patch Size: {PSZ}", loc='right', color='red')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plot_filename = os.path.join(results_dir, f"sample_{plot_idx}_forecast.png")
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved plot: {plot_filename}")

# Create dual-axis plots showing context data and uncertainty
print("\nCreating uncertainty plots...")
for plot_idx, idx in enumerate(sample_indices, 1):  # Use same sample indices
    result = sample_results[idx]
    
    fig, ax1 = plt.subplots(figsize=(15, 8))
    
    # Left y-axis: Context data
    context_len = len(result['context'])
    context_indices = np.arange(-context_len, 0)  # Context before 0
    forecast_indices = np.arange(0, PDT)  # Forecast starts at 0
    
    # Plot context data on left y-axis
    color1 = 'tab:blue'
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Context Value', color=color1)
    line1 = ax1.plot(context_indices, result['context'], color=color1, linewidth=2, 
                     label='Context (Actual)', marker='o', markersize=3, linestyle='-.')
    
    # Plot autoregressive predictions of context
    ar_pred_indices = np.arange(-len(result['context_ar_predictions']), 0)
    ax1.plot(ar_pred_indices, result['context_ar_predictions'], color='orange', linewidth=2, 
             label='Context (AR Predicted)', marker='x', markersize=3, linestyle='-.', alpha=0.8)
    
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Plot forecast data on left y-axis for reference
    ax1.plot(forecast_indices, result['ground_truth'], color='green', 
             linewidth=2, marker='o', markersize=4, alpha=0.7, label='Ground Truth', linestyle='-.')
    ax1.plot(forecast_indices, result['prediction'], color='red', 
             linewidth=2, linestyle='-.', marker='s', markersize=4, alpha=0.7, label='Prediction')
    
    # Right y-axis: Uncertainty
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Uncertainty (CV)', color=color2)
    
    # Plot uncertainty map on right y-axis
    uncertainty_indices = np.arange(-len(result['context_uncertainty']), 0)
    line2 = ax2.plot(uncertainty_indices, result['context_uncertainty'], color=color2, 
                     linewidth=3, alpha=0.8, label='Context Uncertainty', marker='s', markersize=2, linestyle='-.')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Add vertical line at forecast start
    ax1.axvline(x=0, color='black', linestyle=':', alpha=0.7, linewidth=2, label='Forecast Start')
    
    # Title and legends
    fig.suptitle(f"Context Data & Uncertainty Map - Sample {result['window_id']} - MAE: {result['mae']:.4f}", 
                 fontsize=14, x=0.1, ha='left')
    fig.suptitle(f"Patch Size: {PSZ}", fontsize=14, x=0.9, ha='right', color='red')
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Save uncertainty plot
    uncertainty_filename = os.path.join(results_dir, f"sample_{plot_idx}_uncertainty.png")
    plt.tight_layout()
    plt.savefig(uncertainty_filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved uncertainty plot: {uncertainty_filename}")
