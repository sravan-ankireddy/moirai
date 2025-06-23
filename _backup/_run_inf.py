# Simple Moirai Inference and Visualization
# Load model, run inference on full context, and visualize results
# This script demonstrates basic forecasting with the Moirai model

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from huggingface_hub import hf_hub_download
import warnings
warnings.filterwarnings('ignore')

from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

# Configuration
MODEL = "moirai"  # or "moirai-moe"
SIZE = "large"    # small, base, large
CTX = 1000          # Context length
PDT = 8           # Prediction length
BSZ = 32          # Batch size
GPU = 0           # GPU device
PSZ = "auto"

# Data configuration
CSV_PATH = "/home/sa53869/time-series/moirai/time-moe-eval/synthetic_sinusoidal.csv"
COLUMN = 1        # Column to analyze (0-indexed)
TEST_LENGTH = 200 # Test set length

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
print(f"Data shape: {df_selected.shape}")
print(f"Data preview:")
print(df_selected.head())

# Create results directory
results_dir = "results_inf"
os.makedirs(results_dir, exist_ok=True)
print(f"Results will be saved to: {results_dir}")

# Create GluonTS dataset
ds = PandasDataset(dict(df_selected))
train, test_template = split(ds, offset=-TEST_LENGTH)

# Generate test instances
test_data = test_template.generate_instances(
    prediction_length=PDT,
    windows=TEST_LENGTH // PDT,
    distance=PDT,
)

print(f"Number of test windows: {TEST_LENGTH // PDT}")

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


# Create predictor
predictor = model.create_predictor(batch_size=BSZ)
print("Model loaded successfully!")

# Run inference on test data
print("Running inference...")
input_data = list(test_data.input)
label_data = list(test_data.label)

# Run predictions
forecasts = list(predictor.predict(input_data))

print(f"Generated {len(forecasts)} forecasts")

# Prepare data for visualization
sample_results = []
full_data_values = df_selected[selected_column].values

for i, (input_item, label_item, forecast) in enumerate(zip(input_data, label_data, forecasts)):
    # Get context data
    context = input_item['target']
    
    # keep the last `CTX` values for context
    if len(context) > CTX:
        context = context[-CTX:]

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
        'mae': np.mean(np.abs(prediction - ground_truth))
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
    
    plt.plot(context_indices, result['context'], label='Context', color='blue', linewidth=2)
    plt.plot(forecast_indices, result['ground_truth'], label='Ground Truth', color='green', 
             linewidth=3, marker='o', markersize=4)
    plt.plot(forecast_indices, result['prediction'], label='Prediction', color='red', 
             linewidth=2, linestyle='--', marker='s', markersize=4)
    plt.axvline(x=0, color='black', linestyle=':', alpha=0.7, label='Forecast Start')
    plt.title(f"Full Context (len={context_len}) - Sample {result['window_id']} - MAE: {result['mae']:.4f}")
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
