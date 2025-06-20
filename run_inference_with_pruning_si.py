# Moirai Inference and Visualization

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

# Configuration
MODEL = "moirai"  # or "moirai-moe"
SIZE = "large"    # small, base, large
CTX = 2048          # Context length
PDT = 64           # Prediction length
BSZ = 32          # Batch size
GPU = 2           # GPU device
PSZ = "auto"
PSZ_surprisal = "auto"
compression_ratio = 1/2 # 1/2, 1/4, 1/8, etc.

# Control flags
ENABLE_SURPRISAL = False  # Set to False to skip slow self-information computation and surprisal-based methods

# Data configuration
HOME = os.path.expanduser("~")
DATASET_FOLDER = f"{HOME}/time-series/moirai/time-moe-eval/"
MODEL_FOLDER = "Salesforce"

# CSV_PATH = f"{DATASET_FOLDER}/ETT-small/ETTm1.csv"
CSV_PATH = f"{DATASET_FOLDER}/synthetic_sinusoidal.csv"
# CSV_PATH = f"{DATASET_FOLDER}/electricity.csv"

COLUMN = 0        # Column to analyze (0-indexed)

# Load Moirai model
print("Loading Moirai model...")
base_module = MoiraiModule.from_pretrained(f"{MODEL_FOLDER}/{MODEL}-1.0-R-{SIZE}")

# Test configuration
NUM_WINDOWS = 100  # Test set length
TEST_SAMPLES = int(NUM_WINDOWS * PDT)  # Number of test samples
NUM_SAMPLES = 1000  # Number of samples for probabilistic forecasting

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)

print(f"Configuration:")
print(f"  Model: {MODEL}-{SIZE}")
print(f"  Context Length: {CTX}")
print(f"  Prediction Length: {PDT}")
print(f"  Test Length: {NUM_WINDOWS}")
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
results_dir = f"results_prune_si_v2/{dataset_name}_COL_{COLUMN}/{MODEL}-{SIZE}/CTX{CTX}_PDT{PDT}_PSZ{PSZ}_COMP{compression_ratio}/N_{NUM_SAMPLES}"
os.makedirs(results_dir, exist_ok=True)
print(f"Results will be saved to: {results_dir}")

# Create GluonTS dataset
ds = PandasDataset(dict(df_selected))
train, test_template = split(ds, offset=-TEST_SAMPLES)

# Generate test instances
test_data = test_template.generate_instances(
    prediction_length=PDT,
    windows=NUM_WINDOWS,
    distance=PDT,
)

print(f"Number of test windows: {NUM_WINDOWS}")


# Create model with specific configuration
model = MoiraiForecast(
    module=base_module,
    prediction_length=PDT,
    context_length=CTX,
    patch_size=PSZ,
    num_samples=NUM_SAMPLES,  # Number of samples for probabilistic forecasting
    target_dim=1,
    feat_dynamic_real_dim=ds.num_feat_dynamic_real,
    past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
)


# Create model with reduced context length AND reduced prediction length
reduced_ctx = int(compression_ratio * CTX)
reduced_pdt = max(1, int(compression_ratio * PDT))  # Ensure at least 1 prediction step
model_reduced_ctx_pdt = MoiraiForecast(
    module=base_module,
    prediction_length=reduced_pdt,
    context_length=reduced_ctx,
    patch_size=PSZ,
    num_samples=NUM_SAMPLES,
    target_dim=1,
    feat_dynamic_real_dim=ds.num_feat_dynamic_real,
    past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
)

# Create model with only reduced context length
model_reduced_ctx = MoiraiForecast(
    module=base_module,
    prediction_length=PDT,
    context_length=reduced_ctx,
    patch_size=PSZ,
    num_samples=NUM_SAMPLES,
    target_dim=1,
    feat_dynamic_real_dim=ds.num_feat_dynamic_real,
    past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
)

# Create predictor
predictor = model.create_predictor(batch_size=BSZ)
# Create predictor with reduced context length
predictor_reduced_ctx_pdt = model_reduced_ctx_pdt.create_predictor(batch_size=BSZ)
# Create predictor with only reduced context length
predictor_reduced_ctx = model_reduced_ctx.create_predictor(batch_size=BSZ)
print("Model loaded successfully!")
print(f"Reduced model config: context={reduced_ctx}, prediction={reduced_pdt}")

# Run inference on test data
print("Running inference...")
input_data = list(test_data.input)
label_data = list(test_data.label)

# Create a dataset for the downsampling experiment: downsample the input but not the labels
print("\nPreparing downsampled context data...")
input_data_downsampled = []
for item in tqdm(input_data, desc="Processing downsampled context"):
    # Create a deep copy of the item to avoid modifying original
    downsampled_item = item.copy()
    
    # Replace target with downsampled values based on compression_ratio
    original_target = item['target']
    downsample_step = int(1 / compression_ratio)

    downsampled_target = original_target[::downsample_step]  # Take every Nth sample
    downsampled_item['target'] = downsampled_target

    input_data_downsampled.append(downsampled_item)

# Run predictions with progress bars
print("\nRunning main inference...")
forecasts = list(tqdm(predictor.predict(input_data), desc="Main forecasts", total=len(input_data)))

print("Running downsampled context inference...")
forecasts_downsampled = list(tqdm(predictor_reduced_ctx_pdt.predict(input_data_downsampled), desc="Downsampled forecasts", total=len(input_data_downsampled)))

print(f"Generated {len(forecasts)} forecasts")

# Prepare data for visualization
print("\nProcessing main results...")
sample_results = []
full_data_values = df_selected[selected_column].values

for i, (input_item, label_item, forecast) in enumerate(tqdm(zip(input_data, label_data, forecasts), desc="Processing main results", total=len(input_data))):
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

# repeat for reduced context length
print("\nProcessing downsampled results...")
sample_results_downsampled = []
for i, (input_item, label_item, forecast) in enumerate(tqdm(zip(input_data_downsampled, label_data, forecasts_downsampled), desc="Processing downsampled results", total=len(input_data_downsampled))):
    # Get context data
    context = input_item['target']
    
    # keep the last `reduced_ctx` values for context
    if len(context) > reduced_ctx:
        context = context[-reduced_ctx:]

    # Get ground truth (still full PDT length)
    ground_truth = label_item['target'][:PDT]
    
    # Get prediction (mean of samples) - this is now length reduced_pdt
    prediction_downsampled = np.mean(forecast.samples, axis=0)

    # Upsample prediction from reduced_pdt to PDT length using linear interpolation
    reduced_indices = np.linspace(0, PDT-1, reduced_pdt)
    # Create indices for target full prediction
    full_indices = np.arange(PDT)
    # Interpolate to get full-length prediction
    prediction_upsampled = np.interp(full_indices, reduced_indices, prediction_downsampled)

    
    # Store results with both reduced and upsampled predictions
    sample_results_downsampled.append({
        'window_id': i,
        'context': context,
        'ground_truth': ground_truth,
        'prediction': prediction_upsampled,  # Upsampled prediction for MAE calculation
        'prediction_downsampled': prediction_downsampled,  # Original reduced prediction
        'prediction_upsampled': prediction_upsampled,  # Upsampled prediction
        'reduced_pdt': reduced_pdt,  # Store reduced prediction length for plotting
        'mae': np.mean(np.abs(prediction_upsampled - ground_truth))  # MAE against full ground truth
    })

# NEW EXPERIMENT: Direct downsampling of ground truth labels (no resampling of forecast)
# This uses the existing downsampled prediction and compares with downsampled ground truth
print("\nProcessing direct downsampled ground truth experiment...")
sample_results_direct_downsample = []

for i, (input_item, label_item, forecast) in enumerate(tqdm(zip(input_data_downsampled, label_data, forecasts_downsampled), desc="Processing direct downsample results", total=len(input_data_downsampled))):
    # Get context data
    context = input_item['target']
    
    # keep the last `reduced_ctx` values for context
    if len(context) > reduced_ctx:
        context = context[-reduced_ctx:]

    # Get full ground truth
    full_ground_truth = label_item['target'][:PDT]
    
    # Get prediction (mean of samples) - this is the reduced_pdt length prediction
    prediction_downsampled = np.mean(forecast.samples, axis=0)
    
    # Downsample ground truth to match the prediction length
    # Use the same downsampling approach as the context downsampling
    downsample_step = int(1 / compression_ratio)
    downsampled_ground_truth = full_ground_truth[::downsample_step][:reduced_pdt]
    
    # Calculate MAE on the downsampled data directly
    downsampled_mae = np.mean(np.abs(prediction_downsampled - downsampled_ground_truth))
    
    # Store results
    sample_results_direct_downsample.append({
        'window_id': i,
        'context': context,
        'ground_truth': full_ground_truth,  # Store full GT for visualization
        'downsampled_ground_truth': downsampled_ground_truth,  # Downsampled GT
        'prediction_downsampled': prediction_downsampled,  # Downsampled prediction
        'mae': downsampled_mae,  # MAE on downsampled data
        'downsample_step': downsample_step,  # Store for plotting
        'reduced_pdt': reduced_pdt  # Store for plotting
    })

print(f"Processed {len(sample_results_direct_downsample)} direct downsample samples")

# create a new input data with values replaced by 0 and then refilled using interpolation
# This maintains the original context length but tests robustness to missing data
print("\nPreparing interpolated context data...")
input_data_reduced_interpolated = []
for item in tqdm(input_data, desc="Processing interpolated context"):
    # Create a deep copy of the item to avoid modifying original
    reduced_item = item.copy()
    
    # Start with original target
    original_target = item['target']
    target_with_zeros = original_target.copy()

    replace_ratio = 1.0 - compression_ratio
    downsample_step = int(1 / compression_ratio)

    indices_to_replace = []
    for i in range(len(target_with_zeros)):
        if i % downsample_step != 0:  # Keep every downsample_step-th sample, replace others
            indices_to_replace.append(i)
    
    # Set selected indices to 0 (creating missing values)
    target_with_zeros[indices_to_replace] = 0
    
    # Create mask for non-zero values (valid data points)
    valid_mask = target_with_zeros != 0
    valid_indices = np.where(valid_mask)[0]
    valid_values = target_with_zeros[valid_mask]
    
    # Interpolate to fill the zero positions
    all_indices = np.arange(len(target_with_zeros))
    interpolated_target = np.interp(all_indices, valid_indices, valid_values)
    
    # Keep the same context length as original
    if len(interpolated_target) > CTX:
        interpolated_target = interpolated_target[-CTX:]
    
    reduced_item['target'] = interpolated_target
    input_data_reduced_interpolated.append(reduced_item)

# Run predictions with interpolated reduced context
print("Running interpolated context inference...")
forecasts_interpolated = list(tqdm(predictor.predict(input_data_reduced_interpolated), desc="Interpolated forecasts", total=len(input_data_reduced_interpolated)))

# Prepare data for visualization with interpolated reduced context
print("\nProcessing interpolated results...")
sample_results_interpolated = []
for i, (input_item, label_item, forecast) in enumerate(tqdm(zip(input_data_reduced_interpolated, label_data, forecasts_interpolated), desc="Processing interpolated results", total=len(input_data_reduced_interpolated))):
    # Get context data
    context = input_item['target']
    
    # Get ground truth
    ground_truth = label_item['target'][:PDT]
    
    # Get prediction (mean of samples)
    prediction = np.mean(forecast.samples, axis=0)
    
    # Store results
    sample_results_interpolated.append({
        'window_id': i,
        'context': context,
        'ground_truth': ground_truth,
        'prediction': prediction,
        'mae': np.mean(np.abs(prediction - ground_truth))
    })


# Run predictions with reduced model; model will take the normal input but internally use reduced context length
print("Running truncated context inference...")
forecasts_truncated = list(tqdm(predictor_reduced_ctx.predict(input_data), desc="Truncated forecasts", total=len(input_data)))

# Prepare data for visualization with truncated context
print("\nProcessing truncated results...")
sample_results_truncated = []
for i, (input_item, label_item, forecast) in enumerate(tqdm(zip(input_data, label_data, forecasts_truncated), desc="Processing truncated results", total=len(input_data))):
    # Get context data
    context = input_item['target']

    # keep the last `reduced_ctx` values for context
    if len(context) > reduced_ctx:
        context = context[-reduced_ctx:]
    
    # Get ground truth (still full PDT length)
    ground_truth = label_item['target'][:PDT]
    
    # Get prediction
    prediction_truncated = np.mean(forecast.samples, axis=0)

    # Store results with both reduced and upsampled predictions
    sample_results_truncated.append({
        'window_id': i,
        'context': context,
        'ground_truth': ground_truth,
        'prediction': prediction_truncated, 
        'reduced_pdt': reduced_pdt,  # Store reduced prediction length for plotting
        'mae': np.mean(np.abs(prediction_truncated - ground_truth))  # MAE against full ground truth
    })

# All done
print("Inference and processing complete!")

if ENABLE_SURPRISAL:
    # Implement surprisal-based pruning methods
    # Method 1: Drop least important 50% of samples (shorter context)
    # Method 2: Replace least important 50% with interpolation (original length)

    print("\n" + "="*60)
    print("IMPLEMENTING SURPRISAL-BASED PRUNING METHODS")
    print("Using surprisal (self-information) under Gaussian assumption as importance score")
    print("="*60)

# First, create the autoregressive model needed for surprisal computation
print("Creating autoregressive model for surprisal computation...")
model_ar = MoiraiForecast(
    module=base_module,
    prediction_length=1,  # Single-step predictions
    context_length=CTX,
    patch_size=PSZ_surprisal,
    num_samples=NUM_SAMPLES,
    target_dim=1,
    feat_dynamic_real_dim=ds.num_feat_dynamic_real,
    past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
)

predictor_ar = model_ar.create_predictor(batch_size=BSZ)
print("Autoregressive predictor created successfully!")

def compute_surprisal(input_data_context, predictor_ar):
    """
    Compute surprisal/importance for each position in the context windows using surprisal.
    Higher surprisal (surprisal) = higher importance.
    
    For each position, we:
    1. Fit a Gaussian to the model's prediction samples
    2. Compute the surprisal (self-information) of the true value under this Gaussian
    3. Use surprisal as the surprisal/importance score
    
    Surprisal of value x under Gaussian N(μ, σ²):
    surprisal(x) = -log(p(x)) = log(√(2πσ²)) + (x-μ)²/(2σ²)

    Args:
        input_data_context: List of input data items for each context window
        predictor_ar: Autoregressive predictor for single-step predictions

    Returns:
        surprisal_scores: Array of surprisal scores for each position
        predictions_mean: Array of prediction means for each position
    """
    # Get predictions for all context windows
    forecasts_ar = list(tqdm(predictor_ar.predict(input_data_context), desc="Main forecasts", total=len(input_data_context)))

    # Compute surprisal-based surprisal and store predictions for each position
    surprisal_scores = []
    predictions_mean = []
    
    for i, forecast in enumerate(forecasts_ar):
        # Get prediction samples for this position
        samples = forecast.samples[:, 0]  # Single-step prediction
        
        # Fit Gaussian to prediction samples
        mean_pred = np.mean(samples)
        std_pred = np.std(samples)
        
        # Get the true value for this position
        # The true value is the target at the position we're predicting
        input_item = input_data_context[i]
        true_value = input_item['target'][-1]  # Last value in the context (what we're predicting)
        
        # Compute surprisal of the true value under the Gaussian fitted to predictions
        # surprisal(x) = log(√(2πσ²)) + (x-μ)²/(2σ²)
        if std_pred > 0:
            # Standard surprisal computation
            log_normalizer = 0.5 * np.log(2 * np.pi * std_pred**2)
            squared_error_term = (true_value - mean_pred)**2 / (2 * std_pred**2)
            surprisal = log_normalizer + squared_error_term
        else:
            # Handle edge case where std is 0 (all predictions are identical)
            # Use a small epsilon to avoid division by zero
            epsilon = 1e-8
            log_normalizer = 0.5 * np.log(2 * np.pi * epsilon**2)
            squared_error_term = (true_value - mean_pred)**2 / (2 * epsilon**2)
            surprisal = log_normalizer + squared_error_term
        
        surprisal_scores.append(surprisal)
        predictions_mean.append(mean_pred)

    return np.array(surprisal_scores), np.array(predictions_mean)

if ENABLE_SURPRISAL:
    # Compute surprisal for all samples in the test data, so that we can use it for pruning of any context (subset)
    # Extract the CTX+TEST_SAMPLES 
    _, test_template_surprisal = split(ds, offset=-(CTX+TEST_SAMPLES))
    test_data_surprisal = test_template_surprisal.generate_instances(
        prediction_length=1,
        windows=CTX+TEST_SAMPLES,
        distance=1,
    )
    input_data_surprisal = list(test_data_surprisal.input)

    print("Computing surprisal scores for union of all samples in all context windows...")
    surprisal_map, predictions_map = compute_surprisal(input_data_surprisal, predictor_ar)

    # Method 1: Surprisal-based pruning (drop least important 50%)
    print("\nMethod 1: Surprisal-based pruning (drop least important 50%)")

    # Dictionary to store surprisal data for selected samples for later plotting
    selected_samples_data = {}

    input_data_surprisal_short = []
    for j, input_item in enumerate(tqdm(input_data, desc="Processing surprisal-based short context")):
        # Get the last CTX samples for surprisal computation
        original_target = input_item['target']
        
        # Take the last CTX samples
        last_ctx_samples = original_target[-CTX:]

        map_start = j*PDT
        map_end = map_start + CTX

        surprisal_scores = surprisal_map[map_start:map_end]
        prediction_means = predictions_map[map_start:map_end]

        # Store surprisal data for the first few samples for later plotting
        current_idx = len(input_data_surprisal_short)  # Get current sample index
        if current_idx < 10:  # Store data for first 10 samples
            # Calculate signal deltas: delta[i] = x[i] - x[i-1], delta[0] = 0
            signal_deltas = np.zeros(CTX)
            signal_deltas[1:] = np.diff(last_ctx_samples)  # delta[i] = x[i] - x[i-1]
            
            selected_samples_data[current_idx] = {
                'signal': last_ctx_samples.copy(),
                'surprisal': surprisal_scores.copy(), 
                'delta': signal_deltas.copy(),
                'predictions': prediction_means.copy()  # Store predictions for visualization
            }
        
        # Select top 50% most important (highest surprisal) samples
        num_keep = reduced_ctx  # Use reduced_ctx instead of CTX//2 for consistency
        important_indices = np.argsort(surprisal_scores)[-num_keep:]  # Get indices of highest surprisal
        important_indices = np.sort(important_indices)  # Keep them in order
        
        # Create pruned context with only important samples - no padding!
        unc_context = last_ctx_samples[important_indices]
        
        # Create new input item with reduced context
        pruned_item = input_item.copy()
        pruned_item['target'] = unc_context
        input_data_surprisal_short.append(pruned_item)

    print(f"Created {len(input_data_surprisal_short)} inputs with surprisal-based pruning (short context)")

    # Method 2: Surprisal-based pruning with interpolation (original CTX length)
    print("\nMethod 2: Surprisal-based pruning with interpolation (original CTX length)")

    input_data_surprisal_interpolated = []
    for j, input_item in enumerate(tqdm(input_data, desc="Processing surprisal-based short context")):
        # Get the last 2*CTX samples for surprisal computation
        original_target = input_item['target']

        # Compute surprisal for the last CTX samples
        last_ctx_samples = original_target[-CTX:]  # Only analyze the last CTX samples
        
        map_start = j*PDT
        map_end = map_start + CTX

        surprisal_scores = surprisal_map[map_start:map_end]
        prediction_means = predictions_map[map_start:map_end]
        
        # Replace least important 50% samples with interpolated values
        num_replace = CTX // 2
        least_important_indices = np.argsort(surprisal_scores)[:num_replace]  # Get indices of lowest surprisal
        
        # Create interpolated context
        unc_interpolated_context = last_ctx_samples.copy()
        
        # Create mask for interpolation
        unc_interpolated_mask = np.zeros(CTX, dtype=bool)
        unc_interpolated_mask[least_important_indices] = True
        
        # Set least important values to zero temporarily
        target_with_missing = unc_interpolated_context.copy()
        target_with_missing[least_important_indices] = 0
        
        # Interpolate the missing values
        valid_mask = ~unc_interpolated_mask
        valid_indices = np.where(valid_mask)[0]
        valid_values = unc_interpolated_context[valid_mask]
        
        if len(valid_indices) > 1:  # Need at least 2 points for interpolation
            all_indices = np.arange(CTX)
            unc_interpolated_context = np.interp(all_indices, valid_indices, valid_values)
        # If we have too few valid points, keep the original values
        
        # Create new input item
        interpolated_item = input_item.copy()
        interpolated_item['target'] = unc_interpolated_context
        input_data_surprisal_interpolated.append(interpolated_item)

    print(f"Created {len(input_data_surprisal_interpolated)} inputs with surprisal-based interpolation")

    # Run inference with surprisal-based pruning methods
    print("\nRunning inference with surprisal-based pruning methods...")

    print("Running predictions with surprisal-based short context...")
    forecasts_surprisal_short = list(tqdm(predictor_reduced_ctx.predict(input_data_surprisal_short), desc="Surprisal short forecasts", total=len(input_data_surprisal_short)))

    print("Running predictions with surprisal-based interpolation...")
    forecasts_surprisal_interpolated = list(tqdm(predictor.predict(input_data_surprisal_interpolated), desc="Surprisal interpolated forecasts", total=len(input_data_surprisal_interpolated)))

    print(f"Generated {len(forecasts_surprisal_short)} surprisal-based short forecasts")
    print(f"Generated {len(forecasts_surprisal_interpolated)} surprisal-based interpolated forecasts")

    # Process results
    print("\nProcessing surprisal-based pruning results...")

    # Method 1: Surprisal-based short results
    sample_results_surprisal_short = []
    for i, (input_item, label_item, forecast) in enumerate(tqdm(zip(input_data_surprisal_short, label_data, forecasts_surprisal_short), desc="Processing surprisal short results", total=len(input_data_surprisal_short))):
        context = input_item['target']
        ground_truth = label_item['target'][:PDT]
        
        # Get prediction (mean of samples) - this is now length reduced_pdt
        prediction_reduced = np.mean(forecast.samples, axis=0)
        
        sample_results_surprisal_short.append({
            'window_id': i,
            'context': context,
            'ground_truth': ground_truth,
            'prediction': prediction_reduced,  # Use actual reduced prediction
            'prediction_reduced': prediction_reduced,  # Original reduced prediction
            'reduced_pdt': reduced_pdt,  # Store reduced prediction length for plotting
            'mae': np.mean(np.abs(prediction_reduced - ground_truth))  # MAE against full ground truth
        })

    # Method 2: Surprisal-based interpolated results (unchanged)
    sample_results_surprisal_interpolated = []
    for i, (input_item, label_item, forecast) in enumerate(tqdm(zip(input_data_surprisal_interpolated, label_data, forecasts_surprisal_interpolated), desc="Processing surprisal interpolated results", total=len(input_data_surprisal_interpolated))):
        context = input_item['target']
        ground_truth = label_item['target'][:PDT]
        prediction = np.mean(forecast.samples, axis=0)
        
        # Store interpolation mask if available
        interpolated_mask = np.zeros(len(context), dtype=bool)  # Default to no interpolation
        
        sample_results_surprisal_interpolated.append({
            'window_id': i,
            'context': context,
            'ground_truth': ground_truth,
            'prediction': prediction,
            'mae': np.mean(np.abs(prediction - ground_truth)),
            'interpolated_mask': interpolated_mask
        })

    print(f"Processed {len(sample_results_surprisal_short)} surprisal-based short samples")
    print(f"Processed {len(sample_results_surprisal_interpolated)} surprisal-based interpolated samples")

else:
    print("\n" + "="*60)
    print("SURPRISAL-BASED METHODS DISABLED")
    print("="*60)
    # Create empty results for consistency
    sample_results_surprisal_short = []
    sample_results_surprisal_interpolated = []
    selected_samples_data = {}  # Empty dict for plotting

# Calculate average MAE for all methods
print("\n" + "="*60)
print("SUMMARY OF ALL PRUNING METHODS")
print("="*60)

methods_summary = {
    'Full Context': np.mean([r['mae'] for r in sample_results]),
    'Downsampled (50%)': np.mean([r['mae'] for r in sample_results_downsampled]),
    'Direct Downsample Input→Output': np.mean([r['mae'] for r in sample_results_direct_downsample]),
    'Interpolated (50% replaced)': np.mean([r['mae'] for r in sample_results_interpolated]),
    'Truncated (recent values)': np.mean([r['mae'] for r in sample_results_truncated]),
}

# Add surprisal methods only if enabled
if ENABLE_SURPRISAL:
    methods_summary['Surprisal-based Short (50% most important)'] = np.mean([r['mae'] for r in sample_results_surprisal_short])
    methods_summary['Surprisal-based Interpolated (50% replaced)'] = np.mean([r['mae'] for r in sample_results_surprisal_interpolated])

# Calculate standard errors for error bars
methods_std_errors = {
    'Full Context': np.std([r['mae'] for r in sample_results]) / np.sqrt(len(sample_results)),
    'Downsampled (50%)': np.std([r['mae'] for r in sample_results_downsampled]) / np.sqrt(len(sample_results_downsampled)),
    'Direct Downsample Input→Output': np.std([r['mae'] for r in sample_results_direct_downsample]) / np.sqrt(len(sample_results_direct_downsample)),
    'Interpolated (50% replaced)': np.std([r['mae'] for r in sample_results_interpolated]) / np.sqrt(len(sample_results_interpolated)),
    'Truncated (recent values)': np.std([r['mae'] for r in sample_results_truncated]) / np.sqrt(len(sample_results_truncated)),
}

# Add surprisal methods standard errors only if enabled
if ENABLE_SURPRISAL:
    methods_std_errors['Surprisal-based Short (50% most important)'] = np.std([r['mae'] for r in sample_results_surprisal_short]) / np.sqrt(len(sample_results_surprisal_short))
    methods_std_errors['Surprisal-based Interpolated (50% replaced)'] = np.std([r['mae'] for r in sample_results_surprisal_interpolated]) / np.sqrt(len(sample_results_surprisal_interpolated))

print("Method Performance (Mean Absolute Error):")
for method, mae in methods_summary.items():
    print(f"  {method:<40}: {mae:.4f}")

best_method = min(methods_summary.items(), key=lambda x: x[1])
print(f"\nBest performing method: {best_method[0]} (MAE: {best_method[1]:.4f})")



# Plot results for all pruning methods including surprisal-based approaches
# For 3 random samples, plot context, ground truth and prediction for all methods
num_samples = 3
sample_indices = np.random.choice(len(sample_results), num_samples, replace=False)

for plot_idx, idx in enumerate(sample_indices, 1):
    # Get results for all methods
    result = sample_results[idx]
    result_downsampled = sample_results_downsampled[idx]
    result_direct_downsample = sample_results_direct_downsample[idx]
    result_interpolated = sample_results_interpolated[idx]
    result_truncated = sample_results_truncated[idx]
    
    # Conditionally get surprisal results if enabled
    if ENABLE_SURPRISAL:
        result_surprisal_short = sample_results_surprisal_short[idx]
        result_surprisal_interpolated = sample_results_surprisal_interpolated[idx]
        num_subplots = 7
        fig_height = 21
    else:
        result_surprisal_short = None
        result_surprisal_interpolated = None
        num_subplots = 5
        fig_height = 15
    
    plt.figure(figsize=(15, fig_height))  # Adjust height based on number of subplots
    
    # 1. Full context
    plt.subplot(num_subplots, 1, 1)
    context_len = len(result['context'])
    context_indices = np.arange(-context_len, 0)
    forecast_indices = np.arange(0, PDT)
    
    plt.plot(context_indices, result['context'], label='Context', color='blue', linewidth=2, 
             linestyle='-.', marker='o', markersize=6)
    plt.plot(forecast_indices, result['ground_truth'], label='Ground Truth', color='green', 
             linewidth=3, marker='o', markersize=6, linestyle='-.')
    plt.plot(forecast_indices, result['prediction'], label='Prediction', color='red', 
             linewidth=2, linestyle='-.', marker='s', markersize=6)
    plt.axvline(x=0, color='black', linestyle=':', alpha=0.7, label='Forecast Start')
    
    plt.title(f"1. Full Context (len={context_len}) - Sample {result['window_id']} - MAE: {result['mae']:.4f}")
    
    # Add PSZ in red at the right end of the title area
    plt.text(0.98, 1.02, f'PSZ: {PSZ}', transform=plt.gca().transAxes, color='red', 
             fontsize=12, fontweight='bold', ha='right', va='bottom')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Downsampled context - maintain time scale
    plt.subplot(num_subplots, 1, 2)
    context_len_reduced = len(result_downsampled['context'])
    
    # Maintain original time scale by spacing the downsampled points appropriately
    # Use compression_ratio to determine the spacing
    downsample_step = int(1 / compression_ratio)
    downsampled_indices = np.arange(-context_len_reduced * downsample_step, 0, downsample_step)
    
    plt.plot(downsampled_indices, result_downsampled['context'], label='Downsampled Context (50%)', 
             color='blue', linewidth=2, linestyle='-.', marker='o', markersize=6)
    plt.plot(forecast_indices, result_downsampled['ground_truth'], label='Ground Truth', color='green', 
             linewidth=3, marker='o', markersize=6, linestyle='-.')
    plt.plot(forecast_indices, result_downsampled['prediction'], label='Prediction', color='red', 
             linewidth=2, linestyle='-.', marker='s', markersize=6)
    plt.axvline(x=0, color='black', linestyle=':', alpha=0.7, label='Forecast Start')
    
    plt.title(f"2. Downsampled Context (len={context_len_reduced}) - Sample {result_downsampled['window_id']} - MAE: {result_downsampled['mae']:.4f}")
    
    # Add PSZ in red at the right end of the title area
    plt.text(0.98, 1.02, f'PSZ: {PSZ}', transform=plt.gca().transAxes, color='red', 
             fontsize=12, fontweight='bold', ha='right', va='bottom')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Direct Downsampled Ground Truth (new experiment)
    plt.subplot(num_subplots, 1, 3)
    context_len_direct = len(result_direct_downsample['context'])
    
    # Plot downsampled context with proper time scale
    # Use compression_ratio to determine the spacing
    downsample_step_context = int(1 / compression_ratio)
    downsampled_context_indices = np.arange(-context_len_direct * downsample_step_context, 0, downsample_step_context)
    
    plt.plot(downsampled_context_indices, result_direct_downsample['context'], label='Downsampled Context', color='blue', linewidth=2, 
             linestyle='-.', marker='o', markersize=6)
    
    # Plot downsampled forecast vs downsampled ground truth
    # These are the actual comparison points for this experiment
    downsample_step = result_direct_downsample['downsample_step']
    reduced_pdt = result_direct_downsample['reduced_pdt']
    downsampled_forecast_indices = np.arange(0, reduced_pdt * downsample_step, downsample_step)
    
    plt.plot(downsampled_forecast_indices, result_direct_downsample['downsampled_ground_truth'], 
             label='Downsampled GT', color='purple', linewidth=3, linestyle='--', marker='x', markersize=8)
    plt.plot(downsampled_forecast_indices, result_direct_downsample['prediction_downsampled'], 
             label='Downsampled Pred', color='orange', linewidth=2, linestyle='--', marker='s', markersize=8)
    
    # Show full resolution data as light background for context
    forecast_indices = np.arange(0, PDT)
    plt.plot(forecast_indices, result_direct_downsample['ground_truth'], color='green', 
             linewidth=1, alpha=0.3, linestyle='-', label='Full GT (background)')
    
    plt.axvline(x=0, color='black', linestyle=':', alpha=0.7, label='Forecast Start')
    
    plt.title(f"3. Direct Downsample GT (downsampled input→downsampled output) - Sample {result_direct_downsample['window_id']} - MAE: {result_direct_downsample['mae']:.4f}")
    
    # Add PSZ in red at the right end of the title area
    plt.text(0.98, 1.02, f'PSZ: {PSZ}', transform=plt.gca().transAxes, color='red', 
             fontsize=12, fontweight='bold', ha='right', va='bottom')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Interpolated context (50% replaced) - highlight interpolated samples
    plt.subplot(num_subplots, 1, 4)
    context_len_interpolated = len(result_interpolated['context'])
    context_indices_interpolated = np.arange(-context_len_interpolated, 0)
    
    # Get the original data to identify which samples were interpolated
    original_context = input_data[idx]['target'][-context_len_interpolated:]
    interpolated_context = result_interpolated['context']
    
    # Find interpolated positions (where values differ significantly from original)
    interpolated_mask = np.abs(interpolated_context - original_context) > 1e-6
    
    # Plot original samples in blue
    plt.plot(context_indices_interpolated[~interpolated_mask], 
             interpolated_context[~interpolated_mask], 
             'o', color='blue', markersize=6, label='Original Samples')
    
    # Plot interpolated samples in orange
    plt.plot(context_indices_interpolated[interpolated_mask], 
             interpolated_context[interpolated_mask], 
             'o', color='orange', markersize=6, label='Interpolated Samples')
    
    # Connect all points with a line
    plt.plot(context_indices_interpolated, interpolated_context, 
             color='blue', linewidth=2, linestyle='-.', alpha=0.7)
    
    plt.plot(forecast_indices, result_interpolated['ground_truth'], label='Ground Truth', 
             color='green', linewidth=3, marker='o', markersize=6, linestyle='-.')
    plt.plot(forecast_indices, result_interpolated['prediction'], label='Prediction', 
             color='red', linewidth=2, linestyle='-.', marker='s', markersize=6)
    plt.axvline(x=0, color='black', linestyle=':', alpha=0.7, label='Forecast Start')
    
    plt.title(f"4. Interpolated Context (50% samples replaced, len={context_len_interpolated}) - Sample {result_interpolated['window_id']} - MAE: {result_interpolated['mae']:.4f}")
    
    # Add PSZ in red at the right end of the title area
    plt.text(0.98, 1.02, f'PSZ: {PSZ}', transform=plt.gca().transAxes, color='red', 
             fontsize=12, fontweight='bold', ha='right', va='bottom')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Truncated context (most recent values) - plot from correct time index without padding
    plt.subplot(num_subplots, 1, 5)
    context_len_truncated = len(result_truncated['context'])
    
    # Plot from the correct time index without zero padding
    # The truncated context represents the most recent context_len_truncated values
    truncated_start_index = -context_len_truncated
    truncated_indices = np.arange(truncated_start_index, 0)
    
    plt.plot(truncated_indices, result_truncated['context'], label='Truncated Context (most recent)', 
             color='blue', linewidth=2, linestyle='-.', marker='o', markersize=6)
    plt.plot(forecast_indices, result_truncated['ground_truth'], label='Ground Truth', 
             color='green', linewidth=3, marker='o', markersize=6, linestyle='-.')
    plt.plot(forecast_indices, result_truncated['prediction'], label='Prediction', 
             color='red', linewidth=2, linestyle='-.', marker='s', markersize=6)
    plt.axvline(x=0, color='black', linestyle=':', alpha=0.7, label='Forecast Start')
    
    # Get x-axis limits from the full context case to ensure consistency
    full_context_xlim = (-context_len, PDT)
    plt.xlim(full_context_xlim)
    
    plt.title(f"5. Truncated Context (most recent {context_len_truncated} values) - Sample {result_truncated['window_id']} - MAE: {result_truncated['mae']:.4f}")
    
    # Add PSZ in red at the right end of the title area
    plt.text(0.98, 1.02, f'PSZ: {PSZ}', transform=plt.gca().transAxes, color='red', 
             fontsize=12, fontweight='bold', ha='right', va='bottom')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Conditionally add surprisal-based subplots if enabled
    if ENABLE_SURPRISAL:
        # 6. Surprisal-based short context - plot selected samples at original locations
        plt.subplot(num_subplots, 1, 6)
        context_len_unc_short = len(result_surprisal_short['context'])
        
        # Get the original context and surprisal data for this sample
        if idx in selected_samples_data:
            sample_data = selected_samples_data[idx]
            original_signal = sample_data['signal']  # Original full context
            surprisal_scores = sample_data['surprisal']  # Surprisal for each position
            
            # Recreate the selection logic to find which indices were kept
            num_keep = context_len_unc_short  # This should match reduced_ctx
            important_indices = np.argsort(surprisal_scores)[-num_keep:]  # Get indices of highest surprisal
            important_indices = np.sort(important_indices)  # Keep them in order
            
            # Plot only the selected samples at their original time positions
            ctx_indices = np.arange(-CTX, 0)
            selected_positions = ctx_indices[important_indices]
            selected_values = original_signal[important_indices]
            
            plt.plot(selected_positions, selected_values, 'o-', color='blue', linewidth=2, 
                    linestyle='-.', markersize=6, label='Surprisal-based Context (50% most important)')
        else:
            # Fallback: plot as contiguous block if surprisal data not available
            unc_start_index = -context_len_unc_short
            unc_indices = np.arange(unc_start_index, 0)
            plt.plot(unc_indices, result_surprisal_short['context'], 'o-', color='blue', linewidth=2, 
                    linestyle='-.', markersize=6, label='Surprisal-based Context (50% most important)')
        
        plt.plot(forecast_indices, result_surprisal_short['ground_truth'], label='Ground Truth', 
                color='green', linewidth=3, marker='o', markersize=6, linestyle='-.')
        plt.plot(forecast_indices, result_surprisal_short['prediction'], label='Prediction', 
                color='red', linewidth=2, linestyle='-.', marker='s', markersize=6)
        plt.axvline(x=0, color='black', linestyle=':', alpha=0.7, label='Forecast Start')
        
        # Set x-axis limits to match other methods
        plt.xlim(-context_len, PDT)
        
        plt.title(f"6. Surprisal-based Short Context ({context_len_unc_short} most important values) - Sample {result_surprisal_short['window_id']} - MAE: {result_surprisal_short['mae']:.4f}")
        
        # Add PSZ in red at the right end of the title area
        plt.text(0.98, 1.02, f'PSZ: {PSZ}', transform=plt.gca().transAxes, color='red', 
                fontsize=12, fontweight='bold', ha='right', va='bottom')
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 7. Surprisal-based interpolated context - highlight interpolated samples
        plt.subplot(num_subplots, 1, 7)
        context_len_unc_interp = len(result_surprisal_interpolated['context'])
        context_indices_unc_interp = np.arange(-context_len_unc_interp, 0)
        
        # Get the original last CTX samples to identify which were interpolated
        original_last_ctx = input_data[idx]['target'][-CTX:]
        unc_interpolated_context = result_surprisal_interpolated['context']
        
        # Find interpolated positions (where values differ from original)
        unc_interpolated_mask = np.abs(unc_interpolated_context - original_last_ctx) > 1e-6
        
        # Plot original samples in blue
        plt.plot(context_indices_unc_interp[~unc_interpolated_mask], 
                unc_interpolated_context[~unc_interpolated_mask], 
                'o', color='blue', markersize=6, label='Original Samples')
        
        # Plot interpolated samples in orange
        plt.plot(context_indices_unc_interp[unc_interpolated_mask], 
                unc_interpolated_context[unc_interpolated_mask], 
                'o', color='orange', markersize=6, label='Interpolated Samples')
        
        # Connect all points with a line
        plt.plot(context_indices_unc_interp, unc_interpolated_context, 
                color='blue', linewidth=2, linestyle='-.', alpha=0.7)
        
        plt.plot(forecast_indices, result_surprisal_interpolated['ground_truth'], label='Ground Truth', 
                color='green', linewidth=3, marker='o', markersize=6, linestyle='-.')
        plt.plot(forecast_indices, result_surprisal_interpolated['prediction'], label='Prediction', 
                color='red', linewidth=2, linestyle='-.', marker='s', markersize=6)
        plt.axvline(x=0, color='black', linestyle=':', alpha=0.7, label='Forecast Start')
        
        plt.title(f"7. Surprisal-based Interpolated Context (50% least important replaced, len={context_len_unc_interp}) - Sample {result_surprisal_interpolated['window_id']} - MAE: {result_surprisal_interpolated['mae']:.4f}")
        
        # Add PSZ in red at the right end of the title area
        plt.text(0.98, 1.02, f'PSZ: {PSZ}', transform=plt.gca().transAxes, color='red', 
                fontsize=12, fontweight='bold', ha='right', va='bottom')
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
    # Save plot
    plot_filename = os.path.join(results_dir, f"sample_{plot_idx}_all_methods_comparison.png")
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved plot: {plot_filename}")

# Create summary comparison plot
print("\nCreating summary comparison plot for all methods...")
plt.figure(figsize=(15, 8))

methods = list(methods_summary.keys())
mae_values = list(methods_summary.values())
mae_errors = list(methods_std_errors.values())

bars = plt.bar(range(len(methods)), mae_values, yerr=mae_errors, capsize=5,
               color=['steelblue', 'forestgreen', 'darkorange', 'firebrick', 'purple', 'brown'],
               error_kw={'elinewidth': 2, 'capthick': 2})

# Add MAE values on top of bars (adjusted for error bars)
for i, (bar, mae, error) in enumerate(zip(bars, mae_values, mae_errors)):
    plt.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + error + 0.001, 
             f'{mae:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.xlabel('Context Pruning Method', fontsize=12)
plt.ylabel('Mean Absolute Error (MAE)', fontsize=12)

plt.title(f'Context Pruning Methods Comparison\nModel: {MODEL}-{SIZE} | Dataset: {dataset_name} | Context: {CTX} -> Prediction: {PDT}', fontsize=14)

# Add PSZ in red at the right end of the title area
plt.text(0.98, 1.02, f'PSZ: {PSZ}', transform=plt.gca().transAxes, color='red', 
         fontsize=14, fontweight='bold', ha='right', va='bottom')
plt.xticks(range(len(methods)), methods, rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save summary plot
summary_filename = os.path.join(results_dir, "methods_comparison_summary.png")
plt.savefig(summary_filename, dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved summary plot: {summary_filename}")

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
print(f"All results and plots saved to: {results_dir}")
print(f"Best performing method: {best_method[0]} (MAE: {best_method[1]:.4f})")

# Calculate improvement percentages
full_context_mae = methods_summary['Full Context']
print(f"\nPerformance relative to Full Context (MAE: {full_context_mae:.4f}):")
for method, mae in methods_summary.items():
    if method != 'Full Context':
        improvement = ((full_context_mae - mae) / full_context_mae) * 100
        status = "better" if improvement > 0 else "worse"
        print(f"  {method:<40}: {improvement:+.2f}% ({status})")


# Create surprisal and delta analysis plots for the 3 selected samples
if ENABLE_SURPRISAL:
    print("\nCreating surprisal and delta analysis plots for selected samples...")

# Select 3 random samples for detailed analysis from the first 10 (where we saved surprisal data)
num_samples = min(3, len(selected_samples_data))  # Use min to handle cases with < 10 samples
available_indices = list(selected_samples_data.keys())
sample_indices = np.random.choice(available_indices, num_samples, replace=False)

for plot_idx, idx in enumerate(sample_indices, 1):
    # Get the stored surprisal data for this sample
    if idx in selected_samples_data:
        sample_data = selected_samples_data[idx]
        signal = sample_data['signal']
        surprisal = sample_data['surprisal']
        delta = sample_data['delta']  # delta[i] = x[i] - x[i-1], delta[0] = 0
        predictions = sample_data['predictions']  # Prediction means for each position
        
        # Create figure with 3 subplots for each sample
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))
        
        # Plot 1: Signal and Surprisal Map
        ax1_twin = ax1.twinx()  # Create twin axis for surprisal
        
        # Plot the signal (last CTX samples)
        ctx_indices = np.arange(-CTX, 0)
        line1 = ax1.plot(ctx_indices, signal, 'o-', color='blue', linewidth=2, 
                        linestyle='-.', markersize=6, label='Signal (Context)')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Signal Value', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True, alpha=0.3)

        # Plot the surprisal map
        line2 = ax1_twin.plot(ctx_indices, surprisal, 's-', color='red', linewidth=2, 
                             linestyle='-.', markersize=6, alpha=0.7, label='Surprisal Score')
        ax1_twin.set_ylabel('Surprisal Score', color='red')
        ax1_twin.tick_params(axis='y', labelcolor='red')
        # Note: Don't fix Y-axis limits as surprisal can have a wide range
        
        # Add patch size in red text on top right
        ax1.text(0.98, 0.95, f'PSZ: {PSZ}', transform=ax1.transAxes, color='red', 
                fontsize=12, fontweight='bold', ha='right', va='top')
        
        # Create combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        ax1.set_title(f'Signal and Surprisal Map - Sample {plot_idx} (Index {idx})')
        
        # Plot 2: Delta and Surprisal Map
        ax2_twin = ax2.twinx()  # Create twin axis for surprisal
        
        # Plot deltas: delta[i] = x[i] - x[i-1] at position i
        # We skip the first position (i=0) since delta[0] = 0 (no previous value)
        delta_positions = ctx_indices[1:]  # Skip first position
        delta_values = delta[1:]  # Skip delta[0] which is 0
        surprisal_for_delta = surprisal[1:]  # Corresponding surprisal values
        
        line3 = ax2.plot(delta_positions, delta_values, 'o-', color='green', linewidth=2, 
                        linestyle='-.', markersize=6, label='Signal Delta (xi - xi-1)')
        ax2.axhline(y=0, color='black', linestyle=':', alpha=0.5, label='Zero Line')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Delta Value', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        ax2.grid(True, alpha=0.3)
        
        # Plot the surprisal map for delta positions
        line4 = ax2_twin.plot(delta_positions, surprisal_for_delta, 's-', color='red', linewidth=2, 
                             linestyle='-.', markersize=6, alpha=0.7, label='Surprisal Score')
        ax2_twin.set_ylabel('Surprisal Score', color='red')
        ax2_twin.tick_params(axis='y', labelcolor='red')
        # Note: Don't fix Y-axis limits as surprisal can have a wide range
        
        # Add patch size in red text on top right
        ax2.text(0.98, 0.95, f'PSZ: {PSZ}', transform=ax2.transAxes, color='red', 
                fontsize=12, fontweight='bold', ha='right', va='top')
        
        # Create combined legend
        lines3, labels3 = ax2.get_legend_handles_labels()
        lines4, labels4 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines3 + lines4, labels3 + labels4, loc='upper left')
        
        ax2.set_title(f'Signal Delta and Surprisal Map - Sample {plot_idx} (Index {idx})')
        
        # Plot 3: Signal vs Predictions and Surprisal Map
        ax3_twin = ax3.twinx()  # Create twin axis for surprisal
        
        # Plot the actual signal and autoregressive predictions
        line5 = ax3.plot(ctx_indices, signal, 'o-', color='blue', linewidth=2, 
                        linestyle='-.', markersize=6, label='Actual Signal')
        line6 = ax3.plot(ctx_indices, predictions, 's-', color='orange', linewidth=2, 
                        linestyle='-.', markersize=6, alpha=0.8, label='AR Predictions')
        ax3.set_xlabel('Time Steps')
        ax3.set_ylabel('Signal Value', color='blue')
        ax3.tick_params(axis='y', labelcolor='blue')
        ax3.grid(True, alpha=0.3)
        
        # Plot the surprisal map
        line7 = ax3_twin.plot(ctx_indices, surprisal, 's-', color='red', linewidth=2, 
                             linestyle='-.', markersize=6, alpha=0.7, label='Surprisal Score')
        ax3_twin.set_ylabel('Surprisal Score', color='red')
        ax3_twin.tick_params(axis='y', labelcolor='red')
        # Note: Don't fix Y-axis limits as surprisal can have a wide range
        
        # Add patch size in red text on top right
        ax3.text(0.98, 0.95, f'PSZ: {PSZ}', transform=ax3.transAxes, color='red', 
                fontsize=12, fontweight='bold', ha='right', va='top')
        
        # Create combined legend
        lines5, labels5 = ax3.get_legend_handles_labels()
        lines7, labels7 = ax3_twin.get_legend_handles_labels()
        ax3.legend(lines5 + lines7, labels5 + labels7, loc='upper left')
        
        ax3.set_title(f'Signal vs AR Predictions and Surprisal Map - Sample {plot_idx} (Index {idx})')
        
        # Save the surprisal and delta analysis plot
        surprisal_plot_filename = os.path.join(results_dir, f"sample_{plot_idx}_surprisal_delta_predictions_analysis.png")
        plt.tight_layout()
        plt.savefig(surprisal_plot_filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved surprisal and delta analysis plot: {surprisal_plot_filename}")
        
    else:
        print(f"Warning: No surprisal data available for sample {idx} (plot {plot_idx})")

    print("\nSurprisal and delta analysis complete!")
else:
    print("\nSurprisal and delta analysis skipped (ENABLE_SURPRISAL=False)")