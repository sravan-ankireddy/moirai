# Moirai Inference and Visualization - Grid Search Version

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

# Configuration Lists for Grid Search
MODEL = "moirai"  # or "moirai-moe"
SIZE = "large"    # small, base, large
CTX_LIST = [512, 1024, 2048]          # Context length list
PDT = 64           # Prediction length
BSZ = 32          # Batch size
GPU = 2           # GPU device
PSZ_LIST = ["auto", 16, 32, 64]       # Patch size list
PSZ_surprisal = "auto"
COMPRESSION_RATIO_LIST = [1/2, 1/4, 1/8]  # Compression ratio list

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
NUM_WINDOWS = 10  # Test set length
TEST_SAMPLES = int(NUM_WINDOWS * PDT)  # Number of test samples
NUM_SAMPLES = 1000  # Number of samples for probabilistic forecasting

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)

print(f"Grid Search Configuration:")
print(f"  Model: {MODEL}-{SIZE}")
print(f"  Context Lengths: {CTX_LIST}")
print(f"  Patch Sizes: {PSZ_LIST}")
print(f"  Compression Ratios: {COMPRESSION_RATIO_LIST}")
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

# Initialize comprehensive results storage
all_results = []

# Grid search over all parameter combinations
total_combinations = len(CTX_LIST) * len(PSZ_LIST) * len(COMPRESSION_RATIO_LIST)
print(f"\nStarting grid search over {total_combinations} parameter combinations...")

combination_idx = 0
for CTX in CTX_LIST:
    for PSZ in PSZ_LIST:
        for compression_ratio in COMPRESSION_RATIO_LIST:
            combination_idx += 1
            print(f"\n{'='*80}")
            print(f"COMBINATION {combination_idx}/{total_combinations}")
            print(f"CTX={CTX}, PSZ={PSZ}, COMPRESSION_RATIO={compression_ratio}")
            print(f"{'='*80}")
            
            # Create results directory for this combination
            results_dir = f"results_prune_si_grid/{dataset_name}_COL_{COLUMN}/{MODEL}-{SIZE}/CTX{CTX}_PDT{PDT}_PSZ{PSZ}_COMP{compression_ratio}/N_{NUM_SAMPLES}"
            os.makedirs(results_dir, exist_ok=True)
            print(f"Results will be saved to: {results_dir}")

            # Run inference on test data
            print("Running inference...")
            input_data = list(test_data.input)
            label_data = list(test_data.label)

            # Create model with specific configuration
            model = MoiraiForecast(
                module=base_module,
                prediction_length=PDT,
                context_length=CTX,
                patch_size=PSZ,
                num_samples=NUM_SAMPLES,
                target_dim=1,
                feat_dynamic_real_dim=ds.num_feat_dynamic_real,
                past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
            )

            # Create model with reduced context length AND reduced prediction length
            reduced_ctx = int(compression_ratio * CTX)
            reduced_pdt = max(1, int(compression_ratio * PDT))
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

            # Create predictors
            predictor = model.create_predictor(batch_size=BSZ)
            predictor_reduced_ctx_pdt = model_reduced_ctx_pdt.create_predictor(batch_size=BSZ)
            predictor_reduced_ctx = model_reduced_ctx.create_predictor(batch_size=BSZ)
            print("Model loaded successfully!")
            print(f"Reduced model config: context={reduced_ctx}, prediction={reduced_pdt}")

            # Create a dataset for the downsampling experiment: downsample the input but not the labels
            print("\nPreparing downsampled context data...")
            input_data_downsampled = []
            for input_item in input_data:
                downsampled_item = input_item.copy()
                target = input_item['target']
                
                # Downsample the target by taking every nth sample where n = 1/compression_ratio
                n = int(1 / compression_ratio)
                downsampled_target = target[::n]
                
                downsampled_item['target'] = downsampled_target
                input_data_downsampled.append(downsampled_item)

            print("\nRunning main inference...")
            forecasts = list(tqdm(predictor.predict(input_data), desc="Main forecasts", total=len(input_data)))

            print("Running downsampled context inference...")
            forecasts_downsampled = list(tqdm(predictor_reduced_ctx_pdt.predict(input_data_downsampled), desc="Downsampled forecasts", total=len(input_data_downsampled)))

            print(f"Generated {len(forecasts)} forecasts")

            # Prepare data for visualization
            print("\nProcessing main results...")
            sample_results = []

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
                    'prediction': prediction_upsampled,
                    'prediction_downsampled': prediction_downsampled,
                    'prediction_upsampled': prediction_upsampled,
                    'reduced_pdt': reduced_pdt,
                    'mae': np.mean(np.abs(prediction_upsampled - ground_truth))
                })

            # NEW EXPERIMENT: Direct downsampling of ground truth labels (no resampling of forecast)
            print("\nProcessing direct downsampled ground truth experiment...")
            sample_results_direct_downsample = []

            for i, (input_item, label_item, forecast) in enumerate(tqdm(zip(input_data_downsampled, label_data, forecasts_downsampled), desc="Processing direct downsample results", total=len(input_data_downsampled))):
                # Get context data
                context = input_item['target']
                
                # keep the last `reduced_ctx` values for context
                if len(context) > reduced_ctx:
                    context = context[-reduced_ctx:]

                # Get ground truth (still full PDT length for now)
                ground_truth_full = label_item['target'][:PDT]
                
                # Downsample ground truth to match reduced prediction length
                ground_truth_indices = np.linspace(0, PDT-1, reduced_pdt)
                ground_truth_downsampled = np.interp(ground_truth_indices, np.arange(PDT), ground_truth_full)
                
                # Get prediction (mean of samples) - this is length reduced_pdt
                prediction_downsampled = np.mean(forecast.samples, axis=0)

                # Calculate MAE against downsampled ground truth
                mae_downsampled = np.mean(np.abs(prediction_downsampled - ground_truth_downsampled))
                
                # Store results
                sample_results_direct_downsample.append({
                    'window_id': i,
                    'context': context,
                    'ground_truth': ground_truth_full,  # Keep full for consistency
                    'ground_truth_downsampled': ground_truth_downsampled,  # Add downsampled version
                    'prediction': prediction_downsampled,  # Use downsampled prediction
                    'reduced_pdt': reduced_pdt,
                    'mae': mae_downsampled  # MAE against downsampled ground truth
                })

            # Create downsampled input data through random sampling
            print("\nPreparing randomly sampled data...")
            input_data_random_sampled = []
            for input_item in input_data:
                sampled_item = input_item.copy()
                target = input_item['target']
                
                # Keep only the last CTX samples
                if len(target) > CTX:
                    target = target[-CTX:]
                
                # Randomly select reduced_ctx indices
                total_samples = len(target)
                selected_indices = sorted(np.random.choice(total_samples, reduced_ctx, replace=False))
                sampled_target = target[selected_indices]
                
                sampled_item['target'] = sampled_target
                input_data_random_sampled.append(sampled_item)

            print("Running random sampling inference...")
            forecasts_random_sampled = list(tqdm(predictor_reduced_ctx.predict(input_data_random_sampled), desc="Random sampled forecasts", total=len(input_data_random_sampled)))

            # Process random sampling results
            print("\nProcessing random sampling results...")
            sample_results_random_sampled = []
            for i, (input_item, label_item, forecast) in enumerate(tqdm(zip(input_data_random_sampled, label_data, forecasts_random_sampled), desc="Processing random sampling results", total=len(input_data_random_sampled))):
                context = input_item['target']
                ground_truth = label_item['target'][:PDT]
                prediction = np.mean(forecast.samples, axis=0)
                
                sample_results_random_sampled.append({
                    'window_id': i,
                    'context': context,
                    'ground_truth': ground_truth,
                    'prediction': prediction,
                    'mae': np.mean(np.abs(prediction - ground_truth))
                })

            # Create interpolated context experiments
            print("\nPreparing interpolated context data...")
            input_data_reduced_interpolated = []

            for input_item in input_data:
                reduced_item = input_item.copy()
                target = input_item['target']
                
                # Keep only the last CTX samples
                if len(target) > CTX:
                    target = target[-CTX:]
                
                # Calculate how many samples to replace (50% removal)
                num_replace = int(0.5 * len(target))
                indices_to_replace = np.random.choice(len(target), num_replace, replace=False)
                
                # Create a copy and set selected indices to 0 (creating missing values)
                target_with_zeros = target.copy()
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
                    'reduced_pdt': reduced_pdt,
                    'mae': np.mean(np.abs(prediction_truncated - ground_truth))
                })

            # Calculate average MAE for all methods for this combination
            print("\n" + "="*60)
            print(f"SUMMARY FOR CTX={CTX}, PSZ={PSZ}, COMPRESSION_RATIO={compression_ratio}")
            print("="*60)

            methods_summary = {
                'Full Context': np.mean([r['mae'] for r in sample_results]),
                'Downsampled (50%)': np.mean([r['mae'] for r in sample_results_downsampled]),
                'Direct Downsample Input→Output': np.mean([r['mae'] for r in sample_results_direct_downsample]),
                'Interpolated (50% replaced)': np.mean([r['mae'] for r in sample_results_interpolated]),
                'Truncated (recent values)': np.mean([r['mae'] for r in sample_results_truncated]),
                'Random Sampling (50%)': np.mean([r['mae'] for r in sample_results_random_sampled]),
            }

            print("Method Performance (Mean Absolute Error):")
            for method, mae in methods_summary.items():
                print(f"  {method:<40}: {mae:.4f}")

            best_method = min(methods_summary.items(), key=lambda x: x[1])
            print(f"\nBest performing method: {best_method[0]} (MAE: {best_method[1]:.4f})")

            # Store results for this combination
            combination_result = {
                'CTX': CTX,
                'PSZ': PSZ,
                'compression_ratio': compression_ratio,
                'methods_mae': methods_summary.copy(),
                'best_method': best_method[0],
                'best_mae': best_method[1]
            }
            all_results.append(combination_result)

print("\n" + "="*100)
print("COMPREHENSIVE RESULTS SUMMARY ACROSS ALL PARAMETER COMBINATIONS")
print("="*100)

# Create comprehensive summary table
summary_df_data = []
for result in all_results:
    for method, mae in result['methods_mae'].items():
        summary_df_data.append({
            'CTX': result['CTX'],
            'PSZ': result['PSZ'],
            'Compression_Ratio': result['compression_ratio'],
            'Method': method,
            'MAE': mae
        })

summary_df = pd.DataFrame(summary_df_data)

# Print overall best performing combinations
print("\nTOP 10 BEST PERFORMING CONFIGURATIONS:")
print("-" * 80)
best_configs = []
for result in all_results:
    best_configs.append({
        'CTX': result['CTX'],
        'PSZ': result['PSZ'],
        'Compression_Ratio': result['compression_ratio'],
        'Best_Method': result['best_method'],
        'Best_MAE': result['best_mae']
    })

best_configs_df = pd.DataFrame(best_configs)
best_configs_df = best_configs_df.sort_values('Best_MAE').head(10)

for idx, row in best_configs_df.iterrows():
    print(f"Rank {idx+1:2d}: CTX={row['CTX']:4d}, PSZ={str(row['PSZ']):4s}, "
          f"Comp_Ratio={row['Compression_Ratio']:.2f}, "
          f"Method={row['Best_Method']:<30s}, MAE={row['Best_MAE']:.4f}")

# Method comparison across all configurations
print(f"\nMETHOD PERFORMANCE ACROSS ALL {len(all_results)} CONFIGURATIONS:")
print("-" * 80)
method_performance = {}
for result in all_results:
    for method, mae in result['methods_mae'].items():
        if method not in method_performance:
            method_performance[method] = []
        method_performance[method].append(mae)

print("Method                                   | Mean MAE   | Std MAE   | Min MAE   | Max MAE")
print("-" * 80)
for method in sorted(method_performance.keys()):
    maes = method_performance[method]
    mean_mae = np.mean(maes)
    std_mae = np.std(maes)
    min_mae = np.min(maes)
    max_mae = np.max(maes)
    print(f"{method:<40s} | {mean_mae:8.4f} | {std_mae:7.4f} | {min_mae:7.4f} | {max_mae:7.4f}")

# Save comprehensive results
results_summary_file = f"results_prune_si_grid/{dataset_name}_COL_{COLUMN}_comprehensive_summary.csv"
os.makedirs(os.path.dirname(results_summary_file), exist_ok=True)
summary_df.to_csv(results_summary_file, index=False)
print(f"\nComprehensive results saved to: {results_summary_file}")

best_configs_file = f"results_prune_si_grid/{dataset_name}_COL_{COLUMN}_best_configurations.csv"
best_configs_df.to_csv(best_configs_file, index=False)
print(f"Best configurations saved to: {best_configs_file}")

print(f"\nGrid search complete! Processed {total_combinations} parameter combinations.")
