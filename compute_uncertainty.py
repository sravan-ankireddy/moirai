# Moirai Multi-Configuration Uncertainty Analysis
# Tests multiple configurations of context length, prediction length, and patch size
# Analyzes uncertainty in context using autoregressive predictions
# Creates summary plots comparing Mean Absolute Error across all configurations

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
CTX_LIST = [64, 256, 512, 1024]     # List of context lengths to test
PDT_LIST = [8]        # List of prediction lengths to test
PSZ_LIST = ["auto", 1, 8, 16, 32, 64, 128]    # List of patch sizes to test
BSZ = 128                     # Batch size
GPU = 1                       # GPU device
TEST_LENGTH = 10              # Test set length

# Data configuration
CSV_PATH = "/home/sa53869/time-series/moirai/time-moe-eval/synthetic_sinusoidal.csv"
COLUMN = 0        # Column to analyze (0-indexed)

# Load Moirai model
print("Loading Moirai model...")
base_module = MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-{SIZE}")

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)

print(f"Configuration Summary:")
print(f"  Model: {MODEL}-{SIZE}")
print(f"  Context Lengths to Test: {CTX_LIST}")
print(f"  Prediction Lengths to Test: {PDT_LIST}")
print(f"  Patch Sizes to Test: {PSZ_LIST}")
print(f"  Test Windows: {TEST_LENGTH}")
print(f"  GPU Device: {GPU}")

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

# Create GluonTS dataset
ds = PandasDataset(dict(df_selected))

# Store results for all configurations
all_results = []

# Loop through all configurations
total_configs = len(CTX_LIST) * len(PDT_LIST) * len(PSZ_LIST)
config_count = 0

for CTX in CTX_LIST:
    for PDT in PDT_LIST:
        for PSZ in PSZ_LIST:
            config_count += 1
            print(f"\n{'='*50}")
            print(f"Testing Configuration {config_count}/{total_configs}")
            print(f"Context Length: {CTX}, Prediction Length: {PDT}, Patch Size: {PSZ}")
            print(f"{'='*50}")
            
            # Create results directory structure for this configuration
            # Parent folder for all configurations
            main_results_dir = f"results_uncertainty/{dataset_name}/{MODEL}-{SIZE}"
            config_parent_dir = f"{main_results_dir}/configurations"
            results_dir = f"{config_parent_dir}/CTX{CTX}_PDT{PDT}_PSZ{PSZ}"
            os.makedirs(results_dir, exist_ok=True)
            print(f"Configuration results directory: {results_dir}")
            
            # Create train/test split
            train, test_template = split(ds, offset=-TEST_LENGTH*PDT)
            
            # Generate test instances
            test_data = test_template.generate_instances(
                prediction_length=PDT,
                windows=TEST_LENGTH,
                distance=PDT,
            )
            
            print(f"Test windows generated: {TEST_LENGTH}")

            # Create model with specific configuration
            model = MoiraiForecast(
                module=base_module,
                prediction_length=PDT,
                context_length=CTX,
                patch_size=PSZ,
                num_samples=100,  # Number of probabilistic samples
                target_dim=1,
                feat_dynamic_real_dim=ds.num_feat_dynamic_real,
                past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
            )
            
            # Create model for autoregressive inference (single-step predictions)
            model_ar = MoiraiForecast(
                module=base_module,
                prediction_length=1,
                context_length=CTX,
                patch_size=PSZ,
                num_samples=100,  # Number of probabilistic samples
                target_dim=1,
                feat_dynamic_real_dim=ds.num_feat_dynamic_real,
                past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
            )
            
            # Create predictors for both models
            predictor = model.create_predictor(batch_size=BSZ)
            predictor_ar = model_ar.create_predictor(batch_size=BSZ)
            print("Models initialized successfully!")
            
            # Run inference on test data
            print("Running main forecasting...")
            input_data = list(test_data.input)
            label_data = list(test_data.label)
            
            # Generate multi-step predictions
            print("Generating multi-step forecasts...")
            forecasts = list(tqdm(predictor.predict(input_data), 
                                 total=len(input_data), 
                                 desc="Multi-step forecasts"))
            
            print(f"Generated {len(forecasts)} multi-step forecasts")
            
            # Prepare data for analysis
            sample_results = []
            full_data_values = df_selected[selected_column].values
            
            print("Analyzing samples and computing uncertainty...")
            for i, (input_item, label_item, forecast) in enumerate(tqdm(zip(input_data, label_data, forecasts), 
                                                                       total=len(input_data), 
                                                                       desc="Processing samples")):
                # Extract context data
                context = input_item['target']

                # Generate autoregressive predictions for uncertainty analysis
                input_data_context = []
                context_range = range(len(context) - 2*CTX, len(context) - CTX)
                for j in tqdm(context_range, desc=f"AR windows (sample {i+1})", leave=False):
                    # Create input for autoregressive prediction
                    ar_item = input_item.copy()

                    # Extract context window of length CTX starting from position j
                    ar_item['target'] = context[j:j + CTX]
                    input_data_context.append(ar_item)

                # Generate single-step autoregressive predictions
                forecasts_ar = list(tqdm(predictor_ar.predict(input_data_context), 
                                        total=len(input_data_context),
                                        desc=f"AR forecasts (sample {i+1})", 
                                        leave=False))
                
                # Use the last CTX values as context for main prediction
                if len(context) > CTX:
                    context = context[-CTX:]

                # Calculate uncertainty from autoregressive predictions
                # Uncertainty = coefficient of variation (std/mean)
                context_samples = np.array([f.samples for f in forecasts_ar])
                context_mean = np.mean(context_samples, axis=1)
                context_std = np.std(context_samples, axis=1)
                context_uncertainty = context_std / (np.abs(context_mean) + 1e-8)

                # Extract ground truth values
                ground_truth = label_item['target'][:PDT]
                
                # Extract prediction (mean of probabilistic samples)
                prediction = np.mean(forecast.samples, axis=0)
                
                # Store analysis results
                sample_results.append({
                    'window_id': i,
                    'context': context,
                    'ground_truth': ground_truth,
                    'prediction': prediction,
                    'mae': np.mean(np.abs(prediction - ground_truth)),
                    'context_uncertainty': context_uncertainty,
                    'context_ar_predictions': context_mean.flatten()  # AR predictions for context
                })

            print(f"Processed {len(sample_results)} test samples")
            
            # Calculate Mean Absolute Error for this configuration
            config_mae = np.mean([r['mae'] for r in sample_results])
            
            # Store configuration results for comparison
            all_results.append({
                'CTX': CTX,
                'PDT': PDT,
                'PSZ': PSZ,
                'config_name': f"CTX{CTX}_PDT{PDT}_PSZ{PSZ}",
                'mae': config_mae,
                'sample_results': sample_results
            })
            
            print(f"Configuration MAE: {config_mae:.4f}")

print(f"\n{'='*60}")
print("ALL CONFIGURATIONS TESTED SUCCESSFULLY")
print(f"{'='*60}")

# Create summary comparison plot
print("Creating MAE comparison plot...")
config_names = [r['config_name'] for r in all_results]
mae_values = [r['mae'] for r in all_results]

# Calculate standard errors for error bars
mae_std_errors = []
for result in all_results:
    individual_maes = [sample['mae'] for sample in result['sample_results']]
    std_error = np.std(individual_maes) / np.sqrt(len(individual_maes))  # Standard error of the mean
    mae_std_errors.append(std_error)

plt.figure(figsize=(15, 8))
bars = plt.bar(range(len(config_names)), mae_values, 
               yerr=mae_std_errors,  # Add error bars
               capsize=5,  # Error bar cap size
               error_kw={'elinewidth': 2, 'capthick': 2},  # Error bar styling
               color=['steelblue', 'forestgreen', 'firebrick', 'darkorange', 'purple', 'brown', 'hotpink', 'gray', 'olive'][:len(config_names)])

# Add MAE values on top of each bar (adjust position to account for error bars)
for i, (bar, mae, std_err) in enumerate(zip(bars, mae_values, mae_std_errors)):
    plt.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + std_err + 0.001, 
             f'{mae:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.xlabel('Configuration (Context_Prediction_PatchSize)', fontsize=12)
plt.ylabel('Mean Absolute Error (MAE)', fontsize=12)
plt.title(f'Mean Absolute Error Comparison Across All Configurations\nModel: {MODEL}-{SIZE} | Dataset: {dataset_name}', fontsize=14)
plt.xticks(range(len(config_names)), config_names, rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save summary comparison plot
summary_filename = f"results_uncertainty/{dataset_name}/{MODEL}-{SIZE}/summary_mae_comparison.png"
os.makedirs(os.path.dirname(summary_filename), exist_ok=True)
plt.savefig(summary_filename, dpi=300, bbox_inches='tight')
plt.show()
print(f"Summary plot saved: {summary_filename}")

# Print detailed results table
print(f"\n{'='*60}")
print("DETAILED RESULTS TABLE:")
print(f"{'='*60}")
for result in all_results:
    ctx_str = f"{result['CTX']:3d}"
    pdt_str = f"{result['PDT']:2d}"
    psz_str = f"{str(result['PSZ']):>4s}"
    mae_str = f"{result['mae']:.4f}"
    print(f"Context: {ctx_str} | Prediction: {pdt_str} | Patch: {psz_str} | MAE: {mae_str}")

# Identify best performing configuration
best_config = min(all_results, key=lambda x: x['mae'])
print(f"\nBest Performing Configuration:")
print(f"  Context Length: {best_config['CTX']}")
print(f"  Prediction Length: {best_config['PDT']}")
print(f"  Patch Size: {best_config['PSZ']}")
print(f"  Mean Absolute Error: {best_config['mae']:.4f}")

# Optional: Create detailed visualization plots for the best configuration
CREATE_DETAILED_PLOTS = True  # Set to False to skip detailed plotting for best configuration

if CREATE_DETAILED_PLOTS:
    print(f"\nGenerating detailed plots for best performing configuration...")
    best_samples = best_config['sample_results']
    CTX, PDT, PSZ = best_config['CTX'], best_config['PDT'], best_config['PSZ']
    
    # Create results directory for best configuration
    main_results_dir = f"results_uncertainty/{dataset_name}/{MODEL}-{SIZE}"
    best_results_dir = f"{main_results_dir}/best_configuration_CTX{CTX}_PDT{PDT}_PSZ{PSZ}"
    os.makedirs(best_results_dir, exist_ok=True)
    
    # Plot sample visualizations from best configuration
    num_samples = min(3, len(best_samples))
    sample_indices = np.random.choice(len(best_samples), num_samples, replace=False)
    
    for plot_idx, idx in enumerate(sample_indices, 1):
        result = best_samples[idx]
        
        # Forecast comparison plot
        plt.figure(figsize=(15, 6))
        context_len = len(result['context'])
        context_indices = np.arange(-context_len, 0)
        forecast_indices = np.arange(0, PDT)
        
        plt.plot(context_indices, result['context'], label='Actual Context', color='steelblue', linewidth=2, linestyle='--')
        ar_pred_indices = np.arange(-len(result['context_ar_predictions']), 0)
        plt.plot(ar_pred_indices, result['context_ar_predictions'], label='Predicted Context (AR)', 
                 color='darkorange', linewidth=2, linestyle='--', marker='x', markersize=3, alpha=0.8)
        plt.plot(forecast_indices, result['ground_truth'], label='Ground Truth', color='forestgreen', 
                 linewidth=3, marker='o', markersize=4, linestyle='--')
        plt.plot(forecast_indices, result['prediction'], label='Model Prediction', color='firebrick', 
                 linewidth=2, linestyle='--', marker='s', markersize=4)
        plt.axvline(x=0, color='black', linestyle=':', alpha=0.7, label='Prediction Start')
        plt.title(f"Best Configuration - CTX{CTX}_PDT{PDT}_PSZ{PSZ} - Sample {result['window_id']} - MAE: {result['mae']:.4f}", loc='left')
        plt.title(f"Patch Size: {PSZ}", loc='right', color='red')
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        plot_filename = os.path.join(best_results_dir, f"best_sample_{plot_idx}_forecast.png")
        plt.tight_layout()
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Forecast plot saved: {plot_filename}")
        
        # Uncertainty analysis plot
        fig, ax1 = plt.subplots(figsize=(15, 8))
        
        color1 = 'steelblue'
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Time Series Value', color=color1)
        ax1.plot(context_indices, result['context'], color=color1, linewidth=2, 
                 label='Actual Context', marker='o', markersize=3, linestyle='--')
        ax1.plot(ar_pred_indices, result['context_ar_predictions'], color='darkorange', linewidth=2, 
                 label='Predicted Context (AR)', marker='x', markersize=3, linestyle='--', alpha=0.8)
        ax1.plot(forecast_indices, result['ground_truth'], color='forestgreen', 
                 linewidth=2, marker='o', markersize=4, alpha=0.7, label='Ground Truth', linestyle='--')
        ax1.plot(forecast_indices, result['prediction'], color='firebrick', 
                 linewidth=2, linestyle='--', marker='s', markersize=4, alpha=0.7, label='Model Prediction')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)
        
        ax2 = ax1.twinx()
        color2 = 'crimson'
        ax2.set_ylabel('Uncertainty (Coefficient of Variation)', color=color2)
        uncertainty_indices = np.arange(-len(result['context_uncertainty']), 0)
        ax2.plot(uncertainty_indices, result['context_uncertainty'], color=color2, 
                 linewidth=3, alpha=0.8, label='Context Uncertainty', marker='s', markersize=2, linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        ax1.axvline(x=0, color='black', linestyle=':', alpha=0.7, linewidth=2, label='Prediction Start')
        
        fig.suptitle(f"Best Configuration Uncertainty Analysis - CTX{CTX}_PDT{PDT}_PSZ{PSZ} - Sample {result['window_id']} - MAE: {result['mae']:.4f}", 
                     fontsize=14, x=0.1, ha='left')
        fig.suptitle(f"Patch Size: {PSZ}", fontsize=14, x=0.9, ha='right', color='red')
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        uncertainty_filename = os.path.join(best_results_dir, f"best_sample_{plot_idx}_uncertainty.png")
        plt.tight_layout()
        plt.savefig(uncertainty_filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Uncertainty plot saved: {uncertainty_filename}")

print(f"\n{'='*60}")
print("MULTI-CONFIGURATION UNCERTAINTY ANALYSIS COMPLETE")
print(f"{'='*60}")
print(f"Tested {len(all_results)} configurations")
print(f"Best configuration: CTX{best_config['CTX']}_PDT{best_config['PDT']}_PSZ{best_config['PSZ']} (MAE: {best_config['mae']:.4f})")
print(f"Results saved in: results_uncertainty/{dataset_name}/{MODEL}-{SIZE}/")
print(f"  - Summary plots: results_uncertainty/{dataset_name}/{MODEL}-{SIZE}/")
print(f"  - Configuration details: results_uncertainty/{dataset_name}/{MODEL}-{SIZE}/configurations/")
print(f"  - Best config plots: results_uncertainty/{dataset_name}/{MODEL}-{SIZE}/best_configuration_*/")
