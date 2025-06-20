# Moirai Multi-Configuration Inference and Comparison

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

# Configuration Lists for Multiple Experiments
CONTEXT_LENGTHS = [2048]  # Different context lengths to test
PATCH_SIZES = [8, 16, 32, 64]       # Different patch sizes to test  
COMPRESSION_RATIOS = [1/2, 1/4, 1/8]  # Different compression ratios to test

# Control flags
ENABLE_SURPRISAL = False  # Set to False to skip slow self-information computation and surprisal-based methods

# Fixed configuration (will be used as baseline)
MODEL = "moirai"  # or "moirai-moe"
SIZE = "large"    # small, base, large
PDT = 64           # Prediction length
BSZ = 32          # Batch size
GPU = 2           # GPU device

# Data configuration
HOME = os.path.expanduser("~")
DATASET_FOLDER = f"{HOME}/time-series/moirai/time-moe-eval/"
MODEL_FOLDER = "Salesforce"

# CSV_PATH = f"{DATASET_FOLDER}/ETT-small/ETTm1.csv"
CSV_PATH = f"{DATASET_FOLDER}/synthetic_sinusoidal.csv"
# CSV_PATH = f"{DATASET_FOLDER}/electricity.csv"

COLUMN = 0        # Column to analyze (0-indexed)

# Test configuration
NUM_WINDOWS = 10  # Test set length
TEST_SAMPLES = int(NUM_WINDOWS * PDT)  # Number of test samples
NUM_SAMPLES = 1000  # Number of samples for probabilistic forecasting

# Storage for all experiment results
ALL_EXPERIMENT_RESULTS = {}

# Load Moirai model once (before experiments)
print("Loading Moirai model...")
base_module = MoiraiModule.from_pretrained(f"{MODEL_FOLDER}/{MODEL}-1.0-R-{SIZE}")

def run_experiment_configuration(CTX, PSZ, compression_ratio):
    """
    Run a complete experiment for a given configuration.
    Returns a dictionary with MAE results for all methods.
    """
    print(f"\nRunning experiment: CTX={CTX}, PSZ={PSZ}, COMP={compression_ratio}")
    
    # Set GPU for this experiment
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
    
    # Calculate derived parameters
    reduced_ctx = int(compression_ratio * CTX)
    reduced_pdt = max(1, int(compression_ratio * PDT))
    
    print(f"Configuration:")
    print(f"  Model: {MODEL}-{SIZE}")
    print(f"  Context Length: {CTX}")
    print(f"  Prediction Length: {PDT}")
    print(f"  Test Length: {NUM_WINDOWS}")
    print(f"  Using GPU: {GPU}")
    print(f"  Patch Size: {PSZ}")
    print(f"  Compression Ratio: {compression_ratio}")
    print(f"  Reduced context: {reduced_ctx}, Reduced prediction: {reduced_pdt}")

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

    # Create results directory for this configuration
    results_dir = f"results_prune_si_v3/{dataset_name}_COL_{COLUMN}/{MODEL}-{SIZE}/CTX{CTX}_PDT{PDT}_PSZ{PSZ}_COMP{compression_ratio}/N_{NUM_SAMPLES}"
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

    # Create models with different configurations
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
    
    print("Models created successfully!")

    # Run inference on test data
    print("Running inference...")
    input_data = list(test_data.input)
    label_data = list(test_data.label)

    # Method 1: Baseline (full context, full prediction)
    print("Running baseline inference...")
    forecasts_baseline = list(tqdm(predictor.predict(input_data), desc="Baseline forecasts", total=len(input_data)))
    
    # Process baseline results
    mae_baseline = []
    for i, (forecast, label_item) in enumerate(zip(forecasts_baseline, label_data)):
        ground_truth = label_item['target'][:PDT]
        prediction = np.mean(forecast.samples, axis=0)
        mae = np.mean(np.abs(prediction - ground_truth))
        mae_baseline.append(mae)
    
    # Method 2: Reduced context and prediction
    print("Running reduced context+prediction inference...")
    # Create input data with reduced context
    input_data_reduced = []
    for item in input_data:
        reduced_item = item.copy()
        original_target = item['target']
        if len(original_target) > reduced_ctx:
            reduced_item['target'] = original_target[-reduced_ctx:]
        input_data_reduced.append(reduced_item)
    
    forecasts_reduced = list(tqdm(predictor_reduced_ctx_pdt.predict(input_data_reduced), desc="Reduced forecasts", total=len(input_data_reduced)))
    
    # Process reduced results (match prediction length)
    mae_reduced = []
    for i, (forecast, label_item) in enumerate(zip(forecasts_reduced, label_data)):
        full_ground_truth = label_item['target'][:PDT]
        # Downsample ground truth to match reduced prediction length
        downsample_step = int(1 / compression_ratio)
        downsampled_ground_truth = full_ground_truth[::downsample_step][:reduced_pdt]
        
        prediction = np.mean(forecast.samples, axis=0)
        mae = np.mean(np.abs(prediction - downsampled_ground_truth))
        mae_reduced.append(mae)
    
    # Method 3: Truncated context (same prediction length)
    print("Running truncated context inference...")
    forecasts_truncated = list(tqdm(predictor_reduced_ctx.predict(input_data_reduced), desc="Truncated forecasts", total=len(input_data_reduced)))
    
    # Process truncated results
    mae_truncated = []
    for i, (forecast, label_item) in enumerate(zip(forecasts_truncated, label_data)):
        ground_truth = label_item['target'][:PDT]
        prediction = np.mean(forecast.samples, axis=0)
        mae = np.mean(np.abs(prediction - ground_truth))
        mae_truncated.append(mae)

    # Calculate mean MAEs and standard errors
    results = {
        'baseline': {
            'mae_mean': np.mean(mae_baseline),
            'mae_std': np.std(mae_baseline),
            'mae_sem': np.std(mae_baseline) / np.sqrt(len(mae_baseline))
        },
        'reduced_ctx_pdt': {
            'mae_mean': np.mean(mae_reduced),
            'mae_std': np.std(mae_reduced),
            'mae_sem': np.std(mae_reduced) / np.sqrt(len(mae_reduced))
        },
        'truncated_ctx': {
            'mae_mean': np.mean(mae_truncated),
            'mae_std': np.std(mae_truncated),
            'mae_sem': np.std(mae_truncated) / np.sqrt(len(mae_truncated))
        },
        'config': {
            'CTX': CTX,
            'PSZ': PSZ,
            'compression_ratio': compression_ratio,
            'reduced_ctx': reduced_ctx,
            'reduced_pdt': reduced_pdt
        }
    }
    
    print(f"Results for CTX={CTX}, PSZ={PSZ}, COMP={compression_ratio}:")
    print(f"  Baseline MAE: {results['baseline']['mae_mean']:.4f} ± {results['baseline']['mae_sem']:.4f}")
    print(f"  Reduced CTX+PDT MAE: {results['reduced_ctx_pdt']['mae_mean']:.4f} ± {results['reduced_ctx_pdt']['mae_sem']:.4f}")
    print(f"  Truncated CTX MAE: {results['truncated_ctx']['mae_mean']:.4f} ± {results['truncated_ctx']['mae_sem']:.4f}")
    
    return results

def run_all_experiments():
    """Run experiments for all combinations of configurations."""
    
    all_results = {}
    
    for CTX in CONTEXT_LENGTHS:
        for PSZ in PATCH_SIZES:
            for compression_ratio in COMPRESSION_RATIOS:
                
                config_key = f"CTX{CTX}_PSZ{PSZ}_COMP{compression_ratio}"
                print(f"\n{'='*80}")
                print(f"RUNNING EXPERIMENT: {config_key}")
                print(f"{'='*80}")
                
                try:
                    # Execute the experiment
                    experiment_results = run_experiment_configuration(CTX, PSZ, compression_ratio)
                    all_results[config_key] = experiment_results
                    print(f"✓ Completed experiment: {config_key}")
                    
                except Exception as e:
                    print(f"✗ Failed experiment: {config_key} - Error: {e}")
                    all_results[config_key] = None
    
    return all_results

def plot_comparison_results(all_results):
    """Generate comprehensive comparison plot with error bars."""
    
    # Filter successful results
    successful_results = {k: v for k, v in all_results.items() if v is not None}
    
    if not successful_results:
        print("No successful experiments to plot!")
        return
    
    # Prepare data for plotting
    config_labels = []
    baseline_means = []
    baseline_sems = []
    reduced_means = []
    reduced_sems = []
    truncated_means = []
    truncated_sems = []
    
    for config_key, results in successful_results.items():
        # Create readable label
        config = results['config']
        label = f"CTX{config['CTX']}\nPSZ{config['PSZ']}\nCOMP{config['compression_ratio']}"
        config_labels.append(label)
        
        # Extract results
        baseline_means.append(results['baseline']['mae_mean'])
        baseline_sems.append(results['baseline']['mae_sem'])
        
        reduced_means.append(results['reduced_ctx_pdt']['mae_mean'])
        reduced_sems.append(results['reduced_ctx_pdt']['mae_sem'])
        
        truncated_means.append(results['truncated_ctx']['mae_mean'])
        truncated_sems.append(results['truncated_ctx']['mae_sem'])
    
    # Create comparison plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    
    x = np.arange(len(config_labels))
    width = 0.25
    
    # Create bars with error bars
    bars1 = ax.bar(x - width, baseline_means, width, yerr=baseline_sems,
                   label='Baseline (Full CTX+PDT)', alpha=0.8, capsize=5)
    bars2 = ax.bar(x, reduced_means, width, yerr=reduced_sems,
                   label='Reduced CTX+PDT', alpha=0.8, capsize=5)
    bars3 = ax.bar(x + width, truncated_means, width, yerr=truncated_sems,
                   label='Truncated CTX', alpha=0.8, capsize=5)
    
    # Customize plot
    ax.set_xlabel('Configuration (Context Length / Patch Size / Compression Ratio)')
    ax.set_ylabel('Mean Absolute Error (MAE)')
    ax.set_title('MAE Comparison Across Different Configurations\n(Error bars show standard error of the mean)')
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    def add_value_labels(bars, means, sems):
        for bar, mean, sem in zip(bars, means, sems):
            height = bar.get_height()
            ax.annotate(f'{mean:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    add_value_labels(bars1, baseline_means, baseline_sems)
    add_value_labels(bars2, reduced_means, reduced_sems)
    add_value_labels(bars3, truncated_means, truncated_sems)
    
    plt.tight_layout()
    
    # Save plot
    dataset_name = os.path.splitext(os.path.basename(CSV_PATH))[0]
    plot_path = f"results_prune_si_v3/{dataset_name}_comparison_plot.png"
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {plot_path}")
    
    plt.show()
    
    # Print summary table
    print("\n" + "="*100)
    print("EXPERIMENT SUMMARY")
    print("="*100)
    print(f"{'Configuration':<25} {'Baseline MAE':<15} {'Reduced MAE':<15} {'Truncated MAE':<15}")
    print("-"*100)
    
    for i, (config_key, results) in enumerate(successful_results.items()):
        config = results['config']
        config_str = f"CTX{config['CTX']}_PSZ{config['PSZ']}_C{config['compression_ratio']}"
        baseline_str = f"{results['baseline']['mae_mean']:.4f}±{results['baseline']['mae_sem']:.4f}"
        reduced_str = f"{results['reduced_ctx_pdt']['mae_mean']:.4f}±{results['reduced_ctx_pdt']['mae_sem']:.4f}"
        truncated_str = f"{results['truncated_ctx']['mae_mean']:.4f}±{results['truncated_ctx']['mae_sem']:.4f}"
        
        print(f"{config_str:<25} {baseline_str:<15} {reduced_str:<15} {truncated_str:<15}")

if __name__ == "__main__":
    # Run all experiments
    print("Starting multi-configuration experiment...")
    print(f"Will run {len(CONTEXT_LENGTHS)} × {len(PATCH_SIZES)} × {len(COMPRESSION_RATIOS)} = {len(CONTEXT_LENGTHS) * len(PATCH_SIZES) * len(COMPRESSION_RATIOS)} experiments")
    
    all_results = run_all_experiments()
    
    # Generate comparison plot
    print("\nGenerating comparison plot...")
    plot_comparison_results(all_results)
    
    print("\nAll experiments completed!")
