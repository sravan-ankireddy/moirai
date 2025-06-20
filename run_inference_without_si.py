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

def create_visualization_plots(input_data, input_data_reduced, label_data, forecasts_baseline, forecasts_reduced, forecasts_truncated, results_dir, reduced_ctx, reduced_pdt, compression_ratio, CTX, PSZ):
    """Create and save visualization plots for the experiment."""
    
    # Select a few samples to visualize (first 3 windows)
    num_plots = min(3, len(input_data))
    
    for i in range(num_plots):
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Get data for this sample
        input_item = input_data[i]
        input_item_reduced = input_data_reduced[i]
        label_item = label_data[i]
        
        full_context = input_item['target']
        reduced_context = input_item_reduced['target']
        ground_truth = label_item['target'][:PDT]
        
        # Get predictions
        pred_baseline = np.mean(forecasts_baseline[i].samples, axis=0)
        pred_reduced = np.mean(forecasts_reduced[i].samples, axis=0)
        pred_truncated = np.mean(forecasts_truncated[i].samples, axis=0)
        
        # Use consistent time scale: show only last CTX window for all plots
        # This ensures fair visual comparison across all methods
        display_context_length = CTX
        
        # For baseline: take last CTX steps from full context
        baseline_context = full_context[-display_context_length:]
        baseline_context_time = np.arange(-display_context_length, 0)
        
        # For reduced context: it's downsampled, not truncated
        # The reduced_context represents every nth sample from the original context
        # So we need to space them out on the original time scale
        downsample_step = int(1 / compression_ratio)
        
        # Create time indices for downsampled context - maintain original spacing
        # If we have reduced_ctx samples, they represent every downsample_step-th sample
        # from the last (reduced_ctx * downsample_step) samples of the original
        original_context_length = min(len(full_context), reduced_ctx * downsample_step)
        downsampled_context_time = np.arange(-original_context_length, 0, downsample_step)[-len(reduced_context):]
        
        pred_time = np.arange(0, PDT)
        
        # Plot 1: Baseline (Show last CTX window + Full Prediction)
        axes[0].plot(baseline_context_time, baseline_context, label='Context', color='blue', linewidth=2, 
                    linestyle='-.', marker='o', markersize=6)
        axes[0].plot(pred_time, ground_truth, label='Ground Truth', color='green', 
                    linewidth=3, marker='o', markersize=6, linestyle='-.')
        axes[0].plot(pred_time, pred_baseline, label='Prediction', color='red', 
                    linewidth=2, linestyle='-.', marker='s', markersize=6)
        axes[0].axvline(x=0, color='black', linestyle=':', alpha=0.7, label='Forecast Start')
        
        # Add patch size in red text on top right
        axes[0].text(0.98, 0.95, f'PSZ: {PSZ}', transform=axes[0].transAxes, color='red', 
                    fontsize=12, fontweight='bold', ha='right', va='top')
        
        axes[0].set_title(f'1. Full Context (len={len(full_context)}) - Sample {i+1} - MAE: {np.mean(np.abs(pred_baseline - ground_truth)):.4f}')
        axes[0].set_xlabel('Time Steps')
        axes[0].set_ylabel('Value')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Reduced Context + Reduced Prediction  
        # Calculate downsample variables for prediction
        downsampled_ground_truth = ground_truth[::downsample_step][:reduced_pdt]
        reduced_pred_time = np.arange(0, reduced_pdt * downsample_step, downsample_step)
        
        # Plot downsampled context at original time positions (with gaps)
        axes[1].plot(downsampled_context_time, reduced_context, label=f'Downsampled Context (len={reduced_ctx})', 
                    color='blue', linewidth=2, linestyle='-.', marker='o', markersize=6)
        axes[1].plot(pred_time, ground_truth, label='Ground Truth', color='green', 
                    linewidth=3, marker='o', markersize=6, linestyle='-.')
        axes[1].plot(reduced_pred_time[:len(downsampled_ground_truth)], downsampled_ground_truth, 
                    label='Ground Truth (Downsampled)', color='green', 
                    linewidth=3, marker='o', markersize=6, linestyle='-.')
        axes[1].plot(reduced_pred_time[:len(pred_reduced)], pred_reduced, 
                    label='Prediction', color='red', 
                    linewidth=2, linestyle='-.', marker='s', markersize=6)
        axes[1].axvline(x=0, color='black', linestyle=':', alpha=0.7, label='Forecast Start')
        
        # Add patch size in red text on top right
        axes[1].text(0.98, 0.95, f'PSZ: {PSZ}', transform=axes[1].transAxes, color='red', 
                    fontsize=12, fontweight='bold', ha='right', va='top')
        
        axes[1].set_title(f'2. Downsampled Context (len={reduced_ctx}) - Sample {i+1} - MAE: {np.mean(np.abs(pred_reduced - downsampled_ground_truth)):.4f}')
        axes[1].set_xlabel('Time Steps')
        axes[1].set_ylabel('Value')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Truncated Context (Same Prediction Length)
        # For truncated context: show most recent reduced_ctx values at end of time axis
        truncated_context_time = np.arange(-reduced_ctx, 0)
        
        axes[2].plot(truncated_context_time, reduced_context, label=f'Truncated Context (most recent {reduced_ctx})', 
                    color='blue', linewidth=2, linestyle='-.', marker='o', markersize=6)
        axes[2].plot(pred_time, ground_truth, label='Ground Truth', color='green', 
                    linewidth=3, marker='o', markersize=6, linestyle='-.')
        axes[2].plot(pred_time, pred_truncated, label='Prediction', color='red', 
                    linewidth=2, linestyle='-.', marker='s', markersize=6)
        axes[2].axvline(x=0, color='black', linestyle=':', alpha=0.7, label='Forecast Start')
        
        # Add patch size in red text on top right
        axes[2].text(0.98, 0.95, f'PSZ: {PSZ}', transform=axes[2].transAxes, color='red', 
                    fontsize=12, fontweight='bold', ha='right', va='top')
        
        axes[2].set_title(f'3. Truncated Context (most recent {reduced_ctx} values) - Sample {i+1} - MAE: {np.mean(np.abs(pred_truncated - ground_truth)):.4f}')
        axes[2].set_xlabel('Time Steps')
        axes[2].set_ylabel('Value')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Common formatting - ensure all plots have same x-axis range
        x_min = -display_context_length  # Show full context range
        x_max = PDT
        for ax in axes:
            ax.set_xlim(x_min, x_max)
        
        plt.suptitle(f'Sample {i+1}: Comparison of Different Configurations\nCompression Ratio: {compression_ratio}', fontsize=14)
        plt.tight_layout()
        
        # Save the plot
        plot_path = f"{results_dir}/sample_{i+1}_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved visualization: {plot_path}")
    
    # Create summary MAE comparison plot for this configuration
    methods = ['Baseline\n(Full CTX+PDT)', 'Reduced\n(CTX+PDT)', 'Truncated\n(CTX only)']
    mae_values = [
        np.mean([np.mean(np.abs(np.mean(forecasts_baseline[i].samples, axis=0) - label_data[i]['target'][:PDT])) for i in range(len(forecasts_baseline))]),
        np.mean([np.mean(np.abs(np.mean(forecasts_reduced[i].samples, axis=0) - 
                                label_data[i]['target'][::int(1/compression_ratio)][:reduced_pdt])) for i in range(len(forecasts_reduced))]),
        np.mean([np.mean(np.abs(np.mean(forecasts_truncated[i].samples, axis=0) - label_data[i]['target'][:PDT])) for i in range(len(forecasts_truncated))])
    ]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    bars = ax.bar(methods, mae_values, alpha=0.8, color=['blue', 'orange', 'green'])
    
    # Add value labels on bars
    for bar, value in zip(bars, mae_values):
        height = bar.get_height()
        ax.annotate(f'{value:.4f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('Mean Absolute Error (MAE)')
    ax.set_title(f'MAE Comparison for Current Configuration\nCompression Ratio: {compression_ratio}')
    ax.grid(True, alpha=0.3)
    
    # Save the MAE comparison plot
    mae_plot_path = f"{results_dir}/mae_comparison.png"
    plt.savefig(mae_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved MAE comparison: {mae_plot_path}")

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
    
    # Generate visualization plots
    print("Generating visualization plots...")
    create_visualization_plots(
        input_data, input_data_reduced, label_data, 
        forecasts_baseline, forecasts_reduced, forecasts_truncated,
        results_dir, reduced_ctx, reduced_pdt, compression_ratio, CTX, PSZ
    )
    
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
