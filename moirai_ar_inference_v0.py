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
BSZ = 128  # batch size: use 128 as requested
TEST = int(5*PDT)  # test set length: any positive integer
DROP_RATIO = 0.25  # 25% dropout

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

# Split into train/test set
train, test_template = split(ds, offset=-TEST)

# Load the base module once
print("Loading base model module...")
if MODEL == "moirai":
    base_module = MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-{SIZE}")
elif MODEL == "moirai-moe":
    base_module = MoiraiMoEModule.from_pretrained(f"Salesforce/moirai-moe-1.0-R-{SIZE}")

def create_model_with_context_length(context_length):
    """Create a model with specific context length for single-step prediction"""
    if MODEL == "moirai":
        model = MoiraiForecast(
            module=base_module,
            prediction_length=1,  # Single step prediction for importance
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

def create_forecasting_model(context_length):
    """Create a forecasting model with specific context length"""
    if MODEL == "moirai":
        model = MoiraiForecast(
            module=base_module,
            prediction_length=PDT,  # Full prediction length
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
            prediction_length=PDT,
            context_length=context_length,
            patch_size=16,
            num_samples=100,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
    return model.create_predictor(batch_size=BSZ)

def create_window_data(full_series, window_start_idx, context_length, prediction_length):
    """
    Create a single window with specified context and prediction lengths
    """
    context_end = window_start_idx + context_length
    prediction_end = context_end + prediction_length
    
    if prediction_end > len(full_series):
        return None, None
    
    # Extract context and target
    context = full_series[window_start_idx:context_end]
    target = full_series[context_end:prediction_end]
    
    # Create input data structure
    input_data = {
        "target": context,
        "start": pd.Timestamp("2020-01-01"),  # Dummy timestamp
    }
    
    label_data = {
        "target": target,
        "start": pd.Timestamp("2020-01-01") + pd.DateOffset(hours=context_length),
    }
    
    return input_data, label_data

def compute_importance_for_context(input_data):
    """
    Compute importance score for each sample in the context
    For each position i (from PDT to CTX-1): use context[0:i] to predict context[i]
    """
    context = input_data["target"]
    window_context_length = len(context)
    
    print(f"Window context length: {window_context_length}")
    
    # We can analyze positions from PDT to window_context_length-1
    analyzable_positions = list(range(PDT, window_context_length))
    importance_scores = np.zeros(len(analyzable_positions))
    
    print(f"Computing importance for {len(analyzable_positions)} positions in this window (positions {PDT} to {window_context_length-1})")
    
    for idx, pos in enumerate(tqdm(analyzable_positions, desc="Computing importance")):
        # Use context[0:pos] to predict context[pos]
        context_for_prediction = context[:pos]
        true_value = context[pos]
        
        # Create model with appropriate context length
        predictor = create_model_with_context_length(len(context_for_prediction))
        
        # Create input for prediction
        pred_input_data = input_data.copy()
        pred_input_data["target"] = context_for_prediction
        
        # Predict next value
        forecast = next(iter(predictor.predict([pred_input_data])))
        pred_samples = forecast.samples[:, 0]  # Single step prediction
        pred_mean = np.mean(pred_samples)
        
        # Importance = prediction error
        importance_scores[idx] = abs(pred_mean - true_value)
        
        if idx < 5 or idx % 10 == 0:
            print(f"  Position {pos}: context_len={len(context_for_prediction)}, error={importance_scores[idx]:.4f}")
    
    return importance_scores, analyzable_positions

def apply_pruning_strategy(context, importance_scores, analyzable_positions, strategy, drop_ratio=0.25):
    """Apply pruning strategy based on importance scores"""
    context = context.copy()
    
    num_to_modify = int(len(importance_scores) * drop_ratio)
    
    print(f"  Strategy: {strategy}")
    print(f"  Context length: {len(context)}")
    print(f"  Analyzable positions: {len(analyzable_positions)} samples")
    print(f"  Samples to modify: {num_to_modify}")
    
    if strategy == 'least_important_drop':
        # Drop least important samples
        least_important_indices = np.argsort(importance_scores)[:num_to_modify]
        positions_to_drop = [analyzable_positions[idx] for idx in least_important_indices]
        # Remove in reverse order to maintain indices
        for pos in sorted(positions_to_drop, reverse=True):
            context = np.delete(context, pos)
        print(f"  Final context length: {len(context)} (dropped {num_to_modify} samples)")
            
    elif strategy == 'random_drop':
        # Drop random samples
        random_indices = np.random.choice(len(importance_scores), num_to_modify, replace=False)
        positions_to_drop = [analyzable_positions[idx] for idx in random_indices]
        for pos in sorted(positions_to_drop, reverse=True):
            context = np.delete(context, pos)
        print(f"  Final context length: {len(context)} (dropped {num_to_modify} samples)")
            
    elif strategy == 'least_important_replace':
        # Replace least important with mean of neighbors
        least_important_indices = np.argsort(importance_scores)[:num_to_modify]
        positions_to_replace = [analyzable_positions[idx] for idx in least_important_indices]
        for pos in positions_to_replace:
            left_val = context[pos - 1] if pos > 0 else context[pos]
            right_val = context[pos + 1] if pos < len(context) - 1 else context[pos]
            context[pos] = (left_val + right_val) / 2
        print(f"  Final context length: {len(context)} (same length, {num_to_modify} values replaced)")
            
    elif strategy == 'random_replace':
        # Replace random samples with mean of neighbors
        random_indices = np.random.choice(len(importance_scores), num_to_modify, replace=False)
        positions_to_replace = [analyzable_positions[idx] for idx in random_indices]
        for pos in positions_to_replace:
            left_val = context[pos - 1] if pos > 0 else context[pos]
            right_val = context[pos + 1] if pos < len(context) - 1 else context[pos]
            context[pos] = (left_val + right_val) / 2
        print(f"  Final context length: {len(context)} (same length, {num_to_modify} values replaced)")
    
    return context

def forecast_with_context(context, input_data, true_target):
    """Forecast target using given context"""
    # Create forecasting model with appropriate context length
    predictor = create_forecasting_model(len(context))
    
    # Create input with modified context
    forecast_input_data = input_data.copy()
    forecast_input_data["target"] = context
    
    # Forecast
    forecast = next(iter(predictor.predict([forecast_input_data])))
    predicted_target = np.mean(forecast.samples, axis=0)  # Shape: [PDT,]
    
    # Calculate MAE
    mae = np.mean(np.abs(predicted_target - true_target))
    
    return predicted_target, mae

# Main experiment
print(f"\nStarting importance-based pruning validation...")
print(f"Batch size: {BSZ}")
print(f"Context length per window: {CTX}")
print(f"Prediction length: {PDT}")

strategies = ['least_important_drop', 'random_drop', 'least_important_replace', 'random_replace']
all_results = []

# Get the full time series data
full_series_data = next(iter(ds))["target"]
print(f"Full series length: {len(full_series_data)}")

# Create multiple windows manually
num_windows = 3
window_spacing = CTX  # Non-overlapping windows

for window_idx in range(num_windows):
    window_start = window_idx * window_spacing
    
    print(f"\n{'='*80}")
    print(f"PROCESSING WINDOW {window_idx + 1}/{num_windows}")
    print(f"{'='*80}")
    
    # Create window data
    input_data, label_data = create_window_data(
        full_series_data, window_start, CTX, PDT
    )
    
    if input_data is None:
        print(f"Not enough data for window {window_idx + 1}, skipping...")
        continue
    
    original_context = input_data["target"].copy()
    true_target = label_data["target"]
    
    print(f"Window start index: {window_start}")
    print(f"Context length: {len(original_context)} (should be {CTX})")
    print(f"Target length: {len(true_target)} (should be {PDT})")
    
    assert len(original_context) == CTX, f"Expected context length {CTX}, got {len(original_context)}"
    assert len(true_target) == PDT, f"Expected target length {PDT}, got {len(true_target)}"
    
    # Step 1: Compute importance scores for each sample in this window's context
    print(f"\nStep 1: Computing importance scores for this window...")
    importance_scores, analyzable_positions = compute_importance_for_context(input_data)
    
    expected_analyzable = CTX - PDT  # Should be 64 - 8 = 56
    print(f"Computed importance for {len(importance_scores)} samples (expected: {expected_analyzable})")
    print(f"Mean importance: {np.mean(importance_scores):.4f}")
    print(f"Std importance: {np.std(importance_scores):.4f}")
    
    # Step 2: Baseline forecast (no pruning)
    print(f"\nStep 2: Baseline forecast...")
    baseline_pred, baseline_mae = forecast_with_context(original_context, input_data, true_target)
    print(f"Baseline MAE: {baseline_mae:.4f}")
    
    window_results = {
        'window_id': window_idx + 1,
        'window_start': window_start,
        'context_length': len(original_context),
        'importance_scores': importance_scores,
        'analyzable_positions': analyzable_positions,
        'baseline_mae': baseline_mae,
        'strategy_results': {}
    }
    
    # Step 3: Test each pruning strategy
    for strategy in strategies:
        print(f"\nStep 3: Testing strategy: {strategy}")
        
        num_trials = 5 if 'random' in strategy else 1
        strategy_maes = []
        strategy_context_lengths = []
        
        for trial in range(num_trials):
            if 'random' in strategy:
                np.random.seed(42 + trial)
            
            print(f"\n  Trial {trial + 1}/{num_trials}:")
            
            # Apply pruning
            pruned_context = apply_pruning_strategy(
                original_context, importance_scores, analyzable_positions, strategy, DROP_RATIO
            )
            
            try:
                # Forecast with pruned context
                pruned_pred, pruned_mae = forecast_with_context(pruned_context, input_data, true_target)
                
                strategy_maes.append(pruned_mae)
                strategy_context_lengths.append(len(pruned_context))
                
                print(f"  Result: MAE = {pruned_mae:.4f}")
                
            except Exception as e:
                print(f"  Error: {e}")
                strategy_maes.append(np.inf)
                strategy_context_lengths.append(0)
        
        # Store results
        window_results['strategy_results'][strategy] = {
            'maes': strategy_maes,
            'mean_mae': np.mean(strategy_maes),
            'std_mae': np.std(strategy_maes),
            'context_lengths': strategy_context_lengths
        }
        
        print(f"\n  {strategy} Summary:")
        print(f"    Mean MAE: {np.mean(strategy_maes):.4f} ± {np.std(strategy_maes):.4f}")
    
    all_results.append(window_results)

# Final analysis
print(f"\n{'='*80}")
print("FINAL RESULTS")
print(f"{'='*80}")

if len(all_results) == 0:
    print("No windows were processed successfully!")
    exit()

strategy_performance = {strategy: [] for strategy in strategies}
baseline_maes = []

for result in all_results:
    baseline_maes.append(result['baseline_mae'])
    for strategy in strategies:
        strategy_performance[strategy].append(result['strategy_results'][strategy]['mean_mae'])

print(f"\nPerformance Summary:")
print(f"Baseline (no pruning):        {np.mean(baseline_maes):.4f} ± {np.std(baseline_maes):.4f}")

for strategy in strategies:
    mean_mae = np.mean(strategy_performance[strategy])
    std_mae = np.std(strategy_performance[strategy])
    degradation = ((mean_mae - np.mean(baseline_maes)) / np.mean(baseline_maes)) * 100
    print(f"{strategy:25s}: {mean_mae:.4f} ± {std_mae:.4f} ({degradation:+6.1f}% vs baseline)")

# Validation check
print(f"\nValidation Results:")
least_imp_drop = np.array(strategy_performance['least_important_drop'])
random_drop = np.array(strategy_performance['random_drop'])
least_imp_replace = np.array(strategy_performance['least_important_replace'])
random_replace = np.array(strategy_performance['random_replace'])

print(f"Drop strategies:")
improvement = np.mean(random_drop - least_imp_drop)
win_rate = np.sum(least_imp_drop < random_drop) / len(least_imp_drop) * 100
print(f"  Least important vs Random: {improvement:.4f} improvement, {win_rate:.1f}% win rate")

print(f"Replace strategies:")
improvement = np.mean(random_replace - least_imp_replace)
win_rate = np.sum(least_imp_replace < random_replace) / len(least_imp_replace) * 100
print(f"  Least important vs Random: {improvement:.4f} improvement, {win_rate:.1f}% win rate")

print(f"\nConclusion:")
if np.mean(least_imp_drop) < np.mean(random_drop) and np.mean(least_imp_replace) < np.mean(random_replace):
    print("✓ Importance-based pruning performs BETTER than random - importance scores are meaningful!")
else:
    print("✗ Random pruning performs better - importance scores may not be reliable")

print(f"\nExperiment completed successfully!")