import torch
import pandas as pd
import numpy as np
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from typing import Tuple, Dict, Any, Optional
import logging

class MoiraiPerplexityEvaluator:
    """Core class for Moirai model perplexity evaluation and forecasting."""
    
    def __init__(self, model_size: str = "small", prediction_length: int = 96, num_windows: int = 96):
        """
        Initialize the Moirai evaluator.
        
        Args:
            model_size: Size of the Moirai model ("small", "base", "large")
            prediction_length: Number of time steps to forecast
            num_windows: Number of test windows for evaluation
        """
        self.model_size = model_size
        self.prediction_length = prediction_length
        self.num_windows = num_windows
        self.model = None
        self.predictor = None
        
    def load_model(self) -> None:
        """Load the Moirai model."""
        try:
            self.model = MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-{self.model_size}")
            self.predictor = MoiraiForecast(
                module=self.model,
                prediction_length=self.prediction_length,
                context_length=200,
                patch_size="auto",
                num_samples=100,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
            logging.info(f"Successfully loaded Moirai {self.model_size} model")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise
    
    def prepare_dataset(self, df: pd.DataFrame, target_column: str, 
                       timestamp_column: str = "timestamp") -> Tuple[Any, Any]:
        """
        Prepare dataset for training and testing.
        
        Args:
            df: Input dataframe
            target_column: Name of the target column
            timestamp_column: Name of the timestamp column
            
        Returns:
            Tuple of (train_ds, test_ds)
        """
        try:
            # Ensure timestamp is datetime
            df[timestamp_column] = pd.to_datetime(df[timestamp_column])
            df = df.sort_values(timestamp_column)
            
            # Add item_id column if not present
            df_copy = df.copy()
            if 'item_id' not in df_copy.columns:
                df_copy['item_id'] = 'item_0'
            
            # Create GluonTS dataset
            ds = PandasDataset.from_long_dataframe(
                df_copy, 
                target=target_column, 
                timestamp=timestamp_column,
                item_id='item_id'
            )
            
            # Split dataset
            train_ds, test_ds = split(ds, offset=-self.prediction_length)
            
            logging.info(f"Dataset prepared with {len(list(train_ds))} training series")
            return train_ds, test_ds
            
        except Exception as e:
            logging.error(f"Failed to prepare dataset: {e}")
            raise
    
    def prepare_dataset_with_windows(self, df: pd.DataFrame, target_column: str, 
                                   timestamp_column: str = "timestamp") -> Tuple[Any, Any]:
        """
        Prepare dataset with windowed test data generation.
        
        Args:
            df: Input dataframe
            target_column: Name of the target column
            timestamp_column: Name of the timestamp column
            
        Returns:
            Tuple of (train_ds, test_ds_with_windows)
        """
        try:
            # Ensure timestamp is datetime
            df[timestamp_column] = pd.to_datetime(df[timestamp_column])
            df = df.sort_values(timestamp_column)
            
            # Add item_id column if not present
            df_copy = df.copy()
            if 'item_id' not in df_copy.columns:
                df_copy['item_id'] = 'item_0'
            
            # Create GluonTS dataset
            ds = PandasDataset.from_long_dataframe(
                df_copy, 
                target=target_column, 
                timestamp=timestamp_column,
                item_id='item_id'
            )
            
            # Calculate total test length needed for the number of windows
            total_test_length = self.num_windows * self.prediction_length
            
            # Split dataset with larger test portion for windowing
            train_ds, test_template = split(ds, offset=-total_test_length)
            
            # Generate windowed test data
            test_ds_with_windows = test_template.generate_instances(
                prediction_length=self.prediction_length,
                windows=self.num_windows,
                distance=self.prediction_length,
            )
            
            logging.info(f"Dataset prepared with {len(list(train_ds))} training series")
            logging.info(f"Generated {self.num_windows} test windows with prediction length {self.prediction_length}")
            
            return train_ds, test_ds_with_windows
            
        except Exception as e:
            logging.error(f"Failed to prepare dataset with windows: {e}")
            raise
    
    def calculate_perplexity(self, test_ds: Any) -> Dict[str, float]:
        """
        Calculate perplexity metrics for the model.
        
        Args:
            test_ds: Test dataset
            
        Returns:
            Dictionary containing perplexity metrics
        """
        if self.predictor is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            total_log_likelihood = 0.0
            total_length = 0
            series_count = 0
            
            # Handle different dataset structures
            if hasattr(test_ds, 'input') and hasattr(test_ds, 'label'):
                # This is a windowed dataset with input/label structure
                input_iter = iter(test_ds.input)
                label_iter = iter(test_ds.label)
                
                for input_ts, label_ts in zip(input_iter, label_iter):
                    # Get the actual values for the prediction period from label
                    actual_values = label_ts["target"]
                    
                    # Make prediction using input
                    forecast = next(iter(self.predictor.predict([input_ts])))
                    
                    # Calculate log likelihood (simplified approximation)
                    # Using mean prediction and assuming normal distribution
                    predicted_mean = forecast.mean
                    predicted_std = np.std(forecast.samples, axis=0)
                    
                    # Avoid division by zero
                    predicted_std = np.maximum(predicted_std, 1e-6)
                    
                    # Calculate log likelihood for each time step
                    log_likelihood = -0.5 * np.sum(
                        ((actual_values - predicted_mean) / predicted_std) ** 2 +
                        np.log(2 * np.pi * predicted_std ** 2)
                    )
                    
                    total_log_likelihood += log_likelihood
                    total_length += len(actual_values)
                    series_count += 1
            else:
                # Traditional dataset structure
                for ts in test_ds:
                    # Get the actual values for the prediction period
                    target = ts["target"]
                    if len(target) < self.prediction_length:
                        continue
                        
                    # Create input for prediction (excluding last prediction_length points)
                    input_ts = {
                        "target": target[:-self.prediction_length],
                        "start": ts["start"]
                    }
                    
                    # Get actual values for comparison
                    actual_values = target[-self.prediction_length:]
                    
                    # Make prediction
                    forecast = next(iter(self.predictor.predict([input_ts])))
                    
                    # Calculate log likelihood (simplified approximation)
                    # Using mean prediction and assuming normal distribution
                    predicted_mean = forecast.mean
                    predicted_std = np.std(forecast.samples, axis=0)
                    
                    # Avoid division by zero
                    predicted_std = np.maximum(predicted_std, 1e-6)
                    
                    # Calculate log likelihood for each time step
                    log_likelihood = -0.5 * np.sum(
                        ((actual_values - predicted_mean) / predicted_std) ** 2 +
                        np.log(2 * np.pi * predicted_std ** 2)
                    )
                    
                    total_log_likelihood += log_likelihood
                    total_length += len(actual_values)
                    series_count += 1
            
            if total_length == 0:
                raise ValueError("No valid series found for perplexity calculation")
            
            # Calculate perplexity
            avg_log_likelihood = total_log_likelihood / total_length
            perplexity = np.exp(-avg_log_likelihood)
            
            metrics = {
                "perplexity": float(perplexity),
                "avg_log_likelihood": float(avg_log_likelihood),
                "total_series": series_count,
                "total_length": total_length
            }
            
            logging.info(f"Calculated perplexity: {perplexity:.4f}")
            return metrics
            
        except Exception as e:
            logging.error(f"Failed to calculate perplexity: {e}")
            raise
    
    def generate_forecasts(self, train_ds: Any, num_series: int = 5) -> Dict[str, Any]:
        """
        Generate forecasts for a subset of series.
        
        Args:
            train_ds: Training dataset
            num_series: Number of series to forecast
            
        Returns:
            Dictionary containing forecasts and inputs
        """
        if self.predictor is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            forecasts = []
            inputs = []
            
            for i, ts in enumerate(train_ds):
                if i >= num_series:
                    break
                    
                forecast = next(iter(self.predictor.predict([ts])))
                forecasts.append(forecast)
                inputs.append(ts)
            
            logging.info(f"Generated {len(forecasts)} forecasts")
            return {"forecasts": forecasts, "inputs": inputs}
            
        except Exception as e:
            logging.error(f"Failed to generate forecasts: {e}")
            raise
    
    def evaluate_forecast_accuracy(self, test_ds: Any) -> Dict[str, float]:
        """
        Evaluate forecast accuracy using standard metrics.
        
        Args:
            test_ds: Test dataset
            
        Returns:
            Dictionary containing accuracy metrics
        """
        if self.predictor is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            mae_total = 0.0
            mse_total = 0.0
            mape_total = 0.0
            series_count = 0
            
            # Handle different dataset structures
            if hasattr(test_ds, 'input') and hasattr(test_ds, 'label'):
                # This is a windowed dataset with input/label structure
                input_iter = iter(test_ds.input)
                label_iter = iter(test_ds.label)
                
                for input_ts, label_ts in zip(input_iter, label_iter):
                    # Get actual values from label
                    actual_values = label_ts["target"]
                    
                    # Make prediction using input
                    forecast = next(iter(self.predictor.predict([input_ts])))
                    predicted_values = forecast.mean
                    
                    # Calculate metrics
                    mae = np.mean(np.abs(actual_values - predicted_values))
                    mse = np.mean((actual_values - predicted_values) ** 2)
                    mape = np.mean(np.abs((actual_values - predicted_values) / (actual_values + 1e-8))) * 100
                    
                    mae_total += mae
                    mse_total += mse
                    mape_total += mape
                    series_count += 1
            else:
                # Traditional dataset structure
                for ts in test_ds:
                    target = ts["target"]
                    if len(target) < self.prediction_length:
                        continue
                    
                    # Create input for prediction
                    input_ts = {
                        "target": target[:-self.prediction_length],
                        "start": ts["start"]
                    }
                    
                    # Get actual values
                    actual_values = target[-self.prediction_length:]
                    
                    # Make prediction
                    forecast = next(iter(self.predictor.predict([input_ts])))
                    predicted_values = forecast.mean
                    
                    # Calculate metrics
                    mae = np.mean(np.abs(actual_values - predicted_values))
                    mse = np.mean((actual_values - predicted_values) ** 2)
                    mape = np.mean(np.abs((actual_values - predicted_values) / (actual_values + 1e-8))) * 100
                    
                    mae_total += mae
                    mse_total += mse
                    mape_total += mape
                    series_count += 1
            
            if series_count == 0:
                raise ValueError("No valid series found for accuracy evaluation")
            
            metrics = {
                "mae": mae_total / series_count,
                "mse": mse_total / series_count,
                "rmse": np.sqrt(mse_total / series_count),
                "mape": mape_total / series_count,
                "series_count": series_count
            }
            
            logging.info(f"Accuracy metrics calculated for {series_count} series")
            return metrics
            
        except Exception as e:
            logging.error(f"Failed to evaluate forecast accuracy: {e}")
            raise