#!/usr/bin/env python3
"""
Script to generate a CSV file with sinusoidal and cosine waves with hourly timestamps.
Similar to ETTh1.csv format but with synthetic sinusoidal data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

def generate_sinusoidal_data(num_samples=1000, start_date="2024-01-01 00:00:00", 
                           output_dir="/home/sa53869/time-series/moirai/time-moe-eval/ETT-small/",
                           filename="synthetic_sinusoidal.csv"):
    """
    Generate synthetic sinusoidal and cosine wave data with hourly timestamps.
    
    Parameters:
    -----------
    num_samples : int
        Number of samples to generate
    start_date : str
        Starting datetime in format "YYYY-MM-DD HH:MM:SS"
    output_dir : str
        Directory to save the CSV file
    filename : str
        Name of the output CSV file
    
    Returns:
    --------
    pd.DataFrame : Generated dataframe
    str : Path to saved CSV file
    """
    
    # Create hourly timestamps
    start_dt = pd.to_datetime(start_date)
    timestamps = [start_dt + timedelta(hours=i) for i in range(num_samples)]
    
    # Generate time array for wave calculations (normalize to create multiple periods)
    t = np.linspace(0, 10 * np.pi, num_samples)  # 10 periods over the samples
    
    # Generate sinusoidal wave (first column)
    sin_wave = 5 * np.sin(t) + 2 + 0.5 * np.random.normal(0, 0.2, num_samples)  # Add some noise
    
    # Generate cosine wave (second column)
    cos_wave = 3 * np.cos(t) + 1.5 + 0.3 * np.random.normal(0, 0.15, num_samples)  # Add some noise
    
    # Create additional columns for variety (similar to ETTh1 structure)
    # Create waves with different frequencies and phases
    wave3 = 2 * np.sin(2 * t + np.pi/4) + 1 + 0.2 * np.random.normal(0, 0.1, num_samples)
    wave4 = 1.5 * np.cos(0.5 * t + np.pi/3) + 0.5 + 0.15 * np.random.normal(0, 0.1, num_samples)
    wave5 = 4 * np.sin(t + np.pi/2) + 3 + 0.4 * np.random.normal(0, 0.2, num_samples)
    wave6 = 2.5 * np.cos(1.5 * t + np.pi/6) + 2 + 0.25 * np.random.normal(0, 0.15, num_samples)
    
    # Temperature-like column (with daily and seasonal patterns)
    temp_daily = 10 * np.sin(t/12 * 2 * np.pi) + 20  # Daily temperature variation
    temp_seasonal = 5 * np.sin(t/8760 * 2 * np.pi)  # Seasonal variation (approximate)
    temperature = temp_daily + temp_seasonal + np.random.normal(0, 1, num_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': timestamps,
        'SIN_WAVE': sin_wave,
        'COS_WAVE': cos_wave,
        'WAVE3': wave3,
        'WAVE4': wave4,
        'WAVE5': wave5,
        'WAVE6': wave6,
        'TEMPERATURE': temperature
    })
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False)
    
    print(f"Generated {num_samples} samples of sinusoidal data")
    print(f"Data saved to: {output_path}")
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df, output_path

def visualize_data(df, save_plots=True, output_dir="/home/sa53869/time-series/moirai/time-moe-eval/ETT-small/"):
    """
    Create visualizations of the generated sinusoidal data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the generated data
    save_plots : bool
        Whether to save plots to files
    output_dir : str
        Directory to save plot files
    """
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Figure 1: Main sinusoidal and cosine waves
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot first 500 samples for better visibility
    plot_samples = min(500, len(df))
    plot_df = df.head(plot_samples)
    
    # Sinusoidal wave
    axes[0].plot(plot_df['date'], plot_df['SIN_WAVE'], linewidth=2, label='Sinusoidal Wave', color='blue')
    axes[0].set_title('Sinusoidal Wave Over Time', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Amplitude', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Cosine wave
    axes[1].plot(plot_df['date'], plot_df['COS_WAVE'], linewidth=2, label='Cosine Wave', color='red')
    axes[1].set_title('Cosine Wave Over Time', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].set_ylabel('Amplitude', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(os.path.join(output_dir, 'sinusoidal_cosine_waves.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Figure 2: All waves comparison
    fig, ax = plt.subplots(figsize=(15, 10))
    
    wave_columns = ['SIN_WAVE', 'COS_WAVE', 'WAVE3', 'WAVE4', 'WAVE5', 'WAVE6']
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for i, (col, color) in enumerate(zip(wave_columns, colors)):
        ax.plot(plot_df['date'], plot_df[col], linewidth=2, label=col, color=color, alpha=0.8)
    
    ax.set_title('All Generated Waves Comparison', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(os.path.join(output_dir, 'all_waves_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Figure 3: Temperature data
    fig, ax = plt.subplots(figsize=(15, 6))
    
    ax.plot(plot_df['date'], plot_df['TEMPERATURE'], linewidth=2, label='Temperature', color='darkred')
    ax.set_title('Generated Temperature Data', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Temperature', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(os.path.join(output_dir, 'temperature_data.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Figure 4: Distribution plots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for i, col in enumerate(numeric_columns):
        if i < len(axes):
            axes[i].hist(df[col], bins=30, alpha=0.7, color=colors[i % len(colors)])
            axes[i].set_title(f'Distribution of {col}', fontsize=12)
            axes[i].set_xlabel('Value', fontsize=10)
            axes[i].set_ylabel('Frequency', fontsize=10)
            axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(numeric_columns), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(os.path.join(output_dir, 'data_distributions.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    print("\nData Statistics:")
    print("=" * 50)
    print(df.describe())

if __name__ == "__main__":
    # Generate data with default parameters
    # You can modify these parameters as needed
    
    num_samples = 2000  # Number of samples to generate
    start_date = "2024-01-01 00:00:00"  # Starting date
    output_dir = "/home/sa53869/time-series/moirai/time-moe-eval/ETT-small/"
    filename = "synthetic_sinusoidal.csv"
    
    print("Generating sinusoidal data...")
    df, output_path = generate_sinusoidal_data(
        num_samples=num_samples,
        start_date=start_date,
        output_dir=output_dir,
        filename=filename
    )
    
    print("\nCreating visualizations...")
    visualize_data(df, save_plots=True, output_dir=output_dir)
    
    print(f"\nFirst few rows of generated data:")
    print(df.head(10))
    
    print(f"\nLast few rows of generated data:")
    print(df.tail(5))
