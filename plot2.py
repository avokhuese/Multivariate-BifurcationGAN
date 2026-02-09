"""
focused_bifurcationgan_line_plots.py

Simple line plots comparing BifurcationGAN vs baselines with subplots.
"""

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import warnings
import os
import json
import gc
from pathlib import Path
import traceback

# Suppress warnings
warnings.filterwarnings('ignore')

# Import your existing modules
try:
    from config_univariate import config as original_config
    from data_loader_univariate_fixed import load_datasets_for_pipeline, safe_prepare_dataset
    from gan_framework_univariate import create_gan_framework
    HAS_ORIGINAL_CONFIG = True
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    HAS_ORIGINAL_CONFIG = False

# Create a complete config that has all required attributes
class CompleteConfig:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # ONLY THESE 4 MODELS
        self.benchmark_models = [
            "vanilla_gan",    # Simple baseline
            "wgan_gp",        # Strong WGAN variant  
            "tts_gan",        # Time-series specific
            "bifurcation_gan", # Our model
        ]
        
        # Fixed datasets - no automatic selection
        self.dataset_names = ['Epilepsy', 'Heartbeat', 'Plane']
        
        # Training parameters
        self.seq_len = 100
        self.latent_dim = 128
        self.batch_size = 32
        
        # Directory paths
        self.results_dir = "./focused_line_plots"
        self.save_dir = "./saved_models_univariate"
        self.data_dir = "./data/univariate"
        self.logs_dir = "./logs"
        self.cache_dir = "./cache"  # Added missing attribute
        
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

# Use original config if available, otherwise use complete config
if HAS_ORIGINAL_CONFIG:
    # Copy benchmark models from original config but restrict to our 4 models
    our_models = ["vanilla_gan", "wgan_gp", "tts_gan", "bifurcation_gan"]
    original_config.benchmark_models = our_models
    original_config.dataset_names = ['Epilepsy', 'Heartbeat', 'Plane']
    config = original_config
else:
    config = CompleteConfig()

class SimpleLinePlotVisualizer:
    """
    Simple visualizer for line plots comparing BifurcationGAN vs baselines.
    """
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Model colors
        self.model_colors = {
            'vanilla_gan': '#E74C3C',     # Red
            'wgan_gp': '#3498DB',         # Blue
            'tts_gan': '#2ECC71',         # Green
            'bifurcation_gan': '#9B59B6', # Purple - OUR MODEL
        }
        
        # Model names for display
        self.model_names = {
            'vanilla_gan': 'Vanilla GAN',
            'wgan_gp': 'WGAN-GP',
            'tts_gan': 'TTS-GAN',
            'bifurcation_gan': 'BifurcationGAN',
        }
        
        print("=" * 60)
        print("BIFURCATIONGAN LINE PLOT COMPARISON")
        print("=" * 60)
        print(f"Models: {list(self.model_names.values())}")
        print(f"Datasets: {config.dataset_names}")
        print(f"Output: {self.results_dir}")
        print("=" * 60)
    
    def load_real_samples(self, dataset_name: str, n_samples: int = 5):
        """Load real samples from dataset."""
        try:
            datasets = load_datasets_for_pipeline(self.config)
            if dataset_name not in datasets:
                print(f"Dataset {dataset_name} not found")
                return None
            
            dataset_info = datasets[dataset_name]
            train_loader, _, _, _ = safe_prepare_dataset(dataset_info, self.config)
            
            real_samples = []
            for batch in train_loader:
                real_samples.append(batch['data'])
                if len(real_samples) * self.config.batch_size >= n_samples:
                    break
            
            if not real_samples:
                return None
            
            return torch.cat(real_samples, dim=0)[:n_samples]
            
        except Exception as e:
            print(f"Error loading real samples for {dataset_name}: {e}")
            return None
    
    def load_model(self, model_type: str, dataset_name: str):
        """Load a trained model."""
        try:
            # Create model
            gan = create_gan_framework(model_type, self.config)
            
            # Try to load weights
            model_path = Path(self.config.save_dir) / f"{model_type}_{dataset_name}_run0_final.pth"
            if model_path.exists():
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Handle different checkpoint structures
                if 'generator_state_dict' in checkpoint:
                    state_dict = checkpoint['generator_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # Filter out problematic keys
                filtered_dict = {}
                for key, value in state_dict.items():
                    if 'positional_encoding' not in key:
                        filtered_dict[key] = value
                
                # Load filtered state dict
                gan.generator.load_state_dict(filtered_dict, strict=False)
                print(f"  ✓ Loaded {model_type} for {dataset_name}")
            else:
                print(f"  ⚠ No saved model for {model_type} on {dataset_name}")
            
            return gan
            
        except Exception as e:
            print(f"  ✗ Error loading {model_type} for {dataset_name}: {e}")
            return None
    
    def generate_samples(self, model_type: str, dataset_name: str, n_samples: int = 5):
        """Generate samples from a model."""
        gan = self.load_model(model_type, dataset_name)
        if gan is None:
            return None
        
        try:
            gan.generator.eval()
            with torch.no_grad():
                z = torch.randn(n_samples, self.config.latent_dim).to(self.device)
                samples = gan(z).cpu()
            
            # Clean up
            del gan
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return samples
            
        except Exception as e:
            print(f"  ✗ Error generating samples with {model_type}: {e}")
            return None
    
    def create_simple_comparison(self, dataset_name: str):
        """
        Create simple line plots comparing all models.
        """
        print(f"\nCreating plots for {dataset_name}...")
        
        # Get real samples
        real_samples = self.load_real_samples(dataset_name, n_samples=5)
        if real_samples is None:
            print(f"  Could not load real samples for {dataset_name}")
            return
        
        n_samples = min(5, len(real_samples))
        
        # Generate samples for each model
        model_samples = {}
        for model_type in self.config.benchmark_models:
            print(f"  Processing {model_type}...")
            samples = self.generate_samples(model_type, dataset_name, n_samples)
            if samples is not None:
                model_samples[model_type] = samples
        
        if not model_samples:
            print(f"  No models generated samples for {dataset_name}")
            return
        
        # Create main comparison figure (like original code)
        self.create_main_comparison_figure(dataset_name, real_samples, model_samples, n_samples)
        
        # Create side-by-side comparison
        self.create_side_by_side_figure(dataset_name, real_samples, model_samples)
        
        # Create overlay comparison
        self.create_overlay_figure(dataset_name, real_samples, model_samples)
    
    def create_main_comparison_figure(self, dataset_name: str, real_samples, model_samples, n_samples: int):
        """
        Create main comparison figure with subplots (like original code).
        """
        n_models = len(model_samples)
        
        # Create figure: n_samples columns, n_models+1 rows
        fig, axes = plt.subplots(n_models + 1, n_samples, 
                                 figsize=(3*n_samples, 2*(n_models + 1)),
                                 sharex=True, sharey='row',
                                 squeeze=False)
        
        # Plot real samples in first row
        for col in range(n_samples):
            if col < len(real_samples):
                real_data = real_samples[col].numpy().flatten()
                axes[0, col].plot(real_data, linewidth=2.5, alpha=0.8, 
                                 color='#2c3e50')
                axes[0, col].set_title(f"Real {col+1}", fontsize=10, fontweight='bold')
                axes[0, col].grid(True, alpha=0.3, linestyle=':')
        
        # Plot generated samples for each model
        for row_idx, (model_type, samples) in enumerate(model_samples.items(), 1):
            color = self.model_colors[model_type]
            model_name = self.model_names[model_type]
            
            # Add model name as y-label
            axes[row_idx, 0].set_ylabel(model_name, fontsize=10, rotation=0, 
                                       ha='right', va='center')
            
            for col in range(n_samples):
                if col < len(samples):
                    gen_data = samples[col].numpy().flatten()
                    axes[row_idx, col].plot(gen_data, linewidth=2, alpha=0.7,
                                           color=color, linestyle='-')
                    axes[row_idx, col].grid(True, alpha=0.3, linestyle=':')
        
        # Set labels
        for col in range(n_samples):
            axes[-1, col].set_xlabel("Time", fontsize=9)
        
        for row in range(n_models + 1):
            axes[row, 0].set_ylabel("Value", fontsize=9)
        
        plt.suptitle(f"Dataset: {dataset_name}\nReal vs Generated Samples", 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save
        save_path = self.results_dir / f"main_comparison_{dataset_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved main comparison: {save_path}")
    
    def create_side_by_side_figure(self, dataset_name: str, real_samples, model_samples):
        """
        Create side-by-side comparison of all models for first sample.
        """
        if len(real_samples) == 0:
            return
        
        # Use first sample for comparison
        real_data = real_samples[0].numpy().flatten()
        
        # Create figure with subplots for each model
        n_models = len(model_samples)
        fig, axes = plt.subplots(1, n_models + 1, figsize=(4*(n_models + 1), 5))
        
        # Plot real sample in first subplot
        axes[0].plot(real_data, linewidth=3, alpha=0.8, color='#2c3e50')
        axes[0].set_title("Real Sample", fontsize=12, fontweight='bold')
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Value")
        axes[0].grid(True, alpha=0.3)
        
        # Plot each model's generated sample
        for idx, (model_type, samples) in enumerate(model_samples.items(), 1):
            if len(samples) > 0:
                gen_data = samples[0].numpy().flatten()
                color = self.model_colors[model_type]
                
                axes[idx].plot(gen_data, linewidth=2.5, alpha=0.8, color=color)
                axes[idx].set_title(self.model_names[model_type], fontsize=12, fontweight='bold')
                axes[idx].set_xlabel("Time")
                axes[idx].set_ylabel("Value")
                axes[idx].grid(True, alpha=0.3)
        
        plt.suptitle(f"Side-by-Side Comparison - {dataset_name}", 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        save_path = self.results_dir / f"side_by_side_{dataset_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved side-by-side: {save_path}")
    
    def create_overlay_figure(self, dataset_name: str, real_samples, model_samples):
        """
        Create overlay plot comparing all models with real sample.
        """
        if len(real_samples) == 0:
            return
        
        real_data = real_samples[0].numpy().flatten()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: All models with real in background
        ax1.plot(real_data, linewidth=3, alpha=0.3, color='gray', 
                label='Real (background)', linestyle='-')
        
        for model_type, samples in model_samples.items():
            if len(samples) > 0:
                gen_data = samples[0].numpy().flatten()
                color = self.model_colors[model_type]
                ax1.plot(gen_data, linewidth=2, alpha=0.8, color=color,
                        label=self.model_names[model_type], linestyle='-')
        
        ax1.set_title(f"All Generated Samples vs Real Background - {dataset_name}", 
                     fontsize=13, fontweight='bold')
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Value")
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Direct comparison with real
        ax2.plot(real_data, linewidth=3, alpha=0.8, color='black',
                label='Real', linestyle='-')
        
        # Plot each model
        for model_type, samples in model_samples.items():
            if len(samples) > 0:
                gen_data = samples[0].numpy().flatten()
                color = self.model_colors[model_type]
                ax2.plot(gen_data, linewidth=2, alpha=0.7, color=color,
                        label=self.model_names[model_type], linestyle='--')
        
        ax2.set_title(f"Direct Comparison: Real vs All Models - {dataset_name}",
                     fontsize=13, fontweight='bold')
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Value")
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.results_dir / f"overlay_comparison_{dataset_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved overlay: {save_path}")
    
    def create_performance_figure(self):
        """
        Create performance comparison figure if metrics exist.
        """
        metrics_file = Path("./results_univariate/benchmark_summary.csv")
        if not metrics_file.exists():
            return
        
        try:
            df = pd.read_csv(metrics_file)
            
            # Filter for our models and datasets
            our_models = list(self.model_names.keys())
            our_datasets = self.config.dataset_names
            
            df = df[df['Model'].isin(our_models) & df['Dataset'].isin(our_datasets)]
            
            if df.empty:
                return
            
            # Create performance figure
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Plot 1: Bar chart for each dataset
            for dataset in our_datasets:
                dataset_df = df[df['Dataset'] == dataset]
                if not dataset_df.empty:
                    # Sort by quality
                    dataset_df = dataset_df.sort_values('Avg_Quality')
                    
                    # Get colors
                    colors = [self.model_colors.get(m, '#666666') 
                             for m in dataset_df['Model']]
                    
                    # Create horizontal bar chart
                    y_pos = np.arange(len(dataset_df))
                    axes[0].barh(y_pos, dataset_df['Avg_Quality'], 
                                color=colors, alpha=0.7, label=dataset)
                    
                    # Add value labels
                    for i, (bar, quality) in enumerate(zip(dataset_df['Avg_Quality'], dataset_df['Avg_Quality'])):
                        axes[0].text(quality + 0.01, i, f'{quality:.3f}', 
                                    va='center', fontsize=9)
            
            axes[0].set_xlabel('Quality Score')
            axes[0].set_title('Model Performance by Dataset', fontsize=12, fontweight='bold')
            axes[0].set_xlim(0, 1)
            axes[0].grid(True, alpha=0.3, axis='x')
            
            # Add legend
            handles = []
            for dataset in our_datasets:
                handles.append(plt.Rectangle((0,0), 1, 1, alpha=0.7, label=dataset))
            axes[0].legend(handles=handles, fontsize=9)
            
            # Plot 2: BifurcationGAN vs average of others
            axes[1].set_title('BifurcationGAN Performance Advantage', 
                             fontsize=12, fontweight='bold')
            
            bif_advantages = []
            for dataset in our_datasets:
                # Get BifurcationGAN score
                bif_score = df[(df['Dataset'] == dataset) & 
                              (df['Model'] == 'bifurcation_gan')]['Avg_Quality']
                
                # Get average of other models
                other_scores = df[(df['Dataset'] == dataset) & 
                                 (df['Model'] != 'bifurcation_gan')]['Avg_Quality']
                
                if len(bif_score) > 0 and len(other_scores) > 0:
                    bif_val = bif_score.values[0]
                    other_avg = other_scores.mean()
                    advantage = bif_val - other_avg
                    bif_advantages.append((dataset, advantage))
            
            if bif_advantages:
                datasets, advantages = zip(*bif_advantages)
                
                # Color bars based on advantage
                colors = ['#2ECC71' if adv > 0 else '#E74C3C' for adv in advantages]
                
                bars = axes[1].bar(datasets, advantages, color=colors, alpha=0.7)
                
                # Add value labels
                for bar, adv in zip(bars, advantages):
                    height = bar.get_height()
                    axes[1].text(bar.get_x() + bar.get_width()/2., 
                                height + (0.01 if height > 0 else -0.03),
                                f'{adv:+.3f}', ha='center', va='bottom' if height > 0 else 'top',
                                fontsize=9)
                
                axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
                axes[1].set_ylabel('Advantage over baseline average')
                axes[1].grid(True, alpha=0.3, axis='y')
            
            plt.suptitle('BifurcationGAN Performance Analysis', 
                        fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            
            save_path = self.results_dir / "performance_comparison.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved performance comparison: {save_path}")
            
        except Exception as e:
            print(f"  Could not create performance figure: {e}")
    
    def run_all_comparisons(self):
        """
        Run all comparisons for all datasets.
        """
        print("\n" + "=" * 60)
        print("RUNNING ALL COMPARISONS")
        print("=" * 60)
        
        # Create plots for each dataset
        for dataset in self.config.dataset_names:
            self.create_simple_comparison(dataset)
        
        # Create performance figure if metrics exist
        self.create_performance_figure()
        
        print("\n" + "=" * 60)
        print("COMPLETED!")
        print("=" * 60)
        
        # List generated files
        print("\nGenerated files:")
        for file in sorted(self.results_dir.glob("*.png")):
            print(f"  {file.name}")
        
        print(f"\nAll files saved to: {self.results_dir}")

def main():
    """Main execution function."""
    
    print("=" * 60)
    print("BIFURCATIONGAN LINE PLOT COMPARISON")
    print("=" * 60)
    
    # Create visualizer
    visualizer = SimpleLinePlotVisualizer(config)
    
    # Run comparisons
    visualizer.run_all_comparisons()

if __name__ == "__main__":
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run
    try:
        main()
    except Exception as e:
        print(f"Script failed: {e}")
        traceback.print_exc()