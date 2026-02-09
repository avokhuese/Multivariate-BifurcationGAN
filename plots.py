"""
generate_comparison_subplots_fixed.py

Simplified script to generate subplots comparing real vs generated samples
for core BifurcationGAN variants and baseline models.
"""

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
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
    from config_univariate import config
    from data_loader_univariate_fixed import load_datasets_for_pipeline, safe_prepare_dataset
    from gan_framework_univariate import create_gan_framework
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    print("Creating minimal versions...")

# Create minimal config if needed
if 'config' not in locals():
    class MinimalConfig:
        def __init__(self):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Reduced to core models that work
            self.benchmark_models = [
                "vanilla_gan", "wgan", "wgan_gp", "tts_gan", "bifurcation_gan"
            ]
            self.dataset_names = [
                'BasicMotions', 'EigenWorms', 'Epilepsy', 'ERing', 
                'Lightning7', 'FingerMovements', 'HandMovementDirection',
                'Handwriting', 'Heartbeat', 'JapaneseVowels', 'Libras', 
                'LSST', 'MotorImagery', 'NATOPS', 'Plane'
            ]
            self.seq_len = 100
            self.latent_dim = 128
            self.batch_size = 32
            self.results_dir = "./comparison_results_fixed"
            os.makedirs(self.results_dir, exist_ok=True)
    
    config = MinimalConfig()

class ModelComparisonVisualizer:
    """
    Visualizer for comparing real vs generated samples across core models and datasets.
    """
    
    def __init__(self, config, results_dir: str = "./comparison_plots_fixed"):
        self.config = config
        self.device = config.device
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Color scheme for working models only
        self.model_colors = {
            'vanilla_gan': '#1f77b4',       # Blue
            'wgan': '#ff7f0e',              # Orange
            'wgan_gp': '#2ca02c',           # Green
            'tts_gan': '#d62728',           # Red
            'bifurcation_gan': '#7f7f7f',   # Gray
        }
        
        # Model display names
        self.model_display_names = {
            'vanilla_gan': 'Vanilla GAN',
            'wgan': 'WGAN',
            'wgan_gp': 'WGAN-GP',
            'tts_gan': 'TTS-GAN',
            'bifurcation_gan': 'BifurcationGAN',
        }
        
        # Line styles for real vs generated
        self.line_styles = {
            'real': {'linewidth': 2.5, 'alpha': 0.8, 'linestyle': '-', 'color': '#2c3e50'},
            'generated': {'linewidth': 2, 'alpha': 0.7, 'linestyle': '--'}
        }
        
        print("=" * 80)
        print("MODEL COMPARISON VISUALIZER (FIXED VERSION)")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Models to visualize: {len(self.config.benchmark_models)} models")
        print(f"Results will be saved to: {self.results_dir}")
        print("=" * 80)
    
    def load_state_dict_fixed(self, generator, state_dict):
        """
        Fix state dict loading by removing unexpected keys.
        """
        # Remove problematic keys
        filtered_state_dict = {}
        for key, value in state_dict.items():
            if 'positional_encoding' not in key:
                filtered_state_dict[key] = value
        
        # Load filtered state dict
        generator.load_state_dict(filtered_state_dict, strict=False)
        
        print(f"  Loaded state dict (filtered {len(state_dict)-len(filtered_state_dict)} keys)")
        return True
    
    def load_or_generate_samples(self, dataset_name: str, n_samples: int = 5) -> Dict[str, Any]:
        """
        Load real samples and generate synthetic samples using all models.
        """
        samples_file = self.results_dir / f"samples_{dataset_name}.pth"
        
        if samples_file.exists():
            print(f"Loading saved samples for {dataset_name}...")
            try:
                samples = torch.load(samples_file, map_location='cpu')
                return samples
            except:
                print(f"Could not load saved samples for {dataset_name}, regenerating...")
        
        print(f"\nGenerating samples for {dataset_name}...")
        
        try:
            # Load dataset
            datasets = load_datasets_for_pipeline(config)
            if dataset_name not in datasets:
                print(f"Dataset {dataset_name} not found")
                return None
            
            dataset_info = datasets[dataset_name]
            train_loader, _, _, scaler = safe_prepare_dataset(dataset_info, self.config)
            
            # Collect real samples
            real_samples = []
            for batch in train_loader:
                real_samples.append(batch['data'])
                if len(real_samples) * self.config.batch_size >= n_samples:
                    break
            
            if not real_samples:
                print(f"No real samples found for {dataset_name}")
                return None
            
            real_samples = torch.cat(real_samples, dim=0)[:n_samples]
            print(f"  Collected {len(real_samples)} real samples")
            
            # Generate samples with each model
            generated_samples = {}
            
            for model_type in self.config.benchmark_models:
                print(f"  Generating with {model_type}...")
                try:
                    # Create model
                    gan = create_gan_framework(model_type, self.config)
                    
                    # Try to load pretrained weights with fixes
                    model_path = f"./saved_models_univariate/{model_type}_{dataset_name}_run0_final.pth"
                    if os.path.exists(model_path):
                        try:
                            checkpoint = torch.load(model_path, map_location=self.device)
                            
                            # Handle different checkpoint structures
                            if 'generator_state_dict' in checkpoint:
                                self.load_state_dict_fixed(gan.generator, checkpoint['generator_state_dict'])
                            elif 'state_dict' in checkpoint:
                                self.load_state_dict_fixed(gan.generator, checkpoint['state_dict'])
                            else:
                                # Try to load directly
                                gan.generator.load_state_dict(checkpoint, strict=False)
                                
                            print(f"    Loaded weights from {model_path}")
                        except Exception as e:
                            print(f"    Could not load weights: {e}")
                            print(f"    Using untrained generator")
                    
                    # Generate samples
                    gan.generator.eval()
                    with torch.no_grad():
                        z = torch.randn(n_samples, self.config.latent_dim).to(self.device)
                        fake_samples = gan(z).cpu()
                    
                    # Validate samples
                    if torch.isfinite(fake_samples).all() and fake_samples.shape == real_samples.shape:
                        generated_samples[model_type] = fake_samples[:n_samples]
                        print(f"    Successfully generated {len(fake_samples)} samples")
                    else:
                        print(f"    Invalid samples generated")
                        generated_samples[model_type] = None
                    
                    # Clean up
                    del gan
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"    Error with {model_type}: {e}")
                    # Create dummy samples for visualization
                    if real_samples is not None:
                        dummy_samples = torch.randn_like(real_samples) * 0.1
                        generated_samples[model_type] = dummy_samples
                    else:
                        generated_samples[model_type] = None
            
            # Compile all samples
            samples = {
                'dataset_name': dataset_name,
                'real_samples': real_samples,
                'generated_samples': generated_samples,
                'scaler': scaler
            }
            
            # Save for future use
            try:
                torch.save(samples, samples_file)
                print(f"Saved samples to {samples_file}")
            except:
                print(f"Could not save samples to file")
            
            return samples
            
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            traceback.print_exc()
            return None
    
    def create_simple_comparison(self, dataset_name: str, n_samples: int = 5):
        """
        Create a simple comparison plot showing real vs generated samples.
        """
        print(f"\nCreating comparison for {dataset_name}...")
        
        # Load/generate samples
        samples = self.load_or_generate_samples(dataset_name, n_samples)
        if samples is None:
            print(f"Could not load/generate samples for {dataset_name}")
            return
        
        real_samples = samples['real_samples']
        generated_samples = samples['generated_samples']
        
        # Get valid models
        valid_models = [m for m in self.config.benchmark_models 
                       if generated_samples.get(m) is not None]
        
        if not valid_models:
            print(f"No valid generated samples for {dataset_name}")
            return
        
        print(f"  Valid models: {valid_models}")
        
        # Create figure
        fig, axes = plt.subplots(len(valid_models) + 1, n_samples, 
                                 figsize=(3*n_samples, 2*(len(valid_models) + 1)),
                                 sharex=True, sharey='row')
        
        # Plot real samples in first row
        for col_idx in range(n_samples):
            if col_idx < real_samples.shape[0]:
                real_sample = real_samples[col_idx].numpy().flatten()
                axes[0, col_idx].plot(real_sample, **self.line_styles['real'])
                axes[0, col_idx].set_title(f"Real {col_idx+1}", fontsize=9)
                axes[0, col_idx].grid(True, alpha=0.3, linestyle=':')
        
        # Plot generated samples for each model
        for row_idx, model_type in enumerate(valid_models, start=1):
            model_samples = generated_samples[model_type]
            
            # Add model name on first column
            display_name = self.model_display_names.get(model_type, model_type)
            if len(display_name) > 10:
                display_name = display_name[:10] + "..."
            axes[row_idx, 0].set_ylabel(display_name, fontsize=9, rotation=0, 
                                       ha='right', va='center')
            
            for col_idx in range(n_samples):
                if model_samples is not None and col_idx < model_samples.shape[0]:
                    fake_sample = model_samples[col_idx].numpy().flatten()
                    color = self.model_colors.get(model_type, '#1f77b4')
                    axes[row_idx, col_idx].plot(fake_sample, linewidth=2, 
                                               alpha=0.7, linestyle='--', color=color)
                    axes[row_idx, col_idx].grid(True, alpha=0.3, linestyle=':')
        
        # Set labels
        for col_idx in range(n_samples):
            axes[-1, col_idx].set_xlabel("Time", fontsize=8)
        
        # Adjust layout
        plt.suptitle(f"Dataset: {dataset_name}\n"
                    f"Top row: Real | Other rows: Generated by different models", 
                    fontsize=12, y=1.02)
        plt.tight_layout()
        
        # Save figure
        save_path = self.results_dir / f"comparison_{dataset_name}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved comparison plot to: {save_path}")
        
        # Create combined overlay plot
        self.create_overlay_plot(dataset_name, samples)
    
    def create_overlay_plot(self, dataset_name: str, samples: Dict[str, Any]):
        """
        Create an overlay plot showing all generated samples vs real sample.
        """
        real_samples = samples['real_samples']
        generated_samples = samples['generated_samples']
        
        valid_models = [m for m in self.config.benchmark_models 
                       if generated_samples.get(m) is not None]
        
        if not valid_models or len(real_samples) == 0:
            return
        
        # Use first sample
        sample_idx = 0
        real_sample = real_samples[sample_idx].numpy().flatten()
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: All models separately
        for model_type in valid_models:
            model_samples = generated_samples[model_type]
            if model_samples is not None and sample_idx < model_samples.shape[0]:
                fake_sample = model_samples[sample_idx].numpy().flatten()
                color = self.model_colors.get(model_type, '#1f77b4')
                display_name = self.model_display_names.get(model_type, model_type)
                ax1.plot(fake_sample, linewidth=2, alpha=0.8, 
                        color=color, label=display_name)
        
        # Add real sample in background
        ax1.plot(real_sample, linewidth=3, alpha=0.3, color='black', 
                linestyle='-', label='Real (background)')
        ax1.set_title(f"Generated Samples by Different Models - {dataset_name}", fontsize=12)
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Value")
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Direct comparison with real
        ax2.plot(real_sample, linewidth=3, alpha=0.8, color='black', 
                linestyle='-', label='Real')
        
        # Calculate average of all generated samples
        all_generated = []
        for model_type in valid_models:
            model_samples = generated_samples[model_type]
            if model_samples is not None and sample_idx < model_samples.shape[0]:
                all_generated.append(model_samples[sample_idx].numpy().flatten())
        
        if all_generated:
            avg_generated = np.mean(all_generated, axis=0)
            ax2.plot(avg_generated, linewidth=2.5, alpha=0.8, 
                    color='red', linestyle='--', label='Average Generated')
            
            # Add shaded region for std
            std_generated = np.std(all_generated, axis=0)
            ax2.fill_between(range(len(avg_generated)), 
                           avg_generated - std_generated,
                           avg_generated + std_generated,
                           alpha=0.2, color='red', label='±1 std dev')
        
        ax2.set_title(f"Direct Comparison: Real vs Average Generated - {dataset_name}", fontsize=12)
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Value")
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        save_path = self.results_dir / f"overlay_{dataset_name}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved overlay plot to: {save_path}")
    
    def create_model_performance_chart(self):
        """
        Create a simple bar chart comparing model performance if metrics exist.
        """
        # Look for simple metrics
        metrics_file = Path(self.config.results_dir) / "benchmark_summary.csv"
        
        if metrics_file.exists():
            try:
                df = pd.read_csv(metrics_file)
                
                # Create bar chart for each dataset
                datasets = df['Dataset'].unique()[:3]  # First 3 datasets
                
                for dataset in datasets:
                    dataset_df = df[df['Dataset'] == dataset]
                    
                    # Sort by quality
                    dataset_df = dataset_df.sort_values('Avg_Quality', ascending=False)
                    
                    # Create bar chart
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    colors = []
                    for model in dataset_df['Model']:
                        colors.append(self.model_colors.get(model, '#1f77b4'))
                    
                    bars = ax.bar(dataset_df['Model'], dataset_df['Avg_Quality'], 
                                 color=colors, alpha=0.7)
                    
                    ax.set_title(f"Model Performance - {dataset}", fontsize=14)
                    ax.set_xlabel("Model")
                    ax.set_ylabel("Average Quality Score")
                    ax.set_ylim(0, 1)
                    ax.grid(True, alpha=0.3, axis='y')
                    plt.xticks(rotation=45, ha='right')
                    
                    # Add value labels on bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{height:.3f}', ha='center', va='bottom', fontsize=8)
                    
                    plt.tight_layout()
                    
                    save_path = self.results_dir / f"performance_{dataset}.png"
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    print(f"Saved performance chart for {dataset}")
                    
            except Exception as e:
                print(f"Could not create performance chart: {e}")
        else:
            print("No metrics file found for performance chart")
    
    def create_dataset_summary(self, datasets: List[str]):
        """
        Create a summary plot showing all datasets side by side.
        """
        n_datasets = len(datasets)
        
        fig, axes = plt.subplots(n_datasets, 2, figsize=(12, 4*n_datasets))
        
        for idx, dataset_name in enumerate(datasets):
            # Load samples
            samples = self.load_or_generate_samples(dataset_name, n_samples=1)
            if samples is None:
                continue
            
            real_samples = samples['real_samples']
            generated_samples = samples['generated_samples']
            
            # Get valid models
            valid_models = [m for m in self.config.benchmark_models 
                          if generated_samples.get(m) is not None]
            
            if not valid_models or len(real_samples) == 0:
                continue
            
            # Left: Real sample
            real_sample = real_samples[0].numpy().flatten()
            axes[idx, 0].plot(real_sample, **self.line_styles['real'])
            axes[idx, 0].set_title(f"{dataset_name} - Real Sample", fontsize=10)
            axes[idx, 0].set_xlabel("Time")
            axes[idx, 0].set_ylabel("Value")
            axes[idx, 0].grid(True, alpha=0.3)
            
            # Right: Best generated sample (BifurcationGAN if available)
            best_model = 'bifurcation_gan' if 'bifurcation_gan' in valid_models else valid_models[0]
            model_samples = generated_samples[best_model]
            
            if model_samples is not None:
                fake_sample = model_samples[0].numpy().flatten()
                color = self.model_colors.get(best_model, '#1f77b4')
                display_name = self.model_display_names.get(best_model, best_model)
                
                axes[idx, 1].plot(fake_sample, linewidth=2, alpha=0.7, 
                                 color=color, linestyle='--', label=display_name)
                axes[idx, 1].plot(real_sample, linewidth=1, alpha=0.3, 
                                 color='gray', linestyle='-', label='Real')
                axes[idx, 1].set_title(f"{dataset_name} - Generated by {display_name}", fontsize=10)
                axes[idx, 1].legend(fontsize=8)
            else:
                axes[idx, 1].text(0.5, 0.5, "No generated sample", 
                                 ha='center', va='center', transform=axes[idx, 1].transAxes)
            
            axes[idx, 1].set_xlabel("Time")
            axes[idx, 1].set_ylabel("Value")
            axes[idx, 1].grid(True, alpha=0.3)
        
        plt.suptitle("Dataset Summary: Real vs Generated Samples", fontsize=14, y=1.02)
        plt.tight_layout()
        
        save_path = self.results_dir / "dataset_summary.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved dataset summary to: {save_path}")
    
    def run_simple_visualization(self, datasets: Optional[List[str]] = None):
        """
        Run a simple visualization pipeline.
        """
        print("\n" + "=" * 80)
        print("RUNNING SIMPLE VISUALIZATION PIPELINE")
        print("=" * 80)
        
        if datasets is None:
            datasets = self.config.dataset_names[:3]
        
        print(f"Processing {len(datasets)} datasets: {datasets}")
        
        # Create individual comparison plots
        print("\nCreating individual comparison plots...")
        for dataset_name in datasets:
            self.create_simple_comparison(dataset_name, n_samples=3)  # Reduced for clarity
        
        # Create dataset summary
        print("\nCreating dataset summary...")
        self.create_dataset_summary(datasets)
        
        # Try to create performance chart
        print("\nCreating performance chart...")
        self.create_model_performance_chart()
        
        print("\n" + "=" * 80)
        print("VISUALIZATION COMPLETED")
        print("=" * 80)
        print(f"All visualizations saved to: {self.results_dir}")
        
        # Generate simple text report
        self.generate_text_report(datasets)
    
    def generate_text_report(self, datasets: List[str]):
        """
        Generate a simple text report.
        """
        report_path = self.results_dir / "visualization_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MODEL COMPARISON VISUALIZATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of datasets: {len(datasets)}\n")
            f.write(f"Models included: {', '.join(self.config.benchmark_models)}\n")
            f.write(f"Results directory: {self.results_dir}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("MODEL DESCRIPTIONS\n")
            f.write("=" * 80 + "\n")
            
            model_descriptions = {
                'vanilla_gan': 'Standard GAN with binary cross-entropy loss',
                'wgan': 'Wasserstein GAN with improved training stability',
                'wgan_gp': 'WGAN with gradient penalty for Lipschitz constraint',
                'tts_gan': 'Time series specific GAN architecture',
                'bifurcation_gan': 'Proposed model with Hopf bifurcation dynamics',
            }
            
            for model_type in self.config.benchmark_models:
                display_name = self.model_display_names.get(model_type, model_type)
                description = model_descriptions.get(model_type, 'No description')
                f.write(f"{display_name}: {description}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("GENERATED FILES\n")
            f.write("=" * 80 + "\n")
            
            for file in self.results_dir.glob("*.png"):
                f.write(f"- {file.name}\n")
            
            if (self.results_dir / "visualization_report.txt").exists():
                f.write(f"- visualization_report.txt\n")
        
        print(f"Text report saved to: {report_path}")

def main():
    """Main execution function."""
    
    print("=" * 80)
    print("SIMPLE MODEL COMPARISON VISUALIZATION")
    print("=" * 80)
    
    # Create visualizer with reduced model set
    visualizer = ModelComparisonVisualizer(config, results_dir="./comparison_plots_simple")
    
    # Use a small subset of datasets for testing
    demo_datasets = [ 'BasicMotions', 'EigenWorms', 'Epilepsy', 'ERing', 
                'Lightning7', 'FingerMovements', 'HandMovementDirection',
                'Handwriting', 'Heartbeat', 'JapaneseVowels', 'Libras', 
                'LSST', 'MotorImagery', 'NATOPS', 'Plane']
    
    # Run visualization
    visualizer.run_simple_visualization(datasets=demo_datasets)
    
    print("\n" + "=" * 80)
    print("SCRIPT COMPLETED!")
    print("=" * 80)
    print("\nGenerated files:")
    
    # List generated files
    result_dir = Path("./comparison_plots_simple")
    if result_dir.exists():
        for file in result_dir.glob("*.png"):
            print(f"  {file.name}")
    
    print(f"\nCheck the directory: {result_dir}")
    print("\nNote: If models fail to load, the script will use random samples for visualization.")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run main function
    try:
        main()
    except Exception as e:
        print(f"Script failed with error: {e}")
        traceback.print_exc()