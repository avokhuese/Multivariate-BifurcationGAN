"""
Debug script to test model loading and identify issues
"""

import torch
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_univariate import config
from gan_framework_univariate import create_gan_framework
from model_loader_utils import find_best_checkpoint, load_model_checkpoint

def debug_all_models():
    """Debug loading of all trained models"""
    print("\n" + "=" * 80)
    print("DEBUG MODEL LOADING")
    print("=" * 80)
    
    # Models to test
    models = ['vanilla_gan', 'wgan', 'wgan_gp', 'tts_gan', 'bifurcation_gan']
    
    # Find all datasets from checkpoint files
    all_checkpoints = glob.glob(os.path.join(config.save_dir, "*.pth"))
    
    # Extract unique datasets
    datasets = set()
    for checkpoint in all_checkpoints:
        filename = os.path.basename(checkpoint)
        parts = filename.split('_')
        if len(parts) >= 2:
            datasets.add(parts[1])  # Second part is usually dataset name
    
    print(f"Found checkpoints for {len(datasets)} datasets")
    print(f"Datasets: {sorted(list(datasets))}")
    
    # Test each model-dataset combination
    successful_combinations = []
    
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Testing {model_name}")
        print('='*60)
        
        for dataset_name in sorted(datasets):
            print(f"\n  {dataset_name}: ", end='')
            
            # Find checkpoint
            checkpoint_path = find_best_checkpoint(model_name, dataset_name, config.save_dir)
            
            if not checkpoint_path:
                print("No checkpoint found")
                continue
            
            print(f"Found {os.path.basename(checkpoint_path)}")
            
            # Try to load
            try:
                # Create model
                gan = create_gan_framework(model_name, config)
                
                # Try to load
                if load_model_checkpoint(gan.generator, checkpoint_path, config.device):
                    print(f"    ✓ Successfully loaded")
                    
                    # Try to generate a sample
                    try:
                        sample = gan.generate_samples(1)
                        print(f"    ✓ Can generate: {sample.shape}")
                        successful_combinations.append((model_name, dataset_name))
                    except Exception as e:
                        print(f"    ✗ Generation failed: {e}")
                else:
                    print(f"    ✗ Load failed")
                    
            except Exception as e:
                print(f"    ✗ Error: {e}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("LOADING SUMMARY")
    print("=" * 80)
    print(f"\nSuccessful combinations: {len(successful_combinations)}")
    
    # Group by model
    for model_name in models:
        model_success = [d for m, d in successful_combinations if m == model_name]
        print(f"\n{model_name}: {len(model_success)} datasets")
        for dataset in model_success:
            print(f"  - {dataset}")
    
    return successful_combinations

def fix_checkpoint(checkpoint_path: str, output_path: str = None):
    """Fix a checkpoint by removing problematic keys"""
    if output_path is None:
        output_path = checkpoint_path.replace('.pth', '_fixed.pth')
    
    print(f"\nFixing checkpoint: {os.path.basename(checkpoint_path)}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        if 'generator_state_dict' in checkpoint:
            print(f"  Found generator_state_dict with {len(checkpoint['generator_state_dict'])} keys")
            
            # Remove problematic keys
            keys_to_remove = []
            for key in checkpoint['generator_state_dict'].keys():
                if 'positional_encoding' in key or 'pos_encoding' in key:
                    keys_to_remove.append(key)
                    print(f"    Removing: {key}")
            
            for key in keys_to_remove:
                del checkpoint['generator_state_dict'][key]
        
        # Save fixed checkpoint
        torch.save(checkpoint, output_path)
        print(f"  ✓ Saved fixed checkpoint to: {output_path}")
        
        return output_path
        
    except Exception as e:
        print(f"  ✗ Failed to fix checkpoint: {e}")
        return None

def fix_all_checkpoints():
    """Fix all checkpoints in the save directory"""
    print("\n" + "=" * 80)
    print("FIXING ALL CHECKPOINTS")
    print("=" * 80)
    
    checkpoints = glob.glob(os.path.join(config.save_dir, "*.pth"))
    
    fixed_count = 0
    for checkpoint_path in checkpoints:
        if '_fixed' in checkpoint_path:
            continue
        
        fixed_path = fix_checkpoint(checkpoint_path)
        if fixed_path:
            fixed_count += 1
    
    print(f"\nFixed {fixed_count} checkpoints")
    print("\nNow try loading models again:")
    print("  python plot_trained_models_comparison.py")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='Debug model loading')
    parser.add_argument('--fix', action='store_true', help='Fix all checkpoints')
    parser.add_argument('--fix-single', type=str, help='Fix single checkpoint')
    
    args = parser.parse_args()
    
    if args.fix:
        fix_all_checkpoints()
    elif args.fix_single:
        fix_checkpoint(args.fix_single)
    else:
        debug_all_models()