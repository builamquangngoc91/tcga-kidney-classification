"""
Create animated GIF showing similarity scores evolution across samples.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import imageio
from tqdm import tqdm


def create_similarity_gif(results_df, class_names, output_path, n_samples=30, fps=2):
    """
    Create animated GIF showing similarity/probability scores for each sample.
    
    Args:
        results_df: DataFrame with probability columns
        class_names: List of class names
        output_path: Path to save the GIF
        n_samples: Number of samples to animate
        fps: Frames per second
    """
    # Get probability columns
    prob_cols = [col for col in results_df.columns if col.startswith('prob_')]
    
    if not prob_cols:
        print("No probability columns found!")
        return
    
    # Create short labels
    short_names = []
    for name in class_names:
        if 'KIRC' in name:
            short_names.append('KIRC')
        elif 'KICH' in name:
            short_names.append('KICH')
        elif 'KIRP' in name:
            short_names.append('KIRP')
        else:
            short_names.append(name[:10])
    
    # Get slide IDs if available
    slide_ids = results_df['slide_id'].tolist() if 'slide_id' in results_df.columns else [f'Sample {i}' for i in range(len(results_df))]
    
    # Create temp directory for frames
    temp_dir = 'temp_frames'
    os.makedirs(temp_dir, exist_ok=True)
    
    print(f"Creating {n_samples} frames...")
    
    frames = []
    for i in tqdm(range(min(n_samples, len(results_df)))):
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Get probabilities for this sample
        probs = results_df[prob_cols].iloc[i].values
        
        # Create horizontal bar chart
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        bars = ax.barh(short_names, probs, color=colors[:len(short_names)])
        
        # Add value labels
        for bar, prob in zip(bars, probs):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, 
                   f'{prob:.3f}', va='center', fontsize=12, fontweight='bold')
        
        ax.set_xlim(0, 1.15)
        ax.set_xlabel('Probability', fontsize=12)
        ax.set_title(f'Sample: {slide_ids[i]}\nPredicted: {results_df["predicted_class"].iloc[i]}', 
                    fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1.2)
        
        plt.tight_layout()
        
        # Save frame
        frame_path = os.path.join(temp_dir, f'frame_{i:04d}.png')
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        frames.append(imageio.imread(frame_path))
    
    # Create GIF
    print(f"Saving GIF to {output_path}...")
    imageio.mimsave(output_path, frames, fps=fps)
    
    # Cleanup temp files
    import shutil
    shutil.rmtree(temp_dir)
    
    print(f"GIF saved successfully!")


def create_heatmap_timelapse(results_df, class_names, output_path, n_samples=30, fps=2):
    """
    Create animated GIF showing heatmap evolution across samples.
    
    Args:
        results_df: DataFrame with probability columns
        class_names: List of class names
        output_path: Path to save the GIF
        n_samples: Number of samples to animate
        fps: Frames per second
    """
    # Get probability columns
    prob_cols = [col for col in results_df.columns if col.startswith('prob_')]
    
    if not prob_cols:
        print("No probability columns found!")
        return
    
    # Create short labels
    short_names = []
    for name in class_names:
        if 'KIRC' in name:
            short_names.append('KIRC')
        elif 'KICH' in name:
            short_names.append('KICH')
        elif 'KIRP' in name:
            short_names.append('KIRP')
        else:
            short_names.append(name[:10])
    
    # Get slide IDs
    slide_ids = results_df['slide_id'].tolist() if 'slide_id' in results_df.columns else [f'Sample {i}' for i in range(len(results_df))]
    
    # Create temp directory for frames
    temp_dir = 'temp_frames'
    os.makedirs(temp_dir, exist_ok=True)
    
    print(f"Creating {n_samples} heatmap frames...")
    
    frames = []
    for i in tqdm(range(min(n_samples, len(results_df)))):
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Get cumulative probabilities up to current sample
        current_data = results_df[prob_cols].iloc[:i+1].copy()
        current_data.columns = short_names
        
        sns.heatmap(current_data, annot=True, fmt='.2f', cmap='Blues',
                   vmin=0, vmax=1, cbar_kws={'label': 'Probability'}, ax=ax)
        
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Sample', fontsize=12)
        ax.set_title(f'Similarity Scores Over Time\nSample {i+1}/{min(n_samples, len(results_df))}: {slide_ids[i]}', 
                    fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Save frame
        frame_path = os.path.join(temp_dir, f'frame_{i:04d}.png')
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        frames.append(imageio.imread(frame_path))
    
    # Create GIF
    print(f"Saving GIF to {output_path}...")
    imageio.mimsave(output_path, frames, fps=fps)
    
    # Cleanup temp files
    import shutil
    shutil.rmtree(temp_dir)
    
    print(f"Heatmap timelapse GIF saved successfully!")


def create_comparison_gif(results_df, class_names, output_path, n_samples=30, fps=2):
    """
    Create animated GIF with side-by-side bar chart and heatmap.
    
    Args:
        results_df: DataFrame with probability columns
        class_names: List of class names
        output_path: Path to save the GIF
        n_samples: Number of samples to animate
        fps: Frames per second
    """
    # Get probability columns
    prob_cols = [col for col in results_df.columns if col.startswith('prob_')]
    
    if not prob_cols:
        print("No probability columns found!")
        return
    
    # Create short labels
    short_names = []
    for name in class_names:
        if 'KIRC' in name:
            short_names.append('KIRC')
        elif 'KICH' in name:
            short_names.append('KICH')
        elif 'KIRP' in name:
            short_names.append('KIRP')
        else:
            short_names.append(name[:10])
    
    # Get slide IDs
    slide_ids = results_df['slide_id'].tolist() if 'slide_id' in results_df.columns else [f'Sample {i}' for i in range(len(results_df))]
    
    # Create temp directory for frames
    temp_dir = 'temp_frames'
    os.makedirs(temp_dir, exist_ok=True)
    
    print(f"Creating {n_samples} comparison frames...")
    
    frames = []
    for i in tqdm(range(min(n_samples, len(results_df)))):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left: Bar chart for current sample
        probs = results_df[prob_cols].iloc[i].values
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        bars = ax1.barh(short_names, probs, color=colors[:len(short_names)])
        
        for bar, prob in zip(bars, probs):
            ax1.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, 
                   f'{prob:.3f}', va='center', fontsize=11, fontweight='bold')
        
        ax1.set_xlim(0, 1.15)
        ax1.set_xlabel('Probability')
        ax1.set_title(f'Current Sample: {slide_ids[i]}')
        
        # Right: Cumulative heatmap
        current_data = results_df[prob_cols].iloc[:i+1].copy()
        current_data.columns = short_names
        
        sns.heatmap(current_data, annot=True, fmt='.2f', cmap='Blues',
                   vmin=0, vmax=1, cbar_kws={'label': 'Probability'}, ax=ax2)
        
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Sample')
        ax2.set_title(f'Accumulated Samples ({i+1})')
        
        plt.suptitle(f'Zero-Shot Classification: Similarity Scores', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save frame
        frame_path = os.path.join(temp_dir, f'frame_{i:04d}.png')
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        frames.append(imageio.imread(frame_path))
    
    # Create GIF
    print(f"Saving GIF to {output_path}...")
    imageio.mimsave(output_path, frames, fps=fps)
    
    # Cleanup temp files
    import shutil
    shutil.rmtree(temp_dir)
    
    print(f"Comparison GIF saved successfully!")


def main():
    parser = argparse.ArgumentParser(description='Create animated GIF of similarity scores')
    parser.add_argument('--results', type=str, default='results/predictions.csv',
                        help='Path to results CSV file')
    parser.add_argument('--output', type=str, default='results/similarity_animation.gif',
                        help='Output path for GIF')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--n_samples', type=int, default=30,
                        help='Number of samples to animate')
    parser.add_argument('--fps', type=int, default=2,
                        help='Frames per second')
    parser.add_argument('--type', type=str, default='comparison', 
                        choices=['bars', 'heatmap', 'comparison'],
                        help='Type of animation: bars, heatmap, or comparison')
    
    args = parser.parse_args()
    
    # Load results
    if not os.path.exists(args.results):
        print(f"Error: Results file not found: {args.results}")
        return
    
    results_df = pd.read_csv(args.results)
    print(f"Loaded results with {len(results_df)} samples")
    
    # Load class names
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    class_names = config['classification']['class_names']
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Generate GIF
    if args.type == 'bars':
        create_similarity_gif(results_df, class_names, args.output, args.n_samples, args.fps)
    elif args.type == 'heatmap':
        create_heatmap_timelapse(results_df, class_names, args.output, args.n_samples, args.fps)
    else:
        create_comparison_gif(results_df, class_names, args.output, args.n_samples, args.fps)


if __name__ == "__main__":
    main()
