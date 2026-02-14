"""
Heatmap visualization script for TCGA kidney classification results.
Generates various heatmaps from inference results.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import sys


def plot_probability_heatmap(results_df, class_names, output_path=None, top_n=50):
    """
    Plot heatmap of prediction probabilities for top N samples.
    
    Args:
        results_df: DataFrame with probability columns
        class_names: List of class names
        output_path: Path to save the heatmap
        top_n: Number of samples to display
    """
    # Get probability columns
    prob_cols = [col for col in results_df.columns if col.startswith('prob_')]
    
    if not prob_cols:
        print("No probability columns found in results!")
        return
    
    # Extract class names from columns
    prob_data = results_df[prob_cols].head(top_n)
    
    # Create shorter labels for display
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
    
    # Rename columns
    prob_data.columns = short_names
    
    plt.figure(figsize=(10, max(8, top_n * 0.15)))
    sns.heatmap(prob_data, annot=True, fmt='.2f', cmap='Blues',
                vmin=0, vmax=1, cbar_kws={'label': 'Probability'})
    plt.xlabel('Class')
    plt.ylabel('Sample')
    plt.title(f'Prediction Probabilities (Top {top_n} Samples)')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Heatmap saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_class_distribution(results_df, class_names, output_path=None):
    """
    Plot bar chart of predicted class distribution.
    
    Args:
        results_df: DataFrame with predictions
        class_names: List of class names
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Count predictions
    class_counts = results_df['predicted_class'].value_counts()
    
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
    
    # Reorder based on class_names
    counts = [class_counts.get(name, 0) for name in class_names]
    
    plt.bar(short_names, counts, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.xlabel('Class')
    plt.ylabel('Number of Predictions')
    plt.title('Predicted Class Distribution')
    
    # Add count labels on bars
    for i, count in enumerate(counts):
        plt.text(i, count + max(counts) * 0.01, str(count), ha='center')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Class distribution plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(results_df, class_names, output_path=None):
    """
    Plot confusion matrix if ground truth is available.
    
    Args:
        results_df: DataFrame with 'prediction' and optional 'label' column
        class_names: List of class names
        output_path: Path to save the plot
    """
    if 'label' not in results_df.columns:
        print("No ground truth labels found! Skipping confusion matrix.")
        return
    
    from sklearn.metrics import confusion_matrix
    
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
    
    # Compute confusion matrix
    cm = confusion_matrix(results_df['label'], results_df['prediction'])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=short_names, yticklabels=short_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_similarity_scores(results_df, class_names, output_path=None, n_samples=20):
    """
    Plot heatmap of similarity scores (before softmax) if available.
    
    Args:
        results_df: DataFrame with results
        class_names: List of class names
        output_path: Path to save the heatmap
        n_samples: Number of samples to display
    """
    # Check for similarity score columns
    sim_cols = [col for col in results_df.columns if 'sim' in col.lower()]
    
    if not sim_cols:
        print("No similarity score columns found!")
        return
    
    # Get similarity data
    sim_data = results_df[sim_cols].head(n_samples)
    
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
    
    plt.figure(figsize=(10, max(6, n_samples * 0.2)))
    sns.heatmap(sim_data, annot=True, fmt='.3f', cmap='RdYlBu_r',
                center=0, cbar_kws={'label': 'Similarity Score'})
    plt.xlabel('Class')
    plt.ylabel('Sample')
    plt.title(f'Zero-shot Similarity Scores (First {n_samples} Samples)')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Similarity scores heatmap saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate heatmaps from classification results')
    parser.add_argument('--results', type=str, required=True,
                        help='Path to results CSV file')
    parser.add_argument('--output_dir', type=str, default='heatmaps',
                        help='Output directory for heatmaps')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file for class names')
    parser.add_argument('--top_n', type=int, default=50,
                        help='Number of samples to show in probability heatmap')
    
    args = parser.parse_args()
    
    # Load results
    if not os.path.exists(args.results):
        print(f"Error: Results file not found: {args.results}")
        sys.exit(1)
    
    results_df = pd.read_csv(args.results)
    print(f"Loaded results with {len(results_df)} samples")
    
    # Load class names from config
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    class_names = config['classification']['class_names']
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate heatmaps
    print("\nGenerating heatmaps...")
    
    # 1. Probability heatmap
    plot_probability_heatmap(
        results_df, class_names,
        output_path=os.path.join(args.output_dir, 'probability_heatmap.png'),
        top_n=args.top_n
    )
    
    # 2. Class distribution
    plot_class_distribution(
        results_df, class_names,
        output_path=os.path.join(args.output_dir, 'class_distribution.png')
    )
    
    # 3. Confusion matrix (if labels available)
    plot_confusion_matrix(
        results_df, class_names,
        output_path=os.path.join(args.output_dir, 'confusion_matrix.png')
    )
    
    print(f"\nAll heatmaps saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
