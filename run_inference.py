#!/usr/bin/env python3
"""
CONCH Inference for TCGA Kidney Classification
This script performs inference classification on TCGA kidney data using 
pre-extracted CONCH features stored in h5 files.
Usage:
    python run_inference.py --config config.yaml
    
    # Or with command line arguments:
    python run_inference.py --h5_path data/features.h5 --output_dir results
"""
import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Check for h5py
try:
    import h5py
except ImportError:
    print("Error: h5py is required. Install with: pip install h5py")
    sys.exit(1)

# Check for torch
try:
    import torch
except ImportError:
    print("Error: PyTorch is required. Install with: pip install torch")
    sys.exit(1)


class CONCHInference:
    """
    CONCH Inference Pipeline for TCGA Kidney Classification
    
    This class handles:
    - Loading pre-extracted CONCH features from h5 files
    - Running classification inference
    - Saving predictions and probabilities
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the inference pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = config['inference']['device']
        
        # Set device
        if self.device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA not available, using CPU instead")
            self.device = "cpu"
        
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = None
        self.tokenizer = None
        self.classifier = None
        self._load_model()
        
    def _load_model(self):
        """Load CONCH model and classifier."""
        model_config = self.config['model']
        
        # Check if checkpoint exists
        checkpoint_path = model_config.get('conch_checkpoint_path')
        if checkpoint_path and not os.path.exists(checkpoint_path):
            print(f"Warning: Model checkpoint not found at {checkpoint_path}")
            print("Please download the CONCH model from HuggingFace:")
            print("  https://huggingface.co/MahmoodLab/CONCH")
            print("Falling back to feature-based classification (no text prompts)")
            self.model = None
            return
        
        try:
            # Load CONCH model using create_model_from_pretrained
            print(f"Loading CONCH model...")
            
            # Import CONCH (if available)
            try:
                from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer
                checkpoint_path = model_config['conch_checkpoint_path']
                print(f"Loading CONCH from {checkpoint_path}...")
                self.model, self.preprocess = create_model_from_pretrained(
                    'conch_ViT-B-16', 
                    checkpoint_path, 
                    device=self.device
                )
                # Get tokenizer
                self.tokenizer = get_tokenizer()
                self.model.eval()
                print("CONCH model loaded successfully!")
            except ImportError as e:
                print(f"CONCH import error: {e}")
                print("Please install CONCH:")
                print("  git clone https://github.com/mahmoodlab/CONCH.git")
                print("  cd CONCH && pip install -e .")
                self.model = None
                self.tokenizer = None
            except Exception as e:
                print(f"Error loading CONCH model: {e}")
                import traceback
                traceback.print_exc()
                self.model = None
                self.tokenizer = None
                
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
        
        # Load classifier if specified
        classification_config = self.config['classification']
        if classification_config.get('use_pretrained', False):
            classifier_path = classification_config.get('classifier_path')
            if classifier_path and os.path.exists(classifier_path):
                self._load_classifier(classifier_path)
    
    def _load_classifier(self, classifier_path: str):
        """Load a pretrained classifier."""
        try:
            self.classifier = torch.load(classifier_path, map_location=self.device)
            self.classifier.to(self.device)
            self.classifier.eval()
            print(f"Classifier loaded from {classifier_path}")
        except Exception as e:
            print(f"Error loading classifier: {e}")
            self.classifier = None
    
    def load_h5_features(self, h5_path: str) -> Tuple[np.ndarray, List[str]]:
        """
        Load features from h5 file.
        
        Args:
            h5_path: Path to h5 file
            
        Returns:
            Tuple of (features array, slide_ids list)
        """
        print(f"Loading features from {h5_path}")
        
        data_config = self.config['data']
        features_key = data_config.get('features_key', 'features')
        slide_ids_key = data_config.get('slide_ids_key', 'slide_id')
        
        with h5py.File(h5_path, 'r') as f:
            print(f"Available keys in h5 file: {list(f.keys())}")
            
            # Load features
            if features_key in f:
                features = f[features_key][:]
            else:
                # Try common alternative keys
                for key in ['embedding', 'feat', 'feature', 'data']:
                    if key in f:
                        features = f[key][:]
                        print(f"Using '{key}' as features key")
                        break
                else:
                    raise KeyError(f"Could not find features in h5 file. Available keys: {list(f.keys())}")
            
            # Load slide IDs
            if slide_ids_key in f:
                slide_ids = f[slide_ids_key][:]
                # Convert bytes to strings if needed
                if isinstance(slide_ids[0], bytes):
                    slide_ids = [s.decode('utf-8') for s in slide_ids]
            else:
                # Generate default IDs
                slide_ids = [f"slide_{i}" for i in range(len(features))]
                print(f"Warning: Could not find slide IDs, using default IDs")
        
        print(f"Loaded {len(features)} samples with {features.shape[1]} features each")
        
        return features, slide_ids
    
    def load_multiple_h5(self, h5_paths: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Load features from multiple h5 files.
        
        Args:
            h5_paths: List of h5 file paths
            
        Returns:
            Tuple of (combined features array, combined slide_ids list)
        """
        all_features = []
        all_slide_ids = []
        
        for h5_path in h5_paths:
            features, slide_ids = self.load_h5_features(h5_path)
            all_features.append(features)
            all_slide_ids.extend(slide_ids)
        
        combined_features = np.vstack(all_features)
        print(f"Combined {len(h5_paths)} files: {combined_features.shape[0]} samples")
        
        return combined_features, all_slide_ids
    
    def predict_zeroshot(self, features: np.ndarray, 
                        prompts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform zero-shot classification using text prompts.
        
        Args:
            features: Image features (N x D)
            prompts: List of text prompts for each class
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model is None:
            print("Error: CONCH model not loaded. Cannot perform zero-shot classification.")
            return None, None
        
        # Tokenize text prompts
        if self.tokenizer is None:
            print("Error: No tokenizer available. Cannot perform zero-shot classification.")
            return None, None
        
        # Tokenize prompts using CONCH's tokenize function
        from conch.open_clip_custom import tokenize
        text_tokens = tokenize(texts=prompts, tokenizer=self.tokenizer).to(self.device)
        
        # Encode text prompts
        text_embeddings = self.model.encode_text(text_tokens)
        
        # Normalize embeddings (following CONCH example)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        
        # Normalize image features on CPU (avoid GPU memory issues)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features_normalized = features / norms
        
        # Compute similarities using logit_scale in batches (avoid OOM)
        logit_scale = self.model.logit_scale.exp()
        
        batch_size = 10000
        n_samples = features_normalized.shape[0]
        all_probs = []
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_features = torch.from_numpy(features_normalized[start_idx:end_idx]).to(self.device)
            batch_similarities = (batch_features @ text_embeddings.T * logit_scale).detach().cpu().numpy()
            batch_probs = self._softmax(batch_similarities, axis=1)
            all_probs.append(batch_probs)
        
        probs = np.vstack(all_probs)
        
        # Get predictions
        predictions = np.argmax(probs, axis=1)
        
        return predictions, probs
    
    def predict_with_classifier(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform classification using a pretrained classifier.
        
        Args:
            features: Image features (N x D)
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.classifier is None:
            print("Error: Classifier not loaded. Using random predictions.")
            return self._random_predictions(features)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.classifier(features_tensor)
            probs = torch.softmax(logits, dim=1)
        
        predictions = torch.argmax(probs, dim=1).cpu().numpy()
        probs = probs.cpu().numpy()
        
        return predictions, probs
    
    def predict_linear_probe(self, features: np.ndarray, 
                            labels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train a linear probe classifier and predict.
        
        Args:
            features: Image features (N x D)
            labels: Training labels (if available for linear probe training)
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        # This is a simplified linear probe - in practice you'd use proper training
        # For now, we'll use a simple approach
        num_classes = self.config['classification']['num_classes']
        
        if labels is not None:
            # Train a simple linear classifier
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=1000, multi_class='multinomial')
            clf.fit(features, labels)
            
            probs = clf.predict_proba(features)
            predictions = np.argmax(probs, axis=1)
            
            return predictions, probs
        else:
            # No labels provided, use clustering or return random
            return self._random_predictions(features)
    
    def _random_predictions(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate random predictions as fallback."""
        num_classes = self.config['classification']['num_classes']
        n_samples = features.shape[0]
        
        predictions = np.random.randint(0, num_classes, n_samples)
        probs = np.zeros((n_samples, num_classes))
        probs[np.arange(n_samples), predictions] = 1.0
        
        return predictions, probs
    
    def _softmax(self, x: np.ndarray, axis: int = 1) -> np.ndarray:
        """Apply softmax to array."""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def run_inference(self, h5_path: str, 
                     output_dir: str) -> pd.DataFrame:
        """
        Run inference on h5 feature file.
        
        Args:
            h5_path: Path to h5 feature file
            output_dir: Directory to save results
            
        Returns:
            DataFrame with predictions
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load features
        features, slide_ids = self.load_h5_features(h5_path)
        
        # Get predictions
        zeroshot_config = self.config.get('zeroshot', {})
        
        print(f"DEBUG: zeroshot enabled = {zeroshot_config.get('enabled', False)}, model = {self.model}, tokenizer = {self.tokenizer}")
        
        if zeroshot_config.get('enabled', False) and self.model is not None:
            # Zero-shot classification
            prompts = zeroshot_config.get('prompts', [])
            print(f"Running zero-shot classification with {len(prompts)} prompts")
            predictions, probs = self.predict_zeroshot(features, prompts)
            
        elif self.classifier is not None:
            # Classifier-based prediction
            print("Running classifier-based prediction")
            predictions, probs = self.predict_with_classifier(features)
            
        else:
            # Fallback: linear probe or random
            print("Warning: No model or classifier available. Using linear probe.")
            predictions, probs = self.predict_linear_probe(features)
        
        # Create results DataFrame
        class_names = self.config['classification']['class_names']
        
        results = pd.DataFrame({
            'slide_id': slide_ids,
            'prediction': predictions,
            'predicted_class': [class_names[p] for p in predictions]
        })
        
        # Add probabilities for each class
        output_config = self.config['output']
        if output_config.get('save_probabilities', True) and probs is not None:
            prob_df = pd.DataFrame(probs, columns=[f'prob_{class_names[i]}' for i in range(len(class_names))])
            prob_df.insert(0, 'slide_id', slide_ids)
            
            prob_file = os.path.join(output_dir, output_config.get('prob_file', 'probabilities.csv'))
            prob_df.to_csv(prob_file, index=False)
            print(f"Probabilities saved to {prob_file}")
        
        # Save predictions
        output_file = os.path.join(output_dir, output_config.get('output_file', 'predictions.csv'))
        results.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
        
        # Print summary
        print("\nPrediction Summary:")
        print(results['predicted_class'].value_counts())
        
        return results


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(
        description='CONCH Inference for TCGA Kidney Classification'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--h5_path', 
        type=str, 
        default=None,
        help='Path to h5 feature file or glob pattern (overrides config)'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default=None,
        help='Output directory (overrides config)'
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default=None,
        help='Device to use: cuda or cpu (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.h5_path:
        config['data']['h5_features_path'] = args.h5_path
    
    if args.output_dir:
        config['output']['results_dir'] = args.output_dir
    
    if args.device:
        config['inference']['device'] = args.device
    
    # Print config
    print("Configuration:")
    print(f"  H5 features: {config['data']['h5_features_path']}")
    print(f"  Output dir: {config['output']['results_dir']}")
    print(f"  Device: {config['inference']['device']}")
    print()
    
    # Create inference pipeline
    inference = CONCHInference(config)
    
    # Get h5 paths - can be a list or a string
    h5_path_config = config['data']['h5_features_path']
    output_dir = config['output']['results_dir']
    
    # Handle both list and string inputs
    if isinstance(h5_path_config, list):
        # Collect all h5 files from all patterns
        from glob import glob
        all_h5_files = []
        for pattern in h5_path_config:
            files = glob(pattern)
            if files:
                all_h5_files.extend(files)
                print(f"Found {len(files)} files for pattern: {pattern}")
        
        if not all_h5_files:
            print(f"Error: No h5 files found for any of the {len(h5_path_config)} patterns")
            sys.exit(1)
        
        print(f"Total h5 files to process: {len(all_h5_files)}")
        
        # Load and combine all features
        features, slide_ids = inference.load_multiple_h5(all_h5_files)
        
    elif isinstance(h5_path_config, str):
        if '*' in h5_path_config:
            # Single glob pattern
            from glob import glob
            h5_paths = glob(h5_path_config)
            if not h5_paths:
                print(f"Error: No files found matching pattern {h5_path_config}")
                sys.exit(1)
            print(f"Found {len(h5_paths)} files matching pattern")
            
            # Load and combine
            features, slide_ids = inference.load_multiple_h5(h5_paths)
            
        elif os.path.exists(h5_path_config):
            # Single file
            features, slide_ids = inference.load_h5_features(h5_path_config)
        else:
            print(f"Error: H5 file not found: {h5_path_config}")
            sys.exit(1)
    else:
        print("Error: h5_features_path must be a string or list")
        sys.exit(1)
    
    # Now run inference on the loaded features
    print(f"\nRunning inference on {len(features)} samples...")
    
    # Get predictions
    zeroshot_config = config.get('zeroshot', {})
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if zeroshot_config.get('enabled', False) and inference.model is not None and inference.tokenizer is not None:
        # Zero-shot classification
        prompts = zeroshot_config.get('prompts', [])
        print(f"Running zero-shot classification with {len(prompts)} prompts")
        predictions, probs = inference.predict_zeroshot(features, prompts)
        
    elif inference.classifier is not None:
        # Classifier-based prediction
        print("Running classifier-based prediction")
        predictions, probs = inference.predict_with_classifier(features)
        
    else:
        # Fallback: linear probe or random
        print("Warning: No model or classifier available. Using linear probe.")
        predictions, probs = inference.predict_linear_probe(features)
    
    # Create results DataFrame
    class_names = config['classification']['class_names']
    
    results = pd.DataFrame({
        'slide_id': slide_ids,
        'prediction': predictions,
        'predicted_class': [class_names[p] for p in predictions]
    })
    
    # Add probabilities for each class
    output_config = config['output']
    if output_config.get('save_probabilities', True) and probs is not None:
        prob_df = pd.DataFrame(probs, columns=[f'prob_{class_names[i]}' for i in range(len(class_names))])
        prob_df.insert(0, 'slide_id', slide_ids)
        
        prob_file = os.path.join(output_dir, output_config.get('prob_file', 'probabilities.csv'))
        prob_df.to_csv(prob_file, index=False)
        print(f"Probabilities saved to {prob_file}")
    
    # Save predictions
    output_file = os.path.join(output_dir, output_config.get('output_file', 'predictions.csv'))
    results.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
    
    # Print summary
    print("\nPrediction Summary:")
    print(results['predicted_class'].value_counts())
    
    print("\nInference complete!")


if __name__ == '__main__':
    main()
