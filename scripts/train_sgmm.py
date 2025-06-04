#!/usr/bin/env python3
"""
SGMM Training Script
Training using Semi-supervised Gaussian Mixture Model
"""

import argparse
import os
import sys
import json
import numpy as np
from datetime import datetime

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sklearn.metrics import accuracy_score, top_k_accuracy_score
from src.utils.utils import pca_reduce
from src.models.sgmm_core import SemiSupervisedGMM
from src.data.data_loader import DatasetManager

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='SGMM Training')
    parser.add_argument('--dataset', type=str, default='cifar10',
                      help='Dataset name')
    parser.add_argument('--feature_file', type=str, 
                      default='data/vit/cifar10/cifar10_vit_features.pt',
                      help='Path to feature file')
    parser.add_argument('--labeled_per_class', type=int, default=4,
                      help='Number of labeled samples per class')
    parser.add_argument('--n_components_pca', type=int, default=60,
                      help='Number of PCA components')
    parser.add_argument('--n_components_gmm', type=int, default=13,
                      help='Number of GMM components')
    parser.add_argument('--tol', type=float, default=1e1,
                      help='Convergence tolerance')
    parser.add_argument('--max_iter', type=int, default=1024,
                      help='Maximum iterations')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Training device')
    parser.add_argument('--top_k', type=int, default=3,
                      help='Top-K accuracy')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    parser.add_argument('--output_dir', type=str, default='results',
                      help='Output directory')
    return parser.parse_args()

def train_sgmm():
    """Train SGMM model"""
    args = parse_args()
    
    print("="*60)
    print("SGMM Training Configuration")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Feature file: {args.feature_file}")
    print(f"Labeled per class: {args.labeled_per_class}")
    print(f"PCA components: {args.n_components_pca}")
    print(f"GMM components: {args.n_components_gmm}")
    print(f"Tolerance: {args.tol}")
    print(f"Max iterations: {args.max_iter}")
    print(f"Device: {args.device}")
    print(f"Random seed: {args.seed}")
    print("="*60)
    
    # Load data
    print("Loading dataset...")
    dataset_manager = DatasetManager(args.feature_file)
    
    # Get training data (split labeled and unlabeled data)
    X_labeled, y_labeled, X_unlabeled, X_test, y_test = dataset_manager.get_data_for_training(
        label_per_class=args.labeled_per_class,
        seed=args.seed
    )
    
    print(f"Labeled training data: {X_labeled.shape}")
    print(f"Unlabeled training data: {X_unlabeled.shape}")
    print(f"Test data: {X_test.shape}")
    print(f"Labeled samples: {len(y_labeled)}")
    print(f"Unlabeled samples: {len(X_unlabeled)}")
    
    # PCA dimensionality reduction
    print("\nApplying PCA...")
    X_train = np.concatenate((X_labeled, X_unlabeled), axis=0)
    y_train = np.concatenate((y_labeled, np.full(len(X_unlabeled), -1)), axis=0)
    
    X_train_pca, y_train_pca, X_test_pca, y_test_pca = pca_reduce(
        X_train, y_train, X_test, y_test, 
        n_components_pca=args.n_components_pca
    )
    
    # Re-split the data after dimensionality reduction
    X_labeled_pca = X_train_pca[:len(X_labeled)]
    X_unlabeled_pca = X_train_pca[len(X_labeled):]
    y_labeled_pca = y_train_pca[:len(y_labeled)]
    
    print(f"After PCA - Labeled: {X_labeled_pca.shape}, Unlabeled: {X_unlabeled_pca.shape}")
    
    # Determine number of classes
    n_classes = len(np.unique(y_labeled_pca))
    print(f"Number of classes: {n_classes}")
    
    # Initialize SGMM model
    print("\nInitializing SGMM model...")
    sgmm = SemiSupervisedGMM(
        n_components=args.n_components_gmm,
        n_classes=n_classes,
        max_iter=args.max_iter,
        tol=args.tol,
        device=args.device
    )
    
    # Train model
    print("Training SGMM...")
    start_time = datetime.now()
    
    sgmm.fit(X_labeled_pca, y_labeled_pca, X_unlabeled_pca)
    
    training_time = datetime.now() - start_time
    print(f"Training completed in {training_time}")
    
    # Prediction
    print("\nMaking predictions...")
    test_predictions, test_probabilities = sgmm.predict(X_test_pca, return_probs=True)
    
    # Calculate evaluation metrics
    test_accuracy = accuracy_score(y_test_pca, test_predictions)
    test_top_k_accuracy = top_k_accuracy_score(
        y_test_pca, test_probabilities, k=args.top_k
    )
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION ON TESTSET")
    print("="*60)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Top-{args.top_k} Accuracy: {test_top_k_accuracy:.4f}")
    print("="*60)

if __name__ == "__main__":
    train_sgmm()
