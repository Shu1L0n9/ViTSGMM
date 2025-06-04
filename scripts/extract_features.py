#!/usr/bin/env python3
"""
ViT Feature Extraction Script
Extract image features from pre-trained Vision Transformer models
"""

import torch
from timm import create_model
from tqdm import tqdm
import hashlib
import os
import argparse
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_loader import DatasetLoader
from src.config.config import config

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='ViT Feature Extractor')
    parser.add_argument('--dataset_type', type=str, default='cifar10', 
                        choices=['stl10', 'cifar10', 'cifar100'], 
                        help='Dataset type')
    parser.add_argument('--data_root', type=str, default='./data', 
                        help='Dataset root directory')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Batch size')
    parser.add_argument('--image_size', type=int, default=518, 
                        help='Image processing size')
    parser.add_argument('--compute_norm', type=lambda x: x.lower() == 'true', 
                        default=True, help='Whether to recompute normalization parameters')
    return parser.parse_args()

def extract_features():
    """Extract ViT features"""
    args = parse_args()
    
    # Update configuration
    config.data.dataset_type = args.dataset_type
    config.data.data_root = args.data_root
    config.data.batch_size = args.batch_size
    config.data.image_size = args.image_size
    config.data.compute_norm = args.compute_norm
    
    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_path = config.data.feature_file
    model_name = config.vit.model_name
    
    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Initialize data loader
    data_config = {
        'dataset_type': config.data.dataset_type,
        'data_root': config.data.data_root,
        'batch_size': config.data.batch_size,
        'image_size': config.data.image_size,
        'compute_norm': config.data.compute_norm
    }
    
    data_loader = DatasetLoader(data_config)
    train_loader, test_loader = data_loader.get_data_loaders()
    dataset_info = data_loader.get_dataset_info()
    
    print(f"\nDataset Information:")
    print(f"Dataset: {config.data.dataset_type}")
    print(f"Train samples: {dataset_info['train_size']}")
    print(f"Test samples: {dataset_info['test_size']}")
    print(f"Classes: {dataset_info['num_classes']}")
    print(f"Image size: {dataset_info['image_size']}")
    
    # Load pre-trained model
    print(f"\nLoading model: {model_name}")
    model = create_model(model_name, pretrained=True, num_classes=0)
    model = model.to(device)
    model.eval()
    
    # Extract training set features
    print("\nExtracting training features...")
    train_features = []
    train_labels = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc="Training")):
            images = images.to(device)
            features = model(images)
            train_features.append(features.cpu())
            train_labels.append(labels)
    
    train_features = torch.cat(train_features, dim=0)
    train_labels = torch.cat(train_labels, dim=0)
    
    # Extract test set features
    print("Extracting test features...")
    test_features = []
    test_labels = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="Testing")):
            images = images.to(device)
            features = model(images)
            test_features.append(features.cpu())
            test_labels.append(labels)
    
    test_features = torch.cat(test_features, dim=0)
    test_labels = torch.cat(test_labels, dim=0)
    
    # Save features
    print(f"\nSaving features to {save_path}")
    torch.save({
        'train_features': train_features,
        'train_labels': train_labels,
        'test_features': test_features,
        'test_labels': test_labels,
        'dataset_info': dataset_info,
        'model_name': model_name
    }, save_path)
    
    print(f"Features saved successfully!")
    print(f"Train features shape: {train_features.shape}")
    print(f"Test features shape: {test_features.shape}")

if __name__ == "__main__":
    extract_features()
