import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import random

def compute_normalization(dataset, batch_size=1024, input_size=32):
    """Compute normalization parameters for the dataset"""
    # If input is a path, create ImageFolder dataset
    if isinstance(dataset, str):
        dataset_path = dataset
        dataset = torchvision.datasets.ImageFolder(
            dataset_path,
            transform=transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor()
            ])
        )
        print(f"\nComputing normalization parameters based on dataset path: {dataset_path}")
    else:
        print(f"\nComputing normalization parameters based on loaded dataset")
        
    # Create temporary data loader for computation
    temp_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor()
    ])
    
    # Save original transformation
    original_transform = dataset.transform
    # Set temporary transformation
    dataset.transform = temp_transform

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    mean = 0.0
    std = 0.0
    nb_samples = 0.0

    for images, _ in tqdm(loader):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.var(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std = torch.sqrt(std / nb_samples)

    # Restore original transformation
    dataset.transform = original_transform

    print(f"Computed mean: {mean.tolist()}, std: {std.tolist()}")
    return mean.tolist(), std.tolist()

class DatasetLoader:
    """Generic dataset loader class"""
    def __init__(self, config):
        """
        Initialize data loader
        
        Args:
            config: Dictionary containing dataset configuration
                - dataset_type: Dataset type ('stl10', 'cifar10', cifar100', etc.)
                - train_path: Training set path (required for custom datasets)
                - test_path: Test set path (optional)
                - image_size: Input image size
                - batch_size: Batch size
                - num_workers: Number of data loading threads
                - compute_norm: Whether to recompute normalization parameters
        """
        self.config = config
        self.dataset_type = config.get('dataset_type')
        self.batch_size = config.get('batch_size', 256)
        self.num_workers = config.get('num_workers', 4)
        self.image_size = config.get('image_size', 518)  # ViT large models typically use larger input
        self.compute_norm = config.get('compute_norm', False)
        
        # Set default normalization parameters first
        self.norm_mean = config.get('norm_mean', [0.5, 0.5, 0.5])
        self.norm_std = config.get('norm_std', [0.5, 0.5, 0.5])
        
        # If custom dataset with path provided, compute normalization parameters in advance
        if self.compute_norm and 'train_path' in config:
            self.norm_mean, self.norm_std = compute_normalization(
                config['train_path'], 
                self.batch_size, 
                32,  # Use smaller size for normalization computation to improve speed
            )
        
        # Create data transformations
        self._create_transforms()
        
        # Load datasets
        self._load_datasets()
        
        # If need to compute normalization parameters and it's a standard dataset, compute after loading
        if self.compute_norm and 'train_path' not in config:
            self.norm_mean, self.norm_std = compute_normalization(
                self.train_dataset,
                self.batch_size,
                32,  # Use smaller size for normalization computation to improve speed
            )
            # Update data transformations with newly computed normalization parameters
            self._create_transforms()
            # Re-apply transformations to datasets
            self._update_dataset_transforms()
        
    def _create_transforms(self):
        """Create data transformations"""
        # Training set transformation
        self.train_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(self.norm_mean, self.norm_std)
        ])
        
        # Test set transformation
        self.test_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(self.norm_mean, self.norm_std)
        ])
    
    def _update_dataset_transforms(self):
        """Update dataset transformations"""
        if hasattr(self, 'train_dataset'):
            self.train_dataset.transform = self.train_transform
        if hasattr(self, 'test_dataset'):
            self.test_dataset.transform = self.test_transform
    
    def _load_datasets(self):
        """Load datasets according to configuration"""
        if self.dataset_type == 'stl10':
            self.train_dataset = torchvision.datasets.STL10(
                root=self.config.get('data_root', './data'),
                split='train',
                download=True,
                transform=self.train_transform
            )
            
            self.test_dataset = torchvision.datasets.STL10(
                root=self.config.get('data_root', './data'),
                split='test',
                download=True,
                transform=self.test_transform
            )
            
        elif self.dataset_type == 'cifar10':
            self.train_dataset = torchvision.datasets.CIFAR10(
                root=self.config.get('data_root', './data'),
                train=True,
                download=True,
                transform=self.train_transform
            )
            
            self.test_dataset = torchvision.datasets.CIFAR10(
                root=self.config.get('data_root', './data'),
                train=False,
                download=True,
                transform=self.test_transform
            )
            
        elif self.dataset_type == 'cifar100':
            self.train_dataset = torchvision.datasets.CIFAR100(
                root=self.config.get('data_root', './data'),
                train=True,
                download=True,
                transform=self.train_transform
            )
            
            self.test_dataset = torchvision.datasets.CIFAR100(
                root=self.config.get('data_root', './data'),
                train=False,
                download=True,
                transform=self.test_transform
            )
        
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")
            
    def get_data_loaders(self):
        """Return data loaders"""
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.config.get('shuffle_train', False),  # Feature extraction usually doesn't need shuffling
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
        
        test_loader = None
        if self.test_dataset is not None:
            test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=True
            )
            
        return train_loader, test_loader
    
    def get_dataset_info(self):
        """Return dataset information"""
        # Determine number of classes based on dataset type
        num_classes_map = {
            'cifar10': 10,
            'cifar100': 100,
            'stl10': 10
        }
        
        info = {
            'dataset_type': self.dataset_type,
            'train_size': len(self.train_dataset),
            'num_classes': num_classes_map.get(self.dataset_type, 10),
            'image_size': self.image_size,
            'norm_mean': self.norm_mean,
            'norm_std': self.norm_std,
        }
        
        if self.test_dataset is not None:
            info['test_size'] = len(self.test_dataset)
        
        return info


class DatasetManager:
    """
    Dataset manager for handling dataset loading and splitting without saving separate files
    """
    
    def __init__(self, feature_file_path):
        """
        Initialize dataset manager
        
        Args:
            feature_file_path: Path to the feature file
        """
        self.feature_file_path = feature_file_path
        self.full_data = None
        self.train_features = None
        self.train_labels = None
        self.test_features = None
        self.test_labels = None
        self._load_data()
    
    def _load_data(self):
        """Load the full dataset"""
        print(f"Loading dataset from: {self.feature_file_path}")
        self.full_data = torch.load(self.feature_file_path, weights_only=True)
        
        # Separate training set and test set
        self.train_features = self.full_data['train']['features']
        self.train_labels = self.full_data['train']['labels']
        self.test_features = self.full_data['test']['features']
        self.test_labels = self.full_data['test']['labels']
        
        print(f"Dataset loaded successfully:")
        print(f"  Training samples: {len(self.train_features)}")
        print(f"  Test samples: {len(self.test_features)}")
    
    def create_split(self, label_per_class=25, seed=42):
        """
        Create dataset split and return indices
        
        Args:
            label_per_class: Number of labeled samples per class
            seed: Random seed for reproducibility
        
        Returns:
            dict: Dictionary containing split information
        """
        # Set random seed
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Organize indices, ignore samples with label -1
        class_indices = defaultdict(list)
        unlabeled_indices = []
        
        for idx, label in enumerate(self.train_labels.numpy()):
            label_int = int(label)
            if label_int == -1:  # Handle unlabeled data (STL-10 specific)
                unlabeled_indices.append(idx)
            else:
                class_indices[label_int].append(idx)
        
        labeled_indices = []
        split_unlabeled_indices = []
        
        # Safe splitting
        for cls, indices in class_indices.items():
            if len(indices) < label_per_class:
                raise ValueError(f"Class {cls} has only {len(indices)} samples, cannot meet requirement of {label_per_class}")
            
            cls = int(cls)
            perm = torch.randperm(len(indices), generator=torch.Generator().manual_seed(seed + cls))
            selected = perm[:label_per_class].tolist()
            
            labeled = [indices[i] for i in selected]
            unlabeled_from_class = [indices[i] for i in perm[label_per_class:].tolist()]
            
            labeled_indices.extend(labeled)
            split_unlabeled_indices.extend(unlabeled_from_class)
        
        # Merge all unlabeled indices
        all_unlabeled_indices = split_unlabeled_indices + unlabeled_indices
        
        split_info = {
            'labeled_indices': torch.tensor(labeled_indices),
            'unlabeled_indices': torch.tensor(all_unlabeled_indices),
            'original_unlabeled_indices': torch.tensor(unlabeled_indices),
            'metadata': {
                'split_seed': seed,
                'label_per_class': label_per_class,
                'total_labeled': len(labeled_indices),
                'total_unlabeled': len(all_unlabeled_indices),
                'original_unlabeled_count': len(unlabeled_indices),
                'split_unlabeled_count': len(split_unlabeled_indices),
                'num_classes': len(class_indices)
            }
        }
        
        print(f"\nDataset split created:")
        print(f"  Labeled samples: {len(labeled_indices)}")
        print(f"  Unlabeled samples: {len(all_unlabeled_indices)}")
        print(f"  Classes: {len(class_indices)}")
        
        return split_info
    
    def get_split_data(self, split_info):
        """
        Get actual data based on split information
        
        Args:
            split_info: Split information from create_split()
        
        Returns:
            dict: Dictionary containing split datasets
        """
        return {
            'train': {
                'labeled': {
                    'features': self.train_features[split_info['labeled_indices']],
                    'labels': self.train_labels[split_info['labeled_indices']],
                    'indices': split_info['labeled_indices']
                },
                'unlabeled': {
                    'features': self.train_features[split_info['unlabeled_indices']],
                    'labels': self.train_labels[split_info['unlabeled_indices']],
                    'indices': split_info['unlabeled_indices']
                }
            },
            'test': {
                'features': self.test_features,
                'labels': self.test_labels
            },
            'split_info': split_info['metadata']
        }
    
    def get_data_for_training(self, label_per_class=25, seed=42):
        """
        Convenience method to get split data for training
        
        Args:
            label_per_class: Number of labeled samples per class
            seed: Random seed
        
        Returns:
            tuple: (X_labeled, y_labeled, X_unlabeled, X_test, y_test)
        """
        split_info = self.create_split(label_per_class=label_per_class, seed=seed)
        split_data = self.get_split_data(split_info)
        
        X_labeled = split_data['train']['labeled']['features'].numpy()
        y_labeled = split_data['train']['labeled']['labels'].numpy()
        X_unlabeled = split_data['train']['unlabeled']['features'].numpy()
        X_test = split_data['test']['features'].numpy()
        y_test = split_data['test']['labels'].numpy()
        
        return X_labeled, y_labeled, X_unlabeled, X_test, y_test


# Utility functions for backward compatibility
def load_dataset_with_split(feature_file_path, split_config):
    """
    Load dataset and apply splitting based on configuration
    
    Args:
        feature_file_path: Path to the feature file
        split_config: Dictionary containing split configuration
    
    Returns:
        dict: Dictionary containing split datasets
    """
    manager = DatasetManager(feature_file_path)
    split_info = manager.create_split(
        label_per_class=split_config.get('label_per_class', 25),
        seed=split_config.get('seed', 42)
    )
    return manager.get_split_data(split_info)


def get_training_data(feature_file_path, label_per_class=25, seed=42):
    """
    Convenience function to get training data
    
    Args:
        feature_file_path: Path to the feature file
        label_per_class: Number of labeled samples per class
        seed: Random seed
    
    Returns:
        tuple: (X_labeled, y_labeled, X_unlabeled, X_test, y_test)
    """
    manager = DatasetManager(feature_file_path)
    return manager.get_data_for_training(label_per_class=label_per_class, seed=seed)