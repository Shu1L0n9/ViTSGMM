"""
Configuration settings for ViTSGMM project.
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class DataConfig:
    """Data-related configuration"""
    dataset_type: str = 'cifar10'  # 'stl10', 'cifar10', 'cifar100'
    data_root: str = './data'
    batch_size: int = 64
    image_size: int = 518
    compute_norm: bool = True
    
    @property
    def n_classes(self) -> int:
        """Return number of classes based on dataset type"""
        return {
            'cifar10': 10,
            'cifar100': 100,
            'stl10': 10
        }.get(self.dataset_type, 10)
    
    @property
    def feature_file(self) -> str:
        """Feature file path"""
        return f'data/vit/{self.dataset_type}/{self.dataset_type}_vit_features.pt'

@dataclass
class ModelConfig:
    """Model-related configuration"""
    # PCA parameters
    n_components_pca: int = 60
    
    # GMM parameters
    n_components_gmm: int = 13
    tol: float = 1e1
    max_iter: int = 1024
    
    # Training parameters
    labeled_per_class: int = 4
    device: str = 'cuda'
    seed: int = 42
    
    # Evaluation parameters
    top_k: int = 3

@dataclass
class ViTConfig:
    """Vision Transformer configuration"""
    model_name: str = "vit_large_patch14_reg4_dinov2.lvd142m"
    device: str = "cuda"
    
    @property
    def save_path_template(self) -> str:
        """Feature save path template"""
        return "data/vit/{dataset}/{dataset}_vit_features.pt"

class Config:
    """Main configuration class"""
    
    def __init__(self):
        self.data = DataConfig()
        self.model = ModelConfig()
        self.vit = ViTConfig()
        
        # Project paths
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.results_dir = os.path.join(self.project_root, 'results')
        self.models_dir = os.path.join(self.results_dir, 'models')
        self.logs_dir = os.path.join(self.results_dir, 'logs')
        self.figures_dir = os.path.join(self.results_dir, 'figures')
        
        # Ensure directories exist
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
    
    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from dictionary"""
        for key, value in config_dict.items():
            if hasattr(self.data, key):
                setattr(self.data, key, value)
            elif hasattr(self.model, key):
                setattr(self.model, key, value)
            elif hasattr(self.vit, key):
                setattr(self.vit, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'vit': self.vit.__dict__
        }

# Global configuration instance
config = Config()
