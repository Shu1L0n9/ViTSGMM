import numpy as np
from sklearn.decomposition import PCA
import torch

def pca_reduce(train_data, train_labels, test_data, test_labels, n_components_pca=30):
    # Ensure data is on CPU and convert to NumPy arrays
    print(f"Applying PCA to reduce dimensions to {n_components_pca}")
    
    # Move data to CPU and convert to NumPy arrays
    X_train = train_data.cpu().numpy() if torch.is_tensor(train_data) else train_data
    X_test = test_data.cpu().numpy() if torch.is_tensor(test_data) else test_data
    y_train = train_labels.cpu().numpy() if torch.is_tensor(train_labels) else train_labels
    y_test = test_labels.cpu().numpy() if torch.is_tensor(test_labels) else test_labels

    # Apply PCA
    pca = PCA(n_components=n_components_pca)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    print(f"Data shape after PCA - Training: {X_train_pca.shape}, Testing: {X_test_pca.shape}")
    print("Data preprocessing completed.")
    
    return X_train_pca, y_train, X_test_pca, y_test

def split_dataset_by_label_ratio(X, y, labeled_ratio, n_splits, random_state=42):
    """
    Split dataset by label ratio, keeping labeled data the same across all cross-validation sets
    
    Args:
        X: Feature data
        y: Label data 
        labeled_ratio: Ratio of labeled data
        n_splits: Number of cross-validation sets
        random_state: Random seed
    
    Returns:
        cv_splits: List of n_splits dictionaries containing:
            - train_labeled: (X_labeled, y_labeled)
            - unlabeled: X_unlabeled of (n_splits-1)/n_splits
            - val: (X_val, y_val) of 1/n_splits
    """
    np.random.seed(random_state)
    n_samples = len(X)
    n_labeled = int(n_samples * labeled_ratio)
    
    # Separate labeled and unlabeled data
    X_labeled = X[:n_labeled]
    y_labeled = y[:n_labeled]
    X_unlabeled = X[n_labeled:]
    y_unlabeled = y[n_labeled:]
    
    # Randomly shuffle unlabeled data
    indices = np.arange(len(X_unlabeled))
    np.random.shuffle(indices)
    
    # Calculate validation set size
    val_size = len(indices) // n_splits
    
    # Build cross-validation sets
    cv_splits = []
    for i in range(n_splits):
        # Current fold validation set indices
        val_start = i * val_size
        val_end = val_start + val_size if i < n_splits-1 else len(indices)
        val_indices = indices[val_start:val_end]
        
        # Other fold training set indices
        train_indices = np.concatenate([indices[:val_start], indices[val_end:]])
        
        split = {
            'train_labeled': (X_labeled, y_labeled),
            'unlabeled': X_unlabeled[train_indices],
            'val': (X_unlabeled[val_indices], y_unlabeled[val_indices])
        }
        cv_splits.append(split)
    
    return cv_splits