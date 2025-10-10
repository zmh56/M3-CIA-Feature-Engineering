"""
Data loader for .mat format files with obfuscated features.
"""

import torch
import numpy as np
import scipy.io
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import os


class MultiModalDataset(Dataset):
    """
    Dataset class for loading multi-modal .mat files.
    """
    
    def __init__(self, mat_file_path: str, transform=None):
        """
        Initialize dataset from .mat file.
        
        Args:
            mat_file_path (str): Path to the .mat file
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.mat_file_path = mat_file_path
        self.transform = transform
        
        # Load data from .mat or .npz file
        self.data = self._load_data(mat_file_path)
        
        # Extract modalities and labels
        self.mod_a = self.data['mod_a']  # (N, T1, D1)
        self.mod_b = self.data['mod_b']  # (N, T2, D2)
        self.mod_c = self.data['mod_c']  # (N, T3, D3)
        self.mod_d = self.data['mod_d']  # (N, D4)
        self.mod_e = self.data['mod_e']  # (N, D5)
        self.mod_f = self.data['mod_f']  # (N, D6)
        self.labels = self.data['labels'].flatten()  # (N,)
        
        # Load intermediate targets if available (ensure consistent structure)
        if 'intermediate_targets' in self.data and self.data['intermediate_targets'] is not None:
            interm = self.data['intermediate_targets']
            # Support both list-of-arrays and stacked array (layers, N)
            if isinstance(interm, list):
                self.intermediate_targets = interm
                self.intermediate_dim = len(self.intermediate_targets)
            else:
                # assume np.ndarray with shape (layers, N)
                self.intermediate_targets = [interm[i] for i in range(interm.shape[0])]
                self.intermediate_dim = interm.shape[0]
            print(f"Loaded intermediate targets: {self.intermediate_dim} layers")
        else:
            self.intermediate_targets = None
            self.intermediate_dim = 0
            print("No intermediate targets found")
        
        self.num_samples = len(self.labels)
        
        print(f"Loaded dataset with {self.num_samples} samples")

    def _load_data(self, file_path: str) -> Dict[str, np.ndarray]:
        """
        Load data based on file extension. Supports .mat and obfuscated .npz produced by the converter.
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.mat':
            return scipy.io.loadmat(file_path)
        elif ext == '.npz':
            return self._load_npz_obfuscated(file_path)
        else:
            raise ValueError(f"Unsupported data file extension: {ext}")

    def _unpack_npz_array(self, key_prefix: str, npz: Dict[str, np.ndarray], obf_key: int) -> np.ndarray:
        data_key = f"{key_prefix}_data"
        shape_key = f"{key_prefix}_shape"
        dtype_key = f"{key_prefix}_dtype"
        if data_key not in npz or shape_key not in npz or dtype_key not in npz:
            raise KeyError(f"Missing keys for {key_prefix} in NPZ")
        obf_bytes = npz[data_key]
        shape = tuple(npz[shape_key].astype(int).tolist())
        dtype_str = str(npz[dtype_key]) if npz[dtype_key].shape == () else str(npz[dtype_key][()])
        buf = bytes(obf_bytes.tolist())
        if obf_key:
            buf = bytes([b ^ (obf_key & 0xFF) for b in buf])
        arr = np.frombuffer(buf, dtype=np.dtype(dtype_str))
        arr = arr.reshape(shape)
        return arr

    def _load_npz_obfuscated(self, file_path: str) -> Dict[str, np.ndarray]:
        with np.load(file_path, allow_pickle=True) as nd:
            format_tag = str(nd['format']) if nd['format'].shape == () else str(nd['format'][()])
            obf_key = int(nd['obf_key']) if 'obf_key' in nd else 0
            data: Dict[str, np.ndarray] = {}
            # Unpack all required arrays
            for name in ['mod_a', 'mod_b', 'mod_c', 'mod_d', 'mod_e', 'mod_f', 'labels', 'intermediate_targets']:
                data[name] = self._unpack_npz_array(name, nd, obf_key)
            # Additional scalars
            data['num_samples'] = np.array([int(nd['num_samples'])], dtype=np.int32) if 'num_samples' in nd else np.array([data['labels'].shape[0]], dtype=np.int32)
            if 'intermediate_dim' in nd:
                data['intermediate_dim'] = np.array([int(nd['intermediate_dim'])], dtype=np.int32)
            return data
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            Dict[str, torch.Tensor]: Sample dictionary
        """
        # Ensure idx is a scalar integer
        if isinstance(idx, (list, tuple, np.ndarray)):
            idx = idx[0]
        
        # Check and fix NaN/inf values in data
        def clean_data(data):
            if np.isnan(data).any() or np.isinf(data).any():
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            return data
        
        sample = {
            'mod_a': torch.tensor(clean_data(self.mod_a[idx]), dtype=torch.float32),
            'mod_b': torch.tensor(clean_data(self.mod_b[idx]), dtype=torch.float32),
            'mod_c': torch.tensor(clean_data(self.mod_c[idx]), dtype=torch.float32),
            'mod_d': torch.tensor(clean_data(self.mod_d[idx]), dtype=torch.float32),
            'mod_e': torch.tensor(clean_data(self.mod_e[idx]), dtype=torch.float32),
            'mod_f': torch.tensor(clean_data(self.mod_f[idx]), dtype=torch.float32),
            'target': torch.tensor(self.labels[idx], dtype=torch.long)
        }
        
        # Add intermediate targets if available
        if self.intermediate_targets is not None:
            sample['intermediate_targets'] = [
                torch.tensor(target[idx], dtype=torch.long) 
                for target in self.intermediate_targets
            ]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


class MultiMatDataset(Dataset):
    """
    Dataset class for loading multiple separate .mat files.
    """
    
    def __init__(self, data_dir: str, transform=None):
        """
        Initialize dataset from multiple .mat files.
        
        Args:
            data_dir (str): Directory containing .mat files
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Load individual modality files
        self.mod_a = scipy.io.loadmat(f"{data_dir}/mod_a.mat")['mod_a']
        self.mod_b = scipy.io.loadmat(f"{data_dir}/mod_b.mat")['mod_b']
        self.mod_c = scipy.io.loadmat(f"{data_dir}/mod_c.mat")['mod_c']
        self.mod_d = scipy.io.loadmat(f"{data_dir}/mod_d.mat")['mod_d']
        self.mod_e = scipy.io.loadmat(f"{data_dir}/mod_e.mat")['mod_e']
        self.mod_f = scipy.io.loadmat(f"{data_dir}/mod_f.mat")['mod_f']
        self.labels = scipy.io.loadmat(f"{data_dir}/labels.mat")['labels'].flatten()
        
        self.num_samples = len(self.labels)
        
        print(f"Loaded multi-file dataset with {self.num_samples} samples")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            Dict[str, torch.Tensor]: Sample dictionary
        """
        # Ensure idx is a scalar integer
        if isinstance(idx, (list, tuple, np.ndarray)):
            idx = idx[0]
        
        # Check and fix NaN/inf values in data
        def clean_data(data):
            if np.isnan(data).any() or np.isinf(data).any():
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            return data
        
        sample = {
            'mod_a': torch.tensor(clean_data(self.mod_a[idx]), dtype=torch.float32),
            'mod_b': torch.tensor(clean_data(self.mod_b[idx]), dtype=torch.float32),
            'mod_c': torch.tensor(clean_data(self.mod_c[idx]), dtype=torch.float32),
            'mod_d': torch.tensor(clean_data(self.mod_d[idx]), dtype=torch.float32),
            'mod_e': torch.tensor(clean_data(self.mod_e[idx]), dtype=torch.float32),
            'mod_f': torch.tensor(clean_data(self.mod_f[idx]), dtype=torch.float32),
            'target': torch.tensor(self.labels[idx], dtype=torch.long)
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


def create_mat_data_loaders(
    data_path: str,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    shuffle: bool = True,
    num_workers: int = 0,
    use_multi_file: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders from .mat files.
    
    Args:
        data_path (str): Path to .mat file or directory containing .mat files
        batch_size (int): Batch size
        train_ratio (float): Ratio of data to use for training
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of worker processes for data loading
        use_multi_file (bool): Whether to use multiple .mat files or single file
        
    Returns:
        Tuple[DataLoader, DataLoader]: Train and validation data loaders
    """
    # Create dataset
    if use_multi_file:
        dataset = MultiMatDataset(data_path)
    else:
        dataset = MultiModalDataset(data_path)
    
    # Split dataset
    num_train = int(len(dataset) * train_ratio)
    num_val = len(dataset) - num_train
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [num_train, num_val]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"Created data loaders:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Batch size: {batch_size}")
    
    return train_loader, val_loader


def normalize_features(sample: Dict[str, torch.Tensor], 
                      stats: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, torch.Tensor]:
    """
    Normalize features in a sample.
    
    Args:
        sample (Dict[str, torch.Tensor]): Input sample
        stats (Dict[str, Dict[str, float]], optional): Normalization statistics
        
    Returns:
        Dict[str, torch.Tensor]: Normalized sample
    """
    normalized_sample = {}
    
    for key, value in sample.items():
        if key == 'target':
            normalized_sample[key] = value
            continue
            
        if stats and key in stats:
            # Use provided statistics
            mean = stats[key]['mean']
            std = stats[key]['std']
            normalized_sample[key] = (value - mean) / (std + 1e-8)
        else:
            # Compute statistics from current sample
            if len(value.shape) > 1:
                # For multi-dimensional data, normalize along feature dimension
                mean = value.mean(dim=-1, keepdim=True)
                std = value.std(dim=-1, keepdim=True)
                normalized_sample[key] = (value - mean) / (std + 1e-8)
            else:
                # For 1D data
                mean = value.mean()
                std = value.std()
                normalized_sample[key] = (value - mean) / (std + 1e-8)
    
    return normalized_sample


def compute_dataset_stats(dataset: Dataset) -> Dict[str, Dict[str, float]]:
    """
    Compute normalization statistics for a dataset.
    
    Args:
        dataset (Dataset): Input dataset
        
    Returns:
        Dict[str, Dict[str, float]]: Statistics for each modality
    """
    stats = {}
    
    # Get first sample to determine modalities
    sample = dataset[0]
    
    for key, value in sample.items():
        if key == 'target':
            continue
            
        # Collect all values for this modality
        all_values = []
        for i in range(len(dataset)):
            sample_i = dataset[i]
            all_values.append(sample_i[key])
        
        # Stack and compute statistics
        stacked_values = torch.stack(all_values)
        
        if len(stacked_values.shape) > 2:
            # For multi-dimensional data, compute statistics across all dimensions
            mean = stacked_values.mean().item()
            std = stacked_values.std().item()
        else:
            # For 2D data, compute statistics across feature dimension
            mean = stacked_values.mean(dim=-1).mean().item()
            std = stacked_values.std(dim=-1).mean().item()
        
        stats[key] = {'mean': mean, 'std': std}
    
    return stats
