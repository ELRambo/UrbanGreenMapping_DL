# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 21:18:18 2024

@author: 10449
"""

# dataset_loader.py
import numpy as np
import os
import glob
import torch
from torch.utils.data import Dataset

class GeospatialDatasetTorch(Dataset):
    def __init__(self, data):
        """
        Return torch dataset
        
        Parameters:
        -----------
        data: numpy array of shape (n_samples, height, width, channels)
        """
        self.data = torch.FloatTensor(data)
        # Separate features (bands 0-3) from labels (band 4)
        self.features = self.data[..., :4]
        self.labels = self.data[..., 4]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Return features in (channels, height, width) format for PyTorch
        return self.features[idx].permute(2, 0, 1), self.labels[idx]

class GeospatialDataLoader:
    def __init__(self, data_dir, pattern="*.npy", required_shape=(256, 256, 5)):
        """
        Initialize the GeospatialDataLoader
        
        Parameters:
        -----------
        data_dir : str
            Directory containing the .npy files
        pattern : str
            Pattern to match files (default: "*.npy")
        required_shape : tuple
            Expected shape of each sample (excluding batch dimension)
        """
        self.data_dir = data_dir
        self.pattern = pattern
        self.required_shape = required_shape
        self.combined_dataset = None
        
    def load_datasets(self, verbose=True):
        
        """Load and concatenate all valid datasets"""
        
        file_paths = glob.glob(os.path.join(self.data_dir, self.pattern))
        
        if len(file_paths) == 0:
            raise ValueError(f"No .npy files found in {self.data_dir} matching pattern {self.pattern}")
        
        if verbose:
            print(f"Found {len(file_paths)} .npy files:")
        
        valid_datasets = []
        total_samples = 0
        
        for file_path in file_paths:
            filename = os.path.basename(file_path)
            if verbose:
                print(f"Loading {filename}...", end=" ")
            
            data = np.load(file_path, mmap_mode='r+')
            
            # Validate shape
            if self.required_shape and data.shape[1:] != self.required_shape:
                if verbose:
                    print(f"Skipping - Invalid shape {data.shape[1:]}")
                continue
            
            # Handle NaN values
            if np.isnan(data).any():
                if verbose:
                    print(f"Warning: {filename} contains NaN values - replacing with 0")
                data = np.nan_to_num(data, 0)
            
            if verbose:
                print(f"Shape: {data.shape}")
            
            valid_datasets.append(data)
            total_samples += len(data)
        
        if not valid_datasets:
            raise ValueError("No valid datasets found!")
        
        self.combined_dataset = np.concatenate(valid_datasets, axis=0)
        
        if verbose:
            print(f"\nCombined dataset shape: {self.combined_dataset.shape}")
            print(f"Total samples: {total_samples}")
            print(f"Memory usage: {self.combined_dataset.nbytes / (1024**3):.2f} GB")
            print("")
        
        return self.combined_dataset
    
    def get_dataset(self):
        
        """Return the combined dataset"""
        
        if self.combined_dataset is None:
            self.load_datasets()
        return self.combined_dataset
    
    def get_torch_dataset(self):
        
        """Return a PyTorch Dataset object"""
        
        if self.combined_dataset is None:
            self.load_datasets()
        return GeospatialDatasetTorch(self.combined_dataset)
    
    def get_data_stats(self):
        
        """Return basic statistics about the dataset"""
        
        if self.combined_dataset is None:
            self.load_datasets()
            
        stats = {
            'shape': self.combined_dataset.shape,
            'min': np.min(self.combined_dataset),
            'max': np.max(self.combined_dataset),
            'mean': np.mean(self.combined_dataset),
            'std': np.std(self.combined_dataset)
        }
        
        return stats