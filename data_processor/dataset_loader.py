# -*- coding: utf-8 -*-
'''
Created on Sat Dec 28 21:18:18 2024

@author: 10449
'''

# dataset_loader.py
import numpy as np
import os
import glob
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random

class GeospatialDatasetTorch(Dataset):
    def __init__(self, data, is_training=False):
        '''
        Return torch dataset
        
        Parameters:
        -----------
        data: numpy array of shape (n_samples, height, width, channels)
        '''
        self.data = torch.FloatTensor(data)
        
        # Separate features (bands 0-5) from labels (band 6)
        self.features = self.data[..., :5]
        self.labels = self.data[..., 5]
        self.is_training = is_training
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Return features in (channels, height, width) format for PyTorch
        features = self.features[idx].permute(2, 0, 1)  # (C, H, W)
        label = self.labels[idx]  # (H, W)
        
        label_mapping = {
            0: 0,     # Ignore
            1: 0,     # Green
            2: 1,    # Trees
            3: 2,    # Shrubs
            4: 3,    # Grass
        }
        label = label.to(torch.int64)
        label_remapped = torch.zeros_like(label)
        for k, v in label_mapping.items():
            label_remapped[label == k]= v
        
        # Data augmentation
        if self.is_training:
            if random.random() < 0.5:
                features = TF.hflip(features)
                label_remapped = TF.hflip(label_remapped)
            
            if random.random() < 0.5:
                features = TF.vflip(features)
                label_remapped = TF.vflip(label_remapped)
                
            angle = random.choice([0, 90, 180, 270])
            if angle != 0:
                features = TF.rotate(features, angle)
                label_remapped = TF.rotate(
                    label_remapped.unsqueeze(0), 
                    angle, 
                    interpolation=TF.InterpolationMode.NEAREST
                ).squeeze(0).long() 
        
        return features, label_remapped
        
    def calculate_green_percentages(self):
        """
        Calculate the percentage of positive (green) pixels in the entire dataset.
        Returns:
            float: Percentage of positive pixels.
        """
        # Total positive pixels (urban green) across all samples
        total_positive = self.labels.sum().item()
        
        # Total pixels in the dataset (n_samples * height * width)
        total_pixels = self.labels.numel()
        
        # Calculate percentage
        green_percentage = total_positive / total_pixels
        return green_percentage

class GeospatialDataLoader:
    def __init__(self, data_dir, pattern='*.npy', required_shape=(256, 256, 6), 
                 memmap_file='', dataset_type=''):
        '''
        Initialize the GeospatialDataLoader
        
        Parameters:
        -----------
        data_dir : str
            Directory containing the .npy files
        pattern : str
            Pattern to match files (default: '*.npy')
        required_shape : tuple
            Expected shape of each sample (excluding batch dimension)
        memmap_file : str
            Path to store the combined memory-mapped dataset
        dataset_type : str
            train or eval
        '''
        self.data_dir = data_dir
        self.pattern = pattern
        self.required_shape = required_shape
        self.memmap_file = memmap_file
        self.dataset_type = dataset_type
        self.combined_dataset = None
    
    def __len__(self):
        
        '''Return the number of samples in the dataset'''
        
        if self.combined_dataset is None:
            self.load_datasets()
        return self.combined_dataset.shape[0] 
    
    def __getitem__(self, idx):
        
        if self.combined_dataset is None:
            self.load_datasets()
            
        # Get a single sample from the dataset
        sample = self.combined_dataset[idx].reshape(self.required_shape)
        sample_tensor = torch.from_numpy(sample).float()  
        
        feature = sample_tensor[..., :5].permute(2, 0, 1)
        label = sample_tensor[..., 5]
        
        return feature, label
    
    def load_datasets(self, verbose=True):
        
        '''Load and concatenate all valid datasets'''
        
        file_paths = glob.glob(os.path.join(self.data_dir, self.pattern))
        
        if len(file_paths) == 0:
            raise ValueError(f'No .npy files found in {self.data_dir} matching pattern {self.pattern}')
        
        if verbose:
            print(f'Found {len(file_paths)} .npy files:')
        
        valid_datasets = []
        total_samples = 0
        shape = None
        
        try:
            for file_path in file_paths:
                filename = os.path.basename(file_path)
                if verbose:
                    print(f'Loading {filename}...', end=' ')
                
                data = np.load(file_path, mmap_mode='r')
                
                # Validate shape
                if self.required_shape and data.shape[1:] != self.required_shape:
                    if verbose:
                        print(f'Skipping - Invalid shape {data.shape[1:]}')
                    continue
                
                # Handle NaN values
                if np.isnan(data).any():
                    if verbose:
                        print(f'Warning: {filename} contains NaN values - replacing with 0')
                    data = np.nan_to_num(data, 0)
                
                if verbose:
                    print(f'Shape: {data.shape}')
                
                valid_datasets.append(data)
                total_samples += len(data)
                
                if shape is None:
                    shape = data.shape[1:]
                    
                del data
            
            if not valid_datasets:
                raise ValueError('No valid datasets found!')
                
            combined_shape = (total_samples,) + shape
            dtype = valid_datasets[0].dtype
                    
            if self.dataset_type == 'train':
                self.memmap_file = 'memmap_train.memmap'
            else:
                self.memmap_file = 'memmap_eval.memmap'
            
            self.combined_dataset = np.memmap(self.memmap_file, dtype=dtype, 
                                              mode='r+', shape=combined_shape)
            offset = 0
            for dataset in valid_datasets:
                # Calculate the size of each dataset and copy it to the memory-mapped file
                num_samples = len(dataset)
                self.combined_dataset[offset:offset + num_samples] = dataset
                offset += num_samples
            
            # self.combined_dataset = np.concatenate(valid_datasets, axis=0)
            
            if verbose:
                print(f'\nCombined dataset shape: {self.combined_dataset.shape}')
                print(f'Total samples: {total_samples}')
                print(f'Memory usage: {self.combined_dataset.nbytes / (1024**3):.2f} GB')
                print('')
            
            return self.combined_dataset
        
        except KeyboardInterrupt:
            if hasattr(self, 'combined_dataset'):
                self.combined_dataset.flush()  # Ensure changes are saved
                del self.combined_dataset  # Delete reference to close the file
            raise  # Re-raise the exception to allow external handling
    
    def get_dataset(self):
        
        '''Return the combined dataset'''
        
        if self.combined_dataset is None:
            self.load_datasets()
        return self.combined_dataset
    
    def get_torch_dataset(self):
        
        '''Return a PyTorch Dataset object'''
        
        if self.combined_dataset is None:
            self.load_datasets()
        is_training = (self.dataset_type == 'train')
        return GeospatialDatasetTorch(self.combined_dataset, is_training)
    
    def get_data_stats(self):
        
        '''Return basic statistics about the dataset'''
        
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