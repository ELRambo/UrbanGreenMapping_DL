# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 13:10:19 2024

@author: 10449
"""

import os
import numpy as np
import pandas as pd
import rasterio

os.chdir('D:/Msc/Thesis/Data/Relabelled')

def readTiff(file_path):
    with rasterio.open(file_path, 'r') as src:
        tif_as_array = src.read().astype(np.float32)
    return tif_as_array

def normalize(x):
    return 255 * (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))

def genData(zone, file_path, city, isEval, nChannels=6):
    
    arr = readTiff(file_path)
    _, rows, cols = arr.shape

    arr = arr.transpose((1, 2, 0))  # transpose to height, width, channel
    arr = np.nan_to_num(arr)
    
    if np.mean(arr) == 0:
        print('discard empty array ' + file_path)
        os.remove(file_path)
    
    # store bands and label
    data_output = np.zeros((arr.shape[0], arr.shape[1], nChannels))
    
    # normalise spectral bands, 0-B4,B3,B2,B8,B12-4
    for i in range(nChannels - 1):
        data_output[:, :, i] = np.nan_to_num(normalize(arr[:, :, i]))
    # add label-5
    data_output[:, :, 5] = arr[:, :, 5]
    
    sample_size = 256
    
    # pad img
    new_rows = ((rows + sample_size - 1) // sample_size) * sample_size
    new_cols = ((cols + sample_size - 1) // sample_size) * sample_size
    if new_rows != rows or new_cols != cols:
        pad_rows = new_rows - rows
        pad_cols = new_cols - cols
        data_output = np.pad(data_output, ((0, pad_rows), (0, pad_cols), (0, 0)), mode='constant')
    
    r = new_rows // sample_size
    c = new_cols // sample_size
    
    print(data_output.shape, r, c)
    
    overlap = 0.2
    step_size = int(sample_size * (1 - overlap))
        
    # slice img for training
    if isEval != 1:       
        
        # No overlap for train data except a, e
        if zone not in ['a', 'e']:
            step_size = sample_size
            
        dataset_train_list = []
        
        num_tiles_rows = (data_output.shape[0] - sample_size) // step_size + 1
        num_tiles_cols = (data_output.shape[1] - sample_size) // step_size + 1
                
        for i in range(num_tiles_rows):
            for j in range(num_tiles_cols):
                sample = data_output[i * step_size : i * step_size + sample_size, 
                                     j * step_size : j * step_size + sample_size, :]
                if np.count_nonzero(sample.flatten()) == 0:
                    print('discard empty sample')
                else:
                    dataset_train_list.append(sample)
        del data_output
        
        dataset_train = np.stack(dataset_train_list, axis=0)
        np.save(os.path.join(zone, 'train', f'{city}.npy'), dataset_train)
        
        return len(dataset_train)
    
    # slice img for eval
    else:
        
        # No overlap for eval data in c, d
        if zone in ['c', 'd']:
            step_size = sample_size
        
        dataset_eval_list = []
        
        num_tiles_rows = (data_output.shape[0] - sample_size) // step_size + 1
        num_tiles_cols = (data_output.shape[1] - sample_size) // step_size + 1
                
        for i in range(num_tiles_rows):
            for j in range(num_tiles_cols):
                sample = data_output[i * step_size : i * step_size + sample_size, 
                                     j * step_size : j * step_size + sample_size, :]
                if np.count_nonzero(sample.flatten()) == 0:
                    print('discard empty sample')
                else:
                    dataset_eval_list.append(sample)
        del data_output
                
        dataset_eval = np.stack(dataset_eval_list, axis=0)
        np.save(os.path.join(zone, 'eval', f'{city}.npy'), dataset_eval)
        
        return len(dataset_eval)
    
    
if __name__ == '__main__':
    
    nChannels = 6
    zone = 'e'
    
    df = pd.read_csv('D:/Msc/Thesis/Data/GEEDownload/newThresh.csv')
    df = df[(df['zone'] == zone)]
    
    train_size = 0; eval_size = 0
    
    for index, row in df.iterrows():
        
        city = row['city']
        isEval = row['isEval']
        print(city)
        
        file = city + '.tif'
        file_path = os.path.join('F:/Msc/Thesis/Data/Relabelled', zone, file)
        
        if isEval != 1:
            train_size += genData(zone, file_path, city, isEval, nChannels)
        else:
            eval_size += genData(zone, file_path, city, isEval, nChannels)
        
    print(f'Train size: {train_size}, eval size: {eval_size}')