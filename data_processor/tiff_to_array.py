# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 13:10:19 2024

@author: 10449
"""

import os
import numpy as np
import rasterio

os.chdir('D:/Msc/Thesis/Data/GEEDownload')

def readTiff(file_path):
    with rasterio.open(file_path, 'r') as src:
        tif_as_array = src.read().astype(np.float32)
    return tif_as_array

def normalize(x):
    return 255 * (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))

def gen_data(path, folder, file, nChannels=5):
    file_path = os.path.join(path, folder, file)
    arr = readTiff(file_path)
    _, rows, cols = arr.shape

    arr = arr.transpose((1, 2, 0))  # transpose to height, width, channel
    arr = np.nan_to_num(arr)
    
    if np.mean(arr) == 0:
        print('discard empty array ' + file_path)
        os.remove(file_path)
    
    # store bands and label
    data_output = np.zeros((arr.shape[0], arr.shape[1], nChannels))
    
    # normalise spectral bands, 0-B4,B3,B2,B8-3
    for i in range(nChannels - 1):
        data_output[:, :, i] = np.nan_to_num(normalize(arr[:, :, i]))
    # add label-4
    data_output[:, :, 4] = arr[:, :, 4]
    
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
        
    # slice img for training (no overlap)
    if folder == 'train':
        dataset_train_list = []
        for i in range(r):
            for j in range(c):
                sample = data_output[i * sample_size : (i + 1) * sample_size, 
                                     j * sample_size : (j + 1) * sample_size, :]
                if np.count_nonzero(sample.flatten()) == 0:
                    print('discard empty sample')
                else:
                    dataset_train_list.append(sample)
        del data_output
        
        dataset_train = np.stack(dataset_train_list, axis=0)
        np.save(os.path.join(path, folder, f"{file[:-4]}.npy"), dataset_train)
    
    # slice img for eval (overlap)
    elif folder == 'eval':
        overlap = 0.2
        dataset_eval_list = []
        step_size = int(sample_size * (1 - overlap))  # 204
        
        for i in range(r):
            for j in range(c):
                sample = data_output[i * step_size : i * step_size + sample_size, 
                                     j * step_size : j * step_size + sample_size, :]
                if np.count_nonzero(sample.flatten()) == 0:
                    print('discard empty sample')
                else:
                    dataset_eval_list.append(sample)
        del data_output
        
        dataset_eval = np.stack(dataset_eval_list, axis=0)
        np.save(os.path.join(path, folder, f"{file[:-4]}.npy"), dataset_eval)

    
if __name__ == '__main__':
    
    nChannels = 5
    path = 'Polar/'
    folders = ['train', 'eval']
    
    for folder in folders:
        for file in os.listdir(path + folder):
            if file.endswith('tif'):
                print(file)
                gen_data(path, folder, file, nChannels)