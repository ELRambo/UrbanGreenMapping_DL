# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 13:10:19 2024

@author: 10449
"""

import os
import pandas as pd
import numpy as np
import rasterio

os.chdir('D:/Msc/Thesis/Data/GEEDownload')
cities = pd.read_csv('thresholds.csv')

def read_tiff(file_path):
    with rasterio.open(file_path, 'r') as src:
        tif_as_array = src.read().astype(np.float32)
    return tif_as_array

def normalize(x):
    return 255 * (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))

def gen_data(folder, file, nChannels):
    file_path = os.path.join(folder, file)
    arr = read_tiff(file_path)
    _, rows, cols = arr.shape

    arr = arr.transpose((1, 2, 0))  # transpose to height, width, channel
    arr = np.nan_to_num(arr)
    
    if np.mean(arr) == 0:
        print('discard empty array ' + file_path)
        os.remove(file_path)
    
    # store bands and label
    data_output = np.zeros((arr.shape[0], arr.shape[1], nChannels))
    
    # normalise spectral bands, 0-B4,B3,B2,B8-3, NDVI-4
    for i in range(nChannels - 1):
        data_output[:, :, i] = np.nan_to_num(normalize(arr[:, :, i]))
        
    # create label band
    th = cities[cities['city'] == file[:-4]]['threshold'].iloc[0]
    data_output[:, :, nChannels - 1] = arr[:, :, nChannels - 1] > th
    del arr
    
    # slice img for training (no overlap)
    sample_size = 256
    r = rows // sample_size
    c = cols // sample_size
    print(data_output.shape, r, c)
    dataset_train_list = []
    for i in range(r):
        for j in range(c):
            sample = data_output[i * sample_size : (i + 1) * sample_size, 
                             j * sample_size : (j + 1) * sample_size, 
                             0].flatten()
            if np.count_nonzero(sample) == 0:
                print('discard empty sample')
            else:
                dataset_train_list.append(data_output[i * sample_size : (i + 1) * sample_size, 
                                                      j * sample_size : (j + 1) * sample_size, 
                                                      :])
    del data_output
    
    dataset_train = np.stack(dataset_train_list, axis=0)
    np.save(os.path.join(folder, 'dataset', f"{file[:-4]}.npy"), dataset_train)


nChannels = 5
folder = 'Tropical'
if __name__ == '__main__':
    for file in os.listdir(folder):
        if file.endswith('.tif'):
            print(file)
            gen_data(folder, file, 5)