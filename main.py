# -*- coding: utf-8 -*-
'''
Created on Sat Dec 28 21:27:16 2024

@author: 10449
'''

from data_processor.dataset_loader import GeospatialDataLoader
from model_builder.train_evaluate import train_model
import torch
import pandas as pd
import time
import tkinter
from tkinter import messagebox
import winsound
import warnings
warnings.filterwarnings("ignore")

def main():
    zones = ['b']
    for zone in zones:
        sample_size = 256
        train_data_loader = GeospatialDataLoader(
            data_dir=f'D:/Msc/Thesis/Data/GEEDownload/{zone}/train',
            required_shape=(sample_size, sample_size, 6),
            dataset_type = 'train'
        )
        eval_data_loader = GeospatialDataLoader(
            data_dir=f'D:/Msc/Thesis/Data/GEEDownload/{zone}/eval',
            required_shape=(sample_size, sample_size, 6),
            dataset_type='eval'
        )
        dataset_train = train_data_loader.get_torch_dataset()
        dataset_eval = eval_data_loader.get_torch_dataset()
        
        model, final_metrics, epochs, loss_values, oa, iou = train_model(
            dataset_train=dataset_train,
            dataset_eval=dataset_eval,
            batch_size=16,
            num_epochs=50,
            learning_rate=0.001,
        )
        # Save the best model
        torch.save({
            'model_state_dict': model.state_dict(),
            'final_metrics': final_metrics,
        }, f'D:/Msc/Thesis/Models/binary/{zone}_unet31.pth')
        
        dic = {'epoch': epochs, 'loss': loss_values, 'oa': oa, 'iou': iou} 
        df = pd.DataFrame(dic)
        df.to_csv(f'D:/Msc/Thesis/Res/binary/{zone}_unet31.csv')
        
        del dataset_train, dataset_eval, model
    
if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time_minutes = (end_time - start_time) / 60
    print(f'Elapsed time: {elapsed_time_minutes:.1f} minutes')
    
    root = tkinter.Tk()
    root.withdraw()
    messagebox.showinfo('prompt','completed')
    winsound.MessageBeep()