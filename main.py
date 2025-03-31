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

def main():
    zone = 'a'
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
    
    model, final_metrics, epochs, loss_values = train_model(
        dataset_train=dataset_train,
        dataset_eval=dataset_eval,
        batch_size=16,
        num_epochs=100,
        learning_rate=0.001,
        # alpha=0.8,
        # decision_thresh = dataset_train.calculate_green_percentages()
    )
    # Save the best model
    torch.save({
        'model_state_dict': model.state_dict(),
        'final_metrics': final_metrics,
    }, f'D:/Msc/Thesis/Models/{zone}_unet.pth')
    
    dic = {'epoch': epochs, 'loss': loss_values} 
    df = pd.DataFrame(dic)
    df.to_csv(f'D:/Msc/Thesis/Res/{zone}_unet.csv')
    
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