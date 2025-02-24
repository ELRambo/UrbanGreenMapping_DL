# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 21:27:16 2024

@author: 10449
"""

from data_processor.dataset_loader import GeospatialDataLoader
from model_builder.train_evaluate import train_model
import torch
import time
import tkinter
from tkinter import messagebox
import winsound

def main():
    zone = 'Continental'
    train_data_loader = GeospatialDataLoader(
        data_dir=f"D:/Msc/Thesis/Data/GEEDownload/{zone}/train",
        required_shape=(256, 256, 5)
    )
    eval_data_loader = GeospatialDataLoader(
        data_dir=f"D:/Msc/Thesis/Data/GEEDownload/{zone}/eval",
        required_shape=(256, 256, 5)
    )
    dataset_train = train_data_loader.get_torch_dataset()
    dataset_eval = eval_data_loader.get_torch_dataset()
    
    model, final_metrics = train_model(
        dataset_train=dataset_train,
        dataset_eval=dataset_eval,
        batch_size=8,
        num_epochs=50,
        learning_rate=0.001,
    )
    # Save the best model
    torch.save({
        'model_state_dict': model.state_dict(),
        'final_metrics': final_metrics,
    }, f"D:/Msc/Thesis/Models/{zone}_unet.pth")
    
    del dataset_train, dataset_eval, model
    
if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time_minutes = (end_time - start_time) / 60
    print(f"Elapsed time: {elapsed_time_minutes:.1f} minutes")
    
    root = tkinter.Tk()
    root.withdraw()
    messagebox.showinfo('prompt','completed')
    winsound.MessageBeep()