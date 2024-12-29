# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 21:27:16 2024

@author: 10449
"""

from data_processor.dataset_loader import GeospatialDataLoader
from model_builder.train_evaluate import train_model
import torch

def main():
    data_loader = GeospatialDataLoader(
        data_dir="D:/Msc/Thesis/Data/GEEDownload/Tropical/dataset",
        required_shape=(256, 256, 5)
    )
    combined_dataset = data_loader.get_torch_dataset()
    
    model, final_metrics = train_model(
        dataset_train=combined_dataset,
        batch_size=8,
        num_epochs=50,
        learning_rate=0.001,
        val_split=0.2
    )
    # Save the best model
    torch.save({
        'model_state_dict': model.state_dict(),
        'final_metrics': final_metrics,
    }, 'D:/Msc/Thesis/Data/GEEDownload/Tropical/model/geospatial_model.pth')
    
if __name__ == "__main__":
    main()