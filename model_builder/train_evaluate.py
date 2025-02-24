# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 21:30:22 2024

@author: 10449
"""

# train_evaluate.py
from model_builder.unet import SpectralUNet
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from monai.losses.dice import DiceLoss
from monai.metrics import MeanIoU, DiceMetric
from tqdm import tqdm
import numpy as np

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    iou_metric = MeanIoU(include_background=True)
    dice_metric = DiceMetric(include_background=True)
    
    with torch.no_grad():
        for features, labels in data_loader:
            # features: (batch_size, 4, height, width)
            # labels after unsqueezing: (batch_size, 1, height, width)
            features = features.to(device)
            labels = labels.unsqueeze(1).to(device)  # add #channels for Dice loss and metrics calculations
            
            outputs = model(features)
            loss = criterion(outputs.squeeze(1), labels.squeeze(1))  # Remove channel dim for loss
            total_loss += loss.item()
            
            # converts the model's continuous outputs into binary for evaluation
            preds = (outputs > 0.5).float()
            
            iou_metric(preds, labels) 
            dice_metric(preds, labels)
            
            all_preds.extend(preds.cpu().numpy().astype(np.float32).flatten())
            all_labels.extend(labels.squeeze(1).cpu().numpy().astype(np.uint8).flatten())

    metrics = {
        'loss': total_loss / len(data_loader),
        'accuracy': accuracy_score(all_labels, all_preds, zero_division=0),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'iou': iou_metric.aggregate().item(),
        'dice': dice_metric.aggregate().item()
    }
    
    return metrics

def train_model(dataset_train, dataset_eval, batch_size=8, 
                num_epochs=50, learning_rate=0.001):
    train_size = len(dataset_train)
    val_size = len(dataset_eval)
    
    # train_size = total_size - val_size
    # train_dataset, val_dataset = random_split(
    #     dataset_train,
    #     [train_size, val_size],
    #     generator=torch.Generator().manual_seed(42)
    # )
    
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset_eval, batch_size=batch_size)
    
    del dataset_train, dataset_eval
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpectralUNet().to(device)
    criterion = DiceLoss(include_background=True, reduction='mean', sigmoid=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    best_model_state = None
        
    print(f"Training on {device}")
    print(f"Training set size: {train_size}, Validation set size: {val_size}")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
        
        for batch_features, batch_labels in batch_pbar:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs.squeeze(1), batch_labels.squeeze(1))
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            batch_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss = running_loss / len(train_loader)
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        
        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Training Loss: {train_loss:.4f}')
            print(f'Validation Loss: {val_metrics["loss"]:.4f}')
            print(f'Validation F1: {val_metrics["f1"]:.4f}')
            print(f'Validation Precision: {val_metrics["precision"]:.4f}')
            print(f'Validation Recall: {val_metrics["recall"]:.4f}')
            print(f'Validation IoU: {val_metrics["iou"]:.4f}')
            print(f'Validation Dice: {val_metrics["dice"]:.4f}')
            print('-' * 50)
        
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after epoch {epoch+1}')
                break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, val_metrics