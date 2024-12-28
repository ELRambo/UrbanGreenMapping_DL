# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 21:30:22 2024

@author: 10449
"""

from model_builder.unet import SpectralUNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features).squeeze(1)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Convert predictions to binary (0 or 1)
            preds = (outputs > 0.5).float()
            
            # Collect predictions and labels for metric calculation
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    
    # Calculate metrics
    metrics = {
        'loss': total_loss / len(data_loader),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0)
    }
    
    return metrics

def train_model(dataset_train, batch_size=8, num_epochs=50, learning_rate=0.001, val_split=0.2):
    # Split dataset into train and validation
    total_size = len(dataset_train)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(
        dataset_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model, loss function, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpectralUNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # For early stopping
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    best_model_state = None
    
    print(f"Training on {device}")
    print(f"Training set size: {train_size}, Validation set size: {val_size}")
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
        
        for batch_features, batch_labels in batch_pbar:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features).squeeze(1)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            batch_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss = running_loss / len(train_loader)
        
        # Validation phase
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        
        # Print epoch statistics
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Training Loss: {train_loss:.4f}')
            print(f'Validation Loss: {val_metrics["loss"]:.4f}')
            print(f'Validation F1: {val_metrics["f1"]:.4f}')
            print(f'Validation Precision: {val_metrics["precision"]:.4f}')
            print(f'Validation Recall: {val_metrics["recall"]:.4f}')
            print('-' * 50)
        
        # Early stopping
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after epoch {epoch+1}')
                break
    
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, val_metrics