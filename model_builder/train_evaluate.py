# -*- coding: utf-8 -*-
'''
Created on Sat Dec 28 21:30:22 2024

@author: 10449
'''

from model_builder.resnet_attunet import ResNet34CRAUNet
from model_builder.swin_unetr import SwinUNETR
from model_builder.swin_attunetr import SwinAttnUNETR
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import PolynomialLR
from torch.utils.data import DataLoader
from monai.losses import FocalLoss
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.amp import autocast, GradScaler
import gc

# Enable TF32 math (Ampere+ GPUs)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# Optimize conv algorithm selection
torch.backends.cudnn.benchmark = True
# Disable profiling
torch.autograd.profiler.emit_nvtx(False)
torch.autograd.set_detect_anomaly(False)

def evaluate_model(model, data_loader, criterion, device, decision_thresh):
    
    model.eval()
    total_loss = 0
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
    
    with torch.no_grad():
        for features, labels in data_loader:
            # features: (batch_size, channels, height, width)
            # labels: (batch_size, height, width)
            features = features.to(device)
            labels = labels.to(device)
            
            with autocast('cuda'):
                outputs = model(features)
                
            loss = criterion(outputs.squeeze(1), labels)
            total_loss += loss.item()
            
            preds = (outputs > 0.5).float()
            
            # Calculate metrics
            preds_flat = preds.flatten()
            labels_flat = labels.flatten()
            
            total_tp += ((preds_flat == 1) & (labels_flat == 1)).sum().item()
            total_fp += ((preds_flat == 1) & (labels_flat == 0)).sum().item()
            total_fn += ((preds_flat == 0) & (labels_flat == 1)).sum().item()
            total_tn += ((preds_flat == 0) & (labels_flat == 0)).sum().item()
    
    eps = 1e-7  # Avoid division by zero
    accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn + eps)
    precision = total_tp / (total_tp + total_fp + eps)
    recall = total_tp / (total_tp + total_fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    iou = total_tp / (total_tp + total_fp + total_fn + eps)
    # dice = 2 * total_tp / (2 * total_tp + total_fp + total_fn + eps)
    
    metrics = {
        'loss': total_loss / len(data_loader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
    }
        
    return metrics

def print_res(val_metrics):
    print(f'Validation Loss: {val_metrics['loss']:.4f}')
    print(f'Validation Accuracy: {val_metrics['accuracy']:.4f}')
    print(f'Validation Precision: {val_metrics['precision']:.4f}')
    print(f'Validation Recall: {val_metrics['recall']:.4f}')
    print(f'Validation F1: {val_metrics['f1']:.4f}')
    print(f'Validation MIoU: {val_metrics['iou']:.4f}')
    print('-' * 50)

def train_model(dataset_train, dataset_eval, batch_size=8, num_epochs=100, 
                learning_rate=0.001, alpha=0.8, gamma=2, decision_thresh=0.5):
    
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
    
    # model = ResNet34CRAUNet().to(device)
    model = SwinAttnUNETR(
        img_size=(256, 256),
        in_channels=5,
        out_channels=1,
        spatial_dims=2,
    ).to(device)  
    criterion = FocalLoss(reduction='mean', alpha=alpha, gamma=gamma)
    # criterion = DiceLoss()
    # criterion = TverskyLoss(alpha=0.5, beta=0.5)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = PolynomialLR(optimizer, total_iters=50, power=0.9)
    # scheduler= CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    best_val = float('-inf')
    patience = 100
    patience_counter = 0
    best_model_state = None
    
    # print(f'Decision threhold: {decision_thresh:.2f}')
    # print(f'Alpha: {alpha:.2f}')
    print(f'Training on {device}')
    print(f'Training set size: {train_size}, Validation set size: {val_size}')
    
    loss_values = []
    oa = []
    iou = []
    epochs = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        scaler = GradScaler()
        
        batch_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False)
        
        for batch_features, batch_labels in batch_pbar:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            
            with autocast('cuda'):
                outputs = model(batch_features)
                loss = criterion(outputs.squeeze(1), batch_labels.squeeze(1))
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            batch_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss = running_loss / len(train_loader)
        val_metrics = evaluate_model(model, val_loader, criterion, device, decision_thresh)
        
        lr = scheduler.get_last_lr()
        scheduler.step()
        
        gc.collect()  # Force garbage collection
        torch.cuda.empty_cache()  # Clear GPU cache
        
        loss_values.append(val_metrics['loss'])
        oa.append(val_metrics['accuracy'])
        iou.append(val_metrics['iou'])
        epochs.append(epoch+1)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Learning rate: {lr}')
        print(f'Training Loss: {train_loss:.4f}')
        print_res(val_metrics)
        
        if val_metrics['iou'] > best_val:
            best_epoch = epoch
            best_model = model
            best_metrics = val_metrics
            best_train_loss = train_loss
            best_val = val_metrics['iou']
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after epoch {epoch+1}\n')
                break
    
    if best_model_state is not None:
        best_model.load_state_dict(best_model_state)
        
    print(f'Best Epoch: {best_epoch}\n')
    print(f'Training Loss: {best_train_loss:.4f}')
    print_res(best_metrics)
    
    plt.plot(epochs, loss_values)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.ylim(0.02, 0.06)
    plt.show()
     
    return best_model, best_metrics, epochs, loss_values, oa, iou