# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 10:38:51 2025

@author: Jingyi Zhang

"""

from monai.losses import FocalLoss, DiceLoss, TverskyLoss
import torch

class CombinedFocalDiceLoss(torch.nn.Module):
    def __init__(self, weight_focal=1.0, weight_dice=1.0, **kwargs):
        super().__init__()
        # Initialize Focal Loss (adjust parameters as needed)
        self.focal_loss = FocalLoss(
            include_background=False,  # Set True if background class matters
            gamma=2.0,  # Focal Loss gamma (focusing parameter)
            alpha=0.8,  # Balancing factor for class imbalance
            **kwargs
        )
        
        # Initialize Dice Loss
        self.dice_loss = TverskyLoss(
            include_background=False,  # Match with Focal Loss
            sigmoid=True,  # Use sigmoid for binary, softmax for multi-class
            alpha=0.7, beta=0.3,
            **kwargs
        )
        
        self.weight_focal = weight_focal
        self.weight_dice = weight_dice

    def forward(self, inputs, targets):
        # Compute Focal Loss
        focal = self.focal_loss(inputs, targets)
        
        # Compute Dice Loss
        dice = self.dice_loss(inputs, targets)
        
        # Combine losses
        return self.weight_focal * focal + self.weight_dice * dice