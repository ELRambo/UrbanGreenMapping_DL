# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 21:22:47 2024

@author: 10449
"""

# unet.py
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self._make_layer(4, 64)
        self.enc2 = self._make_layer(64, 128)
        self.enc3 = self._make_layer(128, 256)
        
        # Decoder
        self.dec3 = self._make_layer(256, 128)
        self.dec2 = self._make_layer(128, 64)
        self.dec1 = self._make_layer(64, 32)
        
        # Final layer
        self.final = nn.Conv2d(32, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(nn.functional.max_pool2d(enc1, 2))
        enc3 = self.enc3(nn.functional.max_pool2d(enc2, 2))
        
        # Decoder with skip connections
        dec3 = self.dec3(nn.functional.interpolate(enc3, scale_factor=2))
        dec2 = self.dec2(nn.functional.interpolate(dec3 + enc2, scale_factor=2))
        dec1 = self.dec1(dec2 + enc1)
        
        return self.sigmoid(self.final(dec1))
