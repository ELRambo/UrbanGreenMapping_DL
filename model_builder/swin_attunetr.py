# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 05:28:02 2025

@author: 10449
"""

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR
import torch.nn.functional as F

class SwinAttnUNETR(nn.Module):
    def __init__(self, in_channels=5, out_channels=1, feature_size=24, img_size=(256, 256), spatial_dims=2):
        super().__init__()
        
        self.swin = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            spatial_dims=spatial_dims,
        )
        
        # Attention blocks
        self.attn0 = ChannelAttention(feature_size)
        self.attn1 = ChannelAttention(feature_size)
        self.attn2 = ChannelAttention(feature_size*2)
        self.attn3 = ChannelAttention(feature_size*4)
                
    def forward(self, x):
        hidden_states = self.swin.swinViT(x, self.swin.normalize)
        # for i in range(5):
        #     print(hidden_states[i].shape)
        
        # Encoder
        enc0 = self.swin.encoder1(x)
        enc1 = self.swin.encoder2(hidden_states[0])
        enc2 = self.swin.encoder3(hidden_states[1])
        enc3 = self.swin.encoder4(hidden_states[2])
        # print('\n', enc0.shape, enc1.shape, enc2.shape, enc3.shape, '\n')
        
        # Attention
        enc0 = self.attn0(enc0)
        enc1 = self.attn1(enc1)
        enc2 = self.attn2(enc2)
        enc3 = self.attn3(enc3)
                
        # Decoder with fused features
        dec4 = self.swin.encoder10(hidden_states[4])
        dec3 = self.swin.decoder5(dec4, hidden_states[3])
        dec2 = self.swin.decoder4(dec3, enc3)
        dec1 = self.swin.decoder3(dec2, enc2)
        dec0 = self.swin.decoder2(dec1, enc1)
        out = self.swin.decoder1(dec0, enc0)
        # print(dec4.shape, dec3.shape, dec2.shape, dec1.shape, dec0.shape, out.shape, '\n')
                
        logits = self.swin.out(out)
        
        return torch.sigmoid(logits)
    
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        
        x = F.relu(self.conv(x))
        
        # Average pooling
        y_avg = self.avg_pool(x).view(b, c)
        y_avg = self.fc(y_avg)
        
        # Max pooling
        y_max = self.max_pool(x).view(b, c)
        y_max = self.fc(y_max)
        
        # Combine
        weights = y_avg + y_max
        weights = self.sigmoid(weights)
        
        return x + x * weights.view(b, c, 1, 1)  # add the oringinal input to skip connection
    
class SwinAttnUNETRMultiClass(nn.Module):
    def __init__(self, in_channels=5, out_channels=4, feature_size=24, img_size=(256, 256), spatial_dims=2):
        super().__init__()
        
        self.swin = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            spatial_dims=spatial_dims,
        )
                
        # Attention blocks
        self.attn0 = ChannelAttention(feature_size)
        self.attn1 = ChannelAttention(feature_size)
        self.attn2 = ChannelAttention(feature_size*2)
        self.attn3 = ChannelAttention(feature_size*4)
                
    def forward(self, x):
        hidden_states = self.swin.swinViT(x, self.swin.normalize)
        
        # Encoder
        enc0 = self.swin.encoder1(x)
        enc1 = self.swin.encoder2(hidden_states[0])
        enc2 = self.swin.encoder3(hidden_states[1])
        enc3 = self.swin.encoder4(hidden_states[2])
        
        # Attention
        enc0 = self.attn0(enc0)
        enc1 = self.attn1(enc1)
        enc2 = self.attn2(enc2)
        enc3 = self.attn3(enc3)
                
        # Decoder with fused features
        dec4 = self.swin.encoder10(hidden_states[4])
        dec3 = self.swin.decoder5(dec4, hidden_states[3])
        dec2 = self.swin.decoder4(dec3, enc3)
        dec1 = self.swin.decoder3(dec2, enc2)
        dec0 = self.swin.decoder2(dec1, enc1)
        out = self.swin.decoder1(dec0, enc0)
                
        logits = self.swin.out(out)
        
        return logits

    
if __name__ == '__main__':    
    model = SwinAttnUNETR(
        img_size=(256, 256),
        in_channels=5,
        out_channels=1,
        spatial_dims=2,
    )
    x = torch.randn(16, 5, 256, 256)  # Batch size 16
    
    output = model(x)
    print(output.shape)  # torch.Size([16, 1, 256, 256])