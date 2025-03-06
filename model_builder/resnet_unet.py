# -*- coding: utf-8 -*-
'''
Created on Mon Feb 24 14:47:43 2025

@author: 10449
'''

import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y_avg = self.avg_pool(x).view(b, c)
        y_max = self.max_pool(x).view(b, c)
        
        y = self.fc(y_avg) + self.fc(y_max)
        y = torch.sigmoid(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResNetEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        
        resnet = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
        
        # Modify first convolution layer to accept 4 channels
        self.initial_conv = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Initialize new weights using pretrained weights
        if pretrained:
            with torch.no_grad():
                # Average first layer weights across RGB channels and repeat for 4th channel
                new_weight = resnet.conv1.weight.data.mean(dim=1, keepdim=True)
                new_weight = new_weight.repeat(1, 4, 1, 1) * 0.75  # Adjust for 4 channels
                self.initial_conv.weight.data = new_weight
        
        # Initial layers (stem)
        self.initial = nn.Sequential(
            self.initial_conv,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        
        # # Encoder layers
        # self.layer1 = resnet.layer1
        # self.layer2 = resnet.layer2
        # self.layer3 = resnet.layer3
        # self.layer4 = resnet.layer4
        
        # Add attention to encoder layers
        self.layer1 = nn.Sequential(resnet.layer1, ChannelAttention(64))
        self.layer2 = nn.Sequential(resnet.layer2, ChannelAttention(128))
        self.layer3 = nn.Sequential(resnet.layer3, ChannelAttention(256))
        self.layer4 = nn.Sequential(resnet.layer4, ChannelAttention(512))
        
    def forward(self, x):
        # Get intermediate features from different layers
        x0 = self.initial(x)  # Initial features
        x1 = self.layer1(x0)  # 1/4 scale
        x2 = self.layer2(x1)  # 1/8 scale
        x3 = self.layer3(x2)  # 1/16 scale
        x4 = self.layer4(x3)  # 1/32 scale
        return x1, x2, x3, x4

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        
        # Upsampling layer
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.attention = ChannelAttention(skip_channels)
        
        # Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ChannelAttention(out_channels),  # Add attention after convolution
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        skip = self.attention(skip)  # Apply attention to skip connection
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class ResNetUNet(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        # Encoder
        self.encoder = ResNetEncoder(pretrained=pretrained)
        
        # Decoder resnet34
        self.decoder4 = DecoderBlock(512, 256, 256)
        self.decoder3 = DecoderBlock(256, 128, 128)
        self.decoder2 = DecoderBlock(128, 64, 64)
        
        # # Decoder resnet50
        # self.decoder4 = DecoderBlock(2048, 1024, 512)  # layer4 to layer3
        # self.decoder3 = DecoderBlock(512, 512, 256)    # layer3 to layer2
        # self.decoder2 = DecoderBlock(256, 256, 64)     # layer2 to layer1
        
        # Final upsampling and output
        self.final_upsample = nn.Sequential(
            ChannelAttention(64),  # Final attention before upsampling
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, num_classes, kernel_size=1),
            nn.Sigmoid()  # Add sigmoid for binary segmentation
        )

    def forward(self, x):
        # Encoder forward pass
        x1, x2, x3, x4 = self.encoder(x)
        
        # Decoder forward pass
        d4 = self.decoder4(x4, x3)
        d3 = self.decoder3(d4, x2)
        d2 = self.decoder2(d3, x1)
        
        # Final upsampling
        out = self.final_upsample(d2)
        return out