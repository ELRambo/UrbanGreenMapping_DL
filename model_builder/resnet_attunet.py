# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:45:17 2025

@author: 10449
"""

import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights, resnet50, ResNet50_Weights

class SpatialAttention(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(SpatialAttention, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi
    
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
        
        # Average pooling
        y_avg = self.avg_pool(x).view(b, c)
        y_avg = self.fc(y_avg)
        
        # Max pooling
        y_max = self.max_pool(x).view(b, c)
        y_max = self.fc(y_max)
        
        # Combine
        y = y_avg + y_max
        return x * y.view(b, c, 1, 1)

class CombinedAttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        
        # Channel attention for both pathways
        self.ch_att_g = ChannelAttention(F_g)
        self.ch_att_x = ChannelAttention(F_l)
        
        # Spatial attention components
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # Channel attention first
        g_ch = self.ch_att_g(g)  # [B, F_g, H, W]
        x_ch = self.ch_att_x(x)  # [B, F_l, H, W]
        
        # Spatial attention
        g1 = self.W_g(g_ch)
        x1 = self.W_x(x_ch)
        
        # Align spatial dimensions if needed
        if g1.size()[2:] != x1.size()[2:]:
            g1 = nn.functional.interpolate(
                g1, 
                size=x1.size()[2:], 
                mode='bilinear', 
                align_corners=True
            )
            
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)  # [B, 1, H, W]
        
        # Apply combined attention
        return x_ch * psi  # Channel-attended x * spatial mask

class ResNet34Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
        
        # Modify first convolution for 4 channels
        self.initial_conv = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        if pretrained:
            with torch.no_grad():
                new_weight = resnet.conv1.weight.data.mean(dim=1, keepdim=True)
                new_weight = new_weight.repeat(1, 4, 1, 1) * 0.75
                self.initial_conv.weight.data = new_weight

        self.initial = nn.Sequential(
            self.initial_conv,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels

    def forward(self, x):
        x0 = self.initial(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1, x2, x3, x4
    
class ResNet50Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        
        # Modify first convolution for 4 channels
        self.initial_conv = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        if pretrained:
            with torch.no_grad():
                new_weight = resnet.conv1.weight.data.mean(dim=1, keepdim=True)
                new_weight = new_weight.repeat(1, 4, 1, 1) * 0.75
                self.initial_conv.weight.data = new_weight

        self.initial = nn.Sequential(
            self.initial_conv,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        
        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels
        self.layer4 = resnet.layer4  # 2048 channels

    def forward(self, x):
        x0 = self.initial(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1, x2, x3, x4

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)

class ResNet50AttUNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        
        # Encoder
        self.encoder = ResNet50Encoder(pretrained)
        
        # Decoder
        self.Up4 = UpConv(2048, 1024)
        self.Att4 = CombinedAttentionBlock(F_g=1024, F_l=1024, F_int=512)
        self.Up_conv4 = ConvBlock(2048, 1024)

        self.Up3 = UpConv(1024, 512)
        self.Att3 = CombinedAttentionBlock(F_g=512, F_l=512, F_int=256)
        self.Up_conv3 = ConvBlock(1024, 512)

        self.Up2 = UpConv(512, 256)
        self.Att2 = CombinedAttentionBlock(F_g=256, F_l=256, F_int=128)
        self.Up_conv2 = ConvBlock(512, 256)

        # Final upsampling to original size
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        
        # Decoder
        d4 = self.Up4(x4)
        x3_att = self.Att4(g=d4, x=x3)
        d4 = torch.cat([x3_att, d4], dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2_att = self.Att3(g=d3, x=x2)
        d3 = torch.cat([x2_att, d3], dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1_att = self.Att2(g=d2, x=x1)
        d2 = torch.cat([x1_att, d2], dim=1)
        d2 = self.Up_conv2(d2)

        out = self.final_up(d2)
        return out
    
class ResNet34AttUNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        
        # Encoder
        self.encoder = ResNet34Encoder(pretrained)
        
        # Decoder
        self.Up4 = UpConv(512, 256)
        self.Att4 = SpatialAttention(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = ConvBlock(512, 256)

        self.Up3 = UpConv(256, 128)
        self.Att3 = SpatialAttention(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = ConvBlock(256, 128)

        self.Up2 = UpConv(128, 64)
        self.Att2 = SpatialAttention(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = ConvBlock(128, 64)

        # Final upsampling to original size
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        
        # Decoder
        d4 = self.Up4(x4)
        x3_att = self.Att4(g=d4, x=x3)
        d4 = torch.cat([x3_att, d4], dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2_att = self.Att3(g=d3, x=x2)
        d3 = torch.cat([x2_att, d3], dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1_att = self.Att2(g=d2, x=x1)
        d2 = torch.cat([x1_att, d2], dim=1)
        d2 = self.Up_conv2(d2)

        out = self.final_up(d2)
        return out
    
class ResNet34ChannelAttUNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.encoder = ResNet34Encoder(pretrained)
        
        # Decoder with channel attention
        self.Up4 = UpConv(512, 256)
        self.ca4 = ChannelAttention(256)  # Applied to encoder features
        self.Up_conv4 = ConvBlock(512, 256)

        self.Up3 = UpConv(256, 128)
        self.ca3 = ChannelAttention(128)
        self.Up_conv3 = ConvBlock(256, 128)

        self.Up2 = UpConv(128, 64)
        self.ca2 = ChannelAttention(64)
        self.Up_conv2 = ConvBlock(128, 64)

        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        x1, x2, x3, x4 = self.encoder(x)
        
        # Decoder with channel attention
        d4 = self.Up4(x4)
        x3 = self.ca4(x3)  # Apply channel attention to encoder features
        d4 = torch.cat([x3, d4], dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.ca3(x2)
        d3 = torch.cat([x2, d3], dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.ca2(x1)
        d2 = torch.cat([x1, d2], dim=1)
        d2 = self.Up_conv2(d2)

        return self.final_up(d2)