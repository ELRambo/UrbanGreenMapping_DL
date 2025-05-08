# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:45:17 2025

@author: 10449
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights, resnet50, ResNet50_Weights

##########################################
#----------------Encoders----------------# 
##########################################   
class ResNet34Encoder(nn.Module):
    def __init__(self, pretrained=True, nChannels = 5):
        super().__init__()
        resnet = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
        
        # Modify first convolution for channels
        self.initial_conv = nn.Conv2d(nChannels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        if pretrained:
            with torch.no_grad():
                new_weight = resnet.conv1.weight.data.mean(dim=1, keepdim=True)
                new_weight = new_weight.repeat(1, nChannels, 1, 1) * 3 / nChannels
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
    def __init__(self, pretrained=True, nChannels = 5):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        
        # Modify first convolution for 4 channels
        self.initial_conv = nn.Conv2d(nChannels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        if pretrained:
            with torch.no_grad():
                new_weight = resnet.conv1.weight.data.mean(dim=1, keepdim=True)
                new_weight = new_weight.repeat(1, nChannels, 1, 1) * 3 / nChannels
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


##################################################
#----------------Attention Module----------------# 
##################################################
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
        
        return x * weights.view(b, c, 1, 1)


###############################################################################
#--------------------------------Resnet AttUnet-------------------------------# 
############################################################################### 
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

class ResNet34SpatialAttUNet(nn.Module):
    def __init__(self, pretrained=True, output_ch=1):
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
            nn.Conv2d(32, output_ch, kernel_size=1),
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
    
class ResNet50SpatialAttUNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        
        # Encoder
        self.encoder = ResNet50Encoder(pretrained)
        
        # Decoder
        self.Up4 = UpConv(2048, 1024)
        self.Att4 = SpatialAttention(F_g=1024, F_l=1024, F_int=512)
        self.Up_conv4 = ConvBlock(2048, 1024)

        self.Up3 = UpConv(1024, 512)
        self.Att3 = SpatialAttention(F_g=512, F_l=512, F_int=256)
        self.Up_conv3 = ConvBlock(1024, 512)

        self.Up2 = UpConv(512, 256)
        self.Att2 = SpatialAttention(F_g=256, F_l=256, F_int=128)
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


###############################################################################
#-----------------Resnet AttUnet with modified residual block-----------------# 
############################################################################### 
class UpBlock(nn.Module):
    """Decoder block with concatenation"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConcatenatedResidualBlock(in_channels + out_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class ConcatenatedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.identity_conv = nn.Conv2d(in_channels, out_channels, 1)
        
        # Residual path
        self.conv1 = nn.Conv2d(in_channels, out_channels//2, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels//2)
        self.conv2 = nn.Conv2d(out_channels//2, out_channels//2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels//2)
        
        self.final_conv = nn.Conv2d(out_channels, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.identity_conv(x)
        
        # Residual path
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(x1)))
        
        # Concatenate residual path features
        residual_out = torch.cat([x1, x2], dim=1)
        
        # Combine with identity
        out = self.final_conv(residual_out) + identity
        return self.relu(out)

class ResNet34CRAUNet(nn.Module):
    def __init__(self, n_channels=5, base_channels=64):
        super().__init__()
        # Encoder (ResNet34 backbone with modified residual blocks)
        self.encoder = ResNet34Encoder(nChannels=n_channels)
        
        # Skip connections with channel attention
        self.ca1 = ChannelAttention(64)
        self.ca2 = ChannelAttention(128)
        self.ca3 = ChannelAttention(256)
        
        # Decoder with upsampling
        self.decoder = nn.Sequential(
            UpBlock(512, 256),  # 8x8 -> 16x16
            UpBlock(256, 128),   # 16x16 -> 32x32
            UpBlock(128, 64),    # 32x32 -> 64x64
            # Additional upsampling layers
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(64, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(32, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, 1)
            )
        )

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        
        # Apply channel attention to skip connections
        x1 = self.ca1(x1)
        x2 = self.ca2(x2)
        x3 = self.ca3(x3)
                
        # Decoder path
        d = self.decoder[0](x4, x3)
        d = self.decoder[1](d, x2)
        d = self.decoder[2](d, x1)
        d = self.decoder[3](d)
        return torch.sigmoid(self.decoder[4](d))

class ResNet34CRAUNetMultiClass(nn.Module):
    def __init__(self, pretrained_model=None, output_ch=4):
        super().__init__()
        
        # Initialize the original model
        self.base_model = ResNet34CRAUNet()
        if pretrained_model:
            self.base_model.load_state_dict(torch.load(pretrained_model)['model_state_dict'], strict=False)
        
        # Freeze all pretrained layers
        for param in self.base_model.encoder.parameters():
            param.requires_grad = False
                    
        self.base_model.decoder[4] = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, output_ch, kernel_size=1),
        )
        
        nn.init.kaiming_normal_(self.base_model.decoder[4][-1].weight)
        for param in self.base_model.decoder[4].parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.base_model(x)