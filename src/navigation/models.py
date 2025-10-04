"""
CNN model architectures for pose estimation.

This module contains all the model architectures used in Chapter 4,
extracted from demo.ipynb to eliminate code duplication.
"""

import torch
import torch.nn as nn
import torchvision.models as models


# ============================================================================
# Basic Models
# ============================================================================

class PoseNet(nn.Module):
    """
    Original PoseNet from the notebook.
    5-layer CNN with batch normalization.
    """
    
    def __init__(self):
        super(PoseNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        
        # Pooling and activation
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        
        # Fully connected layers
        # Input: 224x224 -> 7x7 after all conv/pool layers
        self.fc1 = nn.Linear(512 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)  # x, y position
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


class ImprovedPoseNet(nn.Module):
    """Enhanced CNN with dropout and batch normalization."""
    
    def __init__(self, dropout_rate=0.5):
        super(ImprovedPoseNet, self).__init__()
        
        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
        self.fc1 = nn.Linear(512 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


# ============================================================================
# Size Variants
# ============================================================================

class SmallPoseNet(nn.Module):
    """Lightweight CNN - fast training, good baseline."""
    
    def __init__(self, dropout_rate=0.3):
        super(SmallPoseNet, self).__init__()
        
        self.features = nn.Sequential(
            # 224x224x3 -> 112x112x32
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 112x112x32 -> 56x56x64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 56x56x64 -> 28x28x128
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 28x28x128 -> 14x14x128
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x


class MediumPoseNet(nn.Module):
    """Medium-sized CNN - balanced accuracy and speed."""
    
    def __init__(self, dropout_rate=0.4):
        super(MediumPoseNet, self).__init__()
        
        self.features = nn.Sequential(
            # 224x224x3 -> 112x112x64
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 112x112x64 -> 56x56x128
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 56x56x128 -> 28x28x256
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # 28x28x256 -> 14x14x256
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x


class LargePoseNet(nn.Module):
    """Large CNN - maximum capacity for best accuracy."""
    
    def __init__(self, dropout_rate=0.5):
        super(LargePoseNet, self).__init__()
        
        self.features = nn.Sequential(
            # 224x224x3 -> 112x112x64
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 112x112x64 -> 56x56x128
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 56x56x128 -> 28x28x256
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # 28x28x256 -> 14x14x512
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            # 14x14x512 -> 7x7x512
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x


# ============================================================================
# Transfer Learning
# ============================================================================

class ResNetPoseNet(nn.Module):
    """Transfer learning with pre-trained ResNet18."""
    
    def __init__(self, dropout_rate=0.5, freeze_backbone=False):
        super(ResNetPoseNet, self).__init__()
        
        # Load pre-trained ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Remove final FC layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Optionally freeze backbone weights
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
        
        # Custom pose regression head
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x


# ============================================================================
# Advanced Architectures
# ============================================================================

class CoordConv2d(nn.Module):
    """Convolution with coordinate channels appended."""
    
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        # Add 2 extra input channels for coordinates
        self.conv = nn.Conv2d(in_channels + 2, out_channels, **kwargs)
    
    def forward(self, x):
        batch_size, _, height, width = x.shape
        
        # Create coordinate channels
        # X coordinates: 0 to 1 from left to right
        x_coords = torch.linspace(0, 1, width, device=x.device)
        x_coords = x_coords.view(1, 1, 1, width).expand(batch_size, 1, height, width)
        
        # Y coordinates: 0 to 1 from top to bottom
        y_coords = torch.linspace(0, 1, height, device=x.device)
        y_coords = y_coords.view(1, 1, height, 1).expand(batch_size, 1, height, width)
        
        # Concatenate with input
        x_with_coords = torch.cat([x, x_coords, y_coords], dim=1)
        
        return self.conv(x_with_coords)


class CoordConvPoseNet(nn.Module):
    """PoseNet with coordinate convolutions - fixes spatial bias!"""
    
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        
        # First layer uses CoordConv to add position awareness
        self.conv1 = CoordConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Rest are regular convolutions
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
        # Regression head
        self.fc1 = nn.Linear(512 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 2)
    
    def forward(self, x):
        # First layer with coordinate info
        x = self.relu(self.bn1(self.conv1(x)))
        
        # Regular conv layers
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        
        # Regression
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class SpatialAttention(nn.Module):
    """Spatial attention mechanism."""
    
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Channel-wise statistics
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        
        # Concatenate and conv
        attention = torch.cat([max_pool, avg_pool], dim=1)
        attention = self.conv(attention)
        attention = self.sigmoid(attention)
        
        # Apply attention
        return x * attention


class AttentionPoseNet(nn.Module):
    """PoseNet with spatial attention."""
    
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.attention1 = SpatialAttention()
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.attention2 = SpatialAttention()
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.attention3 = SpatialAttention()
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.attention4 = SpatialAttention()
        
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
        self.fc1 = nn.Linear(512 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 2)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.attention1(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.attention2(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.attention3(x)
        
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.attention4(x)
        
        x = self.relu(self.bn5(self.conv5(x)))
        
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


# ============================================================================
# Aliases for backwards compatibility
# ============================================================================

# These match the names in deep_learning.py
SimplePoseNet = SmallPoseNet
AccuratePoseNet = ImprovedPoseNet

