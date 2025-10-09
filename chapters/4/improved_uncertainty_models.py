#!/usr/bin/env python3
"""
Improved uncertainty model architectures.

Experiments with different architectural approaches to learn better uncertainty:
1. Separate x/y uncertainties (anisotropic)
2. Deeper uncertainty head with attention
3. Multi-head uncertainty prediction
4. Terrain difficulty as auxiliary task
"""

import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights


class BasicModelWithAnisotropicUncertainty(nn.Module):
    """
    Predict separate uncertainties for x and y coordinates.
    Some terrain may be distinctive in one direction but not the other.
    """

    def __init__(self):
        super().__init__()
        from torchvision.models import densenet121, DenseNet121_Weights

        self.backbone = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)

        # Shared features
        self.shared = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1024, 128),
            nn.ReLU()
        )

        # Coordinate heads
        self.x_coord_head = nn.Linear(128, 1)
        self.y_coord_head = nn.Linear(128, 1)

        # Separate uncertainty for each axis
        self.x_uncertainty_head = nn.Linear(128, 1)
        self.y_uncertainty_head = nn.Linear(128, 1)

        # Initialize uncertainties conservatively
        nn.init.constant_(self.x_uncertainty_head.bias, -2.0)
        nn.init.constant_(self.y_uncertainty_head.bias, -2.0)
        nn.init.normal_(self.x_uncertainty_head.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.y_uncertainty_head.weight, mean=0.0, std=0.01)

    def forward(self, x):
        features = self.backbone.features(x)
        features = torch.nn.functional.relu(features, inplace=True)
        features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)

        shared_features = self.shared(features)

        # Predict each coordinate separately
        x_coord = self.x_coord_head(shared_features)
        y_coord = self.y_coord_head(shared_features)
        coords = torch.cat([x_coord, y_coord], dim=1)

        # Predict separate uncertainties
        x_log_var = self.x_uncertainty_head(shared_features)
        y_log_var = self.y_uncertainty_head(shared_features)
        log_vars = torch.cat([x_log_var, y_log_var], dim=1)

        return coords, log_vars


class BasicModelWithDeepUncertaintyHead(nn.Module):
    """
    Use a deeper, more expressive uncertainty head.
    Maybe single linear layer isn't enough to learn uncertainty patterns.
    """

    def __init__(self):
        super().__init__()
        from torchvision.models import densenet121, DenseNet121_Weights

        self.backbone = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)

        # Shared features
        self.shared = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Coordinate head (simple)
        self.coord_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

        # Deep uncertainty head with attention
        self.uncertainty_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Initialize conservatively
        nn.init.constant_(self.uncertainty_head[-1].bias, -2.0)

    def forward(self, x):
        features = self.backbone.features(x)
        features = torch.nn.functional.relu(features, inplace=True)
        features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)

        shared_features = self.shared(features)

        coords = self.coord_head(shared_features)
        log_var = self.uncertainty_head(shared_features)

        return coords, log_var


class BasicModelWithTerrainDifficulty(nn.Module):
    """
    Add terrain difficulty prediction as auxiliary task.
    Explicitly teach the model to recognize difficult vs easy terrain.
    """

    def __init__(self):
        super().__init__()
        from torchvision.models import densenet121, DenseNet121_Weights

        self.backbone = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)

        # Shared features
        self.shared = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Coordinate head
        self.coord_head = nn.Linear(256, 2)

        # Uncertainty head
        self.uncertainty_head = nn.Linear(256, 1)

        # Terrain difficulty head (0=easy, 1=difficult)
        # This auxiliary task helps learn what makes predictions uncertain
        self.difficulty_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output probability [0, 1]
        )

        # Initialize
        nn.init.constant_(self.uncertainty_head.bias, -2.0)
        nn.init.normal_(self.uncertainty_head.weight, mean=0.0, std=0.01)

    def forward(self, x):
        features = self.backbone.features(x)
        features = torch.nn.functional.relu(features, inplace=True)
        features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)

        shared_features = self.shared(features)

        coords = self.coord_head(shared_features)
        log_var = self.uncertainty_head(shared_features)
        difficulty = self.difficulty_head(shared_features)

        return coords, log_var, difficulty


class BasicModelWithAttentionUncertainty(nn.Module):
    """
    Use spatial attention to identify which parts of the image drive uncertainty.
    The model learns where to look to assess confidence.
    """

    def __init__(self):
        super().__init__()
        from torchvision.models import densenet121, DenseNet121_Weights

        backbone = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        self.features = backbone.features

        # Spatial attention for uncertainty estimation
        self.attention = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Coordinate head (uses global pooling)
        self.coord_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

        # Uncertainty head (uses attention-weighted pooling)
        self.uncertainty_processor = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Initialize
        nn.init.constant_(self.uncertainty_processor[-1].bias, -2.0)

    def forward(self, x):
        # Extract features
        features = self.features(x)  # [B, 1024, H, W]

        # Coordinate prediction (standard global pooling)
        coords = self.coord_head(features)

        # Attention-based uncertainty
        attention_map = self.attention(features)  # [B, 1, H, W]

        # Weighted pooling using attention
        weighted_features = features * attention_map
        pooled_features = torch.nn.functional.adaptive_avg_pool2d(weighted_features, (1, 1))
        pooled_features = torch.flatten(pooled_features, 1)

        log_var = self.uncertainty_processor(pooled_features)

        return coords, log_var


# Loss functions for new architectures

def anisotropic_uncertainty_loss(pred_coords, pred_log_vars, true_coords,
                                 coord_weight=1.0, uncertainty_weight=1.0):
    """
    Loss for anisotropic (separate x, y) uncertainty.

    Args:
        pred_coords: [B, 2] predicted (x, y)
        pred_log_vars: [B, 2] predicted (log_var_x, log_var_y)
        true_coords: [B, 2] ground truth (x, y)
    """
    # Coordinate loss
    coord_loss = torch.mean((pred_coords - true_coords) ** 2)

    # Clamp log variances
    pred_log_vars = torch.clamp(pred_log_vars, min=-5, max=5)

    # Per-axis uncertainty loss
    squared_errors = (pred_coords - true_coords) ** 2  # [B, 2]

    # Per-axis precision
    precisions = torch.exp(-pred_log_vars)
    precisions = torch.clamp(precisions, min=0.01, max=100.0)

    # Uncertainty loss (sum over x and y)
    uncertainty_loss = torch.mean(
        precisions * squared_errors + 0.5 * pred_log_vars
    )

    total_loss = coord_weight * coord_loss + uncertainty_weight * uncertainty_loss

    return total_loss, coord_loss, uncertainty_loss


def terrain_difficulty_loss(pred_coords, pred_log_var, pred_difficulty, true_coords,
                            coord_weight=1.0, uncertainty_weight=1.0, difficulty_weight=0.5):
    """
    Loss with terrain difficulty as auxiliary task.

    The difficulty head learns to predict terrain difficulty (0=easy, 1=hard).
    Difficulty is defined by the actual error.
    """
    # Coordinate loss
    coord_loss = torch.mean((pred_coords - true_coords) ** 2)

    # Uncertainty loss
    pred_log_var = torch.clamp(pred_log_var, min=-5, max=5)
    squared_error = (pred_coords - true_coords) ** 2
    mse_per_sample = torch.mean(squared_error, dim=1, keepdim=True)

    precision = torch.exp(-pred_log_var)
    precision = torch.clamp(precision, min=0.01, max=100.0)
    uncertainty_loss = torch.mean(precision * mse_per_sample + 0.5 * pred_log_var)

    # Terrain difficulty loss
    # Define difficulty as normalized error (high error = difficult)
    actual_errors = torch.sqrt(torch.sum(squared_error, dim=1, keepdim=True))
    # Normalize to [0, 1] range (assume max error ~0.3 in normalized coords)
    difficulty_labels = torch.clamp(actual_errors / 0.3, 0, 1)

    difficulty_loss = nn.functional.binary_cross_entropy(
        pred_difficulty, difficulty_labels
    )

    total_loss = (coord_weight * coord_loss +
                 uncertainty_weight * uncertainty_loss +
                 difficulty_weight * difficulty_loss)

    return total_loss, coord_loss, uncertainty_loss, difficulty_loss


# Quick test to verify architectures work
if __name__ == "__main__":
    print("Testing improved uncertainty architectures...")

    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)

    print("\n1. Anisotropic Uncertainty Model:")
    model1 = BasicModelWithAnisotropicUncertainty()
    coords, log_vars = model1(dummy_input)
    print(f"   Coords: {coords.shape} (x, y for each sample)")
    print(f"   Log vars: {log_vars.shape} (separate for x and y)")
    print(f"   ✅ Works!")

    print("\n2. Deep Uncertainty Head Model:")
    model2 = BasicModelWithDeepUncertaintyHead()
    coords, log_var = model2(dummy_input)
    print(f"   Coords: {coords.shape}")
    print(f"   Log var: {log_var.shape}")
    print(f"   ✅ Works!")

    print("\n3. Terrain Difficulty Model:")
    model3 = BasicModelWithTerrainDifficulty()
    coords, log_var, difficulty = model3(dummy_input)
    print(f"   Coords: {coords.shape}")
    print(f"   Log var: {log_var.shape}")
    print(f"   Difficulty: {difficulty.shape} (terrain difficulty score)")
    print(f"   ✅ Works!")

    print("\n4. Attention Uncertainty Model:")
    model4 = BasicModelWithAttentionUncertainty()
    coords, log_var = model4(dummy_input)
    print(f"   Coords: {coords.shape}")
    print(f"   Log var: {log_var.shape}")
    print(f"   ✅ Works!")

    print("\n✅ All architectures functional!")
    print("\nRecommended order to try:")
    print("  1. Anisotropic (separate x/y) - most likely to help")
    print("  2. Terrain difficulty - adds explicit difficulty learning")
    print("  3. Deep head - more capacity for uncertainty")
    print("  4. Attention - most complex, try last")
