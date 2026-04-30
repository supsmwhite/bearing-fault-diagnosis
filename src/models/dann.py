"""Define DANN with a feature extractor, label classifier, and domain classifier."""

import torch
from torch import nn

from src.models.gradient_reversal import GradientReversalLayer


class DANN(nn.Module):
    def __init__(self, num_classes: int = 10, grl_lambda: float = 1.0):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.label_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )
        self.gradient_reversal = GradientReversalLayer(lambda_value=grl_lambda)
        self.domain_classifier = nn.Sequential(
            self.gradient_reversal,
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def set_grl_lambda(self, lambda_value: float) -> None:
        self.gradient_reversal.set_lambda(lambda_value)

    def extract_features(self, x):
        return self.feature_extractor(x)

    def forward(self, x: torch.Tensor):
        features = self.extract_features(x)
        class_logits = self.label_classifier(features)
        domain_logits = self.domain_classifier(features)
        return class_logits, domain_logits
