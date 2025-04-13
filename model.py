import torch
import torch.nn as nn
from torchvision import models

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.model = models.resnet18(pretrained=True)

        # 1채널 입력(mel spectrogram) 대응 - 기존 conv1 교체
        self.model.conv1 = nn.Conv2d(
            in_channels=1, out_channels=64,
            kernel_size=7, stride=2, padding=3, bias=False
        )

        # 마지막 fc layer 교체
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)