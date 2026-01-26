"""
Implementation of a VGG16-style convolutional neural network with batch normalization.

The network supports CIFAR-style datasets and Tiny ImageNet, with
dataset-specific pooling behavior to accommodate different input sizes.

Reference:
    Diaa, A. et al. (2024).
    "Fast and Private Inference of Deep Neural Networks by Co-designing Activation Functions."
    33rd USENIX Security Symposium.
    https://www.usenix.org/system/files/sec24summer-prepub-373-diaa.pdf
"""

import torch
import torch.nn as nn


class ModulusNet_vgg16_bn(nn.Module):
    """
    VGG16-style convolutional neural network with batch normalization.
    """
    def __init__(self, num_classes, dataset):
        """
        Args:
            num_classes (int): Number of output classes.
            dataset (str): Dataset name. Used to adjust pooling for Tiny ImageNet.
        """
        super().__init__()
        self.dataset = dataset
        self.pool = nn.AvgPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)

        self.conv7 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)

        self.conv8 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)

        self.conv9 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn9 = nn.BatchNorm2d(512)

        self.conv10 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn10 = nn.BatchNorm2d(512)

        self.conv11 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn11 = nn.BatchNorm2d(512)

        self.conv12 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn12 = nn.BatchNorm2d(512)

        self.conv13 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn13 = nn.BatchNorm2d(512)

        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool(x)
        x = self.relu(x)

        # Block 2
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.pool(x)
        x = self.relu(x)

        # Block 3
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.pool(x)
        x = self.relu(x)

        # Block 4
        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = self.relu(x)
        x = self.conv10(x)
        x = self.bn10(x)
        x = self.pool(x)
        x = self.relu(x)

        # Block 5
        x = self.conv11(x)
        x = self.bn11(x)
        x = self.relu(x)
        x = self.conv12(x)
        x = self.bn12(x)
        x = self.relu(x)
        x = self.conv13(x)
        x = self.bn13(x)
        x = self.pool(x)
        x = self.relu(x)

        # Extra pooling for Tiny ImageNet to match spatial resolution
        if self.dataset == "tiny":
            x = self.pool(x)

        # Flatten and classify
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x
