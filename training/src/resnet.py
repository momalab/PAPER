"""
Implementation of ResNet architectures for image classification.

Reference:
    He, K., Zhang, X., Ren, S., & Sun, J. (2016).
    "Deep Residual Learning for Image Recognition."
    Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 770-778.
    https://doi.org/10.1109/CVPR.2016.90
"""

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """
    Basic residual block used in smaller ResNets (e.g., ResNet-18, ResNet-20, ResNet-32).
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        """
        Args:
            in_planes (int): Number of input channels.
            planes (int): Number of output channels before expansion.
            stride (int, optional): Stride for the first convolution. Default: 1.
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut_conv = nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            self.shortcut_bn = nn.BatchNorm2d(self.expansion * planes)
        else:
            self.shortcut_conv = None
            self.shortcut_bn = None

    def forward(self, x):
        """
        Forward pass of the basic block.

        Args:
            x (torch.Tensor): Input feature map.

        Returns:
            torch.Tensor: Output feature map after residual addition.
        """
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        shortcut = x
        if self.shortcut_conv:
            shortcut = self.shortcut_bn(self.shortcut_conv(x))
        out += shortcut
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, dataset):
        """
        General ResNet architecture.

        Args:
            block (nn.Module): Residual block type (BasicBlock or Bottleneck).
            num_blocks (list[int]): Number of blocks in each of the 4 layers.
            num_classes (int): Number of output classes.
            dataset (str): Dataset identifier used to configure pooling size.
        """
        super(ResNet, self).__init__()
        self.in_planes = 64

        # Initial convolution and batch norm. CIFAR-style ResNets use a 3x3 conv with stride=1 (no maxpool).
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.relu = nn.ReLU()

        # Dataset-dependent average pooling
        if dataset == "tiny":
            pool_size = 8
        else:
            pool_size = 4
        self.avg_pool = nn.AvgPool2d(kernel_size=pool_size, stride=pool_size, padding=0)

        if num_blocks[3] == 0:  # For ResNet-20 and ResNet-32
            self.linear = nn.Linear(256 * 2 * 2, num_classes)
        else: # For standard ResNets
            self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        """
        Construct a ResNet layer with stacked residual blocks.

        Args:
            block (nn.Module): Residual block type.
            planes (int): Number of channels in the blocks.
            num_blocks (int): Number of blocks in this layer.
            stride (int): Stride for the first block.

        Returns:
            nn.Sequential: Sequential container of residual blocks.
        """
        if num_blocks == 0:
            return nn.Sequential()
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the ResNet model.

        Args:
            x (torch.Tensor): Input tensor (image batch).

        Returns:
            torch.Tensor: Class logits.
        """
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, start_dim=1)
        out = self.linear(out)
        return out

# Factory functions for ResNet variants

def resnet18(num_classes, dataset):
    """Construct ResNet-18."""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, dataset)


def resnet20(num_classes, dataset):
    """Construct ResNet-20."""
    return ResNet(BasicBlock, [3, 3, 3, 0], num_classes, dataset)


def resnet32(num_classes, dataset):
    """Construct ResNet-32."""
    return ResNet(BasicBlock, [5, 5, 5, 0], num_classes, dataset)
