import torch
import torch.optim as optim
import torchvision.utils as vutils
from torchvision import models
import torchvision.transforms as T
import torch.nn as nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.leaky_relu(out)
        return out


class AdvancedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(AdvancedResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(identity)
        out = self.relu(out)
        return out


class DilatedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=2):
        super(DilatedResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.leaky_relu(out)
        return out


class ConvEncoderV3(nn.Module):
    def __init__(self, embedding_dim=1500):
        super(ConvEncoderV3, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.res_block1 = ResidualBlock(64, 128)
        self.res_block2 = ResidualBlock(128, 256)
        self.res_block3 = DilatedResidualBlock(256, 512, dilation=2)
        self.res_block4 = DilatedResidualBlock(512, 1024, dilation=4)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, embedding_dim)

    def forward(self, x):
        x = self.initial(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ConvEncoderV4(nn.Module):
    def __init__(self, embedding_dim=512):
        super(ConvEncoderV4, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.res_block1 = AdvancedResidualBlock(64, 128)
        self.res_block2 = AdvancedResidualBlock(128, 256)
        self.res_block3 = AdvancedResidualBlock(256, 512)
        self.res_block4 = AdvancedResidualBlock(512, 1024)  # Additional block

        # Optional: Inception block or attention module can be added here

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, embedding_dim)

    def forward(self, x):
        x = self.initial(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)  # Pass through the additional block
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ConvEncoderV5(nn.Module):
    def __init__(self, embedding_dim=2048):
        super(ConvEncoderV5, self).__init__()
        
        # Convolutional layers
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(32)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(128)

        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(256)

        # Adding a linear layer for the embedding
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256 * 16 * 16, embedding_dim)  # Adjust the input features

    def forward(self, x):
        # Forward pass through the network
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))

        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(x)))

        # Flatten and pass through the linear layer
        x = self.flatten(x)
        x = self.fc(x)

        return x


"""Simple CNN V2"""
class ResNetBlock(nn.Module):
    """This one contains more activations"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(identity)
        out = self.relu(out)
        return out


class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttentionBlock, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.shape
        query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)
        key = self.key_conv(x).view(B, -1, H * W)
        value = self.value_conv(x).view(B, -1, H * W)

        attention = self.softmax(torch.bmm(query, key))
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)

        return out + x

class SimpleCNNV2(nn.Module):
    def __init__(self, embedding_dim=1024):
        super(SimpleCNNV2, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.res_block1 = ResNetBlock(64, 128)
        self.attn_block1 = SelfAttentionBlock(128)
        self.res_block2 = ResNetBlock(128, 256)
        self.attn_block2 = SelfAttentionBlock(256)
        self.res_block3 = ResNetBlock(256, 512)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, embedding_dim)

    def forward(self, x):
        x = self.initial(x)
        x = self.res_block1(x)
        x = self.attn_block1(x)
        x = self.res_block2(x)
        x = self.attn_block2(x)
        x = self.res_block3(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SqueezeExcitationBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SqueezeExcitationBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ConvEncoderV6(nn.Module):
    def __init__(self, embedding_dim=1500):
        super(ConvEncoderV6, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.res_block1 = ResidualBlock(64, 128)
        self.attention1 = SqueezeExcitationBlock(128)
        self.res_block2 = ResidualBlock(128, 256)
        self.attention2 = SqueezeExcitationBlock(256)
        self.res_block3 = DilatedResidualBlock(256, 512, dilation=2)
        self.attention3 = SqueezeExcitationBlock(512)
        self.res_block4 = DilatedResidualBlock(512, 1024, dilation=4)
        self.attention4 = SqueezeExcitationBlock(1024)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, embedding_dim)

    def forward(self, x):
        x = self.initial(x)
        x = self.res_block1(x)
        x = self.attention1(x)
        x = self.res_block2(x)
        x = self.attention2(x)
        x = self.res_block3(x)
        x = self.attention3(x)
        x = self.res_block4(x)
        x = self.attention4(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

