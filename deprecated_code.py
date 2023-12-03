import timm
import torch
import torch.optim as optim
import torchvision.utils as vutils
from torchvision import models
import torchvision.transforms as T
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from einops.layers.torch import Rearrange
from einops import rearrange

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


"""ViT Encoder"""
class ViTEncoder(nn.Module):
    def __init__(self, image_size=64, patch_size=8, num_patches=64, embedding_dim=1024, num_heads=8, depth=3, mlp_dim=1024):
        super(ViTEncoder, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embedding_dim = embedding_dim

        self.patch_embedding = nn.Sequential(
            nn.Conv2d(3, embedding_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e')
        )

        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches, embedding_dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=mlp_dim),
            num_layers=depth
        )
        self.to_embedding = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        x = self.patch_embedding(x)
        x += self.position_embedding
        x = rearrange(x, 'b n e -> n b e')
        x = self.transformer(x)
        x = rearrange(x, 'n b e -> b n e')
        embedding = self.to_embedding(x.mean(dim=1))
        return embedding

# Example instantiation
encoder = ViTEncoder(embedding_dim=1024)
encoder = encoder.to(device)


"""Transformer Encoder"""
class GELU(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.gelu(input)

class ResNetBlock(nn.Module):
    # Define a basic ResNet block with two convolutional layers and a residual connection
    def __init__(self, channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.norm1 = nn.BatchNorm2d(channels)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += residual
        return self.relu(out)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff_dim, embed_dim)

        # Update the normalized_shape to match embed_dim
        self.norm1 = nn.LayerNorm(embed_dim) 
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x2 = self.norm1(x)
        x = x + self.dropout1(self.self_attn(x2, x2, x2)[0])
        x2 = self.norm2(x)
        x2 = self.linear2(self.dropout(F.gelu((self.linear1(x2)))))
        x = x + self.dropout2(x2)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, input_channels, num_downsamples, num_blocks, num_heads, ff_dim, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim  # Define the embedding dimension attribute
        self.down_blocks = nn.ModuleList()
        self.transformer_blocks = nn.ModuleList()

        for i in range(num_downsamples):
            out_channels = input_channels * 2
            self.down_blocks.append(
                nn.Sequential(
                    nn.Conv2d(input_channels, out_channels, kernel_size=3, stride=2, padding=1),
                    ResNetBlock(out_channels),
                    GELU()
                )
            )
            input_channels = out_channels

            transformer_blocks = nn.ModuleList()
            for _ in range(num_blocks):
                transformer_blocks.append(TransformerEncoderBlock(embed_dim, num_heads, ff_dim))
            self.transformer_blocks.append(transformer_blocks)

        self.final_linear = nn.Linear(input_channels, embed_dim)

    def forward(self, x):
        for down_block, transformer_block_list in zip(self.down_blocks, self.transformer_blocks):
            x = down_block(x)  # Downsample
            B, C, H, W = x.shape

            # Correctly reshape the tensor for the transformer
            x = x.view(B, C, H * W).permute(0, 2, 1)  # Reshape to [batch_size, sequence_length, embedding_dim]
            x = x.flatten(1)  # Flatten the spatial dimensions
            x = x.view(B, -1, self.embed_dim)  # Reshape to [batch_size, sequence_length, embedding_dim]

            for transformer_block in transformer_block_list:
                x = transformer_block(x)

            # Use .reshape() instead of .view() for the final reshape
            x = x.permute(0, 2, 1).reshape(B, C, H, W)  # Reshape back if needed

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)  # Flatten
        embedding = self.final_linear(x)
        return embedding

# Example usage
input_channels = 3  # for RGB images
num_downsamples = 2
num_blocks = 1
num_heads = 8
ff_dim = 1024
embed_dim = 1024

encoder = TransformerEncoder(input_channels, num_downsamples, num_blocks, num_heads, ff_dim, embed_dim)
encoder = encoder.to(device)


"""Convolutional Encoder V2"""
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

class ConvEncoderV2(nn.Module):
    def __init__(self, embedding_dim=256):
        super(ConvEncoderV2, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Increasing the depth and complexity of the network
        self.res_block1 = ResidualBlock(64, 128)
        self.res_block2 = ResidualBlock(128, 256)
        self.res_block3 = ResidualBlock(256, 512)
        
        # Adaptive pooling to ensure a fixed size output regardless of input image size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Linear(512, embedding_dim)

    def forward(self, x):
        x = self.initial(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc(x)
        return x

# Example usage
encoder = ConvEncoderV2(embedding_dim=768)
encoder.load_state_dict(torch.load('conv_encoder_768_60e_train.pth'))
encoder = encoder.to(device)


"""Simple CNN V1"""
class SimpleCNN(nn.Module):
    def __init__(self, embedding_dim=256):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(32)

        # Adaptive pooling layer
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(32, 512)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, embedding_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        # Apply adaptive pooling
        x = self.adaptive_pool(x)

        # Flatten the output
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Example usage
embedding_dim = 1024
encoder = SimpleCNN(embedding_dim=embedding_dim)
encoder = encoder.to(device)


"""Deep Encoder and Decoder (AutoEncoder)"""
class DeepEncoder(nn.Module):
    def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, act_fn: object = nn.GELU):
        """Encoder.

        Args:
           num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
           base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2),  # 32x32 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 16x16 => 8x8
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 8x8 => 4x4
            act_fn(),
            nn.Flatten(),  # Image grid to single feature vector
            nn.Linear(4096, latent_dim),
        )

    def forward(self, x):
        return self.net(x)

encoder = DeepEncoder(num_input_channels=3, base_channel_size = 64, latent_dim = 2048)
encoder = encoder.to(device)