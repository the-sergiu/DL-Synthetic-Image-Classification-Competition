from collections import Counter

import torch
import torch.optim as optim
import torchvision.utils as vutils
from torchvision import models
import torchvision.transforms as T
import torch.nn as nn
from torchvision import models
from torchvision.models import efficientnet_b3, convnext_small, mobilenet_v3_small, mobilenet_v3_large, maxvit_t, swin_v2_t, efficientnet_b4, efficientnet_v2_m


class EfficientNetB3(nn.Module):
    def __init__(self, num_classes=100, embedding_dim=4096):
        super(EfficientNetB3, self).__init__()
        self.efficientnet_b3 = efficientnet_b3(weights=None)

        # Modify the classifier layer
        in_features = self.efficientnet_b3.classifier[1].in_features
        self.efficientnet_b3.classifier[1] = nn.Linear(in_features, embedding_dim)

    def forward(self, x):
        x = self.efficientnet_b3(x)
        return x


class ConvNextT(nn.Module):
    def __init__(self, embedding_dim=2048):
        super(ConvNextT, self).__init__()
        self.convnext_small = convnext_small(weights=None)

         # Modify the classifier layer
        in_features = self.convnext_small.classifier[2].in_features
        self.convnext_small.classifier[2] = nn.Linear(in_features, embedding_dim)
        
        # print(self.convnext_tiny)
    def forward(self, x):
        x = self.convnext_small(x)
        return x


class MobileNetV3Small(nn.Module):
    def __init__(self, num_classes=100, embedding_dim=2048):
        super(MobileNetV3Small, self).__init__()
        self.mobilenet = mobilenet_v3_small(weights=None)
        self.mobilenet.classifier[3] = nn.Linear(in_features=self.mobilenet.classifier[3].in_features, 
                                                 out_features=embedding_dim)
        # self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.mobilenet(x)
        # x = self.classifier(x)
        return x


class MobileNetV3Large(nn.Module):
    def __init__(self, num_classes=100, embedding_dim=2048):
        super(MobileNetV3Large, self).__init__()
        self.mobilenet = mobilenet_v3_large(weights=None)
        self.mobilenet.classifier[3] = nn.Linear(in_features=self.mobilenet.classifier[3].in_features, 
                                                 out_features=embedding_dim)
        # self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.mobilenet(x)
        # x = self.classifier(x)
        return x


class MaxViTT(nn.Module):
    def __init__(self):
        super(MaxViTT, self).__init__()
        self.maxvit_t = maxvit_t(weights=None)

    def forward(self, x):
        x = self.maxvit_t(x)
        return x


class SwinV2TEncoder(nn.Module):
    def __init__(self, embedding_dim=4096):
        super(SwinV2TEncoder, self).__init__()
        self.swin_v2_t = swin_v2_t(weights=None)

        # Modify the head of the model
        in_features = self.swin_v2_t.head.in_features
        self.swin_v2_t.head = nn.Linear(in_features, embedding_dim)

    def forward(self, x):
        x = self.swin_v2_t(x)
        return x

class EfficientNetB4(nn.Module):
    def __init__(self, num_classes=100, embedding_dim=2048):
        super(EfficientNetB4, self).__init__()
        self.efficientnet_b4 = efficientnet_b4(weights=None)

        # Modify the classifier layer
        in_features = self.efficientnet_b4.classifier[1].in_features
        self.efficientnet_b4.classifier[1] = nn.Linear(in_features, embedding_dim)

    def forward(self, x):
        x = self.efficientnet_b4(x)
        return x


class EfficientNetV2M(nn.Module):
    def __init__(self, num_classes=100, embedding_dim=2048):
        super(EfficientNetV2M, self).__init__()
        self.efficientnet_v2_m = efficientnet_v2_m(weights=None)

        # Modify the classifier layer
        in_features = self.efficientnet_v2_m.classifier[1].in_features
        self.efficientnet_v2_m.classifier[1] = nn.Linear(in_features, embedding_dim)

    def forward(self, x):
        x = self.efficientnet_v2_m(x)
        return x

def ensemble_majority_voting(models, input_data):
    # Ensure models are in evaluation mode
    for model in models:
        model.eval()

    with torch.no_grad():
        all_predictions = [model(input_data).argmax(dim=1) for model in models]
        # Transpose to get predictions per sample across models
        all_predictions = torch.stack(all_predictions).transpose(0, 1)
        majority_votes = [Counter(sample_predictions.tolist()).most_common(1)[0][0] for sample_predictions in all_predictions]
    return torch.tensor(majority_votes)