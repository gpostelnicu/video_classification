import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResnetEncoder(nn.Module):
    def __init__(self, hidden=(512, 512, 300)):
        """
        The last value in the hidden list is the output dimension.
        """
        super().__init__()

        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]  # Delete last FC layer.
        for m in modules:
            m.requires_grad = False
        self.resnet = nn.Sequential(*modules)

        self.fc_layers = []
        self.bn_layers = []

        prev_size = resnet.fc.in_features
        for i, size in enumerate(hidden):
            fc = nn.Linear(prev_size, size)
            self.fc_layers.append(fc)
            if i < len(hidden) - 1:  # Add BatchNorm
                bn = nn.BatchNorm1d(size)
                self.bn_layers.append(bn)
            prev_size = size

    def forward(self, x_3d):
        cnn_seq = []
        for t in range(x_3d.size(1)):
            x = self.resnet(x_3d[:, t, :, :, :])
            x = x.view(x.size(0), -1)

            # Apply FC layers
            for i in range(len(self.bn_layers)):
                fc = self.fc_layers[i]
                bn = self.bn_layers[i]
                x = bn(fc(x))
                x = F.relu(x)
            # Apply final FC layer
            x = self.fc_layers[-1](x)
            cnn_seq.append(x)

        cnn_seq = torch.stack(cnn_seq, dim=0).transpose_(0, 1)  # (batch, time, latent)
        return cnn_seq
