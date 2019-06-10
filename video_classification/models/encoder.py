import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


NETS = {
    'resnet18': models.resnet18,
    'resnet50': models.resnet50,
    'resnet101': models.resnet101,
    'resnet152': models.resnet152
}

class ResnetEncoder(nn.Module):
    def __init__(self, basenet_name='resnet152', fc_hidden1=512, fc_hidden2=512, out_dim=300, pretrained=True):
        super().__init__()

        resnet = models.resnet152(pretrained=pretrained)
        modules = list(resnet.children())[:-1]  # Delete last FC layer.
        self.resnet = nn.Sequential(*modules)
        for p in self.resnet.parameters():  # Only finetuning, disable training for the base net.
            p.requires_grad = False

        # It seems encoding layers inside a list breaks - understand why.
        self.fc1 = nn.Linear(resnet.fc.in_features, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.1)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.1)
        self.fc3 = nn.Linear(fc_hidden2, out_dim)

    def forward(self, x_3d):
        cnn_seq = []
        for t in range(x_3d.size(1)):
            x = self.resnet(x_3d[:, t, :, :, :])
            x = x.view(x.size(0), -1)

            # Apply FC layers
            x = self.bn1(self.fc1(x))
            x = F.relu(x)
            x = self.bn2(self.fc2(x))
            x = F.relu(x)
            x = self.fc3(x)
            cnn_seq.append(x)

        cnn_seq = torch.stack(cnn_seq, dim=0).transpose_(0, 1)  # (batch, time, latent)
        return cnn_seq
