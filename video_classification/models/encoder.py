from collections import namedtuple

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


ResnetEncoderConfig = namedtuple(
    'ResnetEncoderConfig',
    'basenet fc1_dim fc2_dim out_dim pretrained'.split())


CONFIG = 'config'
STATE = 'state'


class ResnetEncoder(nn.Module):
    def __init__(self, config: ResnetEncoderConfig):
        super().__init__()

        self.config = config
        resnet = NETS[config.basenet](pretrained=config.pretrained)
        modules = list(resnet.children())[:-1]  # Delete last FC layer.
        self.resnet = nn.Sequential(*modules)
        for p in self.resnet.parameters():  # Only finetuning, disable training for the base net.
            p.requires_grad = False

        # It seems encoding layers inside a list breaks - understand why.
        self.fc1 = nn.Linear(resnet.fc.in_features, config.fc1_dim)
        self.bn1 = nn.BatchNorm1d(config.fc1_dim, momentum=0.1)
        self.fc2 = nn.Linear(config.fc1_dim, config.fc2_dim)
        self.bn2 = nn.BatchNorm1d(config.fc2_dim, momentum=0.1)
        self.fc3 = nn.Linear(config.fc2_dim, config.out_dim)

    @staticmethod
    def from_dict(checkpoint: dict):
        assert CONFIG in checkpoint
        config = ResnetEncoderConfig(**checkpoint[CONFIG])
        encoder = ResnetEncoder(config)
        assert STATE in checkpoint
        encoder.load_state_dict(checkpoint[STATE])
        return encoder

    def to_dict(self, include_state=True):
        dic = {CONFIG: dict(self.config._asdict())}
        if include_state:
            dic[STATE] = self.state_dict()
        return dic

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
