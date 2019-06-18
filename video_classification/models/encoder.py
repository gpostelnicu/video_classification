from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
from torchvision import models

from video_classification.models.saving_module import SavingModule


NETS = {
    'resnet18': models.resnet18,
    'resnet50': models.resnet50,
    'resnet101': models.resnet101,
    'resnet152': models.resnet152
}


ResnetEncoderConfig = namedtuple(
    'ResnetEncoderConfig',
    'basenet fc1_dim fc2_dim fc_dims out_dim pretrained chunk_size trainable_prefixes'.split())
# Make all fields optional.
ResnetEncoderConfig.__new__.__defaults__ = (None,) * len(ResnetEncoderConfig._fields)


class BasicBlock(nn.Module):
    def __init__(self, in_dim, out_dim, use_batch_norm=True, momentum=0.1):
        super().__init__()

        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = None
        if use_batch_norm:
            self.bn = nn.BatchNorm1d(out_dim, momentum=momentum)

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)
        if self.bn:
            x = self.bn(x)
        return x


class ImageEncoder(SavingModule):
    config_cls = ResnetEncoderConfig

    def __init__(self, config: ResnetEncoderConfig):
        super().__init__()
        self.config = config

        basenet = NETS[config.basenet](pretrained=config.pretrained)
        modules = list(basenet.children())[:-1]  # Delete last FC layer.
        self.basenet = nn.Sequential(*modules)

        trainable_prefixes = set()
        if config.trainable_prefixes:
            trainable_prefixes = set(n for n in config.trainable_prefixes)
        for n, p in basenet.named_parameters():  # Only finetuning, disable training for the base net.
            if n.split('.')[0] not in trainable_prefixes:
                print('Setting as non trainable params name {}'.format(n))
                p.requires_grad = False
            else:
                print('Setting as trainable params name {}'.format(n))

        blocks = []
        prev_dim = basenet.fc.in_features
        if config.fc_dims is None:
            fc_dims = [config.fc1_dim, config.fc2_dim]
        else:
            fc_dims = config.fc_dims

        for fc_dim in fc_dims:
            block = BasicBlock(prev_dim, fc_dim)
            prev_dim = fc_dim
            blocks.append(block)

        self.blocks = nn.Sequential(*blocks)
        self.fc_out = nn.Linear(prev_dim, config.out_dim)

    def forward(self, packed_x: PackedSequence):
        """
        forward expects a PackedSequence as input.
        """
        num_chunks = max(1, packed_x.size(0) // self.config.chunk_size)
        chunks = torch.chunk(packed_x, num_chunks, 0)
        out = [self.process(chunk) for chunk in chunks]
        encoded = torch.cat(out, 0)
        return encoded

    def process(self, x):
        x = self.basenet(x)
        x = x.view(x.size(0), -1)

        x = self.blocks(x)

        x = self.fc_out(x)
        return x


class ResnetEncoder(SavingModule):
    config_cls = ResnetEncoderConfig

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
