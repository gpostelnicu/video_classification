import torch
import yaml

from sklearn.metrics import accuracy_score

import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader

from .dataset import read_list_file, VideoFramesDataset
from .decoder import Decoder
from .encoder import ResnetEncoder


def count_params(lst_params: list):
    total_count = 0
    for params in lst_params:
        np = 1
        for s in list(params.size()):
            np *= s
        total_count += np
    return total_count


class Trainer(object):
    def __init__(self, base_dir: str,
                 train_list_file: str, test_list_file: str,
                 config_file: str, target_size=224):
        train_clips, train_labels = read_list_file(train_list_file)
        test_clips, test_labels = read_list_file(test_list_file)

        # Resnet normalization, see https://github.com/pytorch/vision/issues/39
        basic_tranform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_transform = transforms.Compose([
            transforms.RandomCrop(target_size),
            basic_tranform
        ])
        eval_transform = transforms.Compose([
            transforms.Resize(target_size),
            basic_tranform
        ])

        self.train_dataset = VideoFramesDataset(
            base_dir=base_dir, folders=train_clips, labels=train_labels, transform=train_transform)
        self.test_dataset = VideoFramesDataset(
            base_dir=base_dir, folders=test_clips, labels=test_labels, transform=eval_transform)

        self._load_config(config_file)

        self.encoder = ResnetEncoder(self.encoding_hidden_sizes)
        self.decoder = Decoder(
            input_dim=self.encoding_hidden_sizes[-1], hidden_dim=self.decoder_hidden_dim,
            num_hidden_layers=self.decoder_num_hidden_layers, fc_dim=self.decoder_fc_dim,
            out_dim=self.num_labels
        )
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _load_config(self, config_file):
        with open(config_file) as stream:
            config = yaml.safe_load(stream)
            self.learning_rate = config.learning_rate
            self.batch_size = config.batch_size
            self.encoding_hidden_sizes = self.encoding_hidden_sizes

            self.decoder_hidden_dim = config.decoder_hidden_dim
            self.decoder_num_hidden_layers = config.decoder_num_hidden_layers
            self.decoder_fc_dim = config.decoder_fc_dim
            self.num_labels = config.num_labels


    def accuracy(self, dataset, num_workers=4):
        self.encoder.eval()
        self.decoder.eval()
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        expected = []
        predicted = []
        for data in enumerate(data_loader):
            clips, labels = data
            clips.to_(self.device)

            output = self.decoder(self.encoder(clips))
            pred_labels = output.max(1, keepdims=True)[1]

            expected.extend(labels)
            predicted.extend(pred_labels)

        expected = torch.stack(expected, dim=0)
        predicted = torch.stack(predicted, dim=0)
        score = accuracy_score(expected.cpu().squeeze().numpy(), predicted.cpu().squeeze().numpy())
        return score

    def train(self, num_epochs, num_workers=4, print_every_n=200):
        encoder_params = list(self.encoder.parameters())
        print('Number of encoder params: {}'.format(count_params(encoder_params)))
        decoder_params = list(self.decoder.parameters())
        print('Number of decoder params: {}'.format(count_params(decoder_params)))

        optimizer = torch.optim.Adam(
            encoder_params + decoder_params,
            lr=self.learning_rate)

        criterion = nn.CrossEntropyLoss()

        train_data_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                       shuffle=True, num_workers=num_workers)

        running_loss = 0.0
        for epoch in range(num_epochs):
            # Set models in training mode - for batch norm or dropout.
            self.encoder.train()
            self.decoder.train()

            for i, data in enumerate(train_data_loader):
                clips, labels = data
                clips.to_(self.device)
                labels.to_(self.device)

                optimizer.zero_grad()

                pred_labels = self.decoder(self.encoder(clips))
                loss = criterion(pred_labels, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if (i + 1) % print_every_n == 0:
                    print('epoch {}, step {}: loss {}'.format(epoch, i, running_loss))
                    running_loss = 0.0

            print('Computing accuracy')
            accuracy = self.accuracy(self.test_dataset, num_workers)
            print('Test accuracy: {}'.format(accuracy))





