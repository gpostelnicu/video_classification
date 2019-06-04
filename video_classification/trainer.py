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
    trainable_count = 0
    for params in lst_params:
        np = 1
        for s in list(params.size()):
            np *= s
        total_count += np
        if params.requires_grad:
            trainable_count += np
    return total_count, trainable_count


class Trainer(object):
    def __init__(self, base_dir: str,
                 train_list_file: str, test_list_file: str,
                 config_file: str, target_size=224, num_frames=29):
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
            transforms.Resize([target_size, target_size]),
            basic_tranform
        ])

        self.train_dataset = VideoFramesDataset(
            base_dir=base_dir, folders=train_clips, labels=train_labels,
            num_frames=num_frames, transform=train_transform)
        self.test_dataset = VideoFramesDataset(
            base_dir=base_dir, folders=test_clips, labels=test_labels,
            num_frames=num_frames, transform=eval_transform)

        self._load_config(config_file)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.encoder = ResnetEncoder(self.encoder_fc_dim1, self.encoder_fc_dim2, self.encoder_out_dim).to(self.device)
        self.decoder = Decoder(
            input_dim=self.encoder_out_dim, hidden_dim=self.decoder_hidden_dim,
            num_hidden_layers=self.decoder_num_hidden_layers, fc_dim=self.decoder_fc_dim,
            out_dim=self.num_labels
        ).to(self.device)

    def _load_config(self, config_file):
        with open(config_file) as stream:
            config = yaml.safe_load(stream)
            self.learning_rate = float(config['learning_rate'])  # yaml doesn't seem to parse scientific notation.
            self.batch_size = config['batch_size']
            self.encoder_fc_dim1 = config['encoder_fc_dim1']
            self.encoder_fc_dim2 = config['encoder_fc_dim2']
            self.encoder_out_dim = config['encoder_out_dim']

            self.decoder_hidden_dim = config['decoder_hidden_dim']
            self.decoder_num_hidden_layers = config['decoder_num_hidden_layers']
            self.decoder_fc_dim = config['decoder_fc_dim']
            self.num_labels = config['num_labels']


    def accuracy(self, dataset, num_workers=0):
        self.encoder.eval()
        self.decoder.eval()
        data_loader = DataLoader(dataset, batch_size=self.batch_size,
                                 shuffle=False, num_workers=num_workers,
                                 pin_memory=True)

        expected = []
        predicted = []
        for i, data in enumerate(data_loader):
            clips, labels = data
            clips = clips.to(self.device)

            output = self.decoder(self.encoder(clips))
            pred_labels = output.max(1)[1]

            expected.extend(labels)
            predicted.extend(pred_labels)

        expected = torch.stack(expected, dim=0)
        predicted = torch.stack(predicted, dim=0)
        score = accuracy_score(expected.cpu().squeeze().numpy(), predicted.cpu().squeeze().numpy())
        return score

    def train(self, save_prefix, num_epochs, num_workers=0, print_every_n=200):
        encoder_params = list(self.encoder.parameters())
        print('Number of encoder total/trainable params: {}'.format(count_params(encoder_params)))
        decoder_params = list(self.decoder.parameters())
        print('Number of decoder total/trainable params: {}'.format(count_params(decoder_params)))

        optimizer = torch.optim.Adam(encoder_params + decoder_params,
                                     lr=self.learning_rate)

        criterion = nn.CrossEntropyLoss()

        train_data_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                       shuffle=True, num_workers=num_workers, pin_memory=True)

        best = -1.
        for epoch in range(num_epochs):
            running_loss = 0.0
            count = 0
            # Set models in training mode - for batch norm or dropout.
            self.encoder.train()
            self.decoder.train()

            for i, data in enumerate(train_data_loader):
                clips, labels = data
                # Batchnorm fails for a minibatch of 1: https://github.com/pytorch/pytorch/issues/4534
                if clips.size(0) < 2:
                    print('Encountered minibatch of size 1. Skipping.')
                    continue
                clips = clips.to(self.device)
                labels = labels.to(self.device).view(-1,)

                optimizer.zero_grad()

                encoded = self.encoder(clips)
                pred_labels = self.decoder(encoded)

                loss = criterion(pred_labels, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                count += 1
                if (i + 1) % print_every_n == 0:
                    print('epoch {}, step {}: loss {}'.format(epoch, i, running_loss / count))
                    running_loss = 0.0
                    count = 0

            print('Computing accuracy')
            with torch.no_grad():
                accuracy = self.accuracy(self.test_dataset, num_workers)
            print('Test accuracy: {}'.format(accuracy))
            if accuracy > best:
                print('Saving')
                torch.save(self.encoder.state_dict(), '{}_encoder.pth'.format(save_prefix))
                torch.save(self.decoder.state_dict(), '{}_decoder.pth'.format(save_prefix))





