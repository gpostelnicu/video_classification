import time

import torch
import yaml

import numpy as np
from sklearn.metrics import accuracy_score

import torchvision.transforms as transforms
from torch import nn

from video_classification.logger import FloydLogger
from video_classification.models import ResnetLstm, count_params
from .dataset import VideoFramesDataset, SampledDataset, loader_from_dataset


class Trainer(object):
    def __init__(self, base_dir: str,
                 train_list_file: str, test_list_file: str,
                 config_file: str, target_size=224, num_frames=29):
        self.logger = FloydLogger()
        model_config = self._load_config(config_file)

        # Resnet normalization, see https://github.com/pytorch/vision/issues/39
        basic_tranform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_transform = transforms.Compose([
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(target_size, scale=(.9, 1.)),
            basic_tranform
        ])
        eval_transform = transforms.Compose([
            transforms.Resize([target_size, target_size]),
            basic_tranform
        ])

        self.train_dataset = VideoFramesDataset.from_list(
            base_dir=base_dir, fname=train_list_file,
            max_samples_per_video=self.num_samples_per_folder,
            target_frames=num_frames, transform=train_transform
        )
        print('Train dataset stats')
        self.train_dataset.summary()

        self.test_dataset = VideoFramesDataset.from_list(
            base_dir=base_dir, fname=test_list_file,
            max_samples_per_video=self.num_samples_per_folder,
            target_frames=num_frames, transform=eval_transform
        )
        print('Test dataset stats')
        self.test_dataset.summary()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = ResnetLstm.from_config(model_config)
        self.model.to(self.device)

    def _load_config(self, config_file):
        with open(config_file) as stream:
            config = yaml.safe_load(stream)
            print('Config: {}'.format(config))
            self.learning_rate = float(config['learning_rate'])  # yaml doesn't seem to parse scientific notation.
            self.batch_size = config['batch_size']
            model_config = config['model']

            self.performance_train_max_items = config.get('performance_train_max_items', -1)
            self.num_samples_per_folder = config.get('num_samples_per_folder', 1)
            return model_config

    def peformance(self, dataset, num_workers=0):
        self.model.eval()
        data_loader = loader_from_dataset(dataset=dataset, batch_size=self.batch_size,
                                          shuffle=False, num_workers=num_workers)

        expected = []
        predicted = []
        losses = []
        criterion = nn.CrossEntropyLoss(reduction='none')
        for i, data in enumerate(data_loader):
            clip, labels, weights = data
            clip = clip.to(self.device)
            labels = labels.to(self.device)
            weights = weights.to(self.device)

            output = self.model(clip)
            pred_labels = output.max(1)[1]
            loss = criterion(output, labels.view(-1,))
            loss = loss * weights
            loss = loss.sum()

            expected.extend(labels)
            predicted.extend(pred_labels)
            losses.append(loss.item())

        expected = torch.stack(expected, dim=0)
        predicted = torch.stack(predicted, dim=0)
        score = accuracy_score(expected.cpu().squeeze().numpy(), predicted.cpu().squeeze().numpy())
        loss = np.mean(losses)
        return score, loss

    def train(self, save_fname, num_epochs, num_workers=0, print_every_n=200, max_steps_per_epoch=-1):
        params = list(self.model.parameters())
        print('Total/trainable params: {}'.format(count_params(params)))

        optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss(reduction='none')

        ds = self.train_dataset
        if max_steps_per_epoch > 0:
            training_size = max_steps_per_epoch * self.batch_size
            print('Applying sampling for training dataset size. Size = {}'.format(training_size))
            ds = SampledDataset(self.train_dataset, training_size)

        train_data_loader = loader_from_dataset(dataset=ds, batch_size=self.batch_size,
                                                shuffle=True, num_workers=num_workers)
        sampled_train_ds = SampledDataset(self.train_dataset, self.performance_train_max_items)

        min_loss = 1e9
        for epoch in range(num_epochs):
            print('Epoch {}'.format(epoch))
            running_loss = 0.0
            count = 0
            start_time = time.time()

            self.model.train()  # Set models in training mode - for batch norm or dropout.

            for i, data in enumerate(train_data_loader):
                clip, labels, weights = data
                # Batchnorm fails for a minibatch of 1: https://github.com/pytorch/pytorch/issues/4534
                if labels.size(0) < 2:
                    print('Encountered minibatch of size 1. Skipping.')
                    continue
                clip = clip.to(self.device)
                labels = labels.to(self.device).view(-1,)
                weights = weights.to(self.device)

                # 1. Clear previously set gradients.
                optimizer.zero_grad()

                # 2. Forward pass.
                pred_labels = self.model(clip)
                loss = criterion(pred_labels, labels)
                loss = loss * weights
                loss = loss.sum()

                # 3. Backprop.
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                count += 1
                if (i + 1) % print_every_n == 0:
                    end_time = time.time()
                    delta_seconds = end_time - start_time
                    self.logger.log("running_loss", running_loss / count)
                    print('epoch {}, step {}: loss {}, total time: {} time per sample: {}'.format(
                        epoch, i, running_loss / count, delta_seconds, delta_seconds / (self.batch_size * count)))
                    running_loss = 0.0
                    count = 0
                    start_time = time.time()

            print('Computing model performance.')
            with torch.no_grad():
                train_accuracy, train_loss = self.peformance(sampled_train_ds, num_workers)
                self.logger.log("train_accuracy", train_accuracy, epoch)
                self.logger.log("train_loss", train_loss, epoch)

                test_accuracy, test_loss = self.peformance(self.test_dataset, num_workers)
                self.logger.log("test_accuracy", test_accuracy, epoch)
                self.logger.log("test_loss", test_loss, epoch)

                if test_loss > min_loss:
                    print('Saving')
                    torch.save(self.model.to_dict(include_state=True), save_fname)
                    min_loss = test_loss
