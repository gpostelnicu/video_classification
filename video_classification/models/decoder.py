from collections import namedtuple

import torch.nn as nn
import torch.nn.functional as F

from video_classification.models.saving_module import SavingModule

DecoderConfig = namedtuple(
    'DecoderConfig', 'input_dim hidden_dim num_hidden_layers fc_dim out_dim'.split())


SingleFrameDecoderConfig = namedtuple(
    'SingleFrameDecoderConfig', 'input_dim fc_dim out_dim'.split()
)


class SingleFrameDecoder(SavingModule):
    config_cls = SingleFrameDecoderConfig

    def __init__(self, config: SingleFrameDecoderConfig):
        super().__init__()
        self.config = config

        self.fc = nn.Linear(config.input_dim, config.fc_dim)
        self.fc_out = nn.Linear(config.fc_dim, config.out_dim)

    def forward(self, pack_x):
        unpacked, seq_lens = nn.utils.rnn.pad_packed_sequence(pack_x, batch_first=True)
        x = unpacked[range(unpacked.size(0)), [i - 1 for i in seq_lens.numpy().tolist()], :]  # Last valid time step.
        x = self.fc(x)
        x = F.relu(x)
        x = self.fc_out(x)
        return x


class Decoder(SavingModule):
    config_cls = DecoderConfig

    def __init__(self, config: DecoderConfig):
        super().__init__()

        self.config = config
        self.lstm = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_hidden_layers,
            batch_first=True
        )

        self.fc1 = nn.Linear(config.hidden_dim, config.fc_dim)
        self.fc_out = nn.Linear(config.fc_dim, config.out_dim)

    def forward(self, pack_x):
        lstm_out, _ = self.lstm(pack_x)

        # Unpack the PackedSequence.
        unpacked, seq_lens = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        x = unpacked[range(unpacked.size(0)), [i - 1 for i in seq_lens.numpy().tolist()], :]  # Last valid time step.
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc_out(x)
        return x
