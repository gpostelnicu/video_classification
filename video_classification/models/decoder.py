from collections import namedtuple

import torch.nn as nn
import torch.nn.functional as F


DecoderConfig = namedtuple(
    'DecoderConfig', 'input_dim hidden_dim num_hidden_layers fc_dim out_dim'.split())


CONFIG = 'config'
STATE = 'state'


class Decoder(nn.Module):
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

    @staticmethod
    def from_dict(checkpoint: dict):
        assert CONFIG in checkpoint
        config = DecoderConfig(**checkpoint[CONFIG])
        decoder = Decoder(config)
        assert STATE in checkpoint
        decoder.load_state_dict(checkpoint[STATE])
        return decoder

    def to_dict(self, include_state=True):
        dic = {CONFIG: dict(self.config._asdict())}
        if include_state:
            dic[STATE] = self.state_dict()
        return dic

    def forward(self, pack_x):
        lstm_out, _ = self.lstm(x)

        # Unpack the PackedSequence.
        unpacked, seq_lens = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        x = unpacked[range(unpacked.size(0)), [i - 1 for i in seq_lens.numpy().tolist()], :]  # Last valid time step.
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc_out(x)
        return x

    def old_forward(self, x_seq, x_lens):
        # First, pack sequence so that padded items do not get shown to the lstm.
        x = nn.utils.rnn.pack_padded_sequence(x_seq, x_lens, batch_first=True)

        # Apply lstm
        lstm_out, _ = self.lstm(x)

        # Undo the packing operation
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        x = unpacked[range(x_seq.size(0)), [i - 1 for i in x_lens], :]  # Take the last valid time step.
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc_out(x)

        return x
