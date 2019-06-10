import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_hidden_layers, fc_dim, out_dim):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_hidden_layers,
            batch_first=True
        )

        self.fc1 = nn.Linear(hidden_dim, fc_dim)
        self.fc_out = nn.Linear(fc_dim, out_dim)

    def forward(self, x_seq, x_lens):
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


