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

    def forward(self, x_seq):
        lstm_out, _ = self.lstm(x_seq)

        x = lstm_out[:, -1, :]  # Take last time step.
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc_out(x)

        return x


