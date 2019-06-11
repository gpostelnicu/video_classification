import torch.nn as nn

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


class ResnetLstm(nn.Module):
    def __init__(self, encoder_basenet, encoder_fc1_dim, encoder_fc2_dim, encoder_out_dim,
                 decoder_hidden_dim, decoder_hidden_num, decoder_fc_dim, decoder_out_dim, pretrained=True):
        super().__init__()

        self.encoder = ResnetEncoder(encoder_basenet, encoder_fc1_dim, encoder_fc2_dim, encoder_out_dim, pretrained)
        self.decoder = Decoder(encoder_out_dim, decoder_hidden_dim, decoder_hidden_num,
                               decoder_fc_dim, decoder_out_dim)

    @staticmethod
    def from_config(config, pretrained=True):
        encoder_basenet = config['encoder_basenet']
        encoder_fc1_dim = config['encoder_fc_dim1']
        encoder_fc2_dim = config['encoder_fc_dim2']
        encoder_out_dim = config['encoder_out_dim']

        decoder_hidden_dim = config['decoder_hidden_dim']
        decoder_hidden_num = config['decoder_num_hidden_layers']
        decoder_fc_dim = config['decoder_fc_dim']

        decoder_out_dim = config['num_labels']

        return ResnetLstm(encoder_basenet=encoder_basenet,
                          encoder_fc1_dim=encoder_fc1_dim, encoder_fc2_dim=encoder_fc2_dim,
                          encoder_out_dim=encoder_out_dim, decoder_hidden_dim=decoder_hidden_dim,
                          decoder_hidden_num=decoder_hidden_num, decoder_out_dim=decoder_out_dim,
                          decoder_fc_dim=decoder_fc_dim, pretrained=pretrained)

    def forward(self, x_3d, x_len):
        encoded = self.encoder(x_3d)
        outputs = self.decoder(encoded, x_len)
        return outputs

