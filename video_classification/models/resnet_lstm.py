import torch.nn as nn

from .decoder import Decoder, DecoderConfig
from .encoder import ResnetEncoder, ResnetEncoderConfig, ImageEncoder

ENCODER = 'encoder'
DECODER = 'decoder'


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
    def __init__(self, encoder_config, decoder_config):
        super().__init__()

        self.encoder = ImageEncoder(encoder_config)
        self.decoder = Decoder(decoder_config)

    @staticmethod
    def from_config(config):
        encoder_config = ResnetEncoderConfig(**config[ENCODER])
        decoder_config = DecoderConfig(**config[DECODER])

        return ResnetLstm(encoder_config=encoder_config,
                          decoder_config=decoder_config)

    def to_dict(self, include_state=True):
        return {ENCODER: self.encoder.to_dict(include_state),
                DECODER: self.decoder.to_dict(include_state)}

    def forward(self, packed_x):
        encoded = packed_x._replace(data=self.encoder(packed_x.data))
        outputs = self.decoder(encoded)
        return outputs

