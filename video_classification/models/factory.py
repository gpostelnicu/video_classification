from .resnet_lstm import ResnetLstm


KEY = 'model_class'


MODELS = {
    'resnet_lstm': ResnetLstm,
}


def get_model_by_name(config):
    return MODELS[config[KEY]].from_config(config)
