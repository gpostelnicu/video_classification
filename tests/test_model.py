import yaml

from video_classification.models import ResnetLstm


def test_trainer_init():
    config = yaml.safe_load(open('resources/small_config.yaml'))
    ResnetLstm.from_config(config['model'], pretrained=False)
