import yaml

from video_classification.models import get_model_by_name, ResnetLstm


def test_model_init():
    config = yaml.safe_load(open('tests/resources/test_config.yaml'))
    ResnetLstm.from_config(config['model'])


def test_model_save_load():
    config = yaml.safe_load(open('tests/resources/test_config.yaml'))
    model = get_model_by_name(config['model'])
    dic = model.to_dict(include_state=False)
    assert dic['encoder']['config'] == config['model']['encoder']
    assert dic['decoder']['config'] == config['model']['decoder']
