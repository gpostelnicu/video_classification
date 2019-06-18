import yaml

from video_classification.models import get_model_by_name, ResnetLstm


def filter_null_items(dic):
    return {k: v for k, v in dic.items() if v is not None}


def test_model_init():
    config = yaml.safe_load(open('tests/resources/test_config.yaml'))
    ResnetLstm.from_config(config['model'])


def test_model_save_load():
    config = yaml.safe_load(open('tests/resources/test_config.yaml'))
    model = get_model_by_name(config['model'])
    dic = model.to_dict(include_state=False)
    assert filter_null_items(dic['encoder']['config']) == filter_null_items(config['model']['encoder'])
    assert filter_null_items(dic['decoder']['config']) == filter_null_items(config['model']['decoder'])
