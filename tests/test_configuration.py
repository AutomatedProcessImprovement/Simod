import os

from simod.configuration import config_data_from_file, AndPriorORemove, Configuration


def test_AndPriorORemove_default(entry_point):
    config_path = os.path.join(entry_point, 'optimize_debug_config_2.yml')
    config = config_data_from_file(config_path)
    assert config['strc'] is not None

    structure_config = Configuration(**config['strc'])
    assert structure_config.and_prior is AndPriorORemove.FALSE
    assert structure_config.or_rep is AndPriorORemove.FALSE


def test_AndPriorORemove_to_str():
    args = [
        {'input': [AndPriorORemove.TRUE, AndPriorORemove.FALSE], 'expect': ['true', 'false']},
        {'input': [AndPriorORemove.TRUE], 'expect': ['true']},
        {'input': AndPriorORemove.TRUE, 'expect': 'true'},
        {'input': [], 'expect': []},
    ]

    for arg in args:
        result = AndPriorORemove.to_str(arg['input'])
        assert result == arg['expect']
