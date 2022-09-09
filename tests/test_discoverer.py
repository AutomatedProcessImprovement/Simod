import copy
from pathlib import Path

import pytest
import yaml

from simod.configuration import Configuration, config_data_from_yaml, config_data_from_file
from simod.discoverer import Discoverer
from simod.utilities import get_project_dir

base_config = '''
log_path: resources/event_logs/PurchasingExample.xes
mining_alg: sm3
arr_confidence: 9.2
arr_support: 0.098
epsilon: 0.27747592754346484
eta: 0.3475132024591636
gate_management: discovery
res_confidence: 56.03564752634776
res_support: 0.07334543198001255
res_cal_met: discovered
res_dtype: !!str 247
arr_dtype: !!str 247
rp_similarity: 0.8
pdef_method: automatic
simulator: custom
multitasking: false
'''


def test_data():
    global base_config

    config = yaml.load(base_config, Loader=yaml.FullLoader)
    config['log_path'] = get_project_dir() / 'resources/event_logs/PurchasingExample.xes'

    test_case_1 = copy.deepcopy(config)
    test_case_1['multitasking'] = True

    test_case_2 = copy.deepcopy(config)
    test_case_2['multitasking'] = False

    return [
        test_case_1,
        test_case_2
    ]


@pytest.mark.integration
@pytest.mark.parametrize('config_data', test_data(),
                         ids=map(lambda x: f'multitasking={x["multitasking"]}', test_data()))
def test_execute_pipeline(config_data):
    config_data = config_data_from_yaml(config_data)
    config = Configuration(**config_data)
    discoverer = Discoverer(config)
    discoverer.run()


discover_config_files = [
    'discover_without_model_config.yml',
    'discover_with_model_config.yml',
]


@pytest.mark.integration
@pytest.mark.parametrize('path', discover_config_files)
def test_discover(entry_point, path):
    repository_dir = get_project_dir()
    config_path = entry_point / path
    params = {'config_path': repository_dir.joinpath(config_path).absolute()}

    config_data = config_data_from_file(Path(config_path))
    params['log_path'] = repository_dir.joinpath(config_data['log_path']).absolute()
    model_path = config_data.get('model_path', None)
    if model_path:
        params['model_path'] = repository_dir.joinpath(model_path).absolute()
    config_data.update(params)
    config = Configuration(**config_data)

    discoverer = Discoverer(config)
    discoverer.run()
