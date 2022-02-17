import copy
from pathlib import Path

import pytest
import yaml

from simod.configuration import Configuration, config_data_from_yaml
from simod.discoverer import Discoverer

base_config = '''
log_path: inputs/PurchasingExample.xes
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


@pytest.mark.slow
def test_execute_pipeline(entry_point):
    global base_config
    xes_path = Path(entry_point) / 'PurchasingExample.xes'
    csv_path = Path(entry_point) / 'PurchasingExample.csv'
    config_data = yaml.load(base_config, Loader=yaml.FullLoader)

    config_data_1 = copy.deepcopy(config_data)
    config_data_1['log_path'] = str(xes_path)
    config_data_1['multitasking'] = True
    config_data_1['simulator'] = 'custom'

    config_data_2 = copy.deepcopy(config_data)
    config_data_2['log_path'] = str(xes_path)
    config_data_2['multitasking'] = False
    config_data_2['simulator'] = 'custom'

    config_data_3 = copy.deepcopy(config_data)
    config_data_3['log_path'] = str(xes_path)
    config_data_3['multitasking'] = True
    config_data_3['simulator'] = 'bimp'

    config_data_4 = copy.deepcopy(config_data)
    config_data_4['log_path'] = str(xes_path)
    config_data_4['multitasking'] = False
    config_data_4['simulator'] = 'bimp'

    config_data_5 = copy.deepcopy(config_data)
    config_data_5['log_path'] = str(csv_path)
    config_data_5['multitasking'] = False
    config_data_5['simulator'] = 'custom'

    data = [
        config_data_1,
        config_data_2,
        config_data_3,
        config_data_4,
        config_data_5
    ]

    for config_data in data:
        config_data = config_data_from_yaml(config_data)
        config = Configuration(**config_data)
        config.fill_in_derived_fields()
        discoverer = Discoverer(config)
        discoverer.run()
