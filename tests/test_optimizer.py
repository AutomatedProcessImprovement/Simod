import pytest
import yaml

from simod.configuration import Configuration, config_data_with_datastructures
from simod.optimization.optimizer import Optimizer

config_yaml = """
log_path: assets/PurchasingExample.xes
structure_mining_algorithm: sm3
exec_mode: optimizer
simulation_repetitions: 1
simulation: true
sim_metric: dl
multitasking: false
structure_optimizer:
  max_eval_s: 2
  concurrency:
    - 0.0
    - 1.0
  epsilon:
    - 0.0
    - 1.0
  eta:
    - 0.0
    - 1.0
  gate_management:
    - equiprobable
    - discovery
  or_rep:
    - true
    - false
  and_prior:
    - true
    - false
time_optimizer:
  max_eval_t: 2
  rp_similarity:
    - 0.5
    - 0.9
  res_dtype:
    - dt247
  arr_dtype:
    - dt247
  res_sup_dis:
    - 0.01
    - 0.3
  res_con_dis:
    - 50
    - 85
  arr_support:
    - 0.01
    - 0.1
  arr_confidence:
    - 1
    - 10
  res_cal_met: pool
"""

test_cases = [
    {
        'name': 'A',
        'config_data': config_yaml
    }
]


@pytest.mark.parametrize('test_data', test_cases, ids=list(map(lambda x: x['name'], test_cases)))
def test_optimizer(test_data: dict):
    config = yaml.load(test_data['config_data'], Loader=yaml.FullLoader)

    structure_settings = config.pop('structure_optimizer')
    time_settings = config.pop('time_optimizer')
    common_settings = config_data_with_datastructures(config)

    common_config = Configuration(**common_settings)

    structure_settings.update(common_settings)
    structure_optimizer_config = Configuration(**structure_settings)

    time_settings.update(common_settings)
    time_optimizer_config = Configuration(**time_settings)

    optimizer = Optimizer({'gl': common_config, 'strc': structure_optimizer_config, 'tm': time_optimizer_config})
    optimizer.run(discover_model=common_config.model_path is None)
