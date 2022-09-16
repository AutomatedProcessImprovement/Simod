import pytest

from simod.configuration import ProjectSettings
from simod.optimization.optimizer_new import Optimizer
from simod.optimization.settings import OptimizationSettings
from simod.process_calendars.settings import CalendarOptimizationSettings
from simod.process_structure.settings import StructureOptimizationSettings
from simod.utilities import get_project_dir, folder_id

config_yaml_A = """
log_path: assets/PurchasingExample.xes
exec_mode: optimizer
repetitions: 1
simulation: true
sim_metric: dl
multitasking: false
structure_optimizer:
  mining_algorithm: sm3
  max_evaluations: 2
  concurrency:
    - 0.0
    - 1.0
  epsilon:
    - 0.0
    - 1.0
  eta:
    - 0.0
    - 1.0
  gateway_probabilities:
    - equiprobable
    - discovery
  or_rep:
    - true
    - false
  and_prior:
    - true
    - false
time_optimizer:
  max_evaluations: 2
  gateway_probabilities:
    - equiprobable
    - discovery
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
        'project_settings': ProjectSettings.from_stream(config_yaml_A),
        'structure_settings': StructureOptimizationSettings.from_stream(config_yaml_A),
        'calendar_settings': CalendarOptimizationSettings.from_stream(config_yaml_A)
    }
]


@pytest.mark.parametrize('test_data', test_cases, ids=[test_data['name'] for test_data in test_cases])
def test_optimizer(test_data, entry_point):
    settings = OptimizationSettings(
        project_settings=test_data['project_settings'],
        structure_settings=test_data['structure_settings'],
        calendar_settings=test_data['calendar_settings'])

    settings.project_settings.log_path = entry_point / 'PurchasingExample.xes'
    settings.project_settings.output_dir = get_project_dir() / 'outputs' / folder_id()
    settings.project_settings.project_name = 'PurchasingExample'

    optimizer = Optimizer(settings)

    optimizer.run()
