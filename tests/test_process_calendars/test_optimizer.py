import pytest

from simod.configuration import ProjectSettings
from simod.event_log.reader_writer import LogReaderWriter
from simod.process_calendars.optimizer import CalendarOptimizer
from simod.process_calendars.settings import CalendarOptimizationSettings as OptimizerSettings
from simod.utilities import get_project_dir, folder_id

config_str = """
log_path: resources/event_logs/PurchasingExample.xes
mining_alg: sm3
exec_mode: optimizer
repetitions: 1
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
  simulation_repetitions: 1
  gate_management:
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

test_data = [
    {
        'name': 'A',
        'config_data': config_str
    }
]


@pytest.mark.parametrize('test_data', test_data, ids=[test_data['name'] for test_data in test_data])
def test_optimizer(entry_point, test_data):
    """Smoke test to check that the optimizer can be instantiated and run successfully."""

    project_settings = ProjectSettings.from_stream(test_data['config_data'])
    calendar_settings = OptimizerSettings.from_stream(test_data['config_data'])

    log_path = entry_point / 'PurchasingExample.xes'
    model_path = entry_point / 'PurchasingExampleQBP.bpmn'

    project_settings.log_path = log_path
    project_settings.output_dir = get_project_dir() / 'outputs' / folder_id()

    log = LogReaderWriter(log_path)

    optimizer = CalendarOptimizer(project_settings, calendar_settings, log, model_path)
    optimizer.run()
