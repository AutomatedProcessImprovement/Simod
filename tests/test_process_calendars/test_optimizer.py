import pytest

from simod.configuration import GateManagement
from simod.event_log.reader_writer import LogReaderWriter
from simod.process_calendars.optimizer import CalendarOptimizer
from simod.process_calendars.settings import CalendarOptimizationSettings as OptimizerSettings, PipelineSettings, \
    ArrivalOptimizationSettings, ResourceOptimizationSettings
from simod.utilities import get_project_dir

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
  max_evaluations: 2
  simulation_repetitions: 1
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

test_data = [
    {
        'name': 'A',
        'config_data': config_str
    }
]


@pytest.mark.parametrize('test_data', test_data, ids=[test_data['name'] for test_data in test_data])
def test_optimizer(entry_point, test_data):
    base_dir = get_project_dir() / 'outputs'
    calendar_settings = OptimizerSettings.from_stream(test_data['config_data'], base_dir=base_dir)
    log_path = entry_point / 'PurchasingExample.xes'
    model_path = entry_point / 'PurchasingExampleQBP.bpmn'
    log = LogReaderWriter(log_path)

    optimizer = CalendarOptimizer(calendar_settings, log, model_path)
    result = optimizer.run()

    assert type(result) is PipelineSettings
    assert type(result.arr_cal_met[1]) is ArrivalOptimizationSettings
    assert type(result.gateway_probabilities) is GateManagement
    assert type(result.res_cal_met[1]) is ResourceOptimizationSettings

    # Testing that the returned result actually has the biggest similarity
    assert result.gateway_probabilities == optimizer.evaluation_measurements['gateway_probabilities'].to_list()[0]
    assert result.rp_similarity == optimizer.evaluation_measurements['rp_similarity'].to_list()[0]
