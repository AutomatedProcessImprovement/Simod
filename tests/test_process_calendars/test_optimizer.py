import pytest

from simod.configuration import Configuration
from simod.event_log.column_mapping import STANDARD_COLUMNS
from simod.event_log.reader_writer import LogReaderWriter
from simod.process_calendars.optimizer import CalendarOptimizer
from simod.process_calendars.settings import PipelineSettings, CalendarOptimizationSettings
from simod.utilities import get_project_dir

config_str_A = """
version: 2
common:
  log_path: assets/Production.xes
  exec_mode: optimizer
  repetitions: 1
  simulation: true
  evaluation_metrics: 
    - dl
    - absolute_hourly_emd
    - log_mae
    - mae
preprocessing:
  multitasking: false
structure:
  max_evaluations: 2
  mining_algorithm: sm3
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
calendars:
  max_evaluations: 2
  case_arrival:
    discovery_type: undifferentiated
    granularity: 60
    confidence:
      - 0.01
      - 0.1
    support:
      - 0.01
      - 0.1
    participation: 0.4
  resource_profiles:
    discovery_type: pool
    granularity: 60
    confidence:
      - 0.5
      - 0.85
    support:
      - 0.01 
      - 0.3
    participation: 0.4
"""

config_str_B = """
version: 2
common:
  log_path: assets/Production.xes
  exec_mode: optimizer
  repetitions: 1
  simulation: true
  evaluation_metrics: 
    - dl
    - absolute_hourly_emd
    - log_mae
    - mae
preprocessing:
  multitasking: false
structure:
  max_evaluations: 2
  mining_algorithm: sm3
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
calendars:
  max_evaluations: 2
  case_arrival:
    discovery_type: 24_7
    granularity: 60
    confidence:
      - 0.01
      - 0.1
    support:
      - 0.01
      - 0.1
    participation: 0.4
  resource_profiles:
    discovery_type: pool
    granularity: 60
    confidence:
      - 0.5
      - 0.85
    support:
      - 0.01 
      - 0.3
    participation: 0.4
"""

config_str_C = """
version: 2
common:
  log_path: assets/PurchasingExample.xes
  exec_mode: optimizer
  repetitions: 1
  simulation: true
  evaluation_metrics: 
    - dl
    - absolute_hourly_emd
    - log_mae
    - mae
preprocessing:
  multitasking: false
structure:
  max_evaluations: 2
  mining_algorithm: sm3
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
calendars:
  max_evaluations: 2
  case_arrival:
    discovery_type: 24_7
    granularity: 60
    confidence:
      - 0.01
      - 0.1
    support:
      - 0.01
      - 0.1
    participation: 0.4
  resource_profiles:
    discovery_type: 
      - pool
      - undifferentiated
    granularity: 60
    confidence:
      - 0.5
      - 0.85
    support:
      - 0.01 
      - 0.3
    participation: 0.4
"""

config_str_D = """
version: 2
common:
  log_path: assets/PurchasingExample.xes
  exec_mode: optimizer
  repetitions: 1
  simulation: true
  evaluation_metrics: 
    - dl
    - absolute_hourly_emd
    - log_mae
    - mae
preprocessing:
  multitasking: false
structure:
  max_evaluations: 2
  mining_algorithm: sm3
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
calendars:
  max_evaluations: 2
  case_arrival:
    discovery_type: 24_7
    granularity:
      - 15 
      - 60
    confidence:
      - 0.01
      - 0.1
    support:
      - 0.01
      - 0.1
    participation:
      - 0.1 
      - 0.4
  resource_profiles:
    discovery_type: differentiated
    granularity: 60
    confidence: 0.1
    support: 0.7
    participation: 0.4
"""

test_cases = [
    {
        'name': 'A',
        'config_data': config_str_A
    },
    {
        'name': 'B',
        'config_data': config_str_B
    },
    {
        'name': 'C',
        'config_data': config_str_C
    },
    {
        'name': 'D',
        'config_data': config_str_D
    }
]

base_dir = get_project_dir() / 'outputs'


@pytest.mark.parametrize('test_case', test_cases, ids=[case['name'] for case in test_cases])
def test_optimizer(entry_point, test_case):
    settings = Configuration.from_stream(test_case['config_data'])
    calendar_settings = CalendarOptimizationSettings.from_configuration(settings, base_dir)

    log_path = entry_point / 'PurchasingExample.xes'
    model_path = entry_point / 'PurchasingExampleQBP.bpmn'
    log = LogReaderWriter(log_path, STANDARD_COLUMNS)

    optimizer = CalendarOptimizer(calendar_settings, log, model_path, log_ids=STANDARD_COLUMNS)
    result = optimizer.run()

    assert type(result) is PipelineSettings
