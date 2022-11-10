from pathlib import Path

import pytest

from simod.configuration import Configuration
from simod.optimization.optimizer import Optimizer

config_yaml_A = """
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
    granularity: 
      - 15
      - 60
    confidence:
      - 0.5
      - 0.85
    support:
      - 0.01 
      - 0.3
    participation: 0.4
"""

config_yaml_B = """
version: 2
common:
  log_path: assets/LoanApp_sequential_9-5.xes
  exec_mode: optimizer
  repetitions: 1
  simulation: true
  evaluation_metrics: 
    - dl
    - absolute_hourly_emd
    - day_hour_emd
    - log_mae
    - mae
preprocessing:
  multitasking: false
structure:
  optimization_metric: dl
  max_evaluations: 1
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
  optimization_metric: absolute_hourly_emd
  max_evaluations: 1
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
    discovery_type: undifferentiated
    granularity: 
      - 15
      - 60
    confidence:
      - 0.5
      - 0.85
    support:
      - 0.01 
      - 0.3
    participation: 0.4
"""

test_cases = [
    # {
    #     'name': 'pool',
    #     'settings': Configuration.from_stream(config_yaml_A),
    # },
    {
        'name': 'loan_app_undifferentited',
        'settings': Configuration.from_stream(config_yaml_B),
    },
]


@pytest.mark.parametrize('test_data', test_cases, ids=[test_data['name'] for test_data in test_cases])
def test_optimizer(test_data, entry_point):
    settings: Configuration = test_data['settings']
    settings.common.log_path = entry_point / Path(settings.common.log_path).name
    optimizer = Optimizer(settings)

    optimizer.run()
