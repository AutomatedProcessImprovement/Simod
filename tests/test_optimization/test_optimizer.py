from pathlib import Path

import pytest

from simod.configuration import Configuration
from simod.optimization.optimizer import Optimizer

config_yaml_A = """
version: 2
common:
  log_path: tests/assets/Production_train.csv
  test_log_path: tests/assets/Production_test.csv
  log_ids:
    case: case_id
    activity: Activity
    resource: Resource
    start_time: start_time
    end_time: end_time
  repetitions: 1
  evaluation_metrics: 
    - dl
    - absolute_hourly_emd
    - cycle_time_emd
    - circadian_emd
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
  replace_or_joins:
    - true
    - false
  prioritize_parallelism:
    - true
    - false
calendars:
  optimization_metric: absolute_hourly_emd
  max_evaluations: 1
  resource_profiles:
    discovery_type: differentiated
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
extraneous_activity_delays:
  optimization_metric: absolute_emd
"""

config_yaml_B = """
version: 2
common:
  log_path: tests/assets/LoanApp_sequential_9-5_diffres_timers.csv
  log_ids:
    case: case_id
    activity: Activity
    resource: Resource
    start_time: start_time
    end_time: end_time
  repetitions: 1
  evaluation_metrics: 
    - dl
    - absolute_hourly_emd
    - cycle_time_emd
    - circadian_emd
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
  replace_or_joins:
    - true
    - false
  prioritize_parallelism:
    - true
    - false
calendars:
  optimization_metric: absolute_hourly_emd
  max_evaluations: 1
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

config_yaml_C = """
version: 2
common:
  log_path: tests/assets/LoanApp_sequential_9-5_diffres_filtered.csv
  model_path: tests/assets/LoanApp_sequential_9-5_diffres_filtered.bpmn
  repetitions: 1
  evaluation_metrics: 
    - dl
    - absolute_hourly_emd
    - cycle_time_emd
    - circadian_emd
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
  replace_or_joins:
    - true
    - false
  prioritize_parallelism:
    - true
    - false
calendars:
  optimization_metric: absolute_hourly_emd
  max_evaluations: 1
  resource_profiles:
    discovery_type: differentiated
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

config_yaml_D = """
version: 2
common:
  log_path: tests/assets/LoanApp_sequential_9-5_diffres_filtered.csv
  repetitions: 2
  evaluation_metrics: 
    - dl
    - absolute_hourly_emd
    - cycle_time_emd
    - circadian_emd
preprocessing:
  multitasking: false
structure:
  optimization_metric: dl
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
  replace_or_joins:
    - true
    - false
  prioritize_parallelism:
    - true
    - false
calendars:
  optimization_metric: absolute_hourly_emd
  max_evaluations: 2
  resource_profiles:
    discovery_type: differentiated
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
extraneous_activity_delays:
  optimization_metric: absolute_emd
"""

test_cases = [
    # {
    #     'name': 'loan_app_undifferentiated',
    #     'settings': Configuration.from_stream(config_yaml_B),
    # },
    {
        'name': 'Production_train',
        'settings': Configuration.from_stream(config_yaml_A),
    },
    # {
    #     'name': 'loan_app_differentiated_with_model',
    #     'settings': Configuration.from_stream(config_yaml_C),
    # },
    # {
    #     'name': 'loan_app_differentiated_with_extraneous_delays',
    #     'settings': Configuration.from_stream(config_yaml_D),
    # },
]


@pytest.mark.system
@pytest.mark.parametrize('test_data', test_cases, ids=[test_data['name'] for test_data in test_cases])
def test_optimizer(test_data, entry_point):
    settings: Configuration = test_data['settings']

    settings.common.log_path = (entry_point / Path(settings.common.log_path).name).absolute()

    if settings.common.test_log_path:
        settings.common.test_log_path = (entry_point / Path(settings.common.test_log_path).name).absolute()

    if settings.common.model_path:
        settings.common.model_path = (entry_point / Path(settings.common.model_path).name).absolute()

    optimizer = Optimizer(settings)
    optimizer.run()
