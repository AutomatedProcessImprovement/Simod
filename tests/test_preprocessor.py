from pathlib import Path

import pytest

from simod.configuration import Configuration
from simod.event_log.preprocessor import Preprocessor
from simod.event_log.utilities import read

config_yaml = """
version: 2
common:
  log_path: assets/LoanApp_sequential_9-5_diffres_filtered_no_start_times.csv
  exec_mode: optimizer
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

test_cases = [
    {
        'name': 'A',
        'settings': Configuration.from_stream(config_yaml),
    },
]


@pytest.mark.integration
@pytest.mark.parametrize('test_data', test_cases, ids=[test_data['name'] for test_data in test_cases])
def test_add_start_times(test_data, entry_point):
    settings = test_data['settings']
    settings.common.log_path = entry_point / Path(settings.common.log_path).name

    log, _ = read(settings.common.log_path, settings.common.log_ids)

    preprocessor = Preprocessor(log, settings.common.log_ids)
    log = preprocessor.run()

    assert log[settings.common.log_ids.start_time].isna().sum() == 0
