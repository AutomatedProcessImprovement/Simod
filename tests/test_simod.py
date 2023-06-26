from pathlib import Path

import pytest

from simod.event_log.event_log import EventLog
from simod.settings.simod_settings import SimodSettings
from simod.simod import Simod

config_yaml_A = """
version: 2
common:
  log_path: tests/assets/Insurance_Claims_train.csv
  test_log_path: tests/assets/Insurance_Claims_test.csv
  log_ids:
    case: case_id
    activity: activity
    resource: resource
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
control_flow:
  optimization_metric: dl
  max_evaluations: 3
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
    - discovery
  replace_or_joins:
    - true
    - false
  prioritize_parallelism:
    - true
    - false
resource_model:
  optimization_metric: absolute_hourly_emd
  max_evaluations: 3
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
  num_iterations: 3
"""

test_cases = [
    {
        "name": "Insurance_Claims",
        "settings": SimodSettings.from_stream(config_yaml_A),
    },
]


@pytest.mark.system
@pytest.mark.parametrize("test_data", test_cases, ids=[test_data["name"] for test_data in test_cases])
def test_simod(test_data, entry_point):
    settings: SimodSettings = test_data["settings"]

    settings.common.log_path = (entry_point / Path(settings.common.log_path).name).absolute()

    if settings.common.test_log_path:
        settings.common.test_log_path = (entry_point / Path(settings.common.test_log_path).name).absolute()

    if settings.common.model_path:
        settings.common.model_path = (entry_point / Path(settings.common.model_path).name).absolute()

    event_log = EventLog.from_path(
        path=settings.common.log_path,
        log_ids=settings.common.log_ids,
        process_name=settings.common.log_path.stem,
        test_path=settings.common.test_log_path,
    )
    optimizer = Simod(settings, event_log=event_log)
    optimizer.run()

    assert optimizer._best_bps_model.process_model is not None
    assert optimizer._best_bps_model.resource_model is not None
    assert optimizer._best_bps_model.case_arrival_model is not None
    assert optimizer._best_bps_model.extraneous_delays is not None
