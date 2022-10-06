import pytest
import yaml

from simod.configuration import Configuration

config_yaml_C = """
version: 2
common:
  log_path: assets/Production.xes
  exec_mode: optimizer
  repetitions: 1
  simulation: true
  evaluation_metrics: 
    - dl
    - day_hour_emd
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


@pytest.mark.parametrize('test_case', [config_yaml_C])
def test_configuration(test_case):
    """Smoke test."""

    config = yaml.safe_load(test_case)
    result = Configuration.from_yaml(config)

    assert result is not None
