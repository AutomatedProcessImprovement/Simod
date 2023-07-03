from pathlib import Path

import pytest
import yaml

from simod.settings.simod_settings import SimodSettings

settings = """
version: 2
common:
  log_path: assets/LoanApp_simplified.csv
  repetitions: 1
  evaluation_metrics: 
    - dl
    - absolute_hourly_emd
preprocessing:
  multitasking: false
control_flow:
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
resource_model:
  max_evaluations: 2
  discover_prioritization_rules: true
  resource_profiles:
    discovery_type: differentiated_by_pool
    granularity: 60
    confidence:
      - 0.5
      - 0.85
    support:
      - 0.01 
      - 0.3
    participation: 0.4
"""


@pytest.mark.parametrize("test_case", [settings])
def test_configuration(test_case):
    config = yaml.safe_load(test_case)
    result = SimodSettings.from_yaml(config)

    assert result is not None
    assert_common(config, result)
    assert_preprocessing(config, result)
    assert_control_flow(config, result)
    assert_resource_model(config, result)


def assert_common(config: dict, result: SimodSettings):
    config_common = config["common"]
    result_common = result.common
    for key in config_common:
        if key in ["log_path", "test_log_path"]:
            # path is often modified and expanded internally, so we compare only the last part
            assert Path(config_common[key]).name == result_common.log_path.name
            continue
        assert config_common[key] == getattr(result_common, key)


def assert_preprocessing(config: dict, result: SimodSettings):
    config_preprocessing = config["preprocessing"]
    result_preprocessing = result.preprocessing
    for key in config_preprocessing:
        assert config_preprocessing[key] == getattr(result_preprocessing, key)


def assert_control_flow(config: dict, result: SimodSettings):
    config_control_flow = config["control_flow"]
    result_control_flow = result.control_flow
    for key in config_control_flow:
        if config_control_flow["mining_algorithm"] == "sm3":
            if key == "concurrency":
                # sm3 does not use concurrency
                assert result_control_flow.concurrency is None
                continue
            elif key in ["epsilon", "eta"]:
                assert tuple(config_control_flow[key]) == result_control_flow.epsilon, f"{key} is not equal"
                continue
        elif config_control_flow["mining_algorithm"] == "sm2" and key in [
            "concurrency",
            "epsilon",
            "eta",
            "replace_or_joins",
            "prioritize_parallelism",
        ]:
            if key == "concurrency":
                # pair is stored as a tuple internally, so we need to convert it for comparison
                assert tuple(config_control_flow[key]) == result_control_flow.concurrency, f"{key} is not equal"
                continue
            if config_control_flow["mining_algorithm"] == "sm2":
                # sm2 doesn't use epsilon, eta, replace_or_joins, prioritize_parallelism are None
                assert result_control_flow.epsilon is None, "epsilon is not None for sm2"
                assert result_control_flow.eta is None, "eta is not None for sm2"
                assert result_control_flow.replace_or_joins is None, "replace_or_joins is not None for sm2"
                assert result_control_flow.prioritize_parallelism is None, "prioritize_parallelism is not None for sm2"
                continue
        assert config_control_flow[key] == getattr(result_control_flow, key), f"{key} is not equal"


def assert_resource_model(config: dict, result: SimodSettings):
    config_resource_model = config["resource_model"]
    result_resource_model = result.resource_model

    for key in config_resource_model:
        if key == "resource_profiles":
            for key2 in config_resource_model[key]:
                result_resource_model_value = getattr(result_resource_model, key2)
                if isinstance(result_resource_model_value, tuple):
                    result_resource_model_value = list(result_resource_model_value)
                assert config_resource_model[key][key2] == result_resource_model_value, f"{key2} is not equal"
            continue

        assert config_resource_model[key] == getattr(result_resource_model, key), f"{key} is not equal"
