from pathlib import Path

import yaml

from simod.settings.simod_settings import SimodSettings

settings_5 = """
version: 5
common:
  train_log_path: assets/LoanApp_simplified.csv.gz
  perform_final_evaluation: true
  num_final_evaluations: 1
  evaluation_metrics: 
    - dl
    - absolute_event_distribution
  discover_data_attributes: true
preprocessing:
  multitasking: false
control_flow:
  num_iterations: 2
  mining_algorithm: sm1
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
  num_iterations: 2
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

settings_4 = """
version: 4
common:
  train_log_path: assets/LoanApp_simplified.csv.gz
  perform_final_evaluation: true
  num_final_evaluations: 1
  evaluation_metrics: 
    - dl
    - absolute_event_distribution
  discover_case_attributes: true
preprocessing:
  multitasking: false
control_flow:
  num_iterations: 2
  mining_algorithm: sm1
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
  num_iterations: 2
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


def test_configuration():
    config = yaml.safe_load(settings_5)
    result = SimodSettings.from_yaml(config)

    assert result is not None
    assert_common(config, result)
    assert_preprocessing(config, result)
    assert_control_flow(config, result)
    assert_resource_model(config, result)


def test_configuration_legacy():
    ground_truth = SimodSettings.from_yaml(yaml.safe_load(settings_5))
    legacy = SimodSettings.from_yaml(yaml.safe_load(settings_4))

    assert legacy is not None
    assert ground_truth.to_dict() == legacy.to_dict()


def assert_common(config: dict, result: SimodSettings):
    config_common = config["common"]
    result_common = result.common
    for key in config_common:
        if key in ["train_log_path", "test_log_path"]:
            # path is often modified and expanded internally, so we compare only the last part
            assert Path(config_common[key]).name == result_common.train_log_path.name
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
        if config_control_flow["mining_algorithm"] == "sm1":
            if key in ["epsilon", "eta"]:
                assert tuple(config_control_flow[key]) == result_control_flow.epsilon, f"{key} is not equal"
                continue
        elif config_control_flow["mining_algorithm"] == "sm2" and key in [
            "epsilon",
            "eta",
            "replace_or_joins",
            "prioritize_parallelism",
        ]:
            if config_control_flow["mining_algorithm"] == "sm2":
                assert result_control_flow.epsilon is not None, "epsilon is None for sm2"
                # sm2 doesn't use eta, replace_or_joins, prioritize_parallelism are None
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
