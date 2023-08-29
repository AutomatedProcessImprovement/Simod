import pytest
from pix_framework.discovery.gateway_probabilities import GatewayProbabilitiesDiscoveryMethod

from simod.settings.control_flow_settings import ControlFlowSettings, ProcessModelDiscoveryAlgorithm

settings_single_values_sm2 = {
    "num_iterations": 2,
    "num_evaluations_per_iteration": 3,
    "gateway_probabilities": "equiprobable",
    "mining_algorithm": "Split Miner 2",
    "epsilon": 0.45,
    "eta": 0.34,
    "replace_or_joins": True,
    "prioritize_parallelism": True,
}

settings_interval_values_sm2 = {
    "num_iterations": 10,
    "num_evaluations_per_iteration": 3,
    "gateway_probabilities": ["equiprobable", "discovery"],
    "mining_algorithm": "Split Miner 2",
    "epsilon": [0.12, 0.45],
    "eta": [0.34, 0.55],
    "replace_or_joins": [True, False],
    "prioritize_parallelism": [True, False],
}

settings_single_values_sm1 = {
    "num_iterations": 2,
    "num_evaluations_per_iteration": 3,
    "gateway_probabilities": "equiprobable",
    "mining_algorithm": "Split Miner 1",
    "epsilon": 0.45,
    "eta": 0.34,
    "replace_or_joins": True,
    "prioritize_parallelism": True,
}

settings_interval_values_sm1 = {
    "num_iterations": 10,
    "num_evaluations_per_iteration": 3,
    "gateway_probabilities": ["equiprobable", "discovery"],
    "mining_algorithm": "Split Miner 1",
    "epsilon": [0.12, 0.45],
    "eta": [0.34, 0.55],
    "replace_or_joins": [True, False],
    "prioritize_parallelism": [True, False],
}

test_cases = [
    {"name": "Single values SM2", "control_flow": settings_single_values_sm2},
    {"name": "Intervals SM2", "control_flow": settings_interval_values_sm2},
    {"name": "Single values SM1", "control_flow": settings_single_values_sm1},
    {"name": "Intervals SM1", "control_flow": settings_interval_values_sm1},
]


@pytest.mark.parametrize("test_data", test_cases, ids=list(map(lambda x: x["name"], test_cases)))
def test_control_flow_settings(test_data: dict):
    settings = ControlFlowSettings.from_dict(test_data["control_flow"])

    if test_data["name"] == "Single values SM2":
        assert settings.num_iterations == settings_single_values_sm2["num_iterations"]
        assert settings.num_evaluations_per_iteration == settings_single_values_sm2["num_evaluations_per_iteration"]
        assert settings.gateway_probabilities == GatewayProbabilitiesDiscoveryMethod.EQUIPROBABLE
        assert settings.mining_algorithm == ProcessModelDiscoveryAlgorithm.SPLIT_MINER_V2
        assert settings.epsilon == settings_single_values_sm2["epsilon"]
        assert settings.eta is None
        assert settings.replace_or_joins is None
        assert settings.prioritize_parallelism is None
    elif test_data["name"] == "Intervals SM2":
        assert settings.num_iterations == settings_interval_values_sm2["num_iterations"]
        assert settings.num_evaluations_per_iteration == settings_interval_values_sm2["num_evaluations_per_iteration"]
        assert settings.gateway_probabilities == [
            GatewayProbabilitiesDiscoveryMethod.EQUIPROBABLE,
            GatewayProbabilitiesDiscoveryMethod.DISCOVERY,
        ]
        assert settings.mining_algorithm == ProcessModelDiscoveryAlgorithm.SPLIT_MINER_V2
        assert settings.epsilon == (
            settings_interval_values_sm2["epsilon"][0],
            settings_interval_values_sm2["epsilon"][1],
        )
        assert settings.eta is None
        assert settings.replace_or_joins is None
        assert settings.prioritize_parallelism is None
    elif test_data["name"] == "Single values SM1":
        assert settings.num_iterations == settings_single_values_sm1["num_iterations"]
        assert settings.num_evaluations_per_iteration == settings_single_values_sm1["num_evaluations_per_iteration"]
        assert settings.gateway_probabilities == GatewayProbabilitiesDiscoveryMethod.EQUIPROBABLE
        assert settings.mining_algorithm == ProcessModelDiscoveryAlgorithm.SPLIT_MINER_V1
        assert settings.epsilon == settings_single_values_sm1["epsilon"]
        assert settings.eta == settings_single_values_sm1["eta"]
        assert settings.replace_or_joins == settings_single_values_sm1["replace_or_joins"]
        assert settings.prioritize_parallelism == settings_single_values_sm1["prioritize_parallelism"]
    elif test_data["name"] == "Intervals SM1":
        assert settings.num_iterations == settings_interval_values_sm1["num_iterations"]
        assert settings.num_evaluations_per_iteration == settings_interval_values_sm1["num_evaluations_per_iteration"]
        assert settings.gateway_probabilities == [
            GatewayProbabilitiesDiscoveryMethod.EQUIPROBABLE,
            GatewayProbabilitiesDiscoveryMethod.DISCOVERY,
        ]
        assert settings.mining_algorithm == ProcessModelDiscoveryAlgorithm.SPLIT_MINER_V1
        assert settings.epsilon == (
            settings_interval_values_sm1["epsilon"][0],
            settings_interval_values_sm1["epsilon"][1],
        )
        assert settings.eta == (settings_interval_values_sm1["eta"][0], settings_interval_values_sm1["eta"][1])
        assert settings.replace_or_joins == settings_interval_values_sm1["replace_or_joins"]
        assert settings.prioritize_parallelism == settings_interval_values_sm1["prioritize_parallelism"]
    else:
        assert False
