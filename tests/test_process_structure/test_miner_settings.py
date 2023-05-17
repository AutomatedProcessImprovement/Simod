import pytest
from pix_framework.discovery.gateway_probabilities import GatewayProbabilitiesDiscoveryMethod

from simod.settings.control_flow_settings import ControlFlowSettings, ProcessModelDiscoveryAlgorithm

settings_single_values_sm2 = {
    "max_evaluations": 2,
    "gateway_probabilities": "equiprobable",
    "mining_algorithm": "Split Miner 2",
    "concurrency": 0.87,
    "epsilon": 0.45,
    "eta": 0.34,
    "replace_or_joins": True,
    "prioritize_parallelism": True,
}

settings_interval_values_sm2 = {
    "max_evaluations": 10,
    "gateway_probabilities": ["equiprobable", "discovery"],
    "mining_algorithm": "Split Miner 2",
    "concurrency": [0.0, 1.0],
    "epsilon": [0.12, 0.45],
    "eta": [0.34, 0.55],
    "replace_or_joins": [True, False],
    "prioritize_parallelism": [True, False],
}

settings_single_values_sm3 = {
    "max_evaluations": 2,
    "gateway_probabilities": "equiprobable",
    "mining_algorithm": "Split Miner 3",
    "concurrency": 0.87,
    "epsilon": 0.45,
    "eta": 0.34,
    "replace_or_joins": True,
    "prioritize_parallelism": True,
}

settings_interval_values_sm3 = {
    "max_evaluations": 10,
    "gateway_probabilities": ["equiprobable", "discovery"],
    "mining_algorithm": "Split Miner 3",
    "concurrency": [0.0, 1.0],
    "epsilon": [0.12, 0.45],
    "eta": [0.34, 0.55],
    "replace_or_joins": [True, False],
    "prioritize_parallelism": [True, False],
}

test_cases = [
    {
        'name': "Single values SM2",
        'control_flow': settings_single_values_sm2
    },
    {
        'name': "Intervals SM2",
        'control_flow': settings_interval_values_sm2
    },
    {
        'name': "Single values SM3",
        'control_flow': settings_single_values_sm3
    },
    {
        'name': "Intervals SM3",
        'control_flow': settings_interval_values_sm3
    },
]


@pytest.mark.parametrize('test_data', test_cases, ids=list(map(lambda x: x['name'], test_cases)))
def test_miner_settings(test_data: dict):
    settings = ControlFlowSettings.from_dict(test_data['control_flow'])

    if test_data['name'] == "Single values SM2":
        assert settings.max_evaluations == settings_single_values_sm2['max_evaluations']
        assert settings.gateway_probabilities == GatewayProbabilitiesDiscoveryMethod.EQUIPROBABLE
        assert settings.mining_algorithm == ProcessModelDiscoveryAlgorithm.SPLIT_MINER_2
        assert settings.concurrency == settings_single_values_sm2['concurrency']
        assert settings.epsilon is None
        assert settings.eta is None
        assert settings.replace_or_joins is None
        assert settings.prioritize_parallelism is None
    elif test_data['name'] == "Intervals SM2":
        assert settings.max_evaluations == settings_interval_values_sm2['max_evaluations']
        assert settings.gateway_probabilities == [GatewayProbabilitiesDiscoveryMethod.EQUIPROBABLE,
                                                  GatewayProbabilitiesDiscoveryMethod.DISCOVERY]
        assert settings.mining_algorithm == ProcessModelDiscoveryAlgorithm.SPLIT_MINER_2
        assert settings.concurrency == (
            settings_interval_values_sm2['concurrency'][0], settings_interval_values_sm2['concurrency'][1])
        assert settings.epsilon is None
        assert settings.eta is None
        assert settings.replace_or_joins is None
        assert settings.prioritize_parallelism is None
    elif test_data['name'] == "Single values SM3":
        assert settings.max_evaluations == settings_single_values_sm3['max_evaluations']
        assert settings.gateway_probabilities == GatewayProbabilitiesDiscoveryMethod.EQUIPROBABLE
        assert settings.mining_algorithm == ProcessModelDiscoveryAlgorithm.SPLIT_MINER_3
        assert settings.concurrency is None
        assert settings.epsilon == settings_single_values_sm3['epsilon']
        assert settings.eta == settings_single_values_sm3['eta']
        assert settings.replace_or_joins == settings_single_values_sm3['replace_or_joins']
        assert settings.prioritize_parallelism == settings_single_values_sm3['prioritize_parallelism']
    elif test_data['name'] == "Intervals SM3":
        assert settings.max_evaluations == settings_interval_values_sm3['max_evaluations']
        assert settings.gateway_probabilities == [GatewayProbabilitiesDiscoveryMethod.EQUIPROBABLE,
                                                  GatewayProbabilitiesDiscoveryMethod.DISCOVERY]
        assert settings.mining_algorithm == ProcessModelDiscoveryAlgorithm.SPLIT_MINER_3
        assert settings.concurrency is None
        assert settings.epsilon == (
            settings_interval_values_sm3['epsilon'][0], settings_interval_values_sm3['epsilon'][1])
        assert settings.eta == (settings_interval_values_sm3['eta'][0], settings_interval_values_sm3['eta'][1])
        assert settings.replace_or_joins == settings_interval_values_sm3['replace_or_joins']
        assert settings.prioritize_parallelism == settings_interval_values_sm3['prioritize_parallelism']
    else:
        assert False
