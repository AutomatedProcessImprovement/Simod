import pytest

from simod.configuration import GatewayProbabilitiesDiscoveryMethod, PROJECT_DIR
from simod.process_structure.settings import StructureOptimizationSettings

settings_a = """
structure:
  max_evaluations: 2
  gateway_probabilities:
    - equiprobable
    - discovery
"""

settings_old_gate_management = """
max_evaluations: 2
gate_management:
- equiprobable
- discovery
"""

settings_with_miner_settings = """
structure:
  max_evaluations: 2
  gateway_probabilities:
    - equiprobable
    - discovery
  mining_alg: sm3
  concurrency:
    - 0.0
    - 1.0
  epsilon:
    - 0.0
    - 1.0
  eta:
    - 0.0
    - 1.0
  replace_or_joins:
    - true
    - false
  prioritize_parallelism:
    - true
    - false    
"""

test_cases = [
    {
        'name': 'A',
        'config_data': settings_a,
        'expected_miner_settings': None,
        'structure_output_dir': PROJECT_DIR / 'outputs'
    },
    {
        'name': 'Old Gate Management',
        'config_data': settings_old_gate_management,
        'expected_miner_settings': None,
        'structure_output_dir': PROJECT_DIR / 'outputs'
    },
    {
        'name': 'With Miner Settings',
        'config_data': settings_with_miner_settings,
        'expected_miner_settings': True,
        'structure_output_dir': PROJECT_DIR / 'outputs'
    }
]


@pytest.mark.parametrize('test_data', test_cases, ids=list(map(lambda x: x['name'], test_cases)))
def test_miner_settings(test_data: dict):
    settings = StructureOptimizationSettings.from_stream(
        test_data['config_data'], base_dir=test_data['structure_output_dir'])

    assert settings.max_evaluations == 2
    assert settings.gateway_probabilities_method == [GatewayProbabilitiesDiscoveryMethod.EQUIPROBABLE,
                                                     GatewayProbabilitiesDiscoveryMethod.DISCOVERY]
    if test_data['expected_miner_settings']:
        assert settings.epsilon == [0.0, 1.0]
        assert settings.concurrency == [0.0, 1.0]
        assert settings.eta == [0.0, 1.0]
        assert settings.replace_or_joins == [True, False]
        assert settings.prioritize_parallelism == [True, False]
