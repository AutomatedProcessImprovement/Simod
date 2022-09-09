import pytest

from simod.configuration import GateManagement, AndPriorORemove
from simod.process_structure.settings import StructureOptimizationSettings

settings_a = """
structure_optimizer:
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
structure_optimizer:
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
  or_rep:
    - true
    - false
  and_prior:
    - true
    - false    
"""

test_cases = [
    {
        'name': 'A',
        'config_data': settings_a,
        'expected_miner_settings': None
    },
    {
        'name': 'Old Gate Management',
        'config_data': settings_old_gate_management,
        'expected_miner_settings': None
    },
    {
        'name': 'With Miner Settings',
        'config_data': settings_with_miner_settings,
        'expected_miner_settings': True
    }
]


@pytest.mark.parametrize('test_data', test_cases, ids=list(map(lambda x: x['name'], test_cases)))
def test_miner_settings(test_data: dict):
    settings = StructureOptimizationSettings.from_stream(test_data['config_data'])

    assert settings.max_evaluations == 2
    assert settings.gateway_probabilities == [GateManagement.EQUIPROBABLE, GateManagement.DISCOVERY]
    if test_data['expected_miner_settings']:
        assert settings.epsilon == [0.0, 1.0]
        assert settings.concurrency == [0.0, 1.0]
        assert settings.eta == [0.0, 1.0]
        assert settings.or_rep == [AndPriorORemove.TRUE, AndPriorORemove.FALSE]
        assert settings.and_prior == [AndPriorORemove.TRUE, AndPriorORemove.FALSE]
