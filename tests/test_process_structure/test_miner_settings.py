import pytest

from simod.configuration import StructureMiningAlgorithm
from simod.process_structure.miner import Settings

settings_a = """
structure_optimizer:
  mining_algorithm: Split Miner 3
  max_evaluations: 2
  concurrency: 1.0
  epsilon: 1.0
  eta: 1.0
  gateway_probabilities:
    - equiprobable
    - discovery
  replace_or_joins:
    - true
    - false
  prioritize_parallelism:
    - true
    - false
"""

settings_old_gate_management = """
mining_algorithm: sm3
concurrency: 1.0
epsilon: 1.0
eta: 1.0
replace_or_joins:
- true
- false
prioritize_parallelism:
- true
- false
"""

settings_old_mining_algorithm = """
structure_optimizer:
  mining_alg: sm3
  concurrency: 1.0
  epsilon: 1.0
  eta: 1.0
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
        'config_data': settings_a
    },
    {
        'name': 'Old Gate Management',
        'config_data': settings_old_gate_management
    },
    {
        'name': 'Old Mining Algorithm',
        'config_data': settings_old_mining_algorithm
    }
]


@pytest.mark.parametrize('test_data', test_cases, ids=list(map(lambda x: x['name'], test_cases)))
def test_miner_settings(test_data: dict):
    settings = Settings.from_stream(test_data['config_data'])

    assert settings.mining_algorithm == StructureMiningAlgorithm.SPLIT_MINER_3
    assert settings.concurrency == 1.0
    assert settings.epsilon == 1.0
    assert settings.eta == 1.0
