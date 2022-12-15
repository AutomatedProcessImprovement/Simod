import tempfile
from pathlib import Path

import pytest

from simod.process_structure.miner import Settings, StructureMiner

structure_config_sm1 = """
mining_algorithm: sm1
max_eval_s: 2
concurrency: 0.0
epsilon: 0.0
eta: 1.0
gate_management:
- equiprobable
- discovery
replace_or_joins:
- true
- false
prioritize_parallelism:
- true
- false
"""

structure_config_sm2 = """
mining_algorithm: sm2
max_eval_s: 2
concurrency: 0.0
epsilon: 0.0
eta: 1.0
gate_management:
- equiprobable
- discovery
replace_or_joins:
- true
- false
prioritize_parallelism:
- true
- false
"""

structure_config_sm3 = """
mining_algorithm: sm3
max_eval_s: 2
concurrency: 0.0
epsilon: 0.0
eta: 1.0
gate_management:
- equiprobable
- discovery
replace_or_joins:
- true
- false
prioritize_parallelism:
- true
- false
"""

structure_optimizer_test_data = [
    {
        'name': 'Split Miner 2',
        'config_data': structure_config_sm2
    },
    {
        'name': 'Split Miner 3',
        'config_data': structure_config_sm3
    },
]


@pytest.mark.integration
@pytest.mark.parametrize('test_data', structure_optimizer_test_data,
                         ids=[test_data['name'] for test_data in structure_optimizer_test_data])
def test_miner(entry_point, test_data):
    """Smoke test to check that the structure optimizer can be instantiated and run successfully."""
    settings = Settings.from_stream(test_data['config_data'])
    log_path = entry_point / 'PurchasingExample.xes'

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / 'model.bpmn'

        StructureMiner(settings, log_path, output_path)
        assert output_path.exists()
