import tempfile
from pathlib import Path

import pytest

from simod.process_structure.miner import Settings, StructureMiner

structure_config_sm2 = """
mining_algorithm: sm2
max_eval_s: 2
concurrency: 0.5
epsilon: 0.15
eta: 0.87
gate_management: discovery
replace_or_joins: True
prioritize_parallelism: True
"""

structure_config_sm3 = """
mining_algorithm: sm3
max_eval_s: 2
concurrency: 0.5
epsilon: 0.15
eta: 0.87
gate_management: discovery
replace_or_joins: True
prioritize_parallelism: True
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

        StructureMiner(
            settings.mining_algorithm,
            log_path,
            output_path,
            eta=settings.eta,
            epsilon=settings.epsilon,
            concurrency=settings.concurrency,
            replace_or_joins=settings.replace_or_joins,
            prioritize_parallelism=settings.prioritize_parallelism,
        ).run()

        assert output_path.exists()
