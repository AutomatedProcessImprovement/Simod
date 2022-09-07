import os.path

import pytest

from simod.event_log_processing.reader import EventLogReader
from simod.process_structure.optimizer import StructureOptimizer, Settings

structure_config_sm3 = """
mining_algorithm: sm3
max_eval_s: 2
concurrency:
- 0.0
- 1.0
epsilon:
- 0.0
- 1.0
eta:
- 0.0
- 1.0
gate_management:
- equiprobable
- discovery
or_rep:
- true
- false
and_prior:
- true
- false
"""

structure_optimizer_test_data = [
    {
        'name': 'sm3',
        'config_data': structure_config_sm3
    },
]


@pytest.mark.parametrize('test_data', structure_optimizer_test_data,
                         ids=[test_data['name'] for test_data in structure_optimizer_test_data])
def test_structure_optimizer(entry_point, test_data):
    """Smoke test to check that the structure optimizer can be instantiated and run successfully."""
    settings = Settings.from_stream(test_data['config_data'])
    log_path = entry_point / 'PurchasingExample.xes'
    settings.project_name = os.path.splitext(os.path.basename(log_path))[0]

    log_reader = EventLogReader(log_path)

    optimizer = StructureOptimizer(settings, log_reader)
    optimizer.run()

    assert optimizer.best_parameters is not None
