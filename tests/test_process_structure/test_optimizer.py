import os.path

import pytest

from simod.event_log.reader_writer import LogReaderWriter
from simod.process_structure.optimizer import StructureOptimizer
from simod.process_structure.settings import StructureOptimizationSettings, PipelineSettings

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
    settings = StructureOptimizationSettings.from_stream(test_data['config_data'])
    log_path = entry_point / 'PurchasingExample.xes'
    settings.project_name = os.path.splitext(os.path.basename(log_path))[0]

    log_reader = LogReaderWriter(log_path)

    optimizer = StructureOptimizer(settings, log_reader)
    result = optimizer.run()

    assert type(result) is PipelineSettings
    assert result.output_dir is not None
    assert result.output_dir.exists()
    # for sm3
    assert result.and_prior is not None
    assert result.or_rep is not None
    assert result.eta is not None
    assert result.epsilon is not None
    assert result.concurrency is None
