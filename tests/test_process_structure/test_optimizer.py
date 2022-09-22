import os.path

import pytest

from simod.configuration import AndPriorORemove
from simod.event_log.reader_writer import LogReaderWriter
from simod.process_structure.optimizer import StructureOptimizer
from simod.process_structure.settings import StructureOptimizationSettings, PipelineSettings
from simod.utilities import get_project_dir

structure_config_sm3 = """
mining_algorithm: sm3
max_evaluations: 2
concurrency:
- 0.0
- 1.0
epsilon:
- 0.0
- 1.0
eta:
- 0.0
- 1.0
gateway_probabilities:
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
    base_dir = get_project_dir() / 'outputs'
    settings = StructureOptimizationSettings.from_stream(test_data['config_data'], base_dir=base_dir)
    log_path = entry_point / 'PurchasingExample.xes'
    settings.project_name = os.path.splitext(os.path.basename(log_path))[0]

    log_reader = LogReaderWriter(log_path)

    optimizer = StructureOptimizer(settings, log_reader)
    result: PipelineSettings = optimizer.run()

    assert type(result) is PipelineSettings
    assert result.output_dir is not None
    assert result.output_dir.exists()
    # for sm3
    assert result.and_prior is not None
    assert result.or_rep is not None
    assert result.eta is not None
    assert result.epsilon is not None
    assert result.concurrency is None

    # Testing that the returned result actually has the biggest similarity
    assert result.gateway_probabilities == optimizer.evaluation_measurements['gateway_probabilities'].to_list()[0]
    assert result.and_prior == AndPriorORemove.from_str(optimizer.evaluation_measurements['and_prior'].to_list()[0])
    assert result.or_rep == AndPriorORemove.from_str(optimizer.evaluation_measurements['or_rep'].to_list()[0])
    assert result.eta == optimizer.evaluation_measurements['eta'].to_list()[0]
    assert result.epsilon == optimizer.evaluation_measurements['epsilon'].to_list()[0]
