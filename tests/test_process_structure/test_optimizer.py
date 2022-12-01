import os.path

import pytest

from simod.event_log.column_mapping import STANDARD_COLUMNS
from simod.event_log.reader_writer import LogReaderWriter
from simod.process_structure.optimizer import StructureOptimizer
from simod.process_structure.settings import StructureOptimizationSettings, PipelineSettings
from simod.utilities import get_project_dir

structure_config_sm3 = """
version: 2
common:
  log_path: assets/PurchasingExample.xes
  exec_mode: optimizer
  repetitions: 1
  simulation: true
  evaluation_metrics: 
    - dl
    - absolute_hourly_emd
preprocessing:
  multitasking: false
structure:
  max_evaluations: 1
  mining_algorithm: sm3
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
  replace_or_joins:
    - true
    - false
  prioritize_parallelism:
    - true
    - false
calendars:
  max_evaluations: 1
  resource_profiles:
    discovery_type: differentiated
    granularity: 60
    confidence: 0.1
    support: 0.7
    participation: 0.4
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

    log_reader = LogReaderWriter(log_path, STANDARD_COLUMNS)

    optimizer = StructureOptimizer(settings, log_reader, STANDARD_COLUMNS)
    result: PipelineSettings = optimizer.run()

    assert type(result) is PipelineSettings
    assert result.output_dir is not None
    assert result.output_dir.exists()
    # for sm3
    assert result.prioritize_parallelism is not None
    assert result.replace_or_joins is not None
    assert result.eta is not None
    assert result.epsilon is not None
    assert result.concurrency is None

    # Testing that the returned result actually has the biggest similarity
    assert len(optimizer.evaluation_measurements) > 0
    assert result.gateway_probabilities_method == optimizer.evaluation_measurements['gateway_probabilities'].to_list()[
        0]
    assert result.eta == optimizer.evaluation_measurements['eta'].to_list()[0]
    assert result.epsilon == optimizer.evaluation_measurements['epsilon'].to_list()[0]
