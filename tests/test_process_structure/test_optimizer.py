import os.path

import pytest
from pix_utils.log_ids import DEFAULT_XES_IDS

from simod.configuration import PROJECT_DIR
from simod.event_log.event_log import EventLog
from simod.process_structure.optimizer import StructureOptimizer
from simod.process_structure.settings import StructureOptimizationSettings, PipelineSettings

structure_config_sm3 = """
version: 2
common:
  log_path: assets/PurchasingExample.xes
  repetitions: 1
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


@pytest.mark.integration
@pytest.mark.parametrize('test_data', structure_optimizer_test_data,
                         ids=[test_data['name'] for test_data in structure_optimizer_test_data])
def test_structure_optimizer(entry_point, test_data):
    base_dir = PROJECT_DIR / 'outputs'
    settings = StructureOptimizationSettings.from_stream(test_data['config_data'], base_dir=base_dir)
    log_path = entry_point / 'PurchasingExample.xes'
    settings.project_name = os.path.splitext(os.path.basename(log_path))[0]

    event_log = EventLog.from_path(log_path, DEFAULT_XES_IDS)

    optimizer = StructureOptimizer(settings, event_log)
    result, _, _, _ = optimizer.run()

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


@pytest.mark.integration
@pytest.mark.parametrize('test_data', structure_optimizer_test_data,
                         ids=[test_data['name'] for test_data in structure_optimizer_test_data])
def test_structure_optimizer_with_bpmn(entry_point, test_data):
    base_dir = PROJECT_DIR / 'outputs'
    log_path = entry_point / 'LoanApp_sequential_9-5_diffres_filtered.csv'
    model_path = entry_point / 'LoanApp_sequential_9-5_diffres_filtered.bpmn'

    settings = StructureOptimizationSettings.from_stream(test_data['config_data'], base_dir=base_dir)
    settings.project_name = os.path.splitext(os.path.basename(log_path))[0]

    settings.model_path = model_path

    event_log = EventLog.from_path(log_path, DEFAULT_XES_IDS)

    optimizer = StructureOptimizer(settings, event_log)
    result, best_model_path, _, _ = optimizer.run()

    assert result.model_path == best_model_path
    assert best_model_path == model_path
