import pytest
from pix_utils.filesystem.file_manager import get_random_folder_id, create_folder
from pix_utils.log_ids import DEFAULT_XES_IDS

from simod.event_log.event_log import EventLog
from simod.process_structure.optimizer import StructureOptimizer
from simod.process_structure.settings import HyperoptIterationParams
from simod.settings.control_flow_settings import ControlFlowSettings
from simod.settings.simod_settings import PROJECT_DIR
from simod.settings.temporal_settings import CalendarSettings
from simod.simulation.parameters.BPS_model import BPSModel
from simod.simulation.parameters.case_arrival_model import discover_case_arrival_model
from simod.simulation.parameters.resource_model import discover_resource_model

structure_config_sm3 = {
    "max_evaluations": 1,
    "mining_algorithm": "sm3",
    "concurrency": [0.0, 1.0],
    "epsilon": [0.0, 1.0],
    "eta": [0.0, 1.0],
    "gateway_probabilities": ["equiprobable", "discovery"],
    "replace_or_joins": [True, False],
    "prioritize_parallelism": [True, False]
}

structure_optimizer_test_data = [
    {
        'name': 'sm3',
        'parameters': structure_config_sm3
    },
]


@pytest.mark.integration
@pytest.mark.parametrize('test_data', structure_optimizer_test_data,
                         ids=[test_data['name'] for test_data in structure_optimizer_test_data])
def test_structure_optimizer(entry_point, test_data):
    base_dir = PROJECT_DIR / 'outputs' / get_random_folder_id(prefix='test_control_flow_optimizer_')
    create_folder(base_dir)
    log_path = entry_point / 'PurchasingExample.xes'
    event_log = EventLog.from_path(log_path, DEFAULT_XES_IDS)

    case_arrival_model = discover_case_arrival_model(
        event_log.train_validation_partition,  # No optimization process here, use train + validation
        event_log.log_ids
    )
    resource_model = discover_resource_model(
        event_log.train_validation_partition,  # No optimization process here, use train + validation
        event_log.log_ids,
        CalendarSettings.default()
    )
    bps_model = BPSModel(
        case_arrival_model=case_arrival_model,
        resource_model=resource_model
    )

    settings = ControlFlowSettings.from_dict(test_data['parameters'])
    optimizer = StructureOptimizer(
        event_log=event_log,
        bps_model=bps_model,
        settings=settings,
        base_directory=base_dir
    )
    result = optimizer.run()

    assert type(result) is HyperoptIterationParams
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
    assert result.gateway_probabilities_method in test_data['parameters']['gateway_probabilities']
    assert test_data['parameters']['eta'][0] < result.eta < test_data['parameters']['eta'][1]
    assert test_data['parameters']['epsilon'][0] < result.epsilon < test_data['parameters']['epsilon'][1]


@pytest.mark.integration
@pytest.mark.parametrize('test_data', structure_optimizer_test_data,
                         ids=[test_data['name'] for test_data in structure_optimizer_test_data])
def test_structure_optimizer_with_bpmn(entry_point, test_data):
    base_dir = PROJECT_DIR / 'outputs'
    log_path = entry_point / 'LoanApp_sequential_9-5_diffres_filtered.csv'
    model_path = entry_point / 'LoanApp_sequential_9-5_diffres_filtered.bpmn'

    settings = ControlFlowSettings.from_dict(test_data['parameters'])

    event_log = EventLog.from_path(log_path, DEFAULT_XES_IDS)

    case_arrival_model = discover_case_arrival_model(
        event_log.train_validation_partition,  # No optimization process here, use train + validation
        event_log.log_ids
    )
    resource_model = discover_resource_model(
        event_log.train_validation_partition,  # No optimization process here, use train + validation
        event_log.log_ids,
        CalendarSettings.default()
    )
    bps_model = BPSModel(
        process_model=model_path,
        case_arrival_model=case_arrival_model,
        resource_model=resource_model
    )

    optimizer = StructureOptimizer(
        event_log=event_log,
        bps_model=bps_model,
        settings=settings,
        base_directory=base_dir
    )
    result = optimizer.run()

    assert result.model_path is not None
