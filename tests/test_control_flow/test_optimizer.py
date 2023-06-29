import pandas as pd
import pytest
from hyperopt import STATUS_OK
from pix_framework.discovery.case_arrival import discover_case_arrival_model
from pix_framework.discovery.gateway_probabilities import GatewayProbabilitiesDiscoveryMethod
from pix_framework.discovery.resource_calendars import CalendarDiscoveryParams
from pix_framework.discovery.resource_model import discover_resource_model
from pix_framework.filesystem.file_manager import get_random_folder_id, create_folder
from pix_framework.log_ids import APROMORE_LOG_IDS

from simod.bpm.reader_writer import BPMNReaderWriter
from simod.control_flow.optimizer import ControlFlowOptimizer
from simod.control_flow.settings import HyperoptIterationParams
from simod.event_log.event_log import EventLog
from simod.settings.control_flow_settings import ControlFlowSettings, ProcessModelDiscoveryAlgorithm
from simod.settings.simod_settings import PROJECT_DIR
from simod.simulation.parameters.BPS_model import BPSModel

control_flow_config_sm3 = {
    "max_evaluations": 3,
    "gateway_probabilities": ["equiprobable", "discovery"],
    "mining_algorithm": "sm3",
    "epsilon": (0.2, 0.8),
    "eta": (0.2, 0.8),
    "replace_or_joins": [True, False],
    "prioritize_parallelism": [True, False],
}

control_flow_config_sm2 = {
    "max_evaluations": 3,
    "mining_algorithm": "sm2",
    "concurrency": (0.2, 0.8),
    "gateway_probabilities": ["equiprobable", "discovery"],
}

control_flow_config_model_provided = {
    "max_evaluations": 5,
    "gateway_probabilities": ["equiprobable", "discovery"],
}

control_flow_optimizer_test_data = [
    {
        'name': 'sm3',
        'parameters': control_flow_config_sm3,
        'event_log': "Control_flow_optimization_test.csv",
    },
    {
        'name': 'sm2',
        'parameters': control_flow_config_sm2,
        'event_log': "Control_flow_optimization_test.csv",
    },
]

control_flow_optimizer_model_provided_test_data = [
    {
        'name': 'model_provided',
        'parameters': control_flow_config_model_provided,
        'event_log': "Control_flow_optimization_test.csv",
        'process_model': "Control_flow_optimization_test.bpmn",
    },
]


@pytest.mark.integration
@pytest.mark.parametrize(
    'test_data',
    control_flow_optimizer_test_data,
    ids=[test_data['name'] for test_data in control_flow_optimizer_test_data]
)
def test_control_flow_optimizer(entry_point, test_data):
    base_dir = PROJECT_DIR / 'outputs' / get_random_folder_id(prefix='test_control_flow_optimizer_')
    create_folder(base_dir)
    log_path = entry_point / test_data['event_log']
    event_log = EventLog.from_path(log_path, APROMORE_LOG_IDS)

    case_arrival_model = discover_case_arrival_model(
        event_log.train_validation_partition,
        event_log.log_ids,
    )
    resource_model = discover_resource_model(
        event_log.train_validation_partition,
        event_log.log_ids,
        CalendarDiscoveryParams(),
    )
    bps_model = BPSModel(
        case_arrival_model=case_arrival_model,
        resource_model=resource_model,
    )

    settings = ControlFlowSettings.from_dict(test_data['parameters'])
    optimizer = ControlFlowOptimizer(
        event_log=event_log,
        bps_model=bps_model,
        settings=settings,
        base_directory=base_dir,
    )
    result = optimizer.run()

    # Assert generic result properties and fields
    assert type(result) is HyperoptIterationParams
    assert result.provided_model_path is None
    assert result.output_dir is not None
    assert result.output_dir.exists()
    assert result.gateway_probabilities_method in test_data['parameters']['gateway_probabilities']
    # Assert discovery parameters depending on the algorithm
    if result.mining_algorithm == ProcessModelDiscoveryAlgorithm.SPLIT_MINER_2:
        assert result.concurrency is not None
        assert (
                float(test_data['parameters']['concurrency'][0])
                <= result.concurrency
                <= float(test_data['parameters']['concurrency'][1])
        )
        assert result.prioritize_parallelism is None
        assert result.replace_or_joins is None
        assert result.eta is None
        assert result.epsilon is None
    elif result.mining_algorithm == ProcessModelDiscoveryAlgorithm.SPLIT_MINER_3:
        assert result.prioritize_parallelism is not None
        assert result.replace_or_joins is not None
        assert result.eta is not None
        assert (
                float(test_data['parameters']['eta'][0])
                <= result.eta
                <= float(test_data['parameters']['eta'][1])
        )
        assert result.epsilon is not None
        assert (
                float(test_data['parameters']['epsilon'][0])
                <= result.epsilon
                <= float(test_data['parameters']['epsilon'][1])
        )
        assert result.concurrency is None
    # Assert the discovered model exists and is a BPMN file
    assert optimizer.best_bps_model.process_model is not None
    bpmn_reader = BPMNReaderWriter(optimizer.best_bps_model.process_model)
    activities = bpmn_reader.read_activities()
    assert len(activities) > 0
    # Assert that the returned result actually has the smallest distance
    assert len(optimizer.evaluation_measurements) > 0
    iteration_results = pd.DataFrame(optimizer._bayes_trials.results).sort_values(by="loss", ascending=True)
    assert iteration_results[iteration_results['status'] == STATUS_OK].iloc[0]['output_dir'] == result.output_dir


@pytest.mark.integration
@pytest.mark.parametrize(
    'test_data',
    control_flow_optimizer_model_provided_test_data,
    ids=[test_data['name'] for test_data in control_flow_optimizer_model_provided_test_data])
def test_control_flow_optimizer_model_provided(entry_point, test_data):
    base_dir = PROJECT_DIR / 'outputs' / get_random_folder_id(prefix='test_control_flow_optimizer_')
    create_folder(base_dir)
    log_path = entry_point / test_data['event_log']
    model_path = entry_point / test_data['process_model']
    event_log = EventLog.from_path(log_path, APROMORE_LOG_IDS)

    case_arrival_model = discover_case_arrival_model(
        event_log.train_validation_partition,
        event_log.log_ids,
    )
    resource_model = discover_resource_model(
        event_log.train_validation_partition,
        event_log.log_ids,
        CalendarDiscoveryParams(),
    )
    bps_model = BPSModel(
        process_model=model_path,
        case_arrival_model=case_arrival_model,
        resource_model=resource_model,
    )

    settings = ControlFlowSettings.from_dict(test_data['parameters'])
    optimizer = ControlFlowOptimizer(
        event_log=event_log,
        bps_model=bps_model,
        settings=settings,
        base_directory=base_dir,
    )
    result = optimizer.run()

    # Assert generic result properties and fields
    assert type(result) is HyperoptIterationParams
    assert result.provided_model_path is not None
    assert result.output_dir is not None
    assert result.output_dir.exists()
    # Assert the discovered model exists and is a BPMN file
    assert optimizer.best_bps_model.process_model is not None
    bpmn_reader = BPMNReaderWriter(optimizer.best_bps_model.process_model)
    activities = bpmn_reader.read_activities()
    assert len(activities) > 0
    # Assert that the best gateways probabilities is 'discovery'
    assert result.gateway_probabilities_method == GatewayProbabilitiesDiscoveryMethod.DISCOVERY
    # Assert that the returned result actually has the smallest distance
    assert len(optimizer.evaluation_measurements) > 0
    iteration_results = pd.DataFrame(optimizer._bayes_trials.results).sort_values(by="loss", ascending=True)
    assert iteration_results[iteration_results['status'] == STATUS_OK].iloc[0]['output_dir'] == result.output_dir
