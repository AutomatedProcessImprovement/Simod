import pytest
from pix_framework.discovery.gateway_probabilities import compute_gateway_probabilities
from pix_framework.log_ids import DEFAULT_CSV_IDS

from simod.event_log.event_log import EventLog
from simod.process_calendars.optimizer import ResourceModelOptimizer
from simod.process_calendars.settings import HyperoptIterationParams
from simod.settings.simod_settings import PROJECT_DIR
from simod.settings.temporal_settings import CalendarType, ResourceModelSettings
from simod.simulation.parameters.BPS_model import BPSModel
from simod.simulation.parameters.case_arrival_model import discover_case_arrival_model
from simod.simulation.prosimos_bpm_graph import BPMNGraph

test_cases = [
    {
        'name': 'A',
        'resource_discovery_method': CalendarType.DIFFERENTIATED_BY_POOL,
    },
    {
        'name': 'C',
        'resource_discovery_method': CalendarType.UNDIFFERENTIATED,
    },
    {
        'name': 'D',
        'resource_discovery_method': CalendarType.DIFFERENTIATED_BY_RESOURCE,
    }
]


@pytest.mark.integration
@pytest.mark.parametrize('test_case', test_cases, ids=[case['name'] for case in test_cases])
def test_optimizer(entry_point, test_case):
    base_dir = PROJECT_DIR / 'outputs'
    # Set up settings
    settings = ResourceModelSettings()
    settings.discovery_type = test_case['resource_discovery_method']
    # Read event log
    log_path = entry_point / 'LoanApp_sequential_9-5_diffres_timers.csv'
    event_log = EventLog.from_path(log_path, DEFAULT_CSV_IDS)
    # Build starting BPS model
    model_path = entry_point / 'LoanApp_sequential_9-5_timers.bpmn'
    bpmn_graph = BPMNGraph.from_bpmn_path(model_path)
    bps_model = BPSModel(
        process_model=model_path,
        gateway_probabilities=compute_gateway_probabilities(
            bpmn_graph=bpmn_graph,
            event_log=event_log.train_validation_partition,
            log_ids=event_log.log_ids
        ),
        case_arrival_model=discover_case_arrival_model(
            event_log=event_log.train_validation_partition,
            log_ids=event_log.log_ids
        )
    )

    optimizer = ResourceModelOptimizer(
        event_log=event_log,
        bps_model=bps_model,
        settings=settings,
        base_directory=base_dir
    )
    result = optimizer.run()

    assert type(result) is HyperoptIterationParams
