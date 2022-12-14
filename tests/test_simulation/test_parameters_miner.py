import pytest

from simod.bpm.reader_writer import BPMNReaderWriter
from simod.configuration import GatewayProbabilitiesDiscoveryMethod, PDFMethod
from simod.event_log.column_mapping import EventLogIDs
from simod.event_log.utilities import read
from simod.simulation.parameters.calendars import Calendar
from simod.simulation.parameters.miner import mine_default_24_7


@pytest.mark.parametrize('log_name', ['PurchasingExample.xes'])
def test_mine_default_24_7(entry_point, log_name):
    log_path = entry_point / log_name
    log, log_path_csv = read(log_path)
    log_ids = EventLogIDs(
        case='case:concept:name',
        activity='concept:name',
        resource='org:resource',
        start_time='start_timestamp',
        end_time='time:timestamp'
    )
    model_path = entry_point / 'PurchasingExample.bpmn'
    bpmn_reader = BPMNReaderWriter(model_path)
    process_graph = bpmn_reader.as_graph()
    gateways_probability_type = GatewayProbabilitiesDiscoveryMethod.EQUIPROBABLE

    result = mine_default_24_7(
        log, log_ids, model_path, process_graph, gateways_probability_type)

    assert result is not None
    assert result.resource_calendars is not None
    assert result.task_resource_distributions is not None
    assert result.gateway_branching_probabilities is not None
    assert result.resource_profiles is not None
    assert result.arrival_calendar is not None
    assert result.arrival_distribution is not None

    calendar_24_7 = Calendar.all_day_long()
    assert result.arrival_calendar == calendar_24_7
    assert result.resource_calendars == [calendar_24_7]

    log_path_csv.unlink()
