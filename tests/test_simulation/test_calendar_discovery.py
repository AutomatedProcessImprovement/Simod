import pytest

from simod.event_log.column_mapping import EventLogIDs
from simod.event_log.utilities import read
from simod.simulation.calendar_discovery import case_arrival, resource
from simod.simulation.parameters.calendars import Calendar


@pytest.mark.integration
@pytest.mark.parametrize('log_name', ['DifferentiatedCalendars.xes'])
def test_case_arrival_discover_undifferentiated(entry_point, log_name):
    log_path = entry_point / log_name
    log, log_path_csv = read(log_path)
    log_ids = EventLogIDs(
        case='case:concept:name',
        activity='concept:name',
        resource='org:resource',
        start_time='start_timestamp',
        end_time='time:timestamp'
    )
    result = case_arrival.discover_undifferentiated(log, log_ids)

    assert result
    assert type(result) is Calendar

    log_path_csv.unlink()


@pytest.mark.integration
@pytest.mark.parametrize('log_name', ['DifferentiatedCalendars.xes'])
def test_resource_discover_undifferentiated(entry_point, log_name):
    log_path = entry_point / log_name
    log, log_path_csv = read(log_path)
    log_ids = EventLogIDs(
        case='case:concept:name',
        activity='concept:name',
        resource='org:resource',
        start_time='start_timestamp',
        end_time='time:timestamp'
    )
    result = resource.discover_undifferentiated(log, log_ids)

    assert result
    assert type(result) is Calendar
    assert len(result.timetables) > 0

    log_path_csv.unlink()


@pytest.mark.integration
@pytest.mark.parametrize('log_name', ['DifferentiatedCalendars.xes'])
def test_resource_discover_per_resource_pool(entry_point, log_name):
    log_path = entry_point / log_name
    log, log_path_csv = read(log_path)
    log_ids = EventLogIDs(
        case='case:concept:name',
        activity='concept:name',
        resource='org:resource',
        start_time='start_timestamp',
        end_time='time:timestamp'
    )
    result = resource.discover_per_resource_pool(log, log_ids)

    assert result
    assert len(result) > 0
    assert len(result[0]) > 0

    log_path_csv.unlink()


@pytest.mark.integration
@pytest.mark.parametrize('log_name', ['DifferentiatedCalendars.xes'])
def test_resource_discover_per_resource(entry_point, log_name):
    log_path = entry_point / log_name
    log, log_path_csv = read(log_path)
    log_ids = EventLogIDs(
        case='case:concept:name',
        activity='concept:name',
        resource='org:resource',
        start_time='start_timestamp',
        end_time='time:timestamp'
    )
    result = resource.discover_per_resource(log, log_ids)

    assert result
    assert len(result) > 0
    assert len(result[0].timetables) > 0

    log_path_csv.unlink()
