import pytest
from pix_framework.input import read_csv_log
from pix_framework.log_ids import APROMORE_LOG_IDS

from simod.simulation.calendar_discovery import case_arrival, resource
from simod.simulation.parameters.calendars import Calendar


@pytest.mark.integration
@pytest.mark.parametrize('log_name', ['DifferentiatedCalendars.csv'])
def test_case_arrival_discover_undifferentiated(entry_point, log_name):
    log_path = entry_point / log_name
    log_ids = APROMORE_LOG_IDS
    # Read event log
    log = read_csv_log(log_path, log_ids)
    # Discover arrival calendar
    result = case_arrival.discover_undifferentiated(log, log_ids)
    # Assert
    assert result
    assert type(result) is Calendar


@pytest.mark.integration
@pytest.mark.parametrize('log_name', ['DifferentiatedCalendars.csv'])
def test_resource_discover_undifferentiated(entry_point, log_name):
    log_path = entry_point / log_name
    log_ids = APROMORE_LOG_IDS
    # Read event log
    log = read_csv_log(log_path, log_ids)
    # Discover arrival calendar
    result = resource.discover_undifferentiated(log, log_ids)

    assert result
    assert type(result) is Calendar
    assert len(result.timetables) > 0


@pytest.mark.integration
@pytest.mark.parametrize('log_name', ['DifferentiatedCalendars.csv'])
def test_resource_discover_per_resource_pool(entry_point, log_name):
    log_path = entry_point / log_name
    log_ids = APROMORE_LOG_IDS
    # Read event log
    log = read_csv_log(log_path, log_ids)
    # Discover arrival calendar
    result = resource.discover_per_resource_pool(log, log_ids)

    assert result
    assert len(result) > 0
    assert len(result[0]) > 0


@pytest.mark.integration
@pytest.mark.parametrize('log_name', ['DifferentiatedCalendars.csv'])
def test_resource_discover_per_resource(entry_point, log_name):
    log_path = entry_point / log_name
    log_ids = APROMORE_LOG_IDS
    # Read event log
    log = read_csv_log(log_path, log_ids)
    # Discover arrival calendar
    result = resource.discover_per_resource(log, log_ids)

    assert result
    assert len(result) > 0
    assert len(result[0].timetables) > 0
