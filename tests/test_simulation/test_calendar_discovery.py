import pytest
from pix_framework.calendar.resource_calendar import RCalendar
from pix_framework.discovery.case_arrival import discover_case_arrival_calendar
from pix_framework.input import read_csv_log
from pix_framework.log_ids import APROMORE_LOG_IDS

from simod.simulation.calendar_discovery.resource import discover_undifferentiated, discover_per_resource_pool, \
    discover_per_resource


@pytest.mark.integration
@pytest.mark.parametrize('log_name', ['DifferentiatedCalendars.csv'])
def test_discover_case_arrival_calendar(entry_point, log_name):
    log_path = entry_point / log_name
    log_ids = APROMORE_LOG_IDS
    # Read event log
    log = read_csv_log(log_path, log_ids)
    # Discover arrival calendar
    result = discover_case_arrival_calendar(log, log_ids)
    # Assert
    assert result
    assert type(result) is RCalendar


@pytest.mark.integration
@pytest.mark.parametrize('log_name', ['DifferentiatedCalendars.csv'])
def test_resource_discover_undifferentiated(entry_point, log_name):
    log_path = entry_point / log_name
    log_ids = APROMORE_LOG_IDS
    # Read event log
    log = read_csv_log(log_path, log_ids)
    # Discover arrival calendar
    result = discover_undifferentiated(log, log_ids)

    assert result
    assert type(list(result.items())[0][1]) is RCalendar
    assert len(list(result.items())[0][1].to_json()) > 0


@pytest.mark.integration
@pytest.mark.parametrize('log_name', ['DifferentiatedCalendars.csv'])
def test_resource_discover_per_resource_pool(entry_point, log_name):
    log_path = entry_point / log_name
    log_ids = APROMORE_LOG_IDS
    # Read event log
    log = read_csv_log(log_path, log_ids)
    # Discover arrival calendar
    result = discover_per_resource_pool(log, log_ids)

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
    result = discover_per_resource(log, log_ids)

    assert result
    assert len(result) > 0
    assert len(result['John-000001'].to_json()) > 0
