from pathlib import Path

import pytest
from pix_framework.discovery.gateway_probabilities import GatewayProbabilitiesDiscoveryMethod
from pix_framework.input import read_csv_log
from pix_framework.log_ids import EventLogIDs

from simod.settings.temporal_settings import CalendarSettings, CalendarType
from simod.simulation.parameters.miner import mine_parameters

assets_dir = Path(__file__).parent / "assets"


@pytest.mark.parametrize("calendar_type", [
    CalendarType.DEFAULT_9_5,
    CalendarType.DEFAULT_24_7,
    CalendarType.UNDIFFERENTIATED,
    CalendarType.DIFFERENTIATED_BY_POOL,
    CalendarType.DIFFERENTIATED_BY_RESOURCE,
])
def test_mine_parameters(entry_point, calendar_type):
    case_arrival_settings = CalendarSettings.default()
    model_path = entry_point / "LoanApp_sequential_9-5_diffres_filtered.bpmn"

    resource_profile_settings = CalendarSettings.default()
    resource_profile_settings.discovery_type = calendar_type

    log_path = entry_point / "LoanApp_sequential_9-5_diffres_filtered.csv"
    log_ids = EventLogIDs(
        case="case:concept:name",
        activity="concept:name",
        resource="org:resource",
        start_time="start_timestamp",
        end_time="time:timestamp",
    )
    log = read_csv_log(log_path, log_ids)

    result = mine_parameters(
        case_arrival_settings=case_arrival_settings,
        resource_profiles_settings=resource_profile_settings,
        log=log,
        log_ids=log_ids,
        model_path=model_path,
        gateways_probability_method=GatewayProbabilitiesDiscoveryMethod.DISCOVERY
    )

    parameters = result.to_dict()

    calendar_ids_from_profiles = set()
    for profile in parameters['resource_profiles']:
        for resource in profile['resource_list']:
            calendar_ids_from_profiles.add(resource['calendar'])
    calendar_ids = list(map(lambda c: c["id"], parameters['resource_calendars']))

    assert calendar_ids_from_profiles == set(calendar_ids)
