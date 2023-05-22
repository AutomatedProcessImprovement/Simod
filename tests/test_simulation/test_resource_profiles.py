from pix_framework.log_ids import DEFAULT_CSV_IDS

from simod.event_log.utilities import read
from simod.simulation.parameters.resource_profiles import discover_undifferentiated_resource_profile


def test_resource_profiles_undifferentiated(entry_point):
    log_path = entry_point / 'LoanApp_sequential_9-5_diffres_timers.csv'

    log_ids = DEFAULT_CSV_IDS

    log, _ = read(log_path, log_ids)

    profile = discover_undifferentiated_resource_profile(
        event_log=log,
        log_ids=log_ids
    )

    assert profile is not None
    assert profile.name == 'UNDIFFERENTIATED_RESOURCE_PROFILE'
    assert len(profile.resources) == log[log_ids.resource].nunique()

    distinct_activities = log[log_ids.activity].unique()
    assert len(profile.resources[0].assigned_tasks) == len(distinct_activities)
