from simod.event_log_processing.reader import EventLogReader
from simod.event_log_processing.event_log_ids import EventLogIDs
from simod.process_structure.simulation_parameters.resource_profiles import ResourceProfile


def test_resource_profiles_undifferentiated(entry_point):
    log_path = entry_point / 'PurchasingExample.xes'
    bpmn_path = entry_point / 'PurchasingExample.bpmn'

    log_reader = EventLogReader(log_path)
    log = log_reader.get_traces_df(include_start_end_events=True)

    log_ids = EventLogIDs(
        case='caseid',
        resource='user',
        activity='task',
        end_time='end_timestamp',
    )

    calendar_id = 'foo'

    profile = ResourceProfile.undifferentiated(log, log_ids, bpmn_path, calendar_id)

    assert profile is not None
    assert profile.name == 'UNDIFFERENTIATED_RESOURCE_PROFILE'
    assert len(profile.resources) == log[log_ids.resource].nunique()

    distinct_activities = log[log_ids.activity].unique()
    distinct_activities = list(filter(lambda x: x.lower() != 'start' and x.lower() != 'end', distinct_activities))
    assert len(profile.resources[0].assigned_tasks) == len(distinct_activities)
