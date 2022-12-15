from simod.event_log.column_mapping import STANDARD_COLUMNS, EventLogIDs
from simod.event_log.reader_writer import LogReaderWriter
from simod.simulation.parameters.resource_profiles import ResourceProfile


def test_resource_profiles_undifferentiated(entry_point):
    log_path = entry_point / 'LoanApp_sequential_9-5_timers.csv'
    bpmn_path = entry_point / 'LoanApp_sequential_9-5_timers.bpmn'

    log_ids = EventLogIDs(
        case='case_id',
        activity='Activity',
        resource='Resource',
        start_time='start_time',
        end_time='end_time',
    )

    log_reader = LogReaderWriter(log_path, log_ids)
    log = log_reader.get_traces_df()

    calendar_id = 'foo'

    profile = ResourceProfile.undifferentiated(log, log_ids, bpmn_path, calendar_id)

    assert profile is not None
    assert profile.name == 'UNDIFFERENTIATED_RESOURCE_PROFILE'
    assert len(profile.resources) == log[log_ids.resource].nunique()

    distinct_activities = log[log_ids.activity].unique()
    assert len(profile.resources[0].assigned_tasks) == len(distinct_activities)
