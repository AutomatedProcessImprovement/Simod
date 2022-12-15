import pytest

from simod.configuration import Configuration, CalendarType
from simod.event_log.column_mapping import EventLogIDs
from simod.event_log.reader_writer import LogReaderWriter
from simod.process_calendars.optimizer import CalendarOptimizer
from simod.process_calendars.settings import PipelineSettings, CalendarOptimizationSettings
from simod.utilities import get_project_dir

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

base_dir = get_project_dir() / 'outputs'


@pytest.mark.parametrize('test_case', test_cases, ids=[case['name'] for case in test_cases])
def test_optimizer(entry_point, test_case):
    log_ids = EventLogIDs(
        case='case_id',
        activity='Activity',
        resource='Resource',
        start_time='start_time',
        end_time='end_time',
    )

    settings = Configuration.default()
    settings.calendars.resource_profiles.discovery_type = test_case['resource_discovery_method']
    settings.common.log_ids = log_ids

    calendar_settings = CalendarOptimizationSettings.from_configuration(settings, base_dir)

    log_path = entry_point / 'LoanApp_sequential_9-5_timers.csv'
    model_path = entry_point / 'LoanApp_sequential_9-5_timers.bpmn'
    log = LogReaderWriter(log_path, log_ids)

    optimizer = CalendarOptimizer(calendar_settings, log, model_path=model_path, log_ids=log_ids)
    result = optimizer.run()

    assert type(result) is PipelineSettings
