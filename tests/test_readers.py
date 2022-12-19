import pytest

from simod.bpm.reader_writer import BPMNReaderWriter
from simod.event_log.column_mapping import STANDARD_COLUMNS, EventLogIDs

arguments = [
    {
        'log_path': 'Production.xes',
        'column_names': STANDARD_COLUMNS
    },
    {
        'log_path': 'PurchasingExampleCustomSim.csv',
        'column_names': EventLogIDs(
            case='CaseID',
            activity='Activity',
            resource='Resource',
            start_time='StartTimestamp',
            end_time='EndTimestamp',
            enabled_time='EnableTimestamp',
        )
    }
]


@pytest.mark.integration
def test_bpmn_reader(entry_point):
    bpmn_path = entry_point / 'PurchasingExample.bpmn'
    bpmn_reader = BPMNReaderWriter(bpmn_path)
    activities = bpmn_reader.read_activities()

    assert len(activities) > 0
