import pytest
from pix_framework.log_ids import DEFAULT_XES_IDS, EventLogIDs

from simod.bpm.reader_writer import BPMNReaderWriter

arguments = [
    {
        'log_path': 'Production.xes',
        'column_names': DEFAULT_XES_IDS
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
