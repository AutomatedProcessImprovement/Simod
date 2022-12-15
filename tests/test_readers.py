import pytest

from simod.bpm.reader_writer import BPMNReaderWriter
from simod.event_log.column_mapping import STANDARD_COLUMNS, EventLogIDs
from simod.event_log.reader_writer import LogReaderWriter

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
@pytest.mark.parametrize('arg', arguments, ids=map(lambda x: x['log_path'], arguments))
def test_logreader_new(entry_point, arg):
    log_path = entry_point / arg['log_path']
    log = LogReaderWriter(log_path, arg['column_names'])
    assert len(log.data) != 0


@pytest.mark.parametrize('arg', arguments, ids=map(lambda x: x['log_path'], arguments))
def test_copy_without_data(entry_point, arg):
    log_path = entry_point / arg['log_path']
    copy1 = LogReaderWriter(log_path=log_path, log_ids=STANDARD_COLUMNS, load=False)
    copy1.set_data(['foo'])
    copy2 = LogReaderWriter.copy_without_data(copy1, STANDARD_COLUMNS)
    copy2.set_data(['foo', 'bar'])
    assert copy1.data == ['foo']
    assert copy2.data == ['foo', 'bar']


@pytest.mark.integration
def test_BpmnReader(entry_point):
    bpmn_path = entry_point / 'PurchasingExample.bpmn'
    bpmn_reader = BPMNReaderWriter(bpmn_path)
    activities = bpmn_reader.read_activities()

    assert len(activities) > 0
