import pytest

from simod.event_log_processing.reader import DEFAULT_XES_COLUMNS, EventLogReader
from simod.process_model.bpmn import BPMNReaderWriter

arguments = [
    {'log_path': 'Production.xes', 'column_names': DEFAULT_XES_COLUMNS},
    {'log_path': 'PurchasingExampleCustomSim.csv',
     'column_names': {
         'CaseID': 'caseid',
         'Activity': 'task',
         'EnableTimestamp': 'enabled_timestamp',
         'StartTimestamp': 'start_timestamp',
         'EndTimestamp': 'end_timestamp',
         'Resource': 'user',
     }}
]


@pytest.mark.parametrize('arg', arguments, ids=map(lambda x: x['log_path'], arguments))
def test_logreader_new(entry_point, arg):
    log_path = entry_point / arg['log_path']
    log = EventLogReader(log_path, arg['column_names'])
    assert len(log.data) != 0


@pytest.mark.parametrize('arg', arguments, ids=map(lambda x: x['log_path'], arguments))
def test_copy_without_data(entry_point, arg):
    log_path = entry_point / arg['log_path']
    copy1 = EventLogReader(log_path=log_path, column_names=arg['column_names'], load=False)
    copy1.set_data(['foo'])
    copy2 = EventLogReader.copy_without_data(copy1)
    copy2.set_data(['foo', 'bar'])
    assert copy1._time_format == copy2._time_format
    assert copy1._column_names == copy2._column_names
    assert copy1._column_filter == copy2._column_filter


def test_BpmnReader(entry_point):
    bpmn_path = entry_point / 'PurchasingExample.bpmn'
    bpmn_reader = BPMNReaderWriter(bpmn_path)
    activities = bpmn_reader.read_activities()

    assert len(activities) > 0