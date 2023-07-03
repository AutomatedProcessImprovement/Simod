import pytest

from simod.bpm.reader_writer import BPMNReaderWriter


@pytest.mark.integration
def test_bpmn_reader(entry_point):
    bpmn_path = entry_point / 'LoanApp_simplified.bpmn'
    bpmn_reader = BPMNReaderWriter(bpmn_path)
    activities = bpmn_reader.read_activities()

    assert len(activities) > 0
