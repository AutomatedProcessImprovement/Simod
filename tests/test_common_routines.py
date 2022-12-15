import pytest

from simod.event_log.column_mapping import STANDARD_COLUMNS
from simod.event_log.reader_writer import LogReaderWriter
from simod.event_log.utilities import remove_outliers
from simod.utilities import file_contains


@pytest.fixture
def args(entry_point):
    args = [
        {'model_path': entry_point / 'PurchasingExample.bpmn',
         'log_path': entry_point / 'PurchasingExample.xes'},
    ]
    return args


def test_remove_outliers(args):
    for arg in args:
        log_path = arg['log_path']
        log = LogReaderWriter(log_path, STANDARD_COLUMNS)
        result = remove_outliers(log.df, log_ids=STANDARD_COLUMNS)
        assert result is not None
        assert STANDARD_COLUMNS.case in result.keys()
        assert 'duration_seconds' not in result.keys()


def test_file_contains(entry_point):
    paths_without_inclusive = [
        entry_point / 'PurchasingExample.bpmn',
    ]

    paths_with_inclusive = [
        entry_point / 'ProductionTestFileContains.bpmn',
    ]

    for file_path in paths_without_inclusive:
        assert file_contains(file_path, "exclusiveGateway") is True
        assert file_contains(file_path, "inclusiveGateway") is False

    for file_path in paths_with_inclusive:
        assert file_contains(file_path, "inclusiveGateway") is True
