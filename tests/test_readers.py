import os
from pathlib import Path

import pytest

from simod.configuration import ReadOptions
from simod.event_log import DEFAULT_XES_COLUMNS, LogReader


@pytest.fixture
def args(entry_point) -> list:
    # log_path = os.path.join(entry_point, 'BPI_Challenge_2012_W_Two_TS.xes')
    # log_path = os.path.join(entry_point, 'confidential_1000.xes')
    # log_path = os.path.join(entry_point, 'cvs_pharmacy.xes')
    prosimos_csv_columns = {
        'CaseID': 'caseid',
        'Activity': 'task',
        'EnableTimestamp': 'enabled_timestamp',
        'StartTimestamp': 'start_timestamp',
        'EndTimestamp': 'end_timestamp',
        'Resource': 'user',
    }
    return [
        {'log_path': Path(os.path.join(entry_point, 'Production.xes')), 'column_names': DEFAULT_XES_COLUMNS},
        {'log_path': Path(os.path.join(entry_point, 'PurchasingExampleCustomSim.csv')), 'column_names': prosimos_csv_columns}
    ]


def test_logreader_new(args):
    for arg in args:
        log = LogReader(arg['log_path'], arg['column_names'])
        assert len(log.data) != 0

    # log_df = pd.DataFrame(log.data)
    # assert log_df.isna().all().all() is True  # TODO: is it OK to have NaN values in Activity, Resource, etc. for task = {Start, End, ...}?


def test_copy_without_data(args):
    for arg in args:
        copy1 = LogReader(log_path=arg['log_path'], column_names=arg['column_names'], load=False)
        copy1.set_data(['foo'])
        copy2 = LogReader.copy_without_data(copy1)
        copy2.set_data(['foo', 'bar'])
        assert copy1._time_format == copy2._time_format
        assert copy1._column_names == copy2._column_names
        assert copy1._column_filter == copy2._column_filter
