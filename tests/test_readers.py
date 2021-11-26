import os

import pytest

from simod.configuration import ReadOptions
from simod.readers.log_reader import LogReader


@pytest.fixture
def args(entry_point) -> dict:
    # log_path = os.path.join(entry_point, 'BPI_Challenge_2012_W_Two_TS.xes')
    # log_path = os.path.join(entry_point, 'confidential_1000.xes')
    # log_path = os.path.join(entry_point, 'cvs_pharmacy.xes')
    log_path = os.path.join(entry_point, 'Production.xes')
    options = ReadOptions(column_names=ReadOptions.column_names_default())
    return {'log_path': log_path, 'read_options': options}


def test_logreader_new(args):
    log = LogReader(args['log_path'], args['read_options'])
    assert len(log.data) != 0

    # log_df = pd.DataFrame(log.data)
    # assert log_df.isna().all().all() is True  # TODO: is it OK to have NaN values in Activity, Resource, etc. for task = {Start, End, ...}?


def test_copy_without_data(args):
    copy1 = LogReader(input=args['log_path'], settings=args['read_options'], verbose=True, load=False)
    copy1.set_data(['foo'])
    copy2 = LogReader.copy_without_data(copy1)
    copy2.set_data(['foo', 'bar'])
    assert copy1._time_format == copy2._time_format
    assert copy1._column_names == copy2._column_names
    assert copy1._filter_attributes == copy2._filter_attributes
    assert copy1._verbose == copy2._verbose

    copy1._verbose = False
    assert copy1._verbose != copy2._verbose

