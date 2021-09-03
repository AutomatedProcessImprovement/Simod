import os

import pytest

from simod.configuration import ReadOptions
from simod.readers.log_reader import LogReader, LogReaderOld


@pytest.fixture
def args(entry_point) -> dict:
    # log_path = 'assets/validation_2/BPI_Challenge_2012_W_Two_TS.xes'
    log_path = os.path.join(entry_point, 'validation_1/complete logs/Production.xes')
    # log_path = '../assets/validation_1/complete logs/Production.xes'
    options = ReadOptions(column_names=ReadOptions.column_names_default())
    return {'log_path': log_path, 'read_options': options}


def test_logreader_old(args):
    log = LogReaderOld(args['log_path'], args['read_options'])
    assert len(log.data) != 0


def test_logreader_new(args):
    log = LogReader(args['log_path'], args['read_options'])
    assert len(log.data) != 0


def test_copy_without_data(args):
    copy1 = LogReader(input=args['log_path'], settings=args['read_options'], verbose=True, load=False)
    copy1.set_data(['foo'])
    copy2 = LogReader.copy_without_data(copy1)
    copy2.set_data(['foo', 'bar'])
    assert copy1.timeformat == copy2.timeformat
    assert copy1.column_names == copy2.column_names
    assert copy1.one_timestamp == copy2.one_timestamp
    assert copy1.filter_d_attrib == copy2.filter_d_attrib
    assert copy1.verbose == copy2.verbose

    copy1.verbose = False
    assert copy1.verbose != copy2.verbose
