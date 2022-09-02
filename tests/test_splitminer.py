import os

import pytest

from simod.configuration import ReadOptions, Configuration, AndPriorORemove
from simod.event_log import write_xes, LogReader
from simod.process_structure.miner import StructureMiner

arguments = [
    {'log_path': 'Production.xes',
     'read_options': ReadOptions(column_names=ReadOptions.column_names_default())},
]


@pytest.mark.parametrize('arg', arguments, ids=map(lambda x: x['log_path'], arguments))
def test_splitminer(entry_point, arg, tmp_path):
    log_path = entry_point / arg['log_path']
    read_options = arg['read_options']
    log = LogReader(log_path)
    assert len(log.data) != 0

    config = Configuration()
    config.project_name = os.path.basename(log_path).split('.')[0]
    config.read_options = read_options
    config.output = tmp_path
    output_path = config.output / (config.project_name + '.xes')
    write_xes(log, output_path)
    print(tmp_path.absolute())

    config.epsilon = 0.5
    config.eta = 0.5
    config.and_prior = AndPriorORemove.FALSE
    config.or_rep = AndPriorORemove.FALSE
    exit_code = StructureMiner._sm3_miner(log_path, config)  # TODO: make _sm3_miner to return exit code
    # assert exit_code == 0
