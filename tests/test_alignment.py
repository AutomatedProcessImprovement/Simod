import os

import pytest

from simod.configuration import Configuration, ReadOptions, TraceAlignmentAlgorithm
from simod.readers.log_reader import LogReader
from simod.structure_miner import StructureMiner
from simod.writers import xes_writer


@pytest.fixture
def args(entry_point) -> list:
    options = ReadOptions(column_names=ReadOptions.column_names_default())
    logs = [
        {
            'log_path': os.path.join(entry_point, 'Production.xes'),
            'read_options': options
        },
        # {'log_path': os.path.join(entry_point, 'cvs_pharmacy.xes'), 'read_options': options},
        # {'log_path': os.path.join(entry_point, 'confidential_1000.xes'), 'read_options': options},
    ]
    return logs


def test_alignment(args):
    for arg in args:
        trace_alignment_algorithms = [
            TraceAlignmentAlgorithm.REPLACEMENT,
            TraceAlignmentAlgorithm.REPAIR,
            TraceAlignmentAlgorithm.REMOVAL
        ]

        global_config = Configuration()
        global_config.log_path = arg['log_path']
        global_config.read_options = arg['read_options']
        global_config.fill_in_derived_fields()

        log = LogReader(global_config.log_path, global_config.read_options)

        if not global_config.output.exists():
            os.makedirs(global_config.output)

        structure_config = Configuration(**global_config.__dict__)
        structure_config.epsilon = 0.5
        structure_config.eta = 0.5

        output_path = global_config.output / (global_config.project_name + '.xes')
        xes_writer.XesWriter(log, global_config.read_options, output_path)

        for alg in trace_alignment_algorithms:
            print(f"\n\nTesting {alg}")
            structure_config.alg_manag = alg
            miner = StructureMiner(structure_config, do_trace_alignment=True, log=log)
            miner.execute_pipeline()
