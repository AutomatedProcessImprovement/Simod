import os

import pytest
import yaml

from simod.configuration import ReadOptions, config_data_from_yaml, Configuration, AndPriorORemove
from simod.readers.log_reader import LogReader
from simod.structure_miner import StructureMiner
from simod.writers import xes_writer


@pytest.fixture
def args(entry_point) -> list:
    options = ReadOptions(column_names=ReadOptions.column_names_default())
    logs = [
        {'log_path': os.path.join(entry_point, 'Production.xes'), 'read_options': options},
        # {'log_path': os.path.join(entry_point, 'cvs_pharmacy.xes'), 'read_options': options},
        # {'log_path': os.path.join(entry_point, 'confidential_1000.xes'), 'read_options': options},
    ]
    return logs


@pytest.fixture
def structure_settings() -> Configuration:
    s = """
structure_optimizer:
  max_eval_s: 2
  concurrency:
    - 0.0
    - 1.0
  epsilon:
    - 0.0
    - 1.0
  eta:
    - 0.0
    - 1.0
  gate_management:
    - equiprobable
    - discovery
  or_rep:
    - true
    - false
  and_prior:
    - true
    - false
    """
    config_data = yaml.load(s, Loader=yaml.FullLoader)
    config_data = config_data_from_yaml(config_data)
    config = Configuration(**config_data['strc'])
    return config


def test_splitminer(args, tmp_path):
    for log in args:
        log_path, read_options = log['log_path'], log['read_options']
        log = LogReader(log_path, read_options)
        assert len(log.data) != 0

        config = Configuration()
        config.project_name = os.path.basename(log_path).split('.')[0]
        config.read_options = read_options
        config.output = tmp_path
        output_path = config.output / (config.project_name + '.xes')
        xes_writer.XesWriter(log, config.read_options, output_path)
        print(tmp_path.absolute())

        config.epsilon = 0.5
        config.eta = 0.5
        config.and_prior = AndPriorORemove.FALSE
        config.or_rep = AndPriorORemove.FALSE
        exit_code = StructureMiner._sm3_miner(config)  # TODO: make _sm3_miner to return exit code
        # assert exit_code == 0
