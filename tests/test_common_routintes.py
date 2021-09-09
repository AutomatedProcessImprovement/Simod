import os
from pathlib import Path

import pytest

from simod.common_routines import remove_outliers
from simod.configuration import Configuration
from simod.readers.log_reader import LogReader


@pytest.fixture
def args(entry_point):
    args = [
        {'model_path': Path(os.path.join(entry_point, 'PurchasingExample.bpmn')),
         'log_path': Path(os.path.join(entry_point, 'PurchasingExample.xes'))},
        # {'model_path': Path(os.path.join(entry_point,
        #                                  'validation_1/testing logs and models/20210804_48BA9CAF_B626_44EC_808E_FBEBCC6CF52C/Production.bpmn')),
        #  'log_path': Path(os.path.join(entry_point, 'validation_1/complete logs/Production.xes'))},
        # {'model_path': Path(os.path.join(entry_point,
        #                                  'validation_1/testing logs and models/20210804_672EE52F_F905_4860_9CD2_57F95917D1C9/ConsultaDataMining201618.bpmn')),
        #  'log_path': Path(os.path.join(entry_point, 'validation_1/complete logs/ConsultaDataMining201618.xes'))},
        # {'model_path': Path(os.path.join(entry_point, 'validation_1/testing logs and models/20210804_E7C625FF_E3CA_4AB3_A386_901182018864/BPI_Challenge_2012_W_Two_TS.bpmn')),
        #  'log_path': Path(os.path.join(entry_point, 'validation_1/complete logs/BPI_Challenge_2012_W_Two_TS.xes'))},
    ]
    return args


def test_remove_outliers(args):
    for arg in args:
        settings = Configuration()
        log_path = arg['log_path']
        log = LogReader(log_path, settings.read_options)
        print(f'Running test for {log_path}')
        result = remove_outliers(log)
        assert result is not None
        assert 'caseid' in result.keys()
        assert 'duration_seconds' not in result.keys()
