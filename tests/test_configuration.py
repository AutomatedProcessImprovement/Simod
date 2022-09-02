from pathlib import Path

import yaml

from simod.configuration import config_data_from_file, AndPriorORemove, Configuration, GateManagement, StructureMiningAlgorithm, \
    CalendarType, DataType, PDFMethod, Metric, ExecutionMode, SimulatorKind


class TestConfigurationFromStringConversion:
    def test_MiningAlgorithm_from_str(self):
        args = {
            'sm1': StructureMiningAlgorithm.SPLIT_MINER_1,
            'sm2': StructureMiningAlgorithm.SPLIT_MINER_2,
            'sm3': StructureMiningAlgorithm.SPLIT_MINER_3,
            'SM1': StructureMiningAlgorithm.SPLIT_MINER_1,
            'SM2': StructureMiningAlgorithm.SPLIT_MINER_2,
            'SM3': StructureMiningAlgorithm.SPLIT_MINER_3,
        }

        for arg in args:
            result = StructureMiningAlgorithm.from_str(arg)
            assert result == args[arg]

    def test_GateManagement_from_str(self):
        args = {
            'discovery': GateManagement.DISCOVERY,
            'equiprobable': GateManagement.EQUIPROBABLE,
            'random': GateManagement.RANDOM,
            'DISCOVERY': GateManagement.DISCOVERY,
            'EQUIPROBABLE': GateManagement.EQUIPROBABLE,
            'RANDOM': GateManagement.RANDOM,
        }

        for arg in args:
            result = GateManagement.from_str(arg)
            assert result == args[arg]

        result = GateManagement.from_str(list(args.keys()))
        assert result == list(args.values())

    def test_CalculationMethod_from_str(self):
        args = {
            'undifferentiated': CalendarType.UNDIFFERENTIATED,
            'differentiated_by_pool': CalendarType.DIFFERENTIATED_BY_POOL,
            'differentiated_by_resource': CalendarType.DIFFERENTIATED_BY_RESOURCE,
            'UNDIFFERENTIATED': CalendarType.UNDIFFERENTIATED,
            'DIFFERENTIATED_BY_POOL': CalendarType.DIFFERENTIATED_BY_POOL,
            'DIFFERENTIATED_BY_RESOURCE': CalendarType.DIFFERENTIATED_BY_RESOURCE,
        }

        for arg in args:
            result = CalendarType.from_str(arg)
            assert result == args[arg]

    def test_DataType_from_str(self):
        args = {
            'dt247': DataType.DT247,
            'DT247': DataType.DT247,
            '247': DataType.DT247,
            'lv917': DataType.LV917,
            'LV917': DataType.LV917,
            '917': DataType.LV917,
        }

        for arg in args:
            result = DataType.from_str(arg)
            assert result == args[arg]

        result = DataType.from_str(list(args.keys()))
        assert result == list(args.values())

    def test_PDFMethod_from_str(self):
        args = {
            'automatic': PDFMethod.AUTOMATIC,
            'semiautomatic': PDFMethod.SEMIAUTOMATIC,
            'manual': PDFMethod.MANUAL,
            'default': PDFMethod.DEFAULT,
            'AUTOMATIC': PDFMethod.AUTOMATIC,
            'SEMIAUTOMATIC': PDFMethod.SEMIAUTOMATIC,
            'MANUAL': PDFMethod.MANUAL,
            'DEFAULT': PDFMethod.DEFAULT,
        }

        for arg in args:
            result = PDFMethod.from_str(arg)
            assert result == args[arg]

    def test_Metric_from_str(self):
        args = {
            'tsd': Metric.TSD,
            'day_hour_emd': Metric.DAY_HOUR_EMD,
            'log_mae': Metric.LOG_MAE,
            'dl': Metric.DL,
            'mae': Metric.MAE,
            'day_emd': Metric.DAY_EMD,
            'cal_emd': Metric.CAL_EMD,
            'dl_mae': Metric.DL_MAE,
            'hour_emd': Metric.HOUR_EMD,
            'TSD': Metric.TSD,
            'DAY_HOUR_EMD': Metric.DAY_HOUR_EMD,
            'LOG_MAE': Metric.LOG_MAE,
            'DL': Metric.DL,
            'MAE': Metric.MAE,
            'DAY_EMD': Metric.DAY_EMD,
            'CAL_EMD': Metric.CAL_EMD,
            'DL_MAE': Metric.DL_MAE,
            'HOUR_EMD': Metric.HOUR_EMD,
        }

        for arg in args:
            result = Metric.from_str(arg)
            assert result == args[arg]

        result = Metric.from_str(list(args.keys()))
        assert result == list(args.values())

    def test_ExecutionMode_from_str(self):
        args = {
            'single': ExecutionMode.SINGLE,
            'optimizer': ExecutionMode.OPTIMIZER,
            'SINGLE': ExecutionMode.SINGLE,
            'OPTIMIZER': ExecutionMode.OPTIMIZER,
        }

        for arg in args:
            result = ExecutionMode.from_str(arg)
            assert result == args[arg]

    def test_SimulatorKind_from_str(self):
        args = {
            'custom': SimulatorKind.CUSTOM,
        }

        for arg in args:
            result = SimulatorKind.from_str(arg)
            assert result == args[arg]


class TestConfigurationStringRepresentation:
    def test_GateManagement_str(self):
        args = {
            GateManagement.DISCOVERY: "discovery",
            GateManagement.EQUIPROBABLE: "equiprobable",
            GateManagement.RANDOM: "random",
        }

        for arg in args:
            result = str(arg)
            assert result == args[arg]

    def test_AndPriorORemove_str(self):
        args = {
            AndPriorORemove.TRUE: 'true',
            AndPriorORemove.FALSE: 'false',
        }

        for arg in args:
            result = str(arg)
            assert result == args[arg]

    def test_AndPriorORemove_to_str(self):
        args = [
            {'input': [AndPriorORemove.TRUE, AndPriorORemove.FALSE], 'expect': ['true', 'false']},
            {'input': [AndPriorORemove.TRUE], 'expect': ['true']},
            {'input': AndPriorORemove.TRUE, 'expect': 'true'},
            {'input': [], 'expect': []},
        ]

        for arg in args:
            result = AndPriorORemove.to_str(arg['input'])
            assert result == arg['expect']


def test_AndPriorORemove_default(entry_point):
    config_path = entry_point / 'optimize_debug_config_2.yml'
    config = config_data_from_file(config_path)
    assert config['strc'] is not None

    structure_config = Configuration(**config['strc'])
    assert structure_config.and_prior == [AndPriorORemove.FALSE]
    assert structure_config.or_rep == [AndPriorORemove.FALSE]


def test_config_data_from_file(entry_point):
    paths = [
        entry_point / 'discover_with_model_config.yml',
        entry_point / 'optimize_debug_config.yml'
    ]

    for config_path in paths:
        with config_path.open('r') as f:
            config_data = yaml.load(f, Loader=yaml.FullLoader)

        config = config_data_from_file(config_path)

        if 'dicover' in config_path.name.__str__():
            assert isinstance(config['log_path'], Path)
            assert isinstance(config['gate_management'], GateManagement)
            assert isinstance(config['res_cal_met'], CalendarType)
            assert isinstance(config['pdef_method'], PDFMethod)
            assert config['gate_management'] == GateManagement.from_str(config_data['gate_management'])
            assert config['res_cal_met'] == CalendarType.from_str(config_data['res_cal_met'])
            assert config['pdef_method'] == PDFMethod.from_str(config_data['pdef_method'])
        elif 'optimize' in config_path.name.__str__():
            assert isinstance(config['log_path'], Path)
            assert isinstance(config['mining_alg'], StructureMiningAlgorithm)
            assert config['mining_alg'] == StructureMiningAlgorithm.from_str(config_data['mining_alg'])
