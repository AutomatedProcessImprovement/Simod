import unittest

from simod.configuration import MiningAlgorithm, GateManagement, CalculationMethod, DataType, \
    PDFMethod, Metric, ExecutionMode


class TestConfigurationFromStringConversion(unittest.TestCase):
    def test_MiningAlgorithm_from_str(self):
        args = {
            'sm1': MiningAlgorithm.SM1,
            'sm2': MiningAlgorithm.SM2,
            'sm3': MiningAlgorithm.SM3,
            'SM1': MiningAlgorithm.SM1,
            'SM2': MiningAlgorithm.SM2,
            'SM3': MiningAlgorithm.SM3,
        }

        for arg in args:
            result = MiningAlgorithm.from_str(arg)
            self.assertTrue(result == args[arg])

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
            self.assertTrue(result == args[arg])

        result = GateManagement.from_str(list(args.keys()))
        self.assertTrue(result == list(args.values()))

    def test_CalculationMethod_from_str(self):
        args = {
            'default': CalculationMethod.DEFAULT,
            'discovered': CalculationMethod.DISCOVERED,
            'pool': CalculationMethod.POOL,
            'DEFAULT': CalculationMethod.DEFAULT,
            'DISCOVERED': CalculationMethod.DISCOVERED,
            'POOL': CalculationMethod.POOL,
        }

        for arg in args:
            result = CalculationMethod.from_str(arg)
            self.assertTrue(result == args[arg])

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
            self.assertTrue(result == args[arg])

        result = DataType.from_str(list(args.keys()))
        self.assertTrue(result == list(args.values()))

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
            self.assertTrue(result == args[arg])

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
            self.assertTrue(result == args[arg])

        result = Metric.from_str(list(args.keys()))
        self.assertTrue(result == list(args.values()))

    def test_ExecutionMode_from_str(self):
        args = {
            'single': ExecutionMode.SINGLE,
            'optimizer': ExecutionMode.OPTIMIZER,
            'SINGLE': ExecutionMode.SINGLE,
            'OPTIMIZER': ExecutionMode.OPTIMIZER,
        }

        for arg in args:
            result = ExecutionMode.from_str(arg)
            self.assertTrue(result == args[arg])


class TestConfigurationStringRepresentation(unittest.TestCase):
    def test_GateManagement_str(self):
        args = {
            GateManagement.DISCOVERY: "discovery",
            GateManagement.EQUIPROBABLE: "equiprobable",
            GateManagement.RANDOM: "random",
        }

        for arg in args:
            result = str(arg)
            self.assertTrue(result == args[arg])


if __name__ == '__main__':
    unittest.main()
