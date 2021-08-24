import copy
import logging
import os
import sys
import unittest
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from .common_routines import compute_sequence_flow_frequencies, mine_gateway_probabilities_alternative, \
    mine_gateway_probabilities_alternative_with_gateway_management, remove_outliers
from .configuration import Configuration, GateManagement
from .readers.log_reader import LogReader
from .readers.log_splitter import LogSplitter
from .replayer_datatypes import BPMNGraph
from .structure_optimizer import StructureOptimizer


class TestReplayer(unittest.TestCase):
    validation_1_args: List[dict] = [
        # {'model_path': Path(os.path.dirname(__file__) + '/../../test_assets/PurchasingExample.bpmn'),
        #  'log_path': Path(os.path.dirname(__file__) + '/../../test_assets/PurchasingExample.xes')},
        {'model_path': Path(os.path.dirname(
            __file__) + '/../../test_assets/validation_1/testing logs and models/20210804_48BA9CAF_B626_44EC_808E_FBEBCC6CF52C/Production.bpmn'),
         'log_path': Path(os.path.dirname(
             __file__) + '/../../test_assets/validation_1/complete logs/Production.xes')},
        # {'model_path': Path(os.path.dirname(
        #     __file__) + '/../../test_assets/validation_1/testing logs and models/20210804_672EE52F_F905_4860_9CD2_57F95917D1C9/ConsultaDataMining201618.bpmn'),
        #  'log_path': Path(os.path.dirname(
        #      __file__) + '/../../test_assets/validation_1/complete logs/ConsultaDataMining201618.xes')},
        # {'model_path': Path(os.path.dirname(
        #     __file__) + '/../../test_assets/validation_1/testing logs and models/20210804_E7C625FF_E3CA_4AB3_A386_901182018864/BPI_Challenge_2012_W_Two_TS.bpmn'),
        #  'log_path': Path(os.path.dirname(
        #      __file__) + '/../../test_assets/validation_1/complete logs/BPI_Challenge_2012_W_Two_TS.xes')},
    ]

    # validation_2_args: List[dict] = [
    #     {'log_path': Path(
    #         os.path.dirname(__file__) + '/../../test_assets/validation_2/BPI_Challenge_2017_W_Two_TS.xes')}
    # ]

    @staticmethod
    def setup_data(model_path: Path, log_path: Path):
        settings = Configuration(model_path=Path(model_path), log_path=Path(log_path))
        settings.fill_in_derived_fields()

        log = LogReader(log_path, settings.read_options)
        graph = BPMNGraph.from_bpmn_path(model_path)

        return graph, log, settings

    @staticmethod
    def split_log_buckets(log: LogReader, size: float, one_ts: bool) -> Tuple[pd.DataFrame, LogReader]:
        # Split log data
        splitter = LogSplitter(pd.DataFrame(log.data))
        train, test = splitter.split_log('timeline_contained', size, one_ts)
        total_events = len(log.data)

        # Check size and change time splitting method if necesary
        if len(test) < int(total_events * 0.1):
            train, test = splitter.split_log('timeline_trace', size, one_ts)

        # Set splits
        key = 'end_timestamp' if one_ts else 'start_timestamp'
        test = pd.DataFrame(test)
        train = pd.DataFrame(train)
        log_test = test.sort_values(key, ascending=True).reset_index(drop=True)
        log_train = copy.deepcopy(log)
        log_train.set_data(train.sort_values(key, ascending=True).reset_index(drop=True).to_dict('records'))

        return log_test, log_train

    @staticmethod
    def discover_model(settings: Configuration) -> Tuple[Path, LogReader, pd.DataFrame]:
        log = LogReader(settings.log_path, settings.read_options)

        if not os.path.exists(settings.output.parent):
            os.makedirs(settings.output.parent)

        log_test, log_train = TestReplayer.split_log_buckets(log, 0.8, settings.read_options.one_timestamp)

        structure_optimizer = StructureOptimizer(settings, copy.deepcopy(log_train), discover_model=True)
        structure_optimizer.execute_trials()
        model_path = Path(os.path.join(structure_optimizer.best_output, settings.project_name + '.bpmn'))

        return model_path, log_train, log_test

    def test_replay_trace(self):
        for arg in self.validation_1_args:
            model_path = arg['model_path']
            log_path = arg['log_path']
            print(f'Testing {log_path.name}')

            graph, log, _ = self.setup_data(model_path, log_path)
            traces = log.get_traces()

            try:
                flow_arcs_frequency = dict()
                for trace in traces:
                    task_sequence = [event['task'] for event in trace]
                    graph.replay_trace(task_sequence, flow_arcs_frequency)
            except Exception as e:
                exc_type, exc_value, _ = sys.exc_info()
                logging.exception(e)
                self.fail(f'Should not fail, failed with: {e}')

    def test_compute_sequence_flow_frequencies(self):
        for arg in self.validation_1_args:
            model_path = arg['model_path']
            log_path = arg['log_path']
            print(f'Testing {log_path.name}')

            graph, log, _ = self.setup_data(model_path, log_path)
            traces = log.get_traces()

            try:
                flow_arcs_frequency = compute_sequence_flow_frequencies(traces, graph)
            except Exception as e:
                self.fail(f'Should not fail, failed with: {e}')

            self.assertTrue(flow_arcs_frequency is not None)
            self.assertTrue(len(flow_arcs_frequency) > 0)
            for node_id in flow_arcs_frequency:
                self.assertFalse(flow_arcs_frequency[node_id] == 0)

    def test_compute_sequence_flow_frequencies_without_model(self):
        for arg in self.validation_1_args:
            log_path = arg['log_path']
            print(f'\n\nTesting {log_path.name}')

            config = Configuration(log_path=log_path)
            config.fill_in_derived_fields()

            # settings for StructureOptimizer
            config.max_eval_s = 2
            config.concurrency = [0.0, 1.0]
            config.epsilon = [0.0, 1.0]
            config.eta = [0.0, 1.0]
            config.gate_management = [GateManagement.DISCOVERY]

            model_path, log_train, _ = TestReplayer.discover_model(config)
            print(f'\nmodel_path = {model_path}\n')

            graph = BPMNGraph.from_bpmn_path(model_path)
            traces = log_train.get_traces()

            try:
                flow_arcs_frequency = compute_sequence_flow_frequencies(traces, graph)
            except Exception as e:
                self.fail(f'Should not fail, failed with: {e}')

            self.assertTrue(flow_arcs_frequency is not None)
            self.assertTrue(len(flow_arcs_frequency) > 0)
            for node_id in flow_arcs_frequency:
                self.assertFalse(flow_arcs_frequency[node_id] == 0)

    def test_mine_gateway_probabilities_stochastic_alternative(self):
        for arg in self.validation_1_args:
            model_path = arg['model_path']
            log_path = arg['log_path']
            print(f'\nTesting {log_path.name}')

            graph, log, _ = TestReplayer.setup_data(model_path, log_path)
            traces = log.get_traces()

            try:
                sequences = mine_gateway_probabilities_alternative(traces, graph)
            except Exception as e:
                logging.exception(e)
                self.fail(f'Should not fail, failed with: {e}')

            print(sequences)
            self.assertFalse(len(sequences) == 0)

    def test_mine_gateway_probabilities_alternative_with_gateway_management(self):
        for arg in self.validation_1_args:
            model_path = arg['model_path']
            log_path = arg['log_path']
            print(f'\nTesting {log_path.name}')

            graph, log, _ = TestReplayer.setup_data(model_path, log_path)
            traces = log.get_traces()

            for gateway_management in [GateManagement.DISCOVERY, GateManagement.EQUIPROBABLE]:
                try:
                    sequences = mine_gateway_probabilities_alternative_with_gateway_management(
                        traces, graph, gateway_management)
                except Exception as e:
                    logging.exception(e)
                    self.fail(f'Should not fail, failed with: {e}')
                print(sequences)
                self.assertFalse(len(sequences) == 0)

    # def test_extract_structure_parameters(self):
    #     for arg in self.validation_2_args:
    #         model_path = arg['model_path']
    #         log_path = arg['log_path']
    #         print(f'\nTesting {log_path.name}')
    #
    #         graph, log, settings = TestReplayer.setup_data(model_path, log_path)
    #         process_graph = extract_process_graph(model_path)
    #
    #         try:
    #             parameters = extract_structure_parameters(settings, process_graph, log, model_path)
    #         except Exception as e:
    #             exc_type, exc_value, _ = sys.exc_info()
    #             logging.exception(e)
    #             self.fail(f'Should not fail, failed with: {exc_type} {exc_value}')
    #
    #         self.assertTrue(parameters is not None)
    #         self.assertTrue(parameters.process_stats is not None)
    #         self.assertTrue(parameters.resource_pool is not None)
    #         self.assertTrue(parameters.time_table is not None)
    #         self.assertTrue(parameters.sequences is not None)
    #         self.assertTrue(parameters.elements_data is not None)

    # def test_compute_sequence_flow_frequencies_without_model_validation_2(self):
    #     for arg in self.validation_2_args:
    #         log_path = arg['log_path']
    #         print(f'\n\nTesting {log_path.name}')
    #
    #         config = Configuration(log_path=log_path)
    #         config.fill_in_derived_fields()
    #
    #         # settings for StructureOptimizer
    #         config.max_eval_s = 2
    #         config.concurrency = [0.0, 1.0]
    #         config.epsilon = [0.0, 1.0]
    #         config.eta = [0.0, 1.0]
    #         config.gate_management = [GateManagement.DISCOVERY]
    #
    #         model_path, log_train, _ = TestReplayer.discover_model(config)
    #         print(f'\nmodel_path = {model_path}\n')
    #
    #         graph = BPMNGraph.from_bpmn_path(model_path)
    #         traces = log_train.get_traces()
    #
    #         try:
    #             flow_arcs_frequency = compute_sequence_flow_frequencies(traces, graph)
    #         except Exception as e:
    #             self.fail(f'Should not fail, failed with: {e}')
    #
    #         self.assertTrue(flow_arcs_frequency is not None)
    #         self.assertTrue(len(flow_arcs_frequency) > 0)
    #         for node_id in flow_arcs_frequency:
    #             self.assertFalse(flow_arcs_frequency[node_id] == 0)


class TestOther(unittest.TestCase):
    args: List[dict] = [
        {'model_path': Path(os.path.dirname(__file__) + '/../../test_assets/PurchasingExample.bpmn'),
         'log_path': Path(os.path.dirname(__file__) + '/../../test_assets/PurchasingExample.xes')},
        {'model_path': Path(os.path.dirname(
            __file__) + '/../../test_assets/validation_1/testing logs and models/20210804_48BA9CAF_B626_44EC_808E_FBEBCC6CF52C/Production.bpmn'),
         'log_path': Path(os.path.dirname(
             __file__) + '/../../test_assets/validation_1/complete logs/Production.xes')},
        {'model_path': Path(os.path.dirname(
            __file__) + '/../../test_assets/validation_1/testing logs and models/20210804_672EE52F_F905_4860_9CD2_57F95917D1C9/ConsultaDataMining201618.bpmn'),
         'log_path': Path(os.path.dirname(
             __file__) + '/../../test_assets/validation_1/complete logs/ConsultaDataMining201618.xes')},
        # {'model_path': Path(os.path.dirname(
        #     __file__) + '/../../test_assets/validation_1/testing logs and models/20210804_E7C625FF_E3CA_4AB3_A386_901182018864/BPI_Challenge_2012_W_Two_TS.bpmn'),
        #  'log_path': Path(os.path.dirname(
        #      __file__) + '/../../test_assets/validation_1/complete logs/BPI_Challenge_2012_W_Two_TS.xes')},
    ]

    def test_remove_outliers(self):
        for arg in self.args:
            settings = Configuration()
            log_path = arg['log_path']
            log = LogReader(log_path, settings.read_options)
            print(f'Running test for {log_path}')
            result = remove_outliers(log)
            self.assertFalse(result is None)
            self.assertTrue('caseid' in result.keys())
            self.assertFalse('duration_seconds' in result.keys())


if __name__ == '__main__':
    unittest.main()
