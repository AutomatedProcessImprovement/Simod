import copy
import logging
import os
import sys
import unittest
# from .common_routines import execute_simulator_simple
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from simod.common_routines import compute_sequence_flow_frequencies, extract_structure_parameters, \
    extract_process_graph, mine_gateway_probabilities_stochastic, mine_gateway_probabilities_stochastic_alternative
from simod.configuration import Configuration, GateManagement
from simod.readers.log_reader import LogReader
from simod.readers.log_splitter import LogSplitter
from simod.replayer_datatypes import BPMNGraph
from simod.structure_optimizer import StructureOptimizer


class TestReplayer(unittest.TestCase):
    args: List[dict] = [
        {'model_path': Path(os.path.dirname(__file__) + '/../../test_assets/PurchasingExample.bpmn'),
         'log_path': Path(os.path.dirname(__file__) + '/../../test_assets/PurchasingExample.xes')},
        {'model_path': Path(os.path.dirname(
            __file__) + '/../../test_assets/validate new replayer results/testing logs and models/20210804_48BA9CAF_B626_44EC_808E_FBEBCC6CF52C/Production.bpmn'),
         'log_path': Path(os.path.dirname(
             __file__) + '/../../test_assets/validate new replayer results/testing logs and models/20210804_48BA9CAF_B626_44EC_808E_FBEBCC6CF52C/Production.xes')},
        {'model_path': Path(os.path.dirname(
            __file__) + '/../../test_assets/validate new replayer results/testing logs and models/20210804_672EE52F_F905_4860_9CD2_57F95917D1C9/ConsultaDataMining201618.bpmn'),
         'log_path': Path(os.path.dirname(
             __file__) + '/../../test_assets/validate new replayer results/testing logs and models/20210804_672EE52F_F905_4860_9CD2_57F95917D1C9/ConsultaDataMining201618.xes')},
        {'model_path': Path(os.path.dirname(
            __file__) + '/../../test_assets/validate new replayer results/testing logs and models/20210804_E7C625FF_E3CA_4AB3_A386_901182018864/BPI_Challenge_2012_W_Two_TS.bpmn'),
         'log_path': Path(os.path.dirname(
             __file__) + '/../../test_assets/validate new replayer results/testing logs and models/20210804_E7C625FF_E3CA_4AB3_A386_901182018864/BPI_Challenge_2012_W_Two_TS.xes')},
    ]

    @staticmethod
    def setup_data(model_path: Path, log_path: Path):
        settings = Configuration(model_path=Path(model_path), log_path=Path(log_path))
        settings.fill_in_derived_fields()

        log = LogReader(log_path, settings.read_options)
        graph = BPMNGraph.from_bpmn_path(model_path)

        return graph, log, settings

    def test_replay_trace(self):
        for arg in self.args:
            model_path = arg['model_path']
            log_path = arg['log_path']
            print(f'Testing {log_path.name}')

            graph, log, _ = self.setup_data(model_path, log_path)
            traces = log.get_raw_traces()

            def _collect_task_sequence(trace):
                task_sequence = list()
                for event in trace:
                    task_name = event['task']  # original: concept:name
                    state = event['event_type'].lower()  # original: lifecycle:transition
                    if state in ["start", "assign"]:
                        task_sequence.append(task_name)
                return task_sequence

            task_sequences = map(lambda trace: _collect_task_sequence(trace), traces)

            try:
                flow_arcs_frequency = dict()
                for sequence in task_sequences:
                    graph.replay_trace(sequence, flow_arcs_frequency)
            except Exception as e:
                exc_type, exc_value, _ = sys.exc_info()
                logging.exception(e)
                self.fail(f'Should not fail, failed with: {e}')

    def test_compute_sequence_flow_frequencies(self):
        for arg in self.args:
            model_path = arg['model_path']
            log_path = arg['log_path']
            print(f'Testing {log_path.name}')

            graph, log, _ = self.setup_data(model_path, log_path)
            traces = log.get_raw_traces()

            try:
                flow_arcs_frequency = compute_sequence_flow_frequencies(traces, graph)
            except Exception as e:
                self.fail(f'Should not fail, failed with: {e}')

            self.assertTrue(flow_arcs_frequency is not None)
            self.assertTrue(len(flow_arcs_frequency) > 0)
            for node_id in flow_arcs_frequency:
                self.assertFalse(flow_arcs_frequency[node_id] == 0)

    def test_compute_sequence_flow_frequencies_without_model(self):
        def split_log_buckets(log: LogReader, size: float, one_ts: bool) -> Tuple[pd.DataFrame, LogReader]:
            # Split log data
            splitter = LogSplitter(log.data)
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

        def discover_model(settings: Configuration) -> Tuple[Path, LogReader, pd.DataFrame]:
            log = LogReader(settings.log_path, settings.read_options)

            if not os.path.exists(settings.output.parent):
                os.makedirs(settings.output.parent)

            log_test, log_train = split_log_buckets(log, 0.8, settings.read_options.one_timestamp)

            # settings for StructureOptimizer
            settings.max_eval_s = 2
            settings.concurrency = [0.0, 1.0]
            settings.epsilon = [0.0, 1.0]
            settings.eta = [0.0, 1.0]
            settings.gate_management = [GateManagement.DISCOVERY]

            structure_optimizer = StructureOptimizer(settings, copy.deepcopy(log_train), discover_model=True)
            structure_optimizer.execute_trials()
            model_path = Path(os.path.join(structure_optimizer.best_output, settings.project_name + '.bpmn'))

            return model_path, log_train, log_test

        for arg in self.args:
            log_path = arg['log_path']
            print(f'\n\nTesting {log_path.name}')

            config = Configuration(log_path=log_path)  # TODO: fix output path error
            config.fill_in_derived_fields()
            model_path, log_train, _ = discover_model(config)
            print(f'\nmodel_path = {model_path}\n')

            graph = BPMNGraph.from_bpmn_path(model_path)
            traces = log_train.get_raw_traces()

            try:
                flow_arcs_frequency = compute_sequence_flow_frequencies(traces, graph)
            except Exception as e:
                self.fail(f'Should not fail, failed with: {e}')

            self.assertTrue(flow_arcs_frequency is not None)
            self.assertTrue(len(flow_arcs_frequency) > 0)
            for node_id in flow_arcs_frequency:
                self.assertFalse(flow_arcs_frequency[node_id] == 0)

    # def test_mine_gateway_probabilities_stochastic(self):
    #     for arg in self.args:
    #         model_path = arg['model_path']
    #         log_path = arg['log_path']
    #         print(f'\nTesting {log_path.name}')
    #
    #         graph, log, _ = self.setup_data(model_path, log_path)
    #         traces = log.get_raw_traces()
    #
    #         try:
    #             sequences = mine_gateway_probabilities_stochastic(traces, graph)  # NOTE: this function fails
    #         except Exception as e:
    #             logging.exception(e)
    #             self.fail(f'Should not fail, failed with: {e}')
    #
    #         print(sequences)
    #         self.assertFalse(len(sequences) == 0)

    def test_mine_gateway_probabilities_stochastic_alternative(self):
        for arg in self.args:
            model_path = arg['model_path']
            log_path = arg['log_path']
            print(f'\nTesting {log_path.name}')

            graph, log, _ = self.setup_data(model_path, log_path)
            traces = log.get_raw_traces()

            try:
                sequences = mine_gateway_probabilities_stochastic_alternative(traces, graph)
            except Exception as e:
                logging.exception(e)
                self.fail(f'Should not fail, failed with: {e}')

            print(sequences)
            self.assertFalse(len(sequences) == 0)

    def test_extract_structure_parameters(self):
        for arg in self.args:
            model_path = arg['model_path']
            log_path = arg['log_path']
            print(f'\nTesting {log_path.name}')

            graph, log, settings = self.setup_data(model_path, log_path)
            process_graph = extract_process_graph(model_path)

            try:
                parameters = extract_structure_parameters(settings, process_graph, log, model_path)
            except Exception as e:
                exc_type, exc_value, _ = sys.exc_info()
                logging.exception(e)
                self.fail(f'Should not fail, failed with: {exc_type} {exc_value}')

            self.assertTrue(parameters is not None)
            self.assertTrue(parameters.process_stats is not None)
            self.assertTrue(parameters.resource_pool is not None)
            self.assertTrue(parameters.time_table is not None)
            self.assertTrue(parameters.sequences is not None)
            self.assertTrue(parameters.elements_data is not None)


if __name__ == '__main__':
    unittest.main()
