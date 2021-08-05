import logging
import os
import sys
import unittest
# from .common_routines import execute_simulator_simple
from pathlib import Path
from typing import List

from simod.common_routines import compute_sequence_flow_frequencies, extract_structure_parameters, \
    extract_process_graph, mine_gateway_probabilities_stochastic, mine_gateway_probabilities_stochastic_alternative
from simod.configuration import Configuration
from simod.readers.log_reader import LogReader
from simod.replayer_datatypes import BPMNGraph


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

            print(flow_arcs_frequency)
            self.assertTrue(flow_arcs_frequency is not None)
            self.assertTrue(len(flow_arcs_frequency) > 0)
            for node_id in flow_arcs_frequency:
                self.assertFalse(flow_arcs_frequency[node_id] == 0)

    def test_mine_gateway_probabilities_stochastic(self):
        for arg in self.args:
            model_path = arg['model_path']
            log_path = arg['log_path']
            print(f'\nTesting {log_path.name}')

            graph, log, _ = self.setup_data(model_path, log_path)
            traces = log.get_raw_traces()

            try:
                sequences = mine_gateway_probabilities_stochastic(traces, graph)
            except Exception as e:
                logging.exception(e)
                self.fail(f'Should not fail, failed with: {e}')

            print(sequences)
            self.assertFalse(len(sequences) == 0)

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
