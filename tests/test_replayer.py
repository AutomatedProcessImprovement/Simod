import copy
import logging
import os
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd
import pytest

from simod.common_routines import compute_sequence_flow_frequencies, mine_gateway_probabilities_alternative, \
    mine_gateway_probabilities_alternative_with_gateway_management
from simod.configuration import Configuration, GateManagement
from simod.event_log import LogReader
from simod.readers.log_splitter import LogSplitter
from simod.replayer_datatypes import BPMNGraph
from simod.structure_optimizer import StructureOptimizer


@pytest.fixture
def args(entry_point):
    args = [
        {'model_path': Path(os.path.join(entry_point, 'PurchasingExample.bpmn')),
         'log_path': Path(os.path.join(entry_point, 'PurchasingExample.xes'))},
    ]
    return args


def setup_data(model_path: Path, log_path: Path):
    settings = Configuration(model_path=Path(model_path), log_path=Path(log_path))
    settings.fill_in_derived_fields()

    log = LogReader(log_path)
    graph = BPMNGraph.from_bpmn_path(model_path)

    return graph, log, settings


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


# def discover_model(settings: Configuration) -> Tuple[Path, LogReader, pd.DataFrame]:
#     log = LogReader(settings.log_path)
#
#     if not os.path.exists(settings.output.parent):
#         os.makedirs(settings.output.parent)
#
#     log_test, log_train = split_log_buckets(log, 0.8, settings.read_options.one_timestamp)
#
#     structure_optimizer = StructureOptimizer(settings, copy.deepcopy(log_train))
#     structure_optimizer.run()
#     assert structure_optimizer.best_output is not None
#     model_path = structure_optimizer.best_output / (settings.project_name + '.bpmn')
#
#     return model_path, log_train, log_test


def test_replay_trace(args):
    for arg in args:
        model_path = arg['model_path']
        log_path = arg['log_path']
        print(f'Testing {log_path.name}')

        graph, log, _ = setup_data(model_path, log_path)
        traces = log.get_traces()

        try:
            flow_arcs_frequency = dict()
            for trace in traces:
                task_sequence = [event['task'] for event in trace]
                graph.replay_trace(task_sequence, flow_arcs_frequency)
        except Exception as e:
            exc_type, exc_value, _ = sys.exc_info()
            logging.exception(e)
            pytest.fail(f'Should not fail, failed with: {e}')


def test_compute_sequence_flow_frequencies(args):
    for arg in args:
        model_path = arg['model_path']
        log_path = arg['log_path']
        print(f'Testing {log_path.name}')

        graph, log, _ = setup_data(model_path, log_path)
        traces = log.get_traces()

        try:
            flow_arcs_frequency = compute_sequence_flow_frequencies(traces, graph)
        except Exception as e:
            pytest.fail(f'Should not fail, failed with: {e}')

        assert flow_arcs_frequency is not None
        assert len(flow_arcs_frequency) > 0
        for node_id in flow_arcs_frequency:
            assert flow_arcs_frequency[node_id] != 0


# @pytest.mark.slow
# def test_compute_sequence_flow_frequencies_without_model(args):
#     for arg in args:
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
#         model_path, log_train, _ = discover_model(config)
#         print(f'\nmodel_path = {model_path}\n')
#
#         graph = BPMNGraph.from_bpmn_path(model_path)
#         traces = log_train.get_traces()
#
#         try:
#             flow_arcs_frequency = compute_sequence_flow_frequencies(traces, graph)
#         except Exception as e:
#             pytest.fail(f'Should not fail, failed with: {e}')
#
#         assert flow_arcs_frequency is not None
#         assert len(flow_arcs_frequency) > 0
#         for node_id in flow_arcs_frequency:
#             assert flow_arcs_frequency[node_id] != 0


def test_mine_gateway_probabilities_stochastic_alternative(args):
    for arg in args:
        model_path = arg['model_path']
        log_path = arg['log_path']
        print(f'\nTesting {log_path.name}')

        graph, log, _ = setup_data(model_path, log_path)
        traces = log.get_traces()

        try:
            sequences = mine_gateway_probabilities_alternative(traces, graph)
        except Exception as e:
            logging.exception(e)
            pytest.fail(f'Should not fail, failed with: {e}')

        print(sequences)
        assert len(sequences) != 0


def test_mine_gateway_probabilities_alternative_with_gateway_management(args):
    for arg in args:
        model_path = arg['model_path']
        log_path = arg['log_path']
        print(f'\nTesting {log_path.name}')

        graph, log, _ = setup_data(model_path, log_path)
        traces = log.get_traces()

        for gateway_management in [GateManagement.DISCOVERY, GateManagement.EQUIPROBABLE]:
            try:
                sequences = mine_gateway_probabilities_alternative_with_gateway_management(
                    traces, graph, gateway_management)
            except Exception as e:
                logging.exception(e)
                pytest.fail(f'Should not fail, failed with: {e}')
            print(sequences)
            assert len(sequences) != 0

# NOTE: very long running tests
#
# validation_2_args = [
#     {'log_path': Path('assets/validation_2/BPI_Challenge_2017_W_Two_TS.xes')}
# ]
#
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

# NOTE: too heavy to run each time
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
