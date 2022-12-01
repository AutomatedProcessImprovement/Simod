import copy
import logging
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd
import pytest

from simod.event_log.column_mapping import STANDARD_COLUMNS
from simod.event_log.reader_writer import LogReaderWriter
from simod.event_log.splitter import LogSplitter
from simod.simulation.prosimos_bpm_graph import BPMNGraph


@pytest.fixture
def args(entry_point):
    args = [
        {'model_path': entry_point / 'PurchasingExample.bpmn',
         'log_path': entry_point / 'PurchasingExample.xes'},
    ]
    return args


def setup_data(model_path: Path, log_path: Path):
    log = LogReaderWriter(log_path, STANDARD_COLUMNS)
    graph = BPMNGraph.from_bpmn_path(model_path)

    return graph, log


def split_log_buckets(log: LogReaderWriter, size: float, one_ts: bool) -> Tuple[pd.DataFrame, LogReaderWriter]:
    # Split log data
    splitter = LogSplitter(pd.DataFrame(log.data), STANDARD_COLUMNS)
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


def test_replay_trace(args):
    for arg in args:
        model_path = arg['model_path']
        log_path = arg['log_path']
        print(f'Testing {log_path.name}')

        graph, log = setup_data(model_path, log_path)
        traces = log.get_traces()

        try:
            flow_arcs_frequency = dict()
            for trace in traces:
                task_sequence = [event[STANDARD_COLUMNS.activity] for event in trace]
                graph.replay_trace(task_sequence, flow_arcs_frequency)
        except Exception as e:
            exc_type, exc_value, _ = sys.exc_info()
            logging.exception(e)
            pytest.fail(f'Should not fail, failed with: {e}')
