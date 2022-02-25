import itertools
import os
import shutil
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Union, Optional

import pandas as pd
from hyperopt import STATUS_OK, STATUS_FAIL
from networkx import DiGraph

from simod.readers.log_splitter import LogSplitter
from . import support_utils as sup
from .analyzers import sim_evaluator
from .cli_formatter import print_step, print_notice
from .configuration import CalculationMethod, DataType, GateManagement
from .configuration import Configuration, PDFMethod, Metric
from .extraction.interarrival_definition import InterArrivalEvaluator
from .extraction.role_discovery import ResourcePoolAnalyser
from .extraction.schedule_tables import TimeTablesCreator
from .extraction.tasks_evaluator import TaskEvaluator
from .readers import bpmn_reader
from .readers import process_structure
from .event_log import LogReader
from .replayer_datatypes import BPMNGraph

# NOTE: This module needs better name and possible refactoring. At the moment it contains API, which is suitable
# for discovery and optimization. Before function from here were implemented often as static methods on specific classes
# introducing code duplication. We can stay with the functional approach, like I'm proposing at the moment, or create
# a more general class for Discoverer and Optimizer for them to inherit from it all the general routines.

TIMESTAMP_FORMAT = '%Y-%m-%d %H:%M:%S.%f'  # TODO: should be used from LogReader


@dataclass
class ProcessParameters:
    instances: Union[int, None] = None
    start_time: Union[str, None] = None
    process_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    resource_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    # conformant_traces: list = field(default_factory=list)
    resource_pool: list = field(default_factory=list)
    time_table: List[str] = field(default_factory=list)
    arrival_rate: dict = field(default_factory=dict)
    sequences: list = field(default_factory=list)
    elements_data: list = field(default_factory=list)


def extract_structure_parameters(settings: Configuration, process_graph, log_reader: LogReader,
                                 model_path: Path) -> ProcessParameters:
    settings.pdef_method = PDFMethod.DEFAULT  # TODO: why do we overwrite it here?
    traces = log_reader.get_traces()
    log_df = pd.DataFrame(log_reader.data)

    resource_pool, time_table = mine_resources_wrapper(settings)
    arrival_rate = mine_inter_arrival(process_graph, log_df, settings)
    bpmn_graph = BPMNGraph.from_bpmn_path(model_path)
    sequences = mine_gateway_probabilities_alternative_with_gateway_management(
        traces, bpmn_graph, settings.gate_management)
    log_df['role'] = 'SYSTEM'
    elements_data = process_tasks(process_graph, log_df, resource_pool, settings)

    return ProcessParameters(
        process_stats=log_df,
        resource_pool=resource_pool,
        time_table=time_table,
        arrival_rate=arrival_rate,
        sequences=sequences,
        elements_data=elements_data,
    )


def extract_times_parameters(settings: Configuration, process_graph, log: LogReader, conformant_traces,
                             process_stats) -> ProcessParameters:
    settings.pdef_method = PDFMethod.AUTOMATIC

    time_table, resource_pool, resource_table = mine_resources_with_resource_table(log, settings)
    arrival_rate = mine_inter_arrival(process_graph, conformant_traces, settings)

    process_stats = process_stats.merge(resource_table, left_on='user', right_on='resource', how='left')
    elements_data = process_tasks(process_graph, process_stats, resource_pool, settings)

    return ProcessParameters(
        time_table=time_table,
        resource_pool=resource_pool,
        resource_table=resource_table,
        arrival_rate=arrival_rate,
        elements_data=elements_data,
    )


def mine_resources_wrapper(settings: Configuration) -> Tuple[list, List[str]]:  # TODO: maybe this one is unnecessary
    """Analysing resource pool LV917 or 247"""
    print_step('Resource Miner')
    parameters = mine_resources(settings)
    return parameters['resource_pool'], parameters['time_table']


def mine_inter_arrival(process_graph: DiGraph, conformant_traces: pd.DataFrame, settings: Configuration) -> dict:
    print_step('Inter-arrival Miner')
    inter_evaluator = InterArrivalEvaluator(process_graph, conformant_traces, settings)
    return inter_evaluator.dist


def compute_sequence_flow_frequencies(log_traces: list, bpmn_graph: BPMNGraph):
    flow_arcs_frequency = dict()
    for trace in log_traces:
        task_sequence = [event['task'] for event in trace]
        bpmn_graph.replay_trace(task_sequence, flow_arcs_frequency)
    return flow_arcs_frequency


def mine_gateway_probabilities(log_traces: list, bpmn_graph: BPMNGraph) -> list:
    print_step('Mining gateway probabilities')
    arcs_frequencies = compute_sequence_flow_frequencies(log_traces, bpmn_graph)
    gateways_branching = bpmn_graph.compute_branching_probability(arcs_frequencies)

    sequences = []
    for gateway_id in gateways_branching:
        for seqflow_id in gateways_branching[gateway_id]:
            probability = gateways_branching[gateway_id][seqflow_id]
            sequences.append({'elementid': seqflow_id, 'prob': probability})

    return sequences


# TODO: make it accept gate_management option
def mine_gateway_probabilities_alternative(log_traces: list, bpmn_graph: BPMNGraph) -> list:
    print_step('Mining gateway probabilities')
    arcs_frequencies = compute_sequence_flow_frequencies(log_traces, bpmn_graph)
    gateways_branching = bpmn_graph.compute_branching_probability_alternative_discovery(arcs_frequencies)

    sequences = []
    for gateway_id in gateways_branching:
        for seqflow_id in gateways_branching[gateway_id]:
            probability = gateways_branching[gateway_id][seqflow_id]
            sequences.append({'elementid': seqflow_id, 'prob': probability})

    return sequences


def mine_gateway_probabilities_alternative_with_gateway_management(
        log_traces: list, bpmn_graph: BPMNGraph, gate_management: GateManagement) -> list:
    if isinstance(gate_management, list) and len(gate_management) >= 1:
        print_notice(
            f'A list of gateway management options was provided: {gate_management}, taking the first option: {gate_management[0]}')
        gate_management = gate_management[0]

    print_step(f'Mining gateway probabilities with {gate_management}')
    if gate_management is GateManagement.EQUIPROBABLE:
        gateways_branching = bpmn_graph.compute_branching_probability_alternative_equiprobable()
    elif gate_management is GateManagement.DISCOVERY:
        arcs_frequencies = compute_sequence_flow_frequencies(log_traces, bpmn_graph)
        gateways_branching = bpmn_graph.compute_branching_probability_alternative_discovery(arcs_frequencies)
    else:
        raise Exception(
            f'Only GatewayManagement.DISCOVERY and .EQUIPROBABLE are supported, got {gate_management}, {type(gate_management)}')

    sequences = []
    for gateway_id in gateways_branching:
        for seqflow_id in gateways_branching[gateway_id]:
            probability = gateways_branching[gateway_id][seqflow_id]
            sequences.append({'elementid': seqflow_id, 'prob': probability})

    return sequences


def process_tasks(process_graph: DiGraph, process_stats: pd.DataFrame, resource_pool: list,
                  settings: Configuration):
    print_step('Tasks Processor')
    evaluator = TaskEvaluator(process_graph, process_stats, resource_pool, settings)
    return evaluator.elements_data


def extract_process_graph(model_path) -> DiGraph:
    bpmn = bpmn_reader.BpmnReader(model_path)
    return process_structure.create_process_structure(bpmn)


def execute_shell_cmd(args):
    print_step(f'Executing shell command: {args}')
    os.system(' '.join(args))


def read_stats(args):
    global TIMESTAMP_FORMAT

    # NOTE: extracted from StructureOptimizer static method
    def read(settings: Configuration, rep):
        m_settings = dict()
        m_settings['output'] = settings.output
        column_names = {'resource': 'user'}
        m_settings['read_options'] = settings.read_options
        m_settings['read_options'].timeformat = TIMESTAMP_FORMAT
        m_settings['read_options'].column_names = column_names
        m_settings['project_name'] = settings.project_name
        log_path = Path(os.path.join(m_settings['output'], 'sim_data',
                                     m_settings['project_name'] + '_' + str(rep + 1) + '.csv'))
        temp = LogReader(log_path=log_path, column_names=column_names)
        temp = temp.log
        temp.rename(columns={'user': 'resource'}, inplace=True)
        temp['role'] = temp['resource']
        temp['source'] = 'simulation'
        temp['run_num'] = rep + 1
        temp = temp[~temp.task.isin(['Start', 'End'])]
        return temp

    return read(*args)


def read_stats_alt(args):
    global TIMESTAMP_FORMAT

    # NOTE: extracted from StructureOptimizer static method
    def read(settings: Configuration, rep):
        m_settings = dict()
        m_settings['output'] = settings.output
        m_settings['read_options'] = settings.read_options
        m_settings['read_options'].timeformat = TIMESTAMP_FORMAT
        m_settings['read_options'].column_names = settings.read_options.column_names
        m_settings['project_name'] = settings.project_name
        log_path = Path(os.path.join(m_settings['output'], 'sim_data',
                                     m_settings['project_name'] + '_' + str(rep + 1) + '.csv'))
        temp = LogReader(log_path=log_path, column_names=settings.read_options.column_names)
        temp = temp.log
        temp.rename(columns={'user': 'resource'}, inplace=True)
        temp['role'] = temp['resource']
        temp['source'] = 'simulation'
        temp['run_num'] = rep + 1
        temp = temp[~temp.task.isin(['Start', 'End'])]
        return temp

    return read(*args)


def evaluate_logs(args):
    # NOTE: extracted from StructureOptimizer static method
    def evaluate(settings: Configuration, data: pd.DataFrame, sim_log):
        """Reads the simulation results stats"""
        rep = sim_log.iloc[0].run_num
        sim_values = list()
        evaluator = sim_evaluator.SimilarityEvaluator(data, sim_log, settings, max_cases=1000)
        evaluator.measure_distance(Metric.DL)
        sim_values.append({**{'run_num': rep}, **evaluator.similarity})
        return sim_values

    return evaluate(*args)


def evaluate_logs_with_add_metrics(args):
    def evaluate(settings: Configuration, process_stats: pd.DataFrame, sim_log: pd.DataFrame):
        rep = sim_log.iloc[0].run_num
        sim_values = list()
        evaluator = sim_evaluator.SimilarityEvaluator(process_stats, sim_log, settings, max_cases=1000)
        metrics = [settings.sim_metric]
        if settings.add_metrics:
            metrics = list(set(list(settings.add_metrics) + metrics))
        for metric in metrics:
            evaluator.measure_distance(metric)
            sim_values.append({**{'run_num': rep}, **evaluator.similarity})
        return sim_values

    return evaluate(*args)


def save_times(times, settings: Union[Configuration, dict], temp_output):
    if not times:
        return

    if isinstance(settings, dict):
        settings = Configuration(**settings)

    times = [{**{'output': settings.output}, **times}]
    log_file = os.path.join(temp_output, 'execution_times.csv')
    if not os.path.exists(log_file):
        open(log_file, 'w').close()
    if os.path.getsize(log_file) > 0:
        sup.create_csv_file(times, log_file, mode='a')
    else:
        sup.create_csv_file_header(times, log_file)


def pbar_async(p, msg, reps):
    from tqdm import tqdm
    pbar = tqdm(total=reps, desc=msg)
    processed = 0
    while not p.ready():
        cprocesed = (reps - p._number_left)
        if processed < cprocesed:
            increment = cprocesed - processed
            pbar.update(n=increment)
            processed = cprocesed
    time.sleep(1)
    pbar.update(n=(reps - processed))
    p.wait()
    pbar.close()


def mine_resources(settings: Configuration):
    parameters = dict()
    settings.res_cal_met = CalculationMethod.DEFAULT
    settings.res_dtype = DataType.DT247
    settings.arr_cal_met = CalculationMethod.DEFAULT
    settings.arr_dtype = DataType.DT247
    time_table_creator = TimeTablesCreator(settings)
    args = {'res_cal_met': settings.res_cal_met, 'arr_cal_met': settings.arr_cal_met}

    if not isinstance(args['res_cal_met'], CalculationMethod):
        args['res_cal_met'] = CalculationMethod.from_str(args['res_cal_met'])
    if not isinstance(args['arr_cal_met'], CalculationMethod):
        args['arr_cal_met'] = CalculationMethod.from_str(args['arr_cal_met'])

    time_table_creator.create_timetables(args)
    resource_pool = [
        {'id': 'QBP_DEFAULT_RESOURCE', 'name': 'SYSTEM', 'total_amount': '100000', 'costxhour': '20',
         'timetable_id': time_table_creator.res_ttable_name['arrival']}
    ]

    parameters['resource_pool'] = resource_pool
    parameters['time_table'] = time_table_creator.time_table
    return parameters


def mine_resources_with_resource_table(log: LogReader, settings: Configuration):
    def create_resource_pool(resource_table, table_name) -> list:
        """Creates resource pools and associate them the default timetable in BIMP format"""
        resource_pool = [{'id': 'QBP_DEFAULT_RESOURCE', 'name': 'SYSTEM', 'total_amount': '20', 'costxhour': '20',
                          'timetable_id': table_name['arrival']}]
        data = sorted(resource_table, key=lambda x: x['role'])
        for key, group in itertools.groupby(data, key=lambda x: x['role']):
            res_group = [x['resource'] for x in list(group)]
            r_pool_size = str(len(res_group))
            name = (table_name['resources'] if 'resources' in table_name.keys() else table_name[key])
            resource_pool.append(
                {'id': sup.gen_id(), 'name': key, 'total_amount': r_pool_size, 'costxhour': '20', 'timetable_id': name})
        return resource_pool

    print_step('Resource Miner')

    res_analyzer = ResourcePoolAnalyser(log, sim_threshold=settings.rp_similarity)
    # settings.log_path = log.log_path_xes  # TODO: remove this when CalenderImp Java is removed from Simod
    ttcreator = TimeTablesCreator(settings)

    args = {'res_cal_met': settings.res_cal_met,
            'arr_cal_met': settings.arr_cal_met,
            'resource_table': res_analyzer.resource_table}

    if not isinstance(args['res_cal_met'], CalculationMethod):
        args['res_cal_met'] = CalculationMethod.from_str(settings.res_cal_met)
    if not isinstance(args['arr_cal_met'], CalculationMethod):
        args['arr_cal_met'] = CalculationMethod.from_str(settings.arr_cal_met)

    ttcreator.create_timetables(args)
    resource_pool = create_resource_pool(res_analyzer.resource_table, ttcreator.res_ttable_name)
    resource_table = pd.DataFrame.from_records(res_analyzer.resource_table)

    return ttcreator.time_table, resource_pool, resource_table


def split_timeline(log: Union[LogReader, pd.DataFrame], size: float) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Split an event log dataframe by time to perform split-validation.
    preferred method time splitting removing incomplete traces.
    If the testing set is smaller than the 10% of the log size
    the second method is sort by traces start and split taking the whole
    traces no matter if they are contained in the timeframe or not

    Parameters
    ----------
    log: LogRead, log to split.
    size: float, validation percentage.
    """
    if isinstance(log, LogReader):
        log = pd.DataFrame(log.data)

    # Split log data
    splitter = LogSplitter(log)
    partition1, partition2 = splitter.split_log('timeline_contained', size)
    total_events = len(log)

    # Check size and change time splitting method if necesary
    if len(partition2) < int(total_events * 0.1):
        partition1, partition2 = splitter.split_log('timeline_trace', size)

    # Set splits
    key = 'start_timestamp'
    partition1 = pd.DataFrame(partition1)
    partition2 = pd.DataFrame(partition2)
    return partition1, partition2, key


def remove_outliers(log: Union[LogReader, pd.DataFrame]) -> pd.DataFrame:
    if isinstance(log, LogReader):
        event_log = pd.DataFrame(log.data)
    else:
        event_log = log

    # calculating case durations
    cases_durations = list()
    for id, trace in event_log.groupby('caseid'):
        duration = (trace['end_timestamp'].max() - trace['start_timestamp'].min()).total_seconds()
        cases_durations.append({'caseid': id, 'duration_seconds': duration})
    cases_durations = pd.DataFrame(cases_durations)

    # merging data
    event_log = event_log.merge(cases_durations, how='left', on='caseid')

    # filtering rare events
    unique_cases_durations = event_log[['caseid', 'duration_seconds']].drop_duplicates()
    first_quantile = unique_cases_durations.quantile(0.1)
    last_quantile = unique_cases_durations.quantile(0.9)
    event_log = event_log[(event_log.duration_seconds <= last_quantile.duration_seconds) & (
            event_log.duration_seconds >= first_quantile.duration_seconds)]
    event_log = event_log.drop(columns=['duration_seconds'])

    return event_log


def file_contains(file_path: Path, substr: str) -> Optional[bool]:
    if not file_path.exists():
        return None

    with file_path.open('r') as f:
        contains = next((line for line in f if substr in line), None)

    return contains is not None


def hyperopt_pipeline_step(status: str, fn, *args) -> Tuple[str, object]:
    """Function executes the provided function with arguments in hyperopt safe way."""
    if status == STATUS_OK:
        try:
            return STATUS_OK, fn(*args)
        except Exception as error:
            print(error)
            traceback.print_exc()
            return STATUS_FAIL, None
    else:
        return status, None


def remove_asset(location: Path):
    if location is None or not location.exists():
        return
    print_step(f'Removing {location}')
    if location.is_dir():
        shutil.rmtree(location)
    elif location.is_file():
        location.unlink()


