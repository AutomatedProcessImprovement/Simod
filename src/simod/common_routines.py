import itertools
import multiprocessing
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd
from networkx import DiGraph
from simod.configuration import CalculationMethod, DataType, GateManagement
from simod.extraction.gateways_probabilities import GatewaysEvaluator
from simod.extraction.role_discovery import ResourcePoolAnalyser
from simod.extraction.schedule_tables import TimeTablesCreator
from simod.readers.bpmn_reader import BpmnReader
from tqdm import tqdm
from utils import support as sup

from .analyzers import sim_evaluator
from .cli_formatter import print_step
from .configuration import Configuration, PDFMethod, Metric
from .extraction.interarrival_definition import InterArrivalEvaluator
from .extraction.log_replayer import LogReplayer
from .extraction.tasks_evaluator import TaskEvaluator
from .readers import bpmn_reader
from .readers import process_structure
from .readers.log_reader import LogReader
from .stochastic_miner_datatypes import BPMNGraph


# NOTE: This module needs better name and possible refactoring. At the moment it contains API, which is suitable
# for discovery and optimization. Before function from here were implemented often as static methods on specific classes
# introducing code duplication. We can stay with the functional approach, like I'm proposing at the moment, or create
# a more general class for Discoverer and Optimizer for them to inherit from it all the general routines.

@dataclass
class StructureParameters:
    instances: int
    start_time: str
    process_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    # resource_table: pd.DataFrame = field(default_factory=pd.DataFrame)  # TODO: should be active
    # conformant_traces: list = field(default_factory=list)
    resource_pool: list = field(default_factory=list)
    time_table: List[str] = field(default_factory=list)
    arrival_rate: dict = field(default_factory=dict)
    sequences: list = field(default_factory=list)
    elements_data: list = field(default_factory=list)


def extract_structure_parameters(settings: Configuration, process_graph, log: LogReader,
                                 model_path: Union[Path, None] = None,
                                 bpmn: Union[BpmnReader, None] = None) -> StructureParameters:
    settings.pdef_method = PDFMethod.DEFAULT  # TODO: why do we overwrite it here?
    traces = log.get_traces()  # NOTE: long operation
    traces_raw = log.get_raw_traces()  # NOTE: long operation

    process_stats, conformant_traces = replay_logs(process_graph, traces, settings)
    resource_pool, time_table = mine_resources_wrapper(settings)
    arrival_rate = mine_inter_arrival(process_graph, conformant_traces, settings)
    if model_path:  # new replayer
        bpmn_graph = BPMNGraph.from_bpmn_path(model_path)
        sequences = mine_gateway_probabilities_stochastic(traces_raw, bpmn_graph)
    elif bpmn:  # old replayer
        sequences = mine_gateway_probabilities(bpmn, process_graph, settings.gate_management)
    else:
        raise Exception('Either BPMNGraph or BpmnReader must be used to mine gateway probabilities')
    process_stats['role'] = 'SYSTEM'  # TODO: why is this necessary? in which case?
    elements_data = process_tasks(process_graph, process_stats, resource_pool, settings)

    log_df = pd.DataFrame(log.data)
    num_inst = len(log_df.caseid.unique())
    start_time = log_df.start_timestamp.min().strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")

    return StructureParameters(
        process_stats=process_stats,
        resource_pool=resource_pool,
        time_table=time_table,
        arrival_rate=arrival_rate,
        sequences=sequences,
        elements_data=elements_data,
        instances=num_inst,
        start_time=start_time
    )


# TODO: make it more general and allow new replayer with a flag everywhere
def replay_logs(process_graph: DiGraph,
                log_traces: list,
                settings: Configuration) -> Tuple[Union[list, pd.DataFrame], list]:
    print_step('Log Replayer')
    replayer = LogReplayer(process_graph, log_traces, settings, msg='reading conformant training traces')
    return replayer.process_stats, replayer.conformant_traces


def mine_resources_wrapper(settings: Configuration) -> Tuple[list, List[str]]:  # TODO: maybe this one is unnecessary
    """Analysing resource pool LV917 or 247"""
    print_step('Resource Miner')
    parameters = mine_resources(settings)
    return parameters['resource_pool'], parameters['time_table']


def mine_inter_arrival(process_graph: DiGraph, conformant_traces: list, settings: Configuration) -> dict:
    print_step('Inter-arrival Miner')
    inter_evaluator = InterArrivalEvaluator(process_graph, conformant_traces, settings)
    return inter_evaluator.dist


def mine_gateway_probabilities_stochastic(log_traces_raw: list, bpmn_graph: BPMNGraph) -> list:
    print_step('Mining gateway probabilities')

    def _compute_sequence_flow_frequencies(log_traces_raw: list, bpmn_graph: BPMNGraph):
        flow_arcs_frequency = dict()
        for trace in log_traces_raw:
            task_sequence = list()
            for event in trace:
                task_name = event['task']  # original: concept:name
                state = event['event_type'].lower()  # original: lifecycle:transition
                if state in ["start", "assign"]:
                    task_sequence.append(task_name)
            bpmn_graph.replay_trace(task_sequence, flow_arcs_frequency)
        return flow_arcs_frequency

    arcs_frequencies = _compute_sequence_flow_frequencies(log_traces_raw, bpmn_graph)
    gateways_branching = bpmn_graph.compute_branching_probability(arcs_frequencies)

    sequences = []
    for gateway_id in gateways_branching:
        for seqflow_id in gateways_branching[gateway_id]:
            probability = gateways_branching[gateway_id][seqflow_id]
            sequences.append({'elementid': seqflow_id, 'prob': probability})

    return sequences


def mine_gateway_probabilities(bpmn: BpmnReader, process_graph, gate_management: GateManagement) -> list:
    print_step('Mining gateway probabilities')

    evaluator = GatewaysEvaluator(process_graph, gate_management)
    sequences = evaluator.probabilities
    for seq in sequences:
        seq['elementid'] = bpmn.find_sequence_id(seq['gatewayid'], seq['out_path_id'])
    return sequences


def process_tasks(process_graph: DiGraph, process_stats: Union[list, pd.DataFrame], resource_pool: list,
                  settings: Configuration):
    print_step('Tasks Processor')
    evaluator = TaskEvaluator(process_graph, process_stats, resource_pool, settings)
    return evaluator.elements_data


def extract_process_graph(model_path) -> DiGraph:
    bpmn = bpmn_reader.BpmnReader(model_path)
    return process_structure.create_process_structure(bpmn)


def simulate(settings: Configuration, process_stats: pd.DataFrame, log_test):
    # NOTE: from Discoverer
    def pbar_async(p, msg):
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

    reps = settings.repetitions
    cpu_count = multiprocessing.cpu_count()
    w_count = reps if reps <= cpu_count else cpu_count
    pool = multiprocessing.Pool(processes=w_count)

    # Simulate
    args = [(settings, rep) for rep in range(reps)]
    p = pool.map_async(execute_simulator, args)
    pbar_async(p, 'simulating:')

    # Read simulated logs
    args = [(settings, rep) for rep in range(reps)]
    p = pool.map_async(read_stats, args)
    pbar_async(p, 'reading simulated logs:')

    # Evaluate
    args = [(settings, process_stats, log) for log in p.get()]
    if len(log_test.caseid.unique()) > 1000:
        pool.close()
        results = [evaluate_logs(arg) for arg in tqdm(args, 'evaluating results:')]
        sim_values = list(itertools.chain(*results))
    else:
        p = pool.map_async(evaluate_logs, args)
        pbar_async(p, 'evaluating results:')
        pool.close()
        sim_values = list(itertools.chain(*p.get()))
    return sim_values


def execute_simulator(args):
    # NOTE: extracted from StructureOptimizer static method
    def sim_call(settings: Configuration, rep):
        args = ['java', '-jar', settings.bimp_path,
                os.path.join(settings.output, settings.project_name + '.bpmn'),
                '-csv',
                os.path.join(settings.output, 'sim_data', settings.project_name + '_' + str(rep + 1) + '.csv')]
        # NOTE: the call generates a CSV event log from a model
        # NOTE: might fail silently, because stderr or stdout aren't checked
        subprocess.run(args, check=True, stdout=subprocess.PIPE)

    sim_call(*args)


# def execute_simulator_simple(bimp_path, model_path, csv_output_path):
#     args = ['java', '-jar', bimp_path, model_path, '-csv', csv_output_path]
#     print('args', args)
#     # NOTE: the call generates a CSV event log from a model
#     # NOTE: might fail silently, because stderr or stdout aren't checked
#     subprocess.run(args, check=True, stdout=subprocess.PIPE)


def read_stats(args):
    # NOTE: extracted from StructureOptimizer static method
    def read(settings: Configuration, rep):
        m_settings = dict()
        m_settings['output'] = settings.output
        column_names = {'resource': 'user'}
        m_settings['read_options'] = settings.read_options
        m_settings['read_options'].timeformat = '%Y-%m-%d %H:%M:%S.%f'
        m_settings['read_options'].column_names = column_names
        m_settings['project_name'] = settings.project_name
        temp = LogReader(os.path.join(m_settings['output'], 'sim_data',
                                      m_settings['project_name'] + '_' + str(rep + 1) + '.csv'),
                         m_settings['read_options'],
                         verbose=False)
        temp = pd.DataFrame(temp.data)
        temp.rename(columns={'user': 'resource'}, inplace=True)
        temp['role'] = temp['resource']
        temp['source'] = 'simulation'
        temp['run_num'] = rep + 1
        temp = temp[~temp.task.isin(['Start', 'End'])]
        return temp

    return read(*args)


# TODO: name properly or modify/merge read_stats and read_stats_alt
def read_stats_alt(args):
    # NOTE: extracted from Discoverer and Optimizer static method
    def read(settings: Configuration, rep):
        path = os.path.join(settings.output, 'sim_data')
        log_name = settings.project_name + '_' + str(rep + 1) + '.csv'
        rep_results = pd.read_csv(os.path.join(path, log_name), dtype={'caseid': object})
        rep_results['caseid'] = 'Case' + rep_results['caseid']
        rep_results['run_num'] = rep
        rep_results['source'] = 'simulation'
        rep_results.rename(columns={'resource': 'user'}, inplace=True)
        rep_results['start_timestamp'] = pd.to_datetime(
            rep_results['start_timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
        rep_results['end_timestamp'] = pd.to_datetime(
            rep_results['end_timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
        return rep_results

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

    # TODO: do it after execution of this func
    # Adding role to process stats
    # self.output.process_stats = self.output.process_stats.merge(resource_table, on='resource', how='left')

    return ttcreator.time_table, resource_pool, resource_table
