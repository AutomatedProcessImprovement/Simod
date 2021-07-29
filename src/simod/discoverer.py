import copy
import itertools
import multiprocessing
import os
import time
import types
from dataclasses import dataclass
from multiprocessing import Pool
from operator import itemgetter
from typing import Optional

import pandas as pd
import xmltodict as xtd
from lxml import etree
from networkx import DiGraph
from tqdm import tqdm
from utils import support as sup

from .analyzers import sim_evaluator as sim
from .cli_formatter import print_asset, print_section, print_step
from .common_routines import execute_simulator, read_stats_alt, simulate
from .configuration import Configuration, MiningAlgorithm, CalculationMethod, QBP_NAMESPACE_URI
from .decorators import safe_exec, timeit
from .extraction.log_replayer import LogReplayer
from .extraction.role_discovery import ResourcePoolAnalyser
from .extraction.schedule_tables import TimeTablesCreator
from .parameter_extraction import Operator
from .parameter_extraction import Pipeline, ParameterExtractionOutput, InterArrivalMiner, GatewayProbabilitiesMiner, \
    TasksProcessor
from .readers import log_reader as lr
from .readers import log_splitter as ls
from .readers.bpmn_reader import BpmnReader
from .structure_miner import StructureMiner
from .writers import xes_writer as xes
from .writers import xml_writer as xml


class Discoverer:
    """
    Main class of the Simulation Models Discoverer
    """

    def __init__(self, settings: Configuration):
        self.settings = settings

        self.log = types.SimpleNamespace()
        self.log_train = types.SimpleNamespace()
        self.log_test = types.SimpleNamespace()

        self.sim_values = list()
        self.response = dict()
        self.is_safe = True
        self.output_file = sup.file_id(prefix='SE_')

    def execute_pipeline(self, can=False) -> None:
        exec_times = dict()
        self.is_safe = self.read_inputs(log_time=exec_times, is_safe=self.is_safe)
        self.is_safe = self.temp_path_creation(log_time=exec_times, is_safe=self.is_safe)
        self.is_safe = self.mine_structure(log_time=exec_times, is_safe=self.is_safe)
        self.is_safe = self.replay_process(log_time=exec_times, is_safe=self.is_safe)
        self.is_safe = self.extract_parameters(log_time=exec_times, is_safe=self.is_safe)
        self.is_safe = self.simulate(log_time=exec_times, is_safe=self.is_safe)
        self.mannage_results()
        self.save_times(exec_times, self.settings)
        self.is_safe = self.export_canonical_model(is_safe=self.is_safe)
        print_asset(f"Output folder is at {self.settings.output}")

    @timeit(rec_name='READ_INPUTS')
    @safe_exec
    def read_inputs(self, **kwargs) -> None:
        print_section("Log Parsing")
        # Event log reading
        self.log = lr.LogReader(os.path.join(self.settings.log_path), self.settings.read_options)
        # Time splitting 80-20
        self.split_timeline(0.8, self.settings.read_options.one_timestamp)

    @timeit(rec_name='PATH_DEF')
    @safe_exec
    def temp_path_creation(self, **kwargs) -> None:
        print_section("Log Customization")
        # Output folder creation
        if not os.path.exists(self.settings.output):
            os.makedirs(self.settings.output)
            os.makedirs(os.path.join(self.settings.output, 'sim_data'))
        # Create customized event-log for the external tools
        xes.XesWriter(self.log_train, self.settings)

    @timeit(rec_name='MINING_STRUCTURE')
    @safe_exec
    def mine_structure(self, **kwargs) -> None:
        print_section("Process Structure Mining")
        structure_miner = StructureMiner(self.settings, self.log_train)
        structure_miner.execute_pipeline()
        if structure_miner.is_safe:
            self.bpmn = structure_miner.bpmn
            self.process_graph = structure_miner.process_graph
        else:
            raise RuntimeError('Structure Mining error')

    @timeit(rec_name='REPLAY_PROCESS')
    @safe_exec
    def replay_process(self, **kwargs) -> None:
        """
        Process replaying
        """
        print_section("Log Replaying")
        replayer = LogReplayer(self.process_graph, self.log_train.get_traces(), self.settings,
                               msg='reading conformant training traces')
        self.process_stats = replayer.process_stats
        self.conformant_traces = replayer.conformant_traces

    @timeit(rec_name='EXTRACTION')
    @safe_exec
    def extract_parameters(self, **kwargs) -> None:
        print_section("Simulation Parameters Mining")

        input = ParameterExtractionInputForDiscoverer(
            log=self.log_train, bpmn=self.bpmn, process_graph=self.process_graph, settings=self.settings)
        output = ParameterExtractionOutput()
        parameters_extraction_pipeline = Pipeline(input=input, output=output)
        parameters_extraction_pipeline.set_pipeline([
            LogReplayerForDiscoverer,
            ResourceMinerForDiscoverer,
            InterArrivalMiner,
            GatewayProbabilitiesMiner,
            TasksProcessor
        ])
        parameters_extraction_pipeline.execute()

        self.process_stats = self.process_stats.merge(
            output.resource_table[['resource', 'role']], on='resource', how='left')

        num_inst = len(pd.DataFrame(self.log_test).caseid.unique())
        start_time = (pd.DataFrame(self.log_test).start_timestamp.min().strftime("%Y-%m-%dT%H:%M:%S.%f+00:00"))
        parameters = {
            'instances': num_inst,
            'start_time': start_time,
            'resource_pool': output.resource_pool,
            'time_table': output.time_table,
            'arrival_rate': output.arrival_rate,
            'sequences': output.sequences,
            'elements_data': output.elements_data
        }
        self.parameters = copy.deepcopy(parameters)

        bpmn_path = os.path.join(self.settings.output, self.settings.project_name + '.bpmn')
        xml.print_parameters(bpmn_path, bpmn_path, parameters)

    @timeit(rec_name='SIMULATION_EVAL')
    @safe_exec
    def simulate(self, **kwargs) -> None:
        print_section("Simulation")
        self.sim_values = simulate(self.settings, self.process_stats, self.log_test)

    @staticmethod
    def evaluate_logs(args):
        def evaluate(settings: Configuration, process_stats, sim_log):
            """Reads the simulation results stats
            Args:
                settings (dict): Path to jar and file names
                rep (int): repetition number
            """
            # print('Reading repetition:', (rep+1), sep=' ')
            rep = (sim_log.iloc[0].run_num)
            sim_values = list()
            evaluator = sim.SimilarityEvaluator(process_stats, sim_log, settings, max_cases=1000)
            metrics = [settings.sim_metric]
            if settings.add_metrics:
                metrics = list(set(settings.add_metrics + metrics))
            for metric in metrics:
                evaluator.measure_distance(metric)
                sim_values.append({**{'run_num': rep}, **evaluator.similarity})
            return sim_values

        return evaluate(*args)

    @staticmethod
    def get_traces(data, one_timestamp):
        """
        returns the data splitted by caseid and ordered by start_timestamp
        """
        cases = list(set([x['caseid'] for x in data]))
        traces = list()
        for case in cases:
            order_key = 'end_timestamp' if one_timestamp else 'start_timestamp'
            trace = sorted(
                list(filter(lambda x: (x['caseid'] == case), data)),
                key=itemgetter(order_key))
            traces.append(trace)
        return traces

    def mannage_results(self) -> None:
        self.sim_values = pd.DataFrame.from_records(self.sim_values)
        self.sim_values['output'] = self.settings.output
        self.sim_values.to_csv(os.path.join(self.settings.output, self.output_file), index=False)

    @staticmethod
    def save_times(times, settings: Configuration):
        times = [{**{'output': settings.output}, **times}]
        log_file = os.path.join('outputs', 'execution_times.csv')
        if not os.path.exists(log_file):
            open(log_file, 'w').close()
        if os.path.getsize(log_file) > 0:
            sup.create_csv_file(times, log_file, mode='a')
        else:
            sup.create_csv_file_header(times, log_file)

    @safe_exec
    def export_canonical_model(self, **kwargs):
        ns = {'qbp': QBP_NAMESPACE_URI}
        time_table = etree.tostring(self.parameters['time_table'], pretty_print=True)
        time_table = xtd.parse(time_table, process_namespaces=True, namespaces=ns)
        self.parameters['time_table'] = time_table
        self.parameters['discovery_parameters'] = self.filter_dic_params(self.settings)
        sup.create_json(self.parameters, os.path.join(self.settings.output, self.settings.project_name + '_canon.json'))

    @staticmethod
    def filter_dic_params(settings: Configuration):
        best_params = dict()
        best_params['alg_manag'] = settings.alg_manag.__str__().split('.')[1]
        best_params['gate_management'] = settings.gate_management.__str__().split('.')[1]
        best_params['rp_similarity'] = str(settings.rp_similarity)
        # best structure mining parameters
        if settings.mining_alg in [MiningAlgorithm.SM1, MiningAlgorithm.SM3]:
            best_params['epsilon'] = str(settings.epsilon)
            best_params['eta'] = str(settings.eta)
        elif settings.mining_alg == MiningAlgorithm.SM2:
            best_params['concurrency'] = str(settings.concurrency)
        if settings.res_cal_met == CalculationMethod.DEFAULT:
            best_params['res_dtype'] = settings.res_dtype.__str__().split('.')[1]
        else:
            best_params['res_support'] = str(settings.res_support)
            best_params['res_confidence'] = str(settings.res_confidence)
        if settings.arr_cal_met == CalculationMethod.DEFAULT:
            best_params['arr_dtype'] = settings.res_dtype.__str__().split('.')[1]
        else:
            best_params['arr_support'] = str(settings.arr_support)
            best_params['arr_confidence'] = str(settings.arr_confidence)
        return best_params

    # =============================================================================
    # Support methods
    # =============================================================================

    def split_timeline(self, size: float, one_ts: bool) -> None:
        """
        Split an event log dataframe by time to peform split-validation.
        prefered method time splitting removing incomplete traces.
        If the testing set is smaller than the 10% of the log size
        the second method is sort by traces start and split taking the whole
        traces no matter if they are contained in the timeframe or not

        Parameters
        ----------
        size : float, validation percentage.
        one_ts : bool, Support only one timestamp.
        """
        # Split log data
        splitter = ls.LogSplitter(self.log.data)
        train, test = splitter.split_log('timeline_contained', size, one_ts)
        total_events = len(self.log.data)
        # Check size and change time splitting method if necesary
        if len(test) < int(total_events * 0.1):
            train, test = splitter.split_log('timeline_trace', size, one_ts)
        # Set splits
        key = 'end_timestamp' if one_ts else 'start_timestamp'
        test = pd.DataFrame(test)
        train = pd.DataFrame(train)
        self.log_test = (test.sort_values(key, ascending=True)
                         .reset_index(drop=True))
        self.log_train = copy.deepcopy(self.log)
        self.log_train.set_data(train.sort_values(key, ascending=True)
                                .reset_index(drop=True).to_dict('records'))


# Parameters Extraction Implementations: For Discoverer

@dataclass
class ParameterExtractionInputForDiscoverer:
    log: types.SimpleNamespace
    bpmn: Optional[BpmnReader] = None
    process_graph: Optional[DiGraph] = None
    settings: Optional[Configuration] = None


class LogReplayerForDiscoverer(Operator):
    input: ParameterExtractionInputForDiscoverer
    output: ParameterExtractionOutput

    def __init__(self, input: ParameterExtractionInputForDiscoverer, output: ParameterExtractionOutput):
        self.input = input
        self.output = output
        self._execute()

    def _execute(self):
        print_step('Log Replayer')
        replayer = LogReplayer(self.input.process_graph, self.input.log.get_traces(), self.input.settings,
                               msg='reading conformant training traces')
        self.output.process_stats = replayer.process_stats
        self.output.conformant_traces = replayer.conformant_traces


class ResourceMinerForDiscoverer(Operator):
    input: ParameterExtractionInputForDiscoverer
    output: ParameterExtractionOutput

    def __init__(self, input: ParameterExtractionInputForDiscoverer, output: ParameterExtractionOutput):
        self.input = input
        self.output = output
        self._execute()

    def _execute(self):
        """Analysing resource pool LV917 or 247"""

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
                    {'id': sup.gen_id(), 'name': key, 'total_amount': r_pool_size, 'costxhour': '20',
                     'timetable_id': name})
            return resource_pool

        print_step('Resource Miner')

        res_analyzer = ResourcePoolAnalyser(self.input.log, sim_threshold=self.input.settings.rp_similarity)
        ttcreator = TimeTablesCreator(self.input.settings)
        args = {'res_cal_met': self.input.settings.res_cal_met,
                'arr_cal_met': self.input.settings.arr_cal_met,
                'resource_table': res_analyzer.resource_table}

        if not isinstance(args['res_cal_met'], CalculationMethod):
            args['res_cal_met'] = self.input.settings.res_cal_met
        if not isinstance(args['arr_cal_met'], CalculationMethod):
            args['arr_cal_met'] = self.input.settings.arr_cal_met

        ttcreator.create_timetables(args)
        resource_pool = create_resource_pool(res_analyzer.resource_table, ttcreator.res_ttable_name)
        self.output.resource_pool = resource_pool
        self.output.time_table = ttcreator.time_table

        # Adding role to process stats
        resource_table = pd.DataFrame.from_records(res_analyzer.resource_table)
        self.output.process_stats = self.output.process_stats.merge(resource_table, on='resource', how='left')
        self.output.resource_table = resource_table
