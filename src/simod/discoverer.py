import copy
import os
import types
from pathlib import Path

import pandas as pd
import xmltodict as xtd
from lxml import etree
from simod.stochastic_miner_datatypes import BPMNGraph
from utils import support as sup

from .cli_formatter import print_asset, print_section
from .common_routines import simulate, mine_resources_with_resource_table, \
    mine_inter_arrival, mine_gateway_probabilities_stochastic, process_tasks
from .configuration import Configuration, MiningAlgorithm, CalculationMethod, QBP_NAMESPACE_URI
from .decorators import safe_exec, timeit
from .readers import log_reader as lr
from .readers import log_splitter as ls
from .structure_miner import StructureMiner
from .writers import xes_writer as xes
from .writers import xml_writer as xml


class Discoverer:
    """Main class of the Simulation Models Discoverer"""

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
        structure_miner.execute_pipeline()  # TODO: don't need repair log and evaluate with new replayer
        if structure_miner.is_safe:
            self.bpmn = structure_miner.bpmn
            self.process_graph = structure_miner.process_graph
        else:
            raise RuntimeError('Structure Mining error')

    @timeit(rec_name='EXTRACTION')
    @safe_exec
    def extract_parameters(self, **kwargs) -> None:
        print_section("Simulation Parameters Mining")

        time_table, resource_pool, resource_table = mine_resources_with_resource_table(self.log_train, self.settings)

        log_train_df = pd.DataFrame(self.log_train.data)
        arrival_rate = mine_inter_arrival(self.process_graph, log_train_df, self.settings)

        bpmn_path = os.path.join(self.settings.output, self.settings.project_name + '.bpmn')
        bpmn_graph = BPMNGraph.from_bpmn_path(Path(bpmn_path))
        traces_raw = self.log_train.get_raw_traces()
        sequences = mine_gateway_probabilities_stochastic(traces_raw, bpmn_graph)

        time_delta = log_train_df['end_timestamp'] - log_train_df['start_timestamp']
        time_delta_in_seconds = list(map(lambda x: x.total_seconds(), time_delta))
        log_train_df['processing_time'] = time_delta_in_seconds
        self.process_stats = log_train_df.merge(resource_table[['resource', 'role']],
                                                left_on='user', right_on='resource', how='left')
        elements_data = process_tasks(self.process_graph, self.process_stats, resource_pool, self.settings)

        # rewriting the model file
        num_inst = len(log_train_df.caseid.unique())
        start_time = log_train_df.start_timestamp.min().strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
        parameters = {
            'instances': num_inst,
            'start_time': start_time,
            'resource_pool': resource_pool,
            'time_table': time_table,
            'arrival_rate': arrival_rate,
            'sequences': sequences,
            'elements_data': elements_data
        }
        self.parameters = copy.deepcopy(parameters)

        xml.print_parameters(bpmn_path, bpmn_path, parameters)

    @timeit(rec_name='SIMULATION_EVAL')
    @safe_exec
    def simulate(self, **kwargs) -> None:
        print_section("Simulation")
        self.sim_values = simulate(self.settings, self.process_stats, self.log_test)

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
        best_params['alg_manag'] = str(settings.alg_manag)
        best_params['gate_management'] = str(settings.gate_management)
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
        self.log_test = test.sort_values(key, ascending=True).reset_index(drop=True)
        self.log_train = copy.deepcopy(self.log)
        self.log_train.set_data(train
                                .sort_values(key, ascending=True)
                                .reset_index(drop=True)
                                .to_dict('records'))
