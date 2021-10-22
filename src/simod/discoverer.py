import copy
import os
import types
from pathlib import Path

import pandas as pd
import xmltodict as xtd
from lxml import etree

from simod.replayer_datatypes import BPMNGraph
from . import support_utils as sup
from .cli_formatter import print_asset, print_section, print_notice
from .common_routines import mine_resources_with_resource_table, \
    mine_inter_arrival, mine_gateway_probabilities, process_tasks, evaluate_logs_with_add_metrics, \
    split_timeline, save_times
from .configuration import Configuration, MiningAlgorithm, CalculationMethod, QBP_NAMESPACE_URI
from .decorators import safe_exec, timeit
from .qbp import simulate
from .readers import log_reader as lr
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

    def execute_pipeline(self) -> None:
        print_notice(f'Log path: {self.settings.log_path}')
        exec_times = dict()
        self.is_safe = self._read_inputs(log_time=exec_times, is_safe=self.is_safe)
        self.is_safe = self._temp_path_creation(log_time=exec_times, is_safe=self.is_safe)
        self.is_safe = self._mine_structure(log_time=exec_times, is_safe=self.is_safe)
        self.is_safe = self._extract_parameters(log_time=exec_times, is_safe=self.is_safe)
        self.is_safe = self._simulate(log_time=exec_times, is_safe=self.is_safe)
        self._manage_results()
        save_times(exec_times, self.settings, self.settings.output.parent)
        self.is_safe = self._export_canonical_model(is_safe=self.is_safe)
        print_asset(f"Output folder is at {self.settings.output}")

    @timeit(rec_name='READ_INPUTS')
    @safe_exec
    def _read_inputs(self, **kwargs) -> None:
        print_section("Log Parsing")
        # Event log reading
        self.log = lr.LogReader(os.path.join(self.settings.log_path), self.settings.read_options)
        # Time splitting 80-20
        self._split_timeline(0.8, self.settings.read_options.one_timestamp)

    @timeit(rec_name='PATH_DEF')
    @safe_exec
    def _temp_path_creation(self, **kwargs) -> None:
        print_section("Log Customization")
        # Output folder creation
        if not os.path.exists(self.settings.output):
            os.makedirs(self.settings.output)
            os.makedirs(os.path.join(self.settings.output, 'sim_data'))
        # Create customized event-log for the external tools
        output_path = self.settings.output / (self.settings.project_name + '.xes')
        xes.XesWriter(self.log_train, self.settings.read_options, output_path)

    @timeit(rec_name='MINING_STRUCTURE')
    @safe_exec
    def _mine_structure(self, **kwargs) -> None:
        print_section("Process Structure Mining")
        structure_miner = StructureMiner(self.settings, log=self.log_train)
        structure_miner.execute_pipeline()
        if structure_miner.is_safe:
            self.bpmn = structure_miner.bpmn
            self.process_graph = structure_miner.process_graph
        else:
            raise RuntimeError('Structure Mining error')

    @timeit(rec_name='EXTRACTION')
    @safe_exec
    def _extract_parameters(self, **kwargs) -> None:
        print_section("Simulation Parameters Mining")

        time_table, resource_pool, resource_table = mine_resources_with_resource_table(self.log_train, self.settings)

        log_train_df = pd.DataFrame(self.log_train.data)
        arrival_rate = mine_inter_arrival(self.process_graph, log_train_df, self.settings)

        bpmn_path = os.path.join(self.settings.output, self.settings.project_name + '.bpmn')
        bpmn_graph = BPMNGraph.from_bpmn_path(Path(bpmn_path))
        traces = self.log_train.get_traces()
        sequences = mine_gateway_probabilities(traces, bpmn_graph)

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
    def _simulate(self, **kwargs):
        print_section("Simulation")
        self.sim_values = simulate(self.settings, self.process_stats, self.log_test,
                                   evaluate_fn=evaluate_logs_with_add_metrics)

    def _manage_results(self):
        self.sim_values = pd.DataFrame.from_records(self.sim_values)
        self.sim_values['output'] = self.settings.output
        self.sim_values.to_csv(os.path.join(self.settings.output, self.output_file), index=False)

    @safe_exec
    def _export_canonical_model(self, **kwargs):
        ns = {'qbp': QBP_NAMESPACE_URI}
        time_table = etree.tostring(self.parameters['time_table'], pretty_print=True)
        time_table = xtd.parse(time_table, process_namespaces=True, namespaces=ns)
        self.parameters['time_table'] = time_table
        self.parameters['discovery_parameters'] = self._filter_dic_params(self.settings)
        sup.create_json(self.parameters, os.path.join(self.settings.output, self.settings.project_name + '_canon.json'))

    @staticmethod
    def _filter_dic_params(settings: Configuration) -> dict:
        best_params = dict()
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

    def _split_timeline(self, size: float, one_ts: bool):
        train, test, key = split_timeline(self.log, size, one_ts)
        self.log_test = test.sort_values(key, ascending=True).reset_index(drop=True)
        self.log_train = copy.deepcopy(self.log)
        self.log_train.set_data(train.sort_values(key, ascending=True).reset_index(drop=True).to_dict('records'))
