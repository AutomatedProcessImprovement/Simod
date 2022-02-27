import copy
import os
from pathlib import Path

import pandas as pd
import xmltodict as xtd
from lxml import etree

from . import support_utils as sup
from .cli_formatter import print_asset, print_section, print_notice
from .common_routines import mine_resources_with_resource_table, \
    mine_inter_arrival, mine_gateway_probabilities, process_tasks, evaluate_logs_with_add_metrics, \
    split_timeline, save_times
from .event_log import write_xes, DEFAULT_XES_COLUMNS, LogReader, reformat_timestamps
from .configuration import Configuration, MiningAlgorithm, CalculationMethod, QBP_NAMESPACE_URI
from .preprocessor import Preprocessor
from .replayer_datatypes import BPMNGraph
from .simulator import simulate
from .structure_miner import StructureMiner
from .writers import xml_writer as xml


class Discoverer:
    """Main class of the Simulation Models Discoverer"""
    _settings: Configuration
    _output_file: str
    _log: LogReader
    _log_train: LogReader
    _log_test: LogReader
    _sim_values: list = []

    def __init__(self, settings: Configuration):
        self._settings = settings
        self._output_file = sup.file_id(prefix='SE_')

    def run(self):
        print_notice(f'Log path: {self._settings.log_path}')
        exec_times = dict()
        self._read_inputs()
        self._temp_path_creation()
        self._preprocess()
        self._mine_structure()
        self._extract_parameters()
        self._simulate()
        self._manage_results()
        save_times(exec_times, self._settings, self._settings.output.parent)
        self._export_canonical_model()
        print_asset(f"Output folder is at {self._settings.output}")

    def _preprocess(self):
        processor = Preprocessor(self._settings)
        self._settings = processor.run()

    def _read_inputs(self):
        print_section("Log Parsing")
        # Event log reading
        self._log = LogReader(self._settings.log_path, column_names=DEFAULT_XES_COLUMNS)
        # Time splitting 80-20
        self._split_timeline(0.8)

    def _temp_path_creation(self):
        print_section("Log Customization")
        # Output folder creation
        if not os.path.exists(self._settings.output):
            os.makedirs(self._settings.output)
            os.makedirs(os.path.join(self._settings.output, 'sim_data'))
        # Create customized event-log for the external tools
        output_path = self._settings.output / (self._settings.project_name + '.xes')
        self._settings.log_path = output_path
        write_xes(self._log_train, output_path)
        reformat_timestamps(output_path, output_path)

    def _mine_structure(self):
        print_section("Process Structure Mining")
        structure_miner = StructureMiner(self._settings.log_path, self._settings)
        structure_miner.execute_pipeline()
        self.bpmn = structure_miner.bpmn
        self.process_graph = structure_miner.process_graph

    def _extract_parameters(self):
        print_section("Simulation Parameters Mining")

        time_table, resource_pool, resource_table = mine_resources_with_resource_table(self._log_train, self._settings)

        log_train_df = pd.DataFrame(self._log_train.data)
        arrival_rate = mine_inter_arrival(self.process_graph, log_train_df, self._settings)

        bpmn_path = os.path.join(self._settings.output, self._settings.project_name + '.bpmn')
        bpmn_graph = BPMNGraph.from_bpmn_path(Path(bpmn_path))
        traces = self._log_train.get_traces()
        sequences = mine_gateway_probabilities(traces, bpmn_graph)

        self.process_stats = log_train_df.merge(resource_table[['resource', 'role']],
                                                left_on='user', right_on='resource', how='left')
        elements_data = process_tasks(self.process_graph, self.process_stats, resource_pool, self._settings)

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

    def _simulate(self):
        print_section("Simulation")
        self._sim_values = simulate(self._settings, self.process_stats, evaluate_fn=evaluate_logs_with_add_metrics)

    def _manage_results(self):
        self._sim_values = pd.DataFrame.from_records(self._sim_values)
        self._sim_values['output'] = self._settings.output
        self._sim_values.to_csv(os.path.join(self._settings.output, self._output_file), index=False)

    def _export_canonical_model(self):
        ns = {'qbp': QBP_NAMESPACE_URI}
        time_table = etree.tostring(self.parameters['time_table'], pretty_print=True)
        time_table = xtd.parse(time_table, process_namespaces=True, namespaces=ns)
        self.parameters['time_table'] = time_table
        self.parameters['discovery_parameters'] = self._filter_dic_params(self._settings)
        sup.create_json(self.parameters, os.path.join(self._settings.output, self._settings.project_name + '_canon.json'))

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

    def _split_timeline(self, size: float):
        train, test, key = split_timeline(self._log, size)
        self._log_test = test.sort_values(key, ascending=True).reset_index(drop=True)
        self._log_train = copy.deepcopy(self._log)
        self._log_train.set_data(train.sort_values(key, ascending=True).reset_index(drop=True).to_dict('records'))
