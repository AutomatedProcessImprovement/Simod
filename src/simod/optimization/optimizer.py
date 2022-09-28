import os
import shutil
from collections import namedtuple
from copy import deepcopy
from pathlib import Path
from typing import Optional
from xml.dom import minidom

import pandas as pd

from simod import utilities as sup, xml_writer
from simod.analyzers.sim_evaluator import SimilarityEvaluator
from simod.bpm.reader_writer import BPMNReaderWriter
from simod.cli_formatter import print_section, print_asset, print_subsection, print_notice, print_step
from simod.configuration import Configuration, StructureMiningAlgorithm
from simod.discovery import inter_arrival_distribution
from simod.discovery.gateway_probabilities import discover_with_gateway_management
from simod.discovery.tasks_evaluator import TaskEvaluator
from simod.event_log.preprocessor import Preprocessor
from simod.event_log.reader_writer import LogReaderWriter
from simod.event_log.utilities import remove_outliers
from simod.process_calendars.optimizer import CalendarOptimizer
from simod.process_structure.optimizer import StructureOptimizer
from simod.simulation.calendar_discovery import discover_timetables_and_get_default_arrival_resource_pool
from simod.simulation.prosimos_bpm_graph import BPMNGraph

Parameters = namedtuple('Parameters',
                        ['time_table', 'resource_pool', 'arrival_rate', 'sequences', 'elements_data',
                         'instances', 'start_time', 'process_stats'])


class Optimizer:
    _log_reader: LogReaderWriter
    _log_train: LogReaderWriter
    _log_test: pd.DataFrame
    _settings: Configuration
    _settings_global: Configuration
    _settings_structure: Configuration
    _settings_time: Configuration
    _best_params: dict = {}
    _preprocessor: Optional[Preprocessor] = None

    def __init__(self, settings):
        self._settings = settings
        self._settings_global = settings['gl']
        self._settings_structure = settings['strc']
        self._settings_time = settings['tm']

        if not os.path.exists(self._settings_global.output):
            os.makedirs(self._settings_global.output)

        self._preprocessor = Preprocessor(self._settings_global)
        self._settings_global = self._preprocessor.run()

        print_notice(f'Log path: {self._settings_global.log_path}')
        self._log_reader = LogReaderWriter(self._settings_global.log_path, log=self._preprocessor.log)
        self.split_and_set_log_buckets(0.8)

    def run(self, discover_model: bool = True) -> None:
        print_step('Removing outliers from the training partition')
        self._remove_outliers()

        structure_optimizer = None
        structure_measurements = None

        # optional model discovery if model is not provided
        if discover_model:
            print_section('Model Discovery and Structure Optimization')
            structure_optimizer = StructureOptimizer(self._settings_structure, deepcopy(self._log_train))
            structure_optimizer.run()
            self._redefine_best_params_after_structure_optimization(structure_optimizer)
            structure_measurements = structure_optimizer.measurements_file_path
            model_path = os.path.join(structure_optimizer.best_output, self._settings_global.project_name + '.bpmn')
        else:
            print_section('Parameters Extraction without model discovery')
            model_path = self._extract_parameters_and_rewrite_model()

        print_section('Times Optimization')
        times_optimizer = CalendarOptimizer(self._settings_global, self._settings_time, self._log_train, model_path)
        times_optimizer.run()
        self._redefine_best_params_after_times_optimization(times_optimizer)

        print_section('Final Comparison')
        self._test_model_and_save_simulation_data(times_optimizer.best_output,
                                                  structure_measurements,
                                                  times_optimizer._measurements_file_name)
        self._export_canonical_model(times_optimizer.best_output)

        times_optimizer.cleanup()
        if structure_optimizer:
            structure_optimizer.cleanup()
        if self._preprocessor:
            self._preprocessor.cleanup()

        print_asset(f"Output folder is at {self._settings_global.output}")

    def _remove_outliers(self):
        # removing outliers
        log_train_df = pd.DataFrame(self._log_train.data)
        log_train_df = remove_outliers(log_train_df)
        # converting data back to LogReader format
        key = 'end_timestamp' if self._settings['gl'].read_options.one_timestamp else 'start_timestamp'
        self._log_train.set_data(
            log_train_df.sort_values(key, ascending=True).reset_index(drop=True).to_dict('records'))

    def _redefine_best_params_after_structure_optimization(self, structure_optimizer):
        # receiving mined results and redefining local variables

        best_parameters = structure_optimizer.best_parameters

        self._settings_global.gate_management = self._settings_structure.gate_management[
            best_parameters['gate_management']]
        self._best_params['gate_management'] = self._settings_global.gate_management
        # best structure mining parameters
        if self._settings_global.structure_mining_algorithm in [StructureMiningAlgorithm.SPLIT_MINER_1,
                                                                StructureMiningAlgorithm.SPLIT_MINER_3]:
            self._settings_global.epsilon = best_parameters['epsilon']
            self._settings_global.eta = best_parameters['eta']
            self._settings_global.and_prior = self._settings_structure.and_prior[best_parameters['and_prior']]
            self._settings_global.or_rep = self._settings_structure.or_rep[best_parameters['or_rep']]
            self._best_params['epsilon'] = best_parameters['epsilon']
            self._best_params['eta'] = best_parameters['eta']
            self._best_params['and_prior'] = self._settings_global.and_prior
            self._best_params['or_rep'] = self._settings_global.or_rep
        elif self._settings_global.structure_mining_algorithm is StructureMiningAlgorithm.SPLIT_MINER_2:
            self._settings_global.concurrency = best_parameters['concurrency']
            self._best_params['concurrency'] = best_parameters['concurrency']
        for key in ['rp_similarity', 'res_dtype', 'arr_dtype', 'res_sup_dis', 'res_con_dis', 'arr_support',
                    'arr_confidence', 'res_cal_met', 'arr_cal_met']:  # TODO: seems like this is unnecessary
            self._settings.pop(key, None)
        return

    def _redefine_best_params_after_times_optimization(self, times_optimizer):
        # redefining parameters after times optimizer
        if times_optimizer.best_parameters['res_cal_met'] == 1:
            self._best_params['res_dtype'] = self._settings_time.res_dtype[times_optimizer.best_parameters['res_dtype']]
        else:
            self._best_params['res_support'] = times_optimizer.best_parameters['res_support']
            self._best_params['res_confidence'] = times_optimizer.best_parameters['res_confidence']
        if times_optimizer.best_parameters['arr_cal_met'] == 1:
            self._best_params['arr_dtype'] = self._settings_time.res_dtype[times_optimizer.best_parameters['arr_dtype']]
        else:
            self._best_params['arr_support'] = (times_optimizer.best_parameters['arr_support'])
            self._best_params['arr_confidence'] = (times_optimizer.best_parameters['arr_confidence'])

    def _extract_parameters_and_rewrite_model(self):
        # extracting parameters
        model_path = self._settings_global.model_path

        bpmn = BPMNReaderWriter(model_path)
        process_graph = bpmn.as_graph()

        # TODO: why only arrival resource pool is used and why it's hardcoded in mine_resource_pool_and_calendars?
        arrival_default_resource_pool, time_table = \
            discover_timetables_and_get_default_arrival_resource_pool(self._settings_structure.log_path)

        log_df = pd.DataFrame(self._log_train.data)
        arrival_rate = inter_arrival_distribution.discover(process_graph, log_df, self._settings_structure.pdef_method)

        bpmn_graph = BPMNGraph.from_bpmn_path(model_path)
        traces = self._log_train.get_traces()
        sequences = discover_with_gateway_management(
            traces, bpmn_graph, self._settings_structure.gate_management)

        log_df['role'] = 'SYSTEM'
        elements_data = TaskEvaluator(process_graph, log_df, arrival_default_resource_pool,
                                      self._settings_structure.pdef_method).elements_data

        # TODO: usually, self.log_valdn is used, but we don't have it here, in Discoverer,
        #  self.log_test is used instead. What would be used here?
        num_inst = len(self._log_test.caseid.unique())
        start_time = self._log_test.start_timestamp.min().strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")

        parameters = Parameters(
            resource_pool=arrival_default_resource_pool,
            time_table=time_table,
            arrival_rate=arrival_rate,
            sequences=sequences,
            elements_data=elements_data,
            instances=num_inst,
            start_time=start_time,
            process_stats=log_df
        )

        # copying the original model and rewriting it with extracted parameters
        if not os.path.exists(self._settings_global.output):
            os.makedirs(self._settings_global.output)
        bpmn_path = os.path.join(self._settings_global.output, os.path.basename(model_path))
        shutil.copy(model_path, bpmn_path)
        xml_writer.print_parameters(bpmn_path, bpmn_path, parameters._asdict())
        model_path = bpmn_path

        # simulation
        sim_data_path = os.path.join(self._settings_global.output, 'sim_data')
        if not os.path.exists(sim_data_path):
            os.makedirs(sim_data_path)

        # TODO: why do we simulate here?
        # _ = simulate(self._settings_global, parameters.process_stats)

        return model_path

    def _test_model_and_save_simulation_data(
            self, best_output: Path, structure_measurements: Path, times_measurements: Path):
        output_file = sup.file_id(prefix='SE_')
        output_path = self._settings_global.output.parent / sup.folder_id()
        self._settings_global.output = output_path
        sim_path = output_path / 'sim_data'
        sim_path.mkdir(parents=True, exist_ok=True)

        self._modify_simulation_model(best_output / (self._settings_global.project_name + '.bpmn'))
        self._load_model_and_measures()
        print_subsection("Simulation")

        # TODO: change to the new simulation function
        # self.sim_values = simulate(self._settings_global, self.process_stats,
        #                            evaluate_fn=evaluate_logs_with_add_metrics)
        raise NotImplementedError

        self.sim_values = pd.DataFrame.from_records(self.sim_values)
        self.sim_values['output'] = output_path
        self.sim_values.to_csv(output_path / output_file, index=False)

        # Collecting files
        if structure_measurements:
            shutil.move(structure_measurements, output_path)
        shutil.move(times_measurements, output_path)

    def _export_canonical_model(self, best_output):
        model_path = os.path.join(self._settings_global.output, self._settings_global.project_name + '.bpmn')
        print_asset(f"Model file location: {model_path}")
        canonical_model = BPMNReaderWriter(model_path).serialize_model()

        # Users in rol data
        resource_table = pd.read_pickle(os.path.join(best_output, 'resource_table.pkl'))
        user_rol = dict()
        for key, group in resource_table.groupby('role'):
            user_rol[key] = list(group.resource)
        canonical_model['rol_user'] = user_rol

        # JSON creation
        self._best_params = {k: str(v) for k, v in self._best_params.items()}
        canonical_model['discovery_parameters'] = self._best_params
        sup.create_json(canonical_model, os.path.join(
            self._settings_global.output, self._settings_global.project_name + '_canon.json'))

    def split_and_set_log_buckets(self, size: float):
        key = 'start_timestamp'
        train, test = self._log_reader.split_timeline(size)
        self._log_test = test.sort_values(key, ascending=True).reset_index(drop=True)
        self._log_train = deepcopy(self._log_reader)
        # self.log_train = LogReader.copy_without_data(self.log)
        self._log_train.set_data(train.sort_values(key, ascending=True).reset_index(drop=True).to_dict('records'))

    def _modify_simulation_model(self, model: Path):
        """Modifies the number of instances of the BIMP simulation model
        to be equal to the number of instances in the testing log"""
        num_inst = len(self._log_test.caseid.unique())
        # Get minimum date
        start_time = self._log_test.start_timestamp.min().strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
        mydoc = minidom.parse(str(model))
        items = mydoc.getElementsByTagName('qbp:processSimulationInfo')
        items[0].attributes['processInstances'].value = str(num_inst)
        items[0].attributes['startDateTime'].value = start_time
        new_model_path = os.path.join(self._settings_global.output, os.path.split(str(model))[1])
        with open(new_model_path, 'wb') as f:
            f.write(mydoc.toxml().encode('utf-8'))
        return new_model_path

    def _load_model_and_measures(self):
        self.process_stats = self._log_test
        self.process_stats['source'] = 'log'
        self.process_stats['run_num'] = 0

    @staticmethod
    def evaluate_logs(args):
        settings, process_stats, sim_log = args
        rep = sim_log.iloc[0].run_num
        sim_values = list()
        evaluator = SimilarityEvaluator(process_stats, sim_log, max_cases=1000)
        metrics = [settings.sim_metric]
        if 'add_metrics' in settings.__dict__.keys():
            metrics = list(set(list(settings.add_metrics) + metrics))
        for metric in metrics:
            evaluator.measure_distance(metric)
            sim_values.append({**{'run_num': rep}, **evaluator.similarity})
        return sim_values
