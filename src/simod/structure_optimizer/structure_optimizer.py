import copy
import math
import multiprocessing
import os
import random
from pathlib import Path
from typing import Union, Optional

import numpy as np
import pandas as pd
from hyperopt import Trials, hp, fmin, STATUS_OK, STATUS_FAIL
from hyperopt import tpe

from simod import support_utils as sup
from simod.analyzers.sim_evaluator import evaluate_logs
from simod.cli_formatter import print_message, print_subsection, print_warning
from simod.configuration import Configuration, MiningAlgorithm, Metric, AndPriorORemove, PDFMethod, SimulatorKind
from simod.discovery import inter_arrival_distribution
from simod.discovery.calendar_discovery.adapter import discover_timetables_and_get_default_arrival_resource_pool
from simod.discovery.gateway_probabilities import discover_with_gateway_management
from simod.discovery.tasks_evaluator import TaskEvaluator
from simod.event_log import write_xes, LogReader, EventLogIDs
from simod.hyperopt_pipeline import HyperoptPipeline
from simod.replayer_datatypes import BPMNGraph
from simod.simulator import simulate, diffresbp_simulator
from .simulation import ProsimosSettings, simulate_undifferentiated
from simod.support_utils import get_project_dir, remove_asset
from simod.support_utils import progress_bar_async
from simod.writers import xml_writer as xml
from . import simulation
from .structure_miner import StructureMiner


class StructureOptimizer(HyperoptPipeline):
    best_output: Optional[Path]
    best_parameters: dict
    measurements_file_name: Path

    _bayes_trials: Trials
    _settings: Configuration
    _log_reader: LogReader
    _log_train: LogReader
    _log_validation: pd.DataFrame
    _original_log: LogReader
    _original_log_train: LogReader
    _original_log_validation: pd.DataFrame
    _space: dict
    _temp_output: Path
    _log_ids: EventLogIDs

    def __init__(self, settings: Configuration, log: LogReader):
        self._log_ids = EventLogIDs(
            case='caseid',
            activity='task',
            resource='user',
            end_time='end_timestamp',
            start_time='start_timestamp',
            enabled_time='enabled_timestamp',
        )

        self._settings = settings
        self._settings.read_options.column_names = {  # TODO: deal with that later, no need in overriding configuration
            'CaseID': 'caseid',
            'Activity': 'task',
            'EnableTimestamp': 'enabled_timestamp',
            'StartTimestamp': 'start_timestamp',
            'EndTimestamp': 'end_timestamp',
            'Resource': 'user'
        }

        self._log_reader = log

        self._space = self._define_search_space(self._settings)

        train, validation = self._log_reader.split_timeline(0.8)
        train = StructureOptimizer._sample_log(train)

        self._log_validation = validation.sort_values('start_timestamp', ascending=True).reset_index(drop=True)
        self._log_train = LogReader.copy_without_data(self._log_reader)
        self._log_train.set_data(
            train.sort_values('start_timestamp', ascending=True).reset_index(drop=True).to_dict('records'))

        self._original_log = copy.deepcopy(log)
        self._original_log_train = copy.deepcopy(self._log_train)
        self._original_log_validation = copy.deepcopy(self._log_validation)

        self._temp_output = get_project_dir() / 'outputs' / sup.folder_id()
        self._temp_output.mkdir(parents=True, exist_ok=True)

        # Measurements file
        self.measurements_file_name = self._temp_output / sup.file_id(prefix='OP_')
        with self.measurements_file_name.open('w') as _:
            pass

        # Trials object to track progress
        self._bayes_trials = Trials()
        self.best_output = None
        self.best_parameters = dict()

    def run(self):
        self._log_train = copy.deepcopy(self._original_log_train)

        # resource_pool, timetable = discover_timetables_and_get_default_arrival_resource_pool(self._settings.log_path)
        # parameters = {'resource_pool': resource_pool, 'time_table': timetable}

        def pipeline(trial_stg: Union[Configuration, dict]):
            print_subsection("Trial")
            print_message(f'train split: {len(pd.DataFrame(self._log_train.data).caseid.unique())}, '
                          f'validation split: {len(self._log_validation.caseid.unique())}')

            if isinstance(trial_stg, dict):
                trial_stg = Configuration(**trial_stg)

            status = STATUS_OK

            status, result = self.step(status, self._update_config_and_export_xes, trial_stg)
            if status == STATUS_OK:
                trial_stg = result

            status, result = self.step(status, self._mine_structure, trial_stg)

            # status, result = self.step(status, self._extract_parameters, trial_stg, result, copy.deepcopy(parameters))
            status, result = self.step(status, self._extract_parameters_undifferentiated, trial_stg, result)

            # status, result = self.step(status, self._simulate, trial_stg)
            status, result = self.step(status, self._simulate_undifferentiated, result, trial_stg)

            sim_values = result if status == STATUS_OK else []

            response = self._define_response(trial_stg, status, sim_values)

            # reinstate log
            self._log_reader = copy.deepcopy(self._original_log)
            self._log_train = copy.deepcopy(self._original_log_train)
            self._log_validation = copy.deepcopy(self._original_log_validation)

            print(f'StructureOptimizer pipeline response: {response}')
            return response

        # Optimization
        best = fmin(fn=pipeline,
                    space=self._space,
                    algo=tpe.suggest,
                    max_evals=self._settings.max_eval_s,
                    trials=self._bayes_trials,
                    show_progressbar=False)

        # Saving results
        self.best_parameters = best
        results = pd.DataFrame(self._bayes_trials.results).sort_values('loss')
        results_ok = results[results.status == STATUS_OK]
        try:
            self.best_output = results_ok.iloc[0].output
        except Exception as e:
            raise e

    def cleanup(self):
        remove_asset(self._temp_output)

    @staticmethod
    def _define_search_space(settings: Configuration) -> dict:
        var_dim = {'gate_management': hp.choice('gate_management', settings.gate_management)}
        if settings.mining_alg in [MiningAlgorithm.SM1, MiningAlgorithm.SM3]:
            var_dim['epsilon'] = hp.uniform('epsilon', settings.epsilon[0], settings.epsilon[1])
            var_dim['eta'] = hp.uniform('eta', settings.eta[0], settings.eta[1])
            var_dim['and_prior'] = hp.choice('and_prior', AndPriorORemove.to_str(settings.and_prior))
            var_dim['or_rep'] = hp.choice('or_rep', AndPriorORemove.to_str(settings.or_rep))
        elif settings.mining_alg is MiningAlgorithm.SM2:
            var_dim['concurrency'] = hp.uniform('concurrency', settings.concurrency[0], settings.concurrency[1])
        new_settings = copy.deepcopy(settings.__dict__)
        for key in var_dim.keys():
            new_settings.pop(key, None)
        space = {**var_dim, **new_settings}
        return space

    def _update_config_and_export_xes(self, settings: Configuration) -> Configuration:
        output_path = self._temp_output / sup.folder_id()
        sim_data_path = output_path / 'sim_data'
        sim_data_path.mkdir(parents=True, exist_ok=True)

        # Create customized event-log for the external tools
        xes_path = output_path / (settings.project_name + '.xes')
        write_xes(self._log_train, xes_path)

        settings.output = output_path
        settings.log_path = xes_path
        return settings

    @staticmethod
    def _mine_structure(settings: Configuration):
        structure_miner = StructureMiner(settings.log_path, settings)
        structure_miner.execute_pipeline()
        return [structure_miner.bpmn, structure_miner.process_graph]

    # TODO: use this function in the pipeline and make sure the downstream simulation routine uses output from it
    def _extract_parameters_undifferentiated(self, settings: Configuration, previous_step_result):
        bpmn_reader, process_graph = previous_step_result
        bpmn_path: Path = bpmn_reader.model_path
        log = self._log_train.get_traces_df(include_start_end_events=True)
        pdf_method = self._settings.pdef_method

        simulation_parameters = simulation.undifferentiated_resources_parameters(
            log, self._log_ids, bpmn_path, process_graph, pdf_method, bpmn_reader, settings.gate_management)

        json_path = bpmn_path.with_suffix('.json')
        simulation_parameters.to_json_file(json_path)

        simulation_cases = log[self._log_ids.case].nunique()

        return bpmn_path, json_path, simulation_cases

    # def _extract_parameters(self, settings: Configuration, structure_values, parameters: dict):
    #     _, process_graph = structure_values
    #     num_inst = len(self._log_validation.caseid.unique())  # TODO: why do we use log_valdn instead of log_train?
    #     start_time = self._log_validation.start_timestamp.min().strftime(
    #         "%Y-%m-%dT%H:%M:%S.%f+00:00")  # getting minimum date
    #
    #     model_path = Path(os.path.join(settings.output, settings.project_name + '.bpmn'))
    #
    #     # extract resource pool, resource timetable and arrival timetable
    #     # TODO: why do we mine and overwrite resource pool and time table when it comes from input arg 'parameters'?
    #     resource_pool, time_table = discover_timetables_and_get_default_arrival_resource_pool(settings.log_path)
    #
    #     # extract inter-arrival distribution
    #     log_df = pd.DataFrame(self._log_train.data)
    #     pdef_method = settings.pdef_method
    #     if not pdef_method:
    #         pdef_method = PDFMethod.DEFAULT
    #         print_warning(f'PDFMethod is missing, setting it to the default: {pdef_method}')
    #     arrival_rate = inter_arrival_distribution.discover(process_graph, log_df, pdef_method)
    #
    #     # extract sequences
    #     bpmn_graph = BPMNGraph.from_bpmn_path(model_path)
    #     traces = self._log_train.get_traces()
    #     sequences = discover_with_gateway_management(
    #         traces, bpmn_graph, settings.gate_management)
    #
    #     # extract elements data
    #     log_df['role'] = 'SYSTEM'
    #     elements_data = TaskEvaluator(process_graph, log_df, resource_pool, pdef_method).elements_data
    #
    #     parameters = parameters | {
    #         'resource_pool': resource_pool,
    #         'time_table': time_table,
    #         'arrival_rate': arrival_rate,
    #         'sequences': sequences,
    #         'elements_data': elements_data,
    #         'instances': num_inst,
    #         'start_time': start_time
    #     }
    #     bpmn_path = os.path.join(settings.output, settings.project_name + '.bpmn')
    #     xml.print_parameters(bpmn_path, bpmn_path, parameters)
    #
    #     self._log_validation.rename(columns={'user': 'resource'}, inplace=True)
    #     self._log_validation['source'] = 'log'
    #     self._log_validation['run_num'] = 0
    #     self._log_validation['role'] = 'SYSTEM'
    #     self._log_validation = self._log_validation[~self._log_validation.task.isin(['Start', 'End'])]

    def _simulate_undifferentiated(self, previous_step_result, settings: Configuration):
        self._log_validation.rename(columns={'user': 'resource'}, inplace=True)
        self._log_validation['source'] = 'log'
        self._log_validation['run_num'] = 0
        self._log_validation['role'] = 'SYSTEM'
        self._log_validation = self._log_validation[~self._log_validation.task.isin(['Start', 'End'])]

        return simulate_undifferentiated(
            settings=settings,
            previous_step_result=previous_step_result,
            validation_log=self._log_validation,
        )

    def _simulate(self, trial_stg: Configuration):
        return simulate(trial_stg, self._log_validation, evaluate_fn=evaluate_logs)

    def _define_response(self, settings: Configuration, status, sim_values) -> dict:
        response = {}  # response contains Configuration and additional hyperopt parameters, e.g., 'status', 'loss'
        measurements = list()
        data = {
            'gate_management': settings.gate_management,
            'output': settings.output,
        }
        # Miner parameters
        if settings.mining_alg in [MiningAlgorithm.SM1, MiningAlgorithm.SM3]:
            data['epsilon'] = settings.epsilon
            data['eta'] = settings.eta
            data['and_prior'] = settings.and_prior
            data['or_rep'] = settings.or_rep
        elif settings.mining_alg is MiningAlgorithm.SM2:
            data['concurrency'] = settings.concurrency
        else:
            raise ValueError(settings.mining_alg)
        response['output'] = settings.output

        if status == STATUS_OK:
            similarity = np.mean([x['sim_val'] for x in sim_values])
            loss = (1 - similarity)
            response['loss'] = loss
            response['status'] = status if loss > 0 else STATUS_FAIL
            for sim_val in sim_values:
                measurements.append({
                    **{'similarity': sim_val['sim_val'],
                       'sim_metric': sim_val['metric'],
                       'status': response['status']},
                    **data})
        else:
            response['status'] = status
            measurements.append({**{'similarity': 0,
                                    'sim_metric': Metric.DL,
                                    'status': response['status']},
                                 **data})
        if os.path.getsize(self.measurements_file_name) > 0:
            sup.create_csv_file(measurements, self.measurements_file_name, mode='a')
        else:
            sup.create_csv_file_header(measurements, self.measurements_file_name)
        return response

    @staticmethod
    def _sample_log(train):
        def sample_size(p_size, c_level, c_interval):
            """
            p_size : population size.
            c_level : confidence level.
            c_interval : confidence interval.
            """
            c_level_constant = {50: .67, 68: .99, 90: 1.64, 95: 1.96, 99: 2.57}
            Z = 0.0
            p = 0.5
            e = c_interval / 100.0
            N = p_size
            n_0 = 0.0
            n = 0.0
            # DEVIATIONS FOR THAT CONFIDENCE LEVEL
            Z = c_level_constant[c_level]
            # CALC SAMPLE SIZE
            n_0 = ((Z ** 2) * p * (1 - p)) / (e ** 2)
            # ADJUST SAMPLE SIZE FOR FINITE POPULATION
            n = n_0 / (1 + ((n_0 - 1) / float(N)))
            return int(math.ceil(n))  # THE SAMPLE SIZE

        cases = list(train.caseid.unique())
        if len(cases) > 1000:
            sample_sz = sample_size(len(cases), 95.0, 3.0)
            scases = random.sample(cases, sample_sz)
            train = train[train.caseid.isin(scases)]
        return train
