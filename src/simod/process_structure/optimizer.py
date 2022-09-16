import copy
import itertools
import multiprocessing
import os
from pathlib import Path
from typing import Union, Optional, Tuple

import numpy as np
import pandas as pd
from hyperopt import Trials, hp, fmin, STATUS_OK, STATUS_FAIL
from hyperopt import tpe
from tqdm import tqdm

from simod import utilities as sup
from simod.analyzers.sim_evaluator import SimilarityEvaluator
from simod.cli_formatter import print_message, print_subsection
from simod.configuration import StructureMiningAlgorithm, Metric, AndPriorORemove
from simod.event_log.reader_writer import LogReaderWriter
from simod.hyperopt_pipeline import HyperoptPipeline
from simod.simulation.parameters.miner import mine_simulation_parameters_default_24_7
from simod.utilities import get_project_dir, remove_asset, progress_bar_async
from .miner import StructureMiner, Settings as StructureMinerSettings
from .settings import StructureOptimizationSettings, PipelineSettings
from ..bpm.reader_writer import BPMNReaderWriter
from ..event_log.column_mapping import EventLogIDs
from ..event_log.utilities import sample_log
from ..simulation.prosimos import PROSIMOS_COLUMN_MAPPING, ProsimosSettings, simulate_with_prosimos


class StructureOptimizer(HyperoptPipeline):
    best_output: Optional[Path]
    best_parameters: PipelineSettings
    measurements_file_path: Path

    _bayes_trials: Trials
    _settings: StructureOptimizationSettings
    _log_reader: LogReaderWriter
    _log_train: LogReaderWriter
    _log_validation: pd.DataFrame
    _original_log: LogReaderWriter
    _original_log_train: LogReaderWriter
    _original_log_validation: pd.DataFrame
    _temp_output: Path
    _log_ids: EventLogIDs

    def __init__(
            self,
            settings: StructureOptimizationSettings,
            log: LogReaderWriter,
            log_ids: Optional[EventLogIDs] = None,
    ):
        self._log_ids = EventLogIDs(
            case='caseid',
            activity='task',
            resource='user',
            end_time='end_timestamp',
            start_time='start_timestamp',
            enabled_time='enabled_timestamp',
        ) if log_ids is None else log_ids

        self._settings = settings
        self._log_reader = log

        train, validation = self._log_reader.split_timeline(0.8)
        train = sample_log(train)

        self._log_validation = validation.sort_values('start_timestamp', ascending=True).reset_index(drop=True)
        self._log_train = LogReaderWriter.copy_without_data(self._log_reader)
        self._log_train.set_data(
            train.sort_values('start_timestamp', ascending=True).reset_index(drop=True).to_dict('records'))

        self._original_log = copy.deepcopy(log)
        self._original_log_train = copy.deepcopy(self._log_train)
        self._original_log_validation = copy.deepcopy(self._log_validation)

        self._temp_output = get_project_dir() / 'outputs' / sup.folder_id()
        self._temp_output.mkdir(parents=True, exist_ok=True)

        # Measurements file
        self.measurements_file_path = self._temp_output / sup.file_id(prefix='OP_')
        with self.measurements_file_path.open('w') as _:
            pass

        # Trials object to track progress
        self._bayes_trials = Trials()

    def run(self) -> PipelineSettings:
        self._log_train = copy.deepcopy(self._original_log_train)

        def pipeline(trial_stage_settings: Union[PipelineSettings, dict]):
            print_subsection("Trial")
            print_message(f'train split: {len(pd.DataFrame(self._log_train.data)[self._log_ids.case].unique())}, '
                          f'validation split: {len(self._log_validation[self._log_ids.case].unique())}')

            if isinstance(trial_stage_settings, dict):
                trial_stage_settings = PipelineSettings(
                    model_path=None,
                    output_dir=None,
                    measurements_file_path=self.measurements_file_path,
                    project_name=self._settings.project_name,
                    **trial_stage_settings)
            print_message(f'Parameters: {trial_stage_settings}', capitalize=False)

            status = STATUS_OK

            status, result = self.step(status, self._update_config_and_export_xes, trial_stage_settings.project_name)
            output_dir, log_path = result
            trial_stage_settings.output_dir = output_dir
            trial_stage_settings.model_path = (output_dir / (trial_stage_settings.project_name + '.bpmn')).absolute()

            status, result = self.step(status, self._mine_structure,
                                       trial_stage_settings, log_path, self._settings.mining_algorithm)
            bpmn_reader, process_graph = result

            status, result = self.step(
                status,
                self._extract_parameters_undifferentiated,
                trial_stage_settings,
                bpmn_reader,
                process_graph)
            json_path, simulation_cases = result

            status, result = self.step(status, self._simulate_undifferentiated,
                                       trial_stage_settings,
                                       self._settings.simulation_repetitions,
                                       json_path,
                                       simulation_cases)
            evaluation_measurements = result if status == STATUS_OK else []

            response = self._define_response(
                trial_stage_settings, self._settings.mining_algorithm, status, evaluation_measurements)

            # reset the log
            self._log_reader = copy.deepcopy(self._original_log)
            self._log_train = copy.deepcopy(self._original_log_train)
            self._log_validation = copy.deepcopy(self._original_log_validation)

            print(f'StructureOptimizer pipeline response: {response}')
            return response

        search_space = self._define_search_space(self._settings)

        # Optimization
        best = fmin(fn=pipeline,
                    space=search_space,
                    algo=tpe.suggest,
                    max_evals=self._settings.max_evaluations,
                    trials=self._bayes_trials,
                    show_progressbar=False)

        # Saving results

        results = pd.DataFrame(self._bayes_trials.results).sort_values('loss')
        results_ok = results[results.status == STATUS_OK]
        try:
            self.best_output = results_ok.iloc[0].output
        except Exception as e:
            raise e

        # TODO: ensure that's the path
        best_model_path = Path(results_ok.iloc[0].output) / (self._settings.project_name + '.bpmn')

        best_settings = PipelineSettings.from_hyperopt_dict(
            data=best,
            initial_settings=self._settings,
            model_path=best_model_path,
            project_name=self._settings.project_name,
            measurements_file_path=self.measurements_file_path,
        )

        self.best_parameters = best_settings

        return best_settings

    def cleanup(self):
        remove_asset(self._temp_output)

    @staticmethod
    def _define_search_space(settings: StructureOptimizationSettings) -> dict:
        space = {
            'gateway_probabilities': hp.choice('gateway_probabilities', settings.gateway_probabilities),
        }

        if settings.mining_algorithm in [StructureMiningAlgorithm.SPLIT_MINER_1,
                                         StructureMiningAlgorithm.SPLIT_MINER_3]:
            space |= {
                'epsilon': hp.uniform('epsilon', *settings.epsilon),
                'eta': hp.uniform('eta', *settings.eta),
            }

            if settings.mining_algorithm is StructureMiningAlgorithm.SPLIT_MINER_3:
                space |= {
                    'and_prior': hp.choice('and_prior', AndPriorORemove.to_str(settings.and_prior)),
                    'or_rep': hp.choice('or_rep', AndPriorORemove.to_str(settings.or_rep)),
                }
        elif settings.mining_algorithm is StructureMiningAlgorithm.SPLIT_MINER_2:
            space |= {
                'concurrency': hp.uniform('concurrency', *settings.concurrency)
            }

        return space

    def _define_response(
            self,
            settings: PipelineSettings,
            mining_algorithm: StructureMiningAlgorithm,
            status,
            evaluation_measurements) -> dict:
        response = {
            'output': settings.output_dir.absolute(),
            'model_path': settings.model_path.absolute(),
            'status': status,
            'loss': None,
        }

        # collecting measurements for saving

        measurements = []

        optimization_parameters = {
            'gateway_probabilities': settings.gateway_probabilities,
            'output': str(settings.output_dir.absolute()),
        }

        # structure miner parameters
        if mining_algorithm in [StructureMiningAlgorithm.SPLIT_MINER_1,
                                StructureMiningAlgorithm.SPLIT_MINER_3]:
            optimization_parameters['epsilon'] = settings.epsilon
            optimization_parameters['eta'] = settings.eta
            optimization_parameters['and_prior'] = settings.and_prior
            optimization_parameters['or_rep'] = settings.or_rep
        elif mining_algorithm is StructureMiningAlgorithm.SPLIT_MINER_2:
            optimization_parameters['concurrency'] = settings.concurrency
        else:
            raise ValueError(mining_algorithm)

        # simulation parameters
        if status == STATUS_OK:
            similarity = np.mean([x['sim_val'] for x in evaluation_measurements])
            loss = (1 - similarity)

            response['loss'] = loss
            response['status'] = status if loss > 0 else STATUS_FAIL

            for sim_val in evaluation_measurements:
                values = {
                    'similarity': sim_val['sim_val'],
                    'sim_metric': sim_val['metric'],
                    'status': response['status']
                }
                values = values | optimization_parameters
                measurements.append(values)
        else:
            values = {
                'similarity': 0,
                'sim_metric': Metric.DL,
                'status': status
            }
            values = values | optimization_parameters
            measurements.append(values)

        # writing measurements to file
        if os.path.getsize(self.measurements_file_path) > 0:
            sup.create_csv_file(measurements, self.measurements_file_path, mode='a')
        else:
            sup.create_csv_file_header(measurements, self.measurements_file_path)

        return response

    def _update_config_and_export_xes(self, project_name: str) -> Tuple[Path, Path]:
        output_path = self._temp_output / sup.folder_id()
        sim_data_path = output_path / 'sim_data'
        sim_data_path.mkdir(parents=True, exist_ok=True)

        # Create customized event-log for the external tools
        xes_path = output_path / (project_name + '.xes')
        self._log_train.write_xes(xes_path)

        return output_path, xes_path

    @staticmethod
    def _mine_structure(
            settings: PipelineSettings,
            log_path: Path,
            mining_algorithm: StructureMiningAlgorithm) -> Tuple:

        miner_settings = StructureMinerSettings(
            mining_algorithm=mining_algorithm,
            epsilon=settings.epsilon,
            eta=settings.eta,
            concurrency=settings.concurrency,
            and_prior=settings.and_prior,
            or_rep=settings.or_rep,
        )

        _ = StructureMiner(miner_settings, xes_path=log_path, output_model_path=settings.model_path)

        bpmn_reader = BPMNReaderWriter(settings.model_path)
        process_graph = bpmn_reader.as_graph()

        return bpmn_reader, process_graph

    def _extract_parameters_undifferentiated(self, settings: PipelineSettings, bpmn_reader, process_graph) -> Tuple:
        log = self._log_train.get_traces_df(include_start_end_events=True)
        pdf_method = self._settings.pdef_method

        simulation_parameters = mine_simulation_parameters_default_24_7(
            log,
            self._log_ids,
            settings.model_path,
            process_graph,
            pdf_method,
            bpmn_reader,
            settings.gateway_probabilities)

        json_path = settings.model_path.with_suffix('.json')
        simulation_parameters.to_json_file(json_path)

        simulation_cases = log[self._log_ids.case].nunique()

        return json_path, simulation_cases

    def _simulate_undifferentiated(
            self,
            settings: PipelineSettings,
            simulation_repetitions: int,
            json_path: Path,
            simulation_cases: int):
        self._log_validation.rename(columns={'user': 'resource'}, inplace=True)
        self._log_validation['source'] = 'log'
        self._log_validation['run_num'] = 0
        self._log_validation['role'] = 'SYSTEM'
        self._log_validation = self._log_validation[~self._log_validation.task.isin(['Start', 'End'])]

        num_simulations = simulation_repetitions
        cpu_count = multiprocessing.cpu_count()
        w_count = num_simulations if num_simulations <= cpu_count else cpu_count
        pool = multiprocessing.Pool(processes=w_count)

        # Simulate
        simulation_arguments = [
            ProsimosSettings(
                bpmn_path=settings.model_path,
                parameters_path=json_path,
                output_log_path=settings.output_dir / 'sim_data' / f'{settings.project_name}_{rep}.csv',
                num_simulation_cases=simulation_cases)
            for rep in range(num_simulations)]
        p = pool.map_async(simulate_with_prosimos, simulation_arguments)
        progress_bar_async(p, 'simulating', num_simulations)

        # Read simulated logs
        read_arguments = [(simulation_arguments[index].output_log_path, PROSIMOS_COLUMN_MAPPING, index)
                          for index in range(num_simulations)]
        p = pool.map_async(_read_simulated_log, read_arguments)
        progress_bar_async(p, 'reading simulated logs', num_simulations)

        # Evaluate
        evaluation_arguments = [(settings, self._log_validation, log) for log in p.get()]
        if simulation_cases > 1000:
            pool.close()
            results = [self._evaluate_logs(arg) for arg in tqdm(evaluation_arguments, 'evaluating results')]
            evaluation_measurements = list(itertools.chain(*results))
        else:
            p = pool.map_async(self._evaluate_logs, evaluation_arguments)
            progress_bar_async(p, 'evaluating results', num_simulations)
            pool.close()
            evaluation_measurements = list(itertools.chain(*p.get()))

        return evaluation_measurements

    @staticmethod
    def _evaluate_logs(arguments):
        data: pd.DataFrame
        sim_log: pd.DataFrame
        settings, data, sim_log = arguments

        rep = sim_log.iloc[0].run_num
        evaluator = SimilarityEvaluator(data, sim_log, max_cases=1000)
        evaluator.measure_distance(Metric.DL)
        sim_values = [{'run_num': rep, **evaluator.similarity}]  # TODO: why list for a single dict?
        return sim_values


def _read_simulated_log(arguments: Tuple):
    log_path, log_column_mapping, simulation_repetition_index = arguments

    reader = LogReaderWriter(log_path=log_path, column_names=log_column_mapping)

    reader.df.rename(columns={'user': 'resource'}, inplace=True)
    reader.df['role'] = reader.df['resource']
    reader.df['source'] = 'simulation'
    reader.df['run_num'] = simulation_repetition_index
    reader.df = reader.df[~reader.df.task.isin(['Start', 'End'])]

    return reader.df
