import copy
import itertools
import multiprocessing
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, Optional, List, Tuple

import numpy as np
import pandas as pd
import yaml
from hyperopt import Trials, hp, fmin, STATUS_OK, STATUS_FAIL
from hyperopt import tpe
from tqdm import tqdm

from simod import support_utils as sup
from simod.analyzers.sim_evaluator import evaluate_logs
from simod.cli_formatter import print_message, print_subsection
from simod.configuration import StructureMiningAlgorithm, Metric, AndPriorORemove, GateManagement, PDFMethod
from simod.event_log_processing.reader import EventLogReader
from simod.hyperopt_pipeline import HyperoptPipeline
from simod.support_utils import get_project_dir, remove_asset, progress_bar_async
from .miner import StructureMiner, Settings as StructureMinerSettings
from .simulation import undifferentiated_resources_parameters, ProsimosSettings, simulate_with_prosimos, \
    PROSIMOS_COLUMN_MAPPING
from ..event_log_processing.event_log_ids import EventLogIDs
from ..event_log_processing.utilities import sample_log
from ..process_model.bpmn import BPMNReaderWriter


@dataclass
class Settings:
    """Settings for the structure optimizer."""
    project_name: Optional[str]

    gateway_probabilities: Optional[Union[GateManagement, List[GateManagement]]] = GateManagement.DISCOVERY
    max_evaluations: int = 1
    simulation_repetitions: int = 1
    pdef_method: Optional[PDFMethod] = None  # TODO: rename to distribution_discovery_method

    # Structure Miner Settings can be arrays of values, in that case different values are used for different repetition.
    # Structure Miner accepts only singular values for the following settings:
    #
    # for Split Miner 1 and 3
    epsilon: Optional[Union[float, List[float]]] = None
    eta: Optional[Union[float, List[float]]] = None
    # for Split Miner 2
    concurrency: Optional[Union[float, List[float]]] = 0.0
    #
    # Singular Structure Miner configuration used to compose the search space and split epsilon, eta and concurrency
    # lists into singular values.
    mining_algorithm: StructureMiningAlgorithm = StructureMiningAlgorithm.SPLIT_MINER_3
    #
    # Split Miner 3
    and_prior: List[AndPriorORemove] = field(default_factory=lambda: [AndPriorORemove.FALSE])
    or_rep: List[AndPriorORemove] = field(default_factory=lambda: [AndPriorORemove.FALSE])

    @staticmethod
    def from_stream(stream: Union[str, bytes]) -> 'Settings':
        settings = yaml.load(stream, Loader=yaml.FullLoader)

        project_name = settings.get('project_name', None)

        if 'structure_optimizer' in settings:
            settings = settings['structure_optimizer']

        gateway_probabilities = settings.get('gateway_probabilities', None)
        if gateway_probabilities is None:
            gateway_probabilities = settings.get('gate_management', None)  # legacy key support
        if gateway_probabilities is not None:
            if isinstance(gateway_probabilities, list):
                gateway_probabilities = [GateManagement.from_str(g) for g in gateway_probabilities]
            elif isinstance(gateway_probabilities, str):
                gateway_probabilities = GateManagement.from_str(gateway_probabilities)
            else:
                raise ValueError('Gateway probabilities must be a list or a string.')

        max_evaluations = settings.get('max_evaluations', None)
        if max_evaluations is None:
            max_evaluations = settings.get('max_eval_s', 1)  # legacy key support

        simulation_repetitions = settings.get('simulation_repetitions', 1)

        pdef_method = settings.get('pdef_method', None)
        if pdef_method is not None:
            pdef_method = PDFMethod.from_str(pdef_method)
        else:
            pdef_method = PDFMethod.DEFAULT

        epsilon = settings.get('epsilon', None)

        eta = settings.get('eta', None)

        concurrency = settings.get('concurrency', 0.0)

        mining_algorithm = settings.get('mining_algorithm', None)
        if mining_algorithm is None:
            mining_algorithm = settings.get('mining_alg', None)  # legacy key support
        if mining_algorithm is not None:
            mining_algorithm = StructureMiningAlgorithm.from_str(mining_algorithm)

        and_prior = settings.get('and_prior', None)
        if and_prior is not None:
            if isinstance(and_prior, list):
                and_prior = [AndPriorORemove.from_str(a) for a in and_prior]
            elif isinstance(and_prior, str):
                and_prior = [AndPriorORemove.from_str(and_prior)]
            else:
                raise ValueError('and_prior must be a list or a string.')

        or_rep = settings.get('or_rep', None)
        if or_rep is not None:
            if isinstance(or_rep, list):
                or_rep = [AndPriorORemove.from_str(o) for o in or_rep]
            elif isinstance(or_rep, str):
                or_rep = [AndPriorORemove.from_str(or_rep)]
            else:
                raise ValueError('or_rep must be a list or a string.')

        return Settings(
            project_name=project_name,
            gateway_probabilities=gateway_probabilities,
            max_evaluations=max_evaluations,
            simulation_repetitions=simulation_repetitions,
            pdef_method=pdef_method,
            epsilon=epsilon,
            eta=eta,
            concurrency=concurrency,
            mining_algorithm=mining_algorithm,
            and_prior=and_prior,
            or_rep=or_rep
        )


class StructureOptimizer(HyperoptPipeline):
    best_output: Optional[Path]
    best_parameters: dict
    measurements_file_name: Path

    _bayes_trials: Trials
    _settings: Settings
    _log_reader: EventLogReader
    _log_train: EventLogReader
    _log_validation: pd.DataFrame
    _original_log: EventLogReader
    _original_log_train: EventLogReader
    _original_log_validation: pd.DataFrame
    _temp_output: Path
    _log_ids: EventLogIDs

    def __init__(self, settings: Settings, log: EventLogReader, log_ids: Optional[EventLogIDs] = None):
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
        self._log_train = EventLogReader.copy_without_data(self._log_reader)
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

        def pipeline(trial_stage_settings: Union[Settings, dict]):
            print_subsection("Trial")
            print_message(f'train split: {len(pd.DataFrame(self._log_train.data)[self._log_ids.case].unique())}, '
                          f'validation split: {len(self._log_validation[self._log_ids.case].unique())}')

            if isinstance(trial_stage_settings, dict):
                trial_stage_settings = Settings(**trial_stage_settings)
            print_message(f'Parameters: {trial_stage_settings}', capitalize=False)

            status = STATUS_OK

            status, result = self.step(status, self._update_config_and_export_xes, trial_stage_settings.project_name)
            output_dir, log_path = result

            status, result = self.step(status, self._mine_structure, trial_stage_settings, output_dir, log_path)
            bpmn_reader, process_graph = result

            status, result = self.step(
                status,
                self._extract_parameters_undifferentiated,
                trial_stage_settings,
                bpmn_reader,
                process_graph)
            bpmn_path, json_path, simulation_cases = result

            status, result = self.step(
                status, self._simulate_undifferentiated,
                trial_stage_settings, bpmn_path, json_path, simulation_cases, output_dir)
            evaluation_measurements = result if status == STATUS_OK else []

            response = self._define_response(trial_stage_settings, status, evaluation_measurements, output_dir)

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
    def _define_search_space(settings: Settings) -> dict:
        split_miner_dependent_settings = {
            'gateway_probabilities': hp.choice('gateway_probabilities', settings.gateway_probabilities),
            'epsilon': hp.uniform('epsilon', *settings.epsilon), 'eta': hp.uniform('eta', *settings.eta),
            'and_prior': hp.choice('and_prior', AndPriorORemove.to_str(settings.and_prior)),
            'or_rep': hp.choice('or_rep', AndPriorORemove.to_str(settings.or_rep)),
            'concurrency': hp.uniform('concurrency', *settings.concurrency)
        }

        # TODO: better be more specific and specify the keys to add to space
        other_settings = copy.deepcopy(settings.__dict__)
        for key in split_miner_dependent_settings:
            other_settings.pop(key, None)

        space = split_miner_dependent_settings | other_settings

        return space

    def _update_config_and_export_xes(self, project_name: str) -> Tuple[Path, Path]:
        output_path = self._temp_output / sup.folder_id()
        sim_data_path = output_path / 'sim_data'
        sim_data_path.mkdir(parents=True, exist_ok=True)

        # Create customized event-log for the external tools
        xes_path = output_path / (project_name + '.xes')
        self._log_train.write_xes(xes_path)

        return output_path, xes_path

    @staticmethod
    def _mine_structure(settings: Settings, output_dir: Path, log_path: Path) -> Tuple:
        model_path = (output_dir / (settings.project_name + '.bpmn')).absolute()

        miner_settings = StructureMinerSettings(
            mining_algorithm=settings.mining_algorithm,
            epsilon=settings.epsilon,
            eta=settings.eta,
            concurrency=settings.concurrency,
            and_prior=settings.and_prior,
            or_rep=settings.or_rep,
        )

        _ = StructureMiner(miner_settings, xes_path=log_path, output_model_path=model_path)

        bpmn_reader = BPMNReaderWriter(model_path)
        process_graph = bpmn_reader.as_graph()

        return bpmn_reader, process_graph

    def _extract_parameters_undifferentiated(self, settings: Settings, bpmn_reader, process_graph) -> Tuple:
        bpmn_path: Path = bpmn_reader.model_path
        log = self._log_train.get_traces_df(include_start_end_events=True)
        pdf_method = self._settings.pdef_method

        simulation_parameters = undifferentiated_resources_parameters(
            log, self._log_ids, bpmn_path, process_graph, pdf_method, bpmn_reader, settings.gateway_probabilities)

        json_path = bpmn_path.with_suffix('.json')
        simulation_parameters.to_json_file(json_path)

        simulation_cases = log[self._log_ids.case].nunique()

        return bpmn_path, json_path, simulation_cases

    def _simulate_undifferentiated(
            self,
            settings: Settings,
            bpmn_path: Path,
            json_path: Path,
            simulation_cases: int,
            output_dir: Path):
        self._log_validation.rename(columns={'user': 'resource'}, inplace=True)
        self._log_validation['source'] = 'log'
        self._log_validation['run_num'] = 0
        self._log_validation['role'] = 'SYSTEM'
        self._log_validation = self._log_validation[~self._log_validation.task.isin(['Start', 'End'])]

        num_simulations = settings.simulation_repetitions
        cpu_count = multiprocessing.cpu_count()
        w_count = num_simulations if num_simulations <= cpu_count else cpu_count
        pool = multiprocessing.Pool(processes=w_count)

        # Simulate
        simulation_arguments = [
            ProsimosSettings(
                bpmn_path=bpmn_path,
                parameters_path=json_path,
                output_log_path=output_dir / 'sim_data' / f'{settings.project_name}_{rep}.csv',
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
            results = [evaluate_logs(arg) for arg in tqdm(evaluation_arguments, 'evaluating results')]
            evaluation_measurements = list(itertools.chain(*results))
        else:
            p = pool.map_async(evaluate_logs, evaluation_arguments)
            progress_bar_async(p, 'evaluating results', num_simulations)
            pool.close()
            evaluation_measurements = list(itertools.chain(*p.get()))

        return evaluation_measurements

    def _define_response(self, settings: Settings, status, simulation_results, output_dir: Path) -> dict:
        response = {
            'output': str(output_dir.absolute()),
            'status': status,
            'loss': None,
        }

        measurements = []

        # collecting measurements for saving

        optimization_parameters = {
            'gateway_probabilities': settings.gateway_probabilities,
            'output': str(output_dir.absolute()),
        }

        # structure miner parameters
        if settings.mining_algorithm in [StructureMiningAlgorithm.SPLIT_MINER_1,
                                         StructureMiningAlgorithm.SPLIT_MINER_3]:
            optimization_parameters['epsilon'] = settings.epsilon
            optimization_parameters['eta'] = settings.eta
            optimization_parameters['and_prior'] = settings.and_prior
            optimization_parameters['or_rep'] = settings.or_rep
        elif settings.mining_algorithm is StructureMiningAlgorithm.SPLIT_MINER_2:
            optimization_parameters['concurrency'] = settings.concurrency
        else:
            raise ValueError(settings.mining_algorithm)

        # simulation parameters
        if status == STATUS_OK:
            similarity = np.mean([x['sim_val'] for x in simulation_results])
            loss = (1 - similarity)

            response['loss'] = loss
            response['status'] = status if loss > 0 else STATUS_FAIL

            for sim_val in simulation_results:
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
        if os.path.getsize(self.measurements_file_name) > 0:
            sup.create_csv_file(measurements, self.measurements_file_name, mode='a')
        else:
            sup.create_csv_file_header(measurements, self.measurements_file_name)

        return response


def _read_simulated_log(arguments: Tuple):
    log_path, log_column_mapping, simulation_repetition_index = arguments

    reader = EventLogReader(log_path=log_path, column_names=log_column_mapping)

    reader.df.rename(columns={'user': 'resource'}, inplace=True)
    reader.df['role'] = reader.df['resource']
    reader.df['source'] = 'simulation'
    reader.df['run_num'] = simulation_repetition_index
    reader.df = reader.df[~reader.df.task.isin(['Start', 'End'])]

    return reader.df
