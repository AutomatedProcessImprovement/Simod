import json
import shutil
from pathlib import Path
from typing import Tuple, Optional

import hyperopt
import numpy as np
import pandas as pd
from hyperopt import Trials, hp, fmin, STATUS_OK, STATUS_FAIL
from hyperopt import tpe
from pix_utils.filesystem.file_manager import get_random_folder_id, get_random_file_id, remove_asset, create_folder

from .miner import StructureMiner
from .settings import HyperoptIterationParams
from ..bpm.reader_writer import BPMNReaderWriter
from ..cli_formatter import print_message, print_subsection, print_step
from ..event_log.event_log import EventLog
from ..settings.common_settings import Metric
from ..settings.control_flow_settings import ProcessModelDiscoveryAlgorithm, ControlFlowSettings
from ..simulation.parameters.miner import mine_default_24_7
from ..simulation.prosimos import simulate_and_evaluate
from ..utilities import hyperopt_step


class StructureOptimizer:
    # Event log with train/test partitions
    event_log: EventLog
    # Configuration settings
    settings: ControlFlowSettings
    # Root directory for the output files
    output_dir: Path
    # Path to the discovered process model
    model_path: Optional[Path]
    # Gateway probabilities
    gateway_probabilities: Optional[list]
    # Quality measure of each hyperopt iteration
    evaluation_measurements: pd.DataFrame

    def __init__(
            self,
            event_log: EventLog,
            settings: ControlFlowSettings,
            base_dir: Path,
            model_path: Optional[Path] = None,
    ):
        # Save parameters
        self.event_log = event_log
        self.settings = settings
        # Create and save output directoy path
        self.output_dir = base_dir / get_random_folder_id(prefix='control-flow')
        create_folder(self.output_dir)
        # Save model path and read activities-IDs map if it exists
        self.model_path = model_path
        self._process_graph = BPMNReaderWriter(model_path).as_graph() if model_path else None
        # Initialize table to store quality measures
        self.evaluation_measurements = pd.DataFrame(columns=[
            'value', 'metric', 'status', 'gateway_probabilities', 'epsilon',
            'eta', 'prioritize_parallelism', 'replace_or_joins', 'output_dir'
        ])

        # If needed, create path to export training log (XES format) for SplitMiner
        self._train_log_path = self.output_dir / (event_log.process_name + '.xes') if self.model_path is None else None
        # Instantiate trials for hyper-optimization process
        self._bayes_trials = Trials()

    def _optimization_objective(self, hyperopt_iteration_dict: dict):
        print_subsection("Structure Optimization Trial")

        # current trial folder
        output_dir = self.output_dir / get_random_folder_id(prefix='structure_trial_')
        output_dir.mkdir(parents=True, exist_ok=True)

        # casting a dictionary provided by hyperopt to PipelineSettings for convenience
        hyperopt_iteration_params = HyperoptIterationParams.from_hyperopt_dict(
            hyperopt_dict=hyperopt_iteration_dict,
            optimization_metric=self.settings.optimization_metric,
            mining_algorithm=self.settings.mining_algorithm,
            model_path=self.model_path,
            output_dir=output_dir,
            project_name=self.event_log.process_name,
        )
        print_message(f'Parameters: {hyperopt_iteration_params}', capitalize=False)

        # initializing status
        status = STATUS_OK

        # structure mining
        try:
            model_path = output_dir / (hyperopt_iteration_params.project_name + '.bpmn')
            if hyperopt_iteration_params.model_path is None:
                hyperopt_iteration_params.model_path = model_path
                print_step('Executing SplitMiner')
                status, result = hyperopt_step(status, self._mine_structure,
                                               hyperopt_iteration_params, self._train_log_path)

                self._process_graph = BPMNReaderWriter(hyperopt_iteration_params.model_path).as_graph()
            else:
                print_step('Model is provided, skipping SplitMiner execution')
                # We copy the model mostly for debugging purposes, so we have the model always in the output folder
                shutil.copy(self.model_path, model_path)
        except Exception as e:
            print_message(f'Process Discovery failed: {e}')
            status = STATUS_FAIL

        # simulation parameters mining
        # TODO perform this (except the gateway probabilities) just once before getting in the structure optimizer
        status, result = hyperopt_step(
            status,
            self._extract_parameters_undifferentiated,
            hyperopt_iteration_params,
            self._process_graph)

        json_path = result if status is STATUS_OK else None

        # simulation
        status, result = hyperopt_step(status, self._simulate_undifferentiated,
                                       hyperopt_iteration_params,
                                       self.settings.num_evaluations_per_iteration,
                                       json_path)
        evaluation_measurements = result if status == STATUS_OK else []

        # loss
        response, status = self._define_response(status, evaluation_measurements, hyperopt_iteration_params)
        print(f'StructureOptimizer pipeline response: {response}')

        # saving results
        self._process_measurements(hyperopt_iteration_params, status, evaluation_measurements)

        return response

    def run(self) -> Tuple[HyperoptIterationParams, Path]:
        """
        Run Control-Flow & Gateway Probabilities discovery
        :return: Tuple of the best settings, the path to the best model and the list of evaluation measurements.
        """
        # Check if model already provided
        need_to_discover_model = self.model_path is None
        # Define search space
        search_space = self._define_search_space(
            settings=self.settings,
            discover_model=need_to_discover_model
        )
        # If needed, write training event log to xes (SplitMiner needs XES as input)
        if need_to_discover_model:
            self.event_log.train_to_xes(self._train_log_path)
        # Launch optimization process
        best_hyperopt_params = fmin(
            fn=self._optimization_objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=self.settings.max_evaluations,
            trials=self._bayes_trials,
            show_progressbar=False
        )
        best_hyperopt_params = hyperopt.space_eval(search_space, best_hyperopt_params)
        # Process best results
        results = pd.DataFrame(self._bayes_trials.results).sort_values('loss')
        results_ok = results[results.status == STATUS_OK]
        best_model_path = results_ok.iloc[0].model_path
        assert best_model_path.exists(), f'Best model path {best_model_path} does not exist'
        best_settings = HyperoptIterationParams.from_hyperopt_dict(
            hyperopt_dict=best_hyperopt_params,
            optimization_metric=self.settings.optimization_metric,
            mining_algorithm=self.settings.mining_algorithm,
            model_path=None if need_to_discover_model else self.model_path,
            output_dir=best_model_path.parent,
            project_name=self.event_log.process_name,
        )
        # Save discovered gateway probabilities
        best_parameters_path = best_model_path.parent / 'simulation_parameters.json'
        self.gateway_probabilities = json.load(open(best_parameters_path, 'r'))['gateway_branching_probabilities']
        # Save best model path
        self.model_path = best_model_path
        # Save evaluation measurements
        self.evaluation_measurements.sort_values('value', ascending=False, inplace=True)
        self.evaluation_measurements.to_csv(self.output_dir / get_random_file_id(extension="csv", prefix="evaluation_"), index=False)
        # Return settings of the best iteration and path to the best simulation parameters
        return best_settings, best_parameters_path

    def cleanup(self):
        remove_asset(self.output_dir)

    @staticmethod
    def _define_search_space(settings: ControlFlowSettings, discover_model: bool) -> dict:
        space = {}
        # Add gateway probabilities method
        if isinstance(settings.gateway_probabilities, list):
            space['gateway_probabilities_method'] = hp.choice('gateway_probabilities_method', settings.gateway_probabilities)
        else:
            space['gateway_probabilities_method'] = settings.gateway_probabilities
        # Process model discovery parameters if we need to discover it
        if discover_model:
            if settings.mining_algorithm is ProcessModelDiscoveryAlgorithm.SPLIT_MINER_2:
                # Split Miner 2, concurrency parameter
                if isinstance(settings.concurrency, tuple):
                    space['concurrency'] = hp.uniform('concurrency', settings.concurrency[0], settings.concurrency[1])
                else:
                    space['concurrency'] = settings.concurrency
            elif settings.mining_algorithm is ProcessModelDiscoveryAlgorithm.SPLIT_MINER_3:
                # Split Miner 3
                # epsilon
                if isinstance(settings.epsilon, tuple):
                    space['epsilon'] = hp.uniform('epsilon', settings.epsilon[0], settings.epsilon[1])
                else:
                    space['epsilon'] = settings.epsilon
                # eta
                if isinstance(settings.eta, tuple):
                    space['eta'] = hp.uniform('eta', settings.eta[0], settings.eta[1])
                else:
                    space['eta'] = settings.eta
                # prioritize_parallelism
                if isinstance(settings.prioritize_parallelism, list):
                    space['prioritize_parallelism'] = hp.choice(
                        'prioritize_parallelism',
                        [str(value) for value in settings.prioritize_parallelism]
                    )
                else:
                    space['prioritize_parallelism'] = str(settings.prioritize_parallelism)
                # replace_or_joins
                if isinstance(settings.replace_or_joins, list):
                    space['replace_or_joins'] = hp.choice(
                        'replace_or_joins',
                        [str(value) for value in settings.replace_or_joins]
                    )
                else:
                    space['replace_or_joins'] = str(settings.replace_or_joins)
        # Return search space
        return space

    @staticmethod
    def _define_response(
            status: str,
            evaluation_measurements: list,
            pipeline_settings: HyperoptIterationParams,
    ) -> Tuple[dict, str]:
        distance = np.mean([x['value'] for x in evaluation_measurements])
        status = status if distance > 0 else STATUS_FAIL

        response = {
            'loss': distance,
            'status': status,
            'output_dir': pipeline_settings.output_dir,
            'model_path': pipeline_settings.model_path,
        }

        return response, status

    def _process_measurements(
            self,
            settings: HyperoptIterationParams,
            status,
            evaluation_measurements):
        optimization_parameters = settings.to_dict()
        optimization_parameters['status'] = status

        if status == STATUS_OK:
            for measurement in evaluation_measurements:
                values = {
                    'value': measurement['value'],
                    'metric': measurement['metric'],
                }
                values = values | optimization_parameters
                self.evaluation_measurements = pd.concat([self.evaluation_measurements, pd.DataFrame([values])])
        else:
            values = {
                'value': 0,
                'metric': Metric.DL,
            }
            values = values | optimization_parameters
            self.evaluation_measurements = pd.concat([self.evaluation_measurements, pd.DataFrame([values])])

    @staticmethod
    def _mine_structure(
            settings: HyperoptIterationParams,
            log_path: Path) -> None:
        StructureMiner(
            settings.mining_algorithm,
            log_path,
            settings.model_path,
            concurrency=settings.concurrency,
            eta=settings.eta,
            epsilon=settings.epsilon,
            prioritize_parallelism=settings.prioritize_parallelism,
            replace_or_joins=settings.replace_or_joins,
        ).run()

    def _extract_parameters_undifferentiated(self, settings: HyperoptIterationParams, process_graph) -> Path:
        # Below, we mine simulation parameters with undifferentiated resources, because we optimize the structure,
        # not calendars. So, we do not need to differentiate resources.
        simulation_parameters = mine_default_24_7(
            self.event_log.train_partition,
            self.event_log.log_ids,
            settings.model_path,
            process_graph,
            settings.gateway_probabilities_method)

        json_path = settings.model_path.parent / 'simulation_parameters.json'
        simulation_parameters.to_json_file(json_path)

        return json_path

    def _simulate_undifferentiated(
            self,
            settings: HyperoptIterationParams,
            simulation_repetitions: int,
            json_path: Path):
        self.event_log.validation_partition['source'] = 'log'
        self.event_log.validation_partition['run_num'] = 0
        self.event_log.validation_partition['role'] = 'SYSTEM'

        return simulate_and_evaluate(
            model_path=settings.model_path,
            parameters_path=json_path,
            output_dir=settings.output_dir,
            simulation_cases=self.event_log.validation_partition[self.event_log.log_ids.case].nunique(),
            simulation_start_time=self.event_log.validation_partition[self.event_log.log_ids.start_time].min(),
            validation_log=self.event_log.validation_partition,
            validation_log_ids=self.event_log.log_ids,
            metrics=[self.settings.optimization_metric],
            num_simulations=simulation_repetitions,
        )
