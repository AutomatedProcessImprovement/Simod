import json
from pathlib import Path
from typing import Tuple, Optional, List

import hyperopt
import numpy as np
import pandas as pd
from hyperopt import Trials, hp, fmin, STATUS_OK, STATUS_FAIL
from hyperopt import tpe
from pix_utils.filesystem.file_manager import get_random_folder_id, remove_asset, create_folder

from .miner import StructureMiner
from .settings import HyperoptIterationParams
from ..bpm.reader_writer import BPMNReaderWriter
from ..cli_formatter import print_message, print_subsection, print_step
from ..event_log.event_log import EventLog
from ..settings.common_settings import Metric
from ..settings.control_flow_settings import ProcessModelDiscoveryAlgorithm, ControlFlowSettings
from ..simulation.parameters.gateway_probabilities import GatewayProbabilities
from ..simulation.parameters.miner import mine_default_24_7
from ..simulation.prosimos import simulate_and_evaluate
from ..utilities import hyperopt_step


class StructureOptimizer:
    # Event log with train/test partitions
    event_log: EventLog
    # Configuration settings
    settings: ControlFlowSettings
    # Root directory for the output files
    base_directory: Path
    # Path to the process model
    model_path: Path
    # Gateway probabilities
    gateway_probabilities: List[GatewayProbabilities]
    # Quality measure of each hyperopt iteration
    evaluation_measurements: pd.DataFrame

    # Flag indicating if the model is provided of it needs to be discovered
    _need_to_discover_model: bool
    # Path to the training log in XES format, needed for Split Miner
    _xes_train_log_path: Optional[Path] = None

    def __init__(
            self,
            event_log: EventLog,
            settings: ControlFlowSettings,
            base_directory: Path,
            model_path: Optional[Path] = None,
    ):
        # Save event log, optimization settings, and output directory
        self.event_log = event_log
        self.settings = settings
        self.base_directory = base_directory
        # Save model path
        if model_path is not None:
            # Provided
            self.model_path = model_path
            self._need_to_discover_model = False
        else:
            # Not provided, create path to best discovered model
            self.model_path = self.base_directory / f"{self.event_log.process_name}.bpmn"
            self._need_to_discover_model = True
            # Create path to export training log (XES format) for SplitMiner
            self._xes_train_log_path = self.base_directory / (self.event_log.process_name + '.xes')
        # Initialize empty list for gateway probabilities
        self.gateway_probabilities = []
        # Initialize table to store quality measures
        self.evaluation_measurements = pd.DataFrame(columns=[
            'distance', 'metric', 'status', 'gateway_probabilities', 'epsilon',
            'eta', 'prioritize_parallelism', 'replace_or_joins', 'output_dir'
        ])

        # Instantiate trials for hyper-optimization process
        self._bayes_trials = Trials()

    def _hyperopt_iteration(self, hyperopt_iteration_dict: dict):
        print_subsection("Control-flow optimization iteration")
        # Initializing status
        status = STATUS_OK

        # Create folder for this iteration
        output_dir = self.base_directory / get_random_folder_id(prefix='iteration_')
        create_folder(output_dir)

        # Parse the parameters of this iteration of the hyperopt process
        hyperopt_iteration_params = HyperoptIterationParams.from_hyperopt_dict(
            hyperopt_dict=hyperopt_iteration_dict,
            optimization_metric=self.settings.optimization_metric,
            mining_algorithm=self.settings.mining_algorithm,
            model_path=None if self._need_to_discover_model else self.model_path,
            output_dir=output_dir,
            project_name=self.event_log.process_name,
        )
        print_message(f'Parameters: {hyperopt_iteration_params}', capitalize=False)

        # Discover process model if needed
        if self._need_to_discover_model:
            try:
                hyperopt_iteration_params.model_path = output_dir / f"{self.event_log.process_name}.bpmn"
                status, _ = hyperopt_step(status, self._discover_process_model, hyperopt_iteration_params)
            except Exception as e:
                print_message(f'Process Discovery failed: {e}')
                status = STATUS_FAIL
        else:
            hyperopt_iteration_params.model_path = self.model_path

        # simulation parameters mining
        # TODO only discover gateway probabilities here, the rest are given by SIMOD
        status, result = hyperopt_step(
            status,
            self._extract_parameters_undifferentiated,
            hyperopt_iteration_params)

        json_path = result if status is STATUS_OK else None

        # Simulate BPS model of this iteration and evaluate its quality
        status, evaluation_measurements = hyperopt_step(status, self._simulate_undifferentiated,
                                                        hyperopt_iteration_params,
                                                        self.settings.num_evaluations_per_iteration,
                                                        json_path)

        # Define the response of this iteration
        status, response = self._define_response(status, evaluation_measurements, hyperopt_iteration_params)
        print(f"Control-flow iteration response: {response}")

        # Save quality of this evaluation
        self._process_measurements(hyperopt_iteration_params, status, evaluation_measurements)

        return response

    def run(self) -> Tuple[HyperoptIterationParams, Path]:
        """
        Run Control-Flow & Gateway Probabilities discovery
        :return: Tuple of the best settings, the path to the best model and the list of evaluation measurements.
        """
        # Define search space
        search_space = self._define_search_space(settings=self.settings)
        # If needed, write training event log to xes (SplitMiner needs XES as input)
        if self._need_to_discover_model:
            self.event_log.train_to_xes(self._xes_train_log_path)
        # Launch optimization process
        best_hyperopt_params = fmin(
            fn=self._hyperopt_iteration,
            space=search_space,
            algo=tpe.suggest,
            max_evals=self.settings.max_evaluations,
            trials=self._bayes_trials,
            show_progressbar=False
        )
        best_hyperopt_params = hyperopt.space_eval(search_space, best_hyperopt_params)
        # Process best results
        results = pd.DataFrame(self._bayes_trials.results).sort_values('loss')
        best_model_path = results[results.status == STATUS_OK].iloc[0].model_path
        assert best_model_path.exists(), f'Best model path {best_model_path} does not exist'
        best_settings = HyperoptIterationParams.from_hyperopt_dict(
            hyperopt_dict=best_hyperopt_params,
            optimization_metric=self.settings.optimization_metric,
            mining_algorithm=self.settings.mining_algorithm,
            model_path=None if self._need_to_discover_model else self.model_path,
            output_dir=best_model_path.parent,
            project_name=self.event_log.process_name,
        )
        # Save discovered gateway probabilities
        best_parameters_path = best_model_path.parent / 'simulation_parameters.json'
        self.gateway_probabilities = [
            GatewayProbabilities.from_dict(gateway_probabilities)
            for gateway_probabilities in json.load(open(best_parameters_path, 'r'))['gateway_branching_probabilities']
        ]
        # Save best model path
        self.model_path = best_model_path
        # Save evaluation measurements
        self.evaluation_measurements.sort_values('distance', ascending=True, inplace=True)
        self.evaluation_measurements.to_csv(self.base_directory / "evaluation_measures.csv", index=False)
        # Return settings of the best iteration and path to the best simulation parameters
        return best_settings, best_parameters_path

    def _define_search_space(self, settings: ControlFlowSettings) -> dict:
        space = {}
        # Add gateway probabilities method
        if isinstance(settings.gateway_probabilities, list):
            space['gateway_probabilities_method'] = hp.choice('gateway_probabilities_method', settings.gateway_probabilities)
        else:
            space['gateway_probabilities_method'] = settings.gateway_probabilities
        # Process model discovery parameters if we need to discover it
        if self._need_to_discover_model:
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

    def cleanup(self):
        remove_asset(self.base_directory)

    @staticmethod
    def _define_response(
            status: str,
            evaluation_measurements: list,
            pipeline_settings: HyperoptIterationParams,
    ) -> Tuple[str, dict]:
        # Compute mean distance if status is OK
        if status is STATUS_OK:
            distance = np.mean([x['distance'] for x in evaluation_measurements])
            # Change status if distance value is negative
            if distance < 0.0:
                status = STATUS_FAIL
        else:
            distance = 1.0
        # Define response dict
        response = {
            'loss': distance,  # Loss value for the fmin function
            'status': status,  # Status of the optimization iteration
            'output_dir': pipeline_settings.output_dir,
            'model_path': pipeline_settings.model_path,
        }

        return status, response

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
                    'distance': measurement['distance'],
                    'metric': measurement['metric'],
                }
                values = values | optimization_parameters
                self.evaluation_measurements = pd.concat([self.evaluation_measurements, pd.DataFrame([values])])
        else:
            values = {
                'distance': 0,
                'metric': Metric.DL,
            }
            values = values | optimization_parameters
            self.evaluation_measurements = pd.concat([self.evaluation_measurements, pd.DataFrame([values])])

    def _discover_process_model(self, params: HyperoptIterationParams):
        print_step('Discovering Process Model with SplitMiner')
        StructureMiner(
            params.mining_algorithm,
            self._xes_train_log_path,
            params.model_path,
            concurrency=params.concurrency,
            eta=params.eta,
            epsilon=params.epsilon,
            prioritize_parallelism=params.prioritize_parallelism,
            replace_or_joins=params.replace_or_joins,
        ).run()

    def _extract_parameters_undifferentiated(self, settings: HyperoptIterationParams) -> Path:
        # Below, we mine simulation parameters with undifferentiated resources, because we optimize the structure,
        # not calendars. So, we do not need to differentiate resources.
        process_graph = self._process_graph = BPMNReaderWriter(settings.model_path).as_graph()
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
