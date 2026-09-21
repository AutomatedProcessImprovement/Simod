import json
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import hyperopt
import numpy as np
import pandas as pd
from hyperopt import STATUS_FAIL, STATUS_OK, Trials, fmin, hp, tpe
from pix_framework.discovery.case_arrival import discover_case_arrival_model, CaseArrivalModel
from pix_framework.filesystem.file_manager import create_folder, get_random_folder_id, remove_asset

from .settings import HyperoptIterationParams
from ..cli_formatter import print_message, print_step, print_subsection
from ..event_log.event_log import EventLog
from ..settings.case_arrival_settings import CaseArrivalSettings
from ..simulation.parameters.BPS_model import BPSModel
from ..simulation.prosimos import simulate_and_evaluate
from ..utilities import get_process_model_path, get_simulation_parameters_path, hyperopt_step


class CaseArrivalOptimizer:
    """
    Optimizes the case-arrival of a business process simulation (BPS) model using hyperparameter
    optimization.

    This class performs iterative optimization to refine the case arrival model of a BPS model..

    The search space is built based on the parameters ranges in [settings].

    Attributes
    ----------
    event_log : :class:`EventLog`
        Event log containing train and validation partitions.
    initial_bps_model : :class:`BPSModel`
        Business process simulation (BPS) model to use as a base, by replacing its control-flow model
        with the discovered one in each iteration.
    settings : :class:`~simod.settings.case_arrival_settings.CaseArrivalSettings`
        Configuration settings to build the search space for the optimization process.
    base_directory : :class:`pathlib.Path`
        Root directory where output files will be stored.
    best_bps_model : :class:`BPSModel`, optional
        Best discovered BPS model after the optimization process.
    evaluation_measurements : :class:`pandas.DataFrame`
        Quality measures recorded for each hyperopt iteration.

    Notes
    -----
    - Currently, this process only optimizes the threshold to discard outliers when learning the
    inter-arrival probability distribution.
    """

    # Event log with train/validation partitions
    event_log: EventLog
    # BPS model taken as starting point
    initial_bps_model: BPSModel
    # Configuration settings
    settings: CaseArrivalSettings
    # Root directory for the output files
    base_directory: Path
    # Path to the best process model
    best_bps_model: Optional[BPSModel]
    # Quality measure of each hyperopt iteration
    evaluation_measurements: pd.DataFrame

    # Set of trials for the hyperparameter optimization process
    _bayes_trials = Trials

    def __init__(
        self,
        event_log: EventLog,
        bps_model: BPSModel,
        settings: CaseArrivalSettings,
        base_directory: Path,
    ):
        # Save event log, optimization settings, and output directory
        self.event_log = event_log
        self.initial_bps_model = bps_model.deep_copy()
        self.settings = settings
        self.base_directory = base_directory
        # Check if it is needed to discover the process model
        self.best_bps_model = None
        # Initialize table to store quality measures of each iteration
        self.evaluation_measurements = pd.DataFrame(
            columns=[
                "distance",
                "metric",
                "status",
                "outlier_threshold",
            ]
        )
        # Instantiate trials for hyper-optimization process
        self._bayes_trials = Trials()
        self.iteration_index = 0

    def _hyperopt_iteration(self, hyperopt_iteration_dict: dict):
        # Report new iteration
        print_subsection(f"Case Arrival Model optimization iteration {self.iteration_index}")
        # Initialize status
        status = STATUS_OK
        # Create folder for this iteration
        output_dir = self.base_directory / get_random_folder_id(prefix="iteration_")
        create_folder(output_dir)
        # Initialize BPS model for this iteration
        current_bps_model = self.initial_bps_model.deep_copy()
        # Parameters of this iteration
        hyperopt_iteration_params = HyperoptIterationParams.from_hyperopt_dict(
            hyperopt_dict=hyperopt_iteration_dict,
            optimization_metric=self.settings.optimization_metric,
            output_dir=output_dir,
            project_name=self.event_log.process_name,
        )
        print_message(f"Parameters: {hyperopt_iteration_params}")

        # Discover case arrival model
        status, current_bps_model.case_arrival_model = hyperopt_step(
            status,
            self._discover_case_arrival_model,
            hyperopt_iteration_params,
        )

        # Simulate candidate and evaluate its quality
        status, evaluation_measurements = hyperopt_step(
            status,
            self._simulate_bps_model,
            current_bps_model,
            hyperopt_iteration_params.output_dir
        )

        # Define the response of this iteration
        status, response = self._define_response(
            status, evaluation_measurements, hyperopt_iteration_params.output_dir, current_bps_model.process_model
        )
        print(f"Case Arrival Model optimization iteration response: {response}")

        # Save the quality of this evaluation and increase iteration index
        self._process_measurements(hyperopt_iteration_params, status, evaluation_measurements)
        self.iteration_index += 1

        return response

    def run(self) -> HyperoptIterationParams:
        """
        Run the case arrival optimization process.

        This method defines the hyperparameter search space and executes a
        TPE-hyperparameter optimization process to discover the best case arrival model.
        It evaluates multiple iterations and selects the best-performing set of parameters
        for its discovery.

        Returns
        -------
        :class:`~simod.case_arrival.settings.HyperoptIterationParams`
            The parameters of the best iteration of the optimization process.
        """
        # Define search space
        self.iteration_index = 0
        search_space = self._define_search_space()

        # Launch optimization process
        params_best_iteration = fmin(
            fn=self._hyperopt_iteration,
            space=search_space,
            algo=tpe.suggest,
            max_evals=self.settings.num_iterations,
            trials=self._bayes_trials,
            show_progressbar=False,
        )
        params_best_iteration = hyperopt.space_eval(search_space, params_best_iteration)

        # Process best results
        results = pd.DataFrame(self._bayes_trials.results).sort_values("loss")
        best_result = results[results.status == STATUS_OK].iloc[0]

        # Re-build parameters of the best hyperopt iteration
        best_hyperopt_parameters = HyperoptIterationParams.from_hyperopt_dict(
            hyperopt_dict=params_best_iteration,
            optimization_metric=self.settings.optimization_metric,
            output_dir=best_result["output_dir"],
            project_name=self.event_log.process_name,
        )

        # Instantiate best BPS model
        self.best_bps_model = self.initial_bps_model.deep_copy()
        # Update best process model (save it in base directory)
        self.best_bps_model.process_model = get_process_model_path(self.base_directory, self.event_log.process_name)
        shutil.copyfile(best_result["process_model_path"], self.best_bps_model.process_model)
        # Update simulation parameters (save them in base directory)
        best_parameters_path = get_simulation_parameters_path(self.base_directory, self.event_log.process_name)
        shutil.copyfile(
            get_simulation_parameters_path(best_result["output_dir"], self.event_log.process_name), best_parameters_path
        )
        # Update case arrival model
        self.best_bps_model.case_arrival_model = CaseArrivalModel.from_dict(json.load(open(best_parameters_path, "r")))

        # Save evaluation measurements
        self.evaluation_measurements.sort_values("distance", ascending=True, inplace=True)
        self.evaluation_measurements.to_csv(self.base_directory / "evaluation_measures.csv", index=False)

        # Return settings of the best iteration
        return best_hyperopt_parameters

    def _define_search_space(self) -> dict:
        space = {}
        # Outlier threshold
        if isinstance(self.settings.outlier_threshold, tuple):
            space["outlier_threshold"] = hp.uniform(
                "outlier_threshold",
                self.settings.outlier_threshold[0],
                self.settings.outlier_threshold[1]
            )
        else:
            space["outlier_threshold"] = self.settings.outlier_threshold
        return space

    def cleanup(self):
        remove_asset(self.base_directory)

    @staticmethod
    def _define_response(
        status: str,
        evaluation_measurements: list,
        output_dir: Path,
        process_model_path: Path
    ) -> Tuple[str, dict]:
        # Compute mean distance if status is OK
        if status is STATUS_OK:
            distance = np.mean([x["distance"] for x in evaluation_measurements])
            # Change status if distance value is negative
            if distance < 0.0:
                status = STATUS_FAIL
        else:
            distance = 1.0
        # Define response dict
        response = {
            "loss": distance,  # Loss value for the fmin function
            "status": status,  # Status of the optimization iteration
            "output_dir": output_dir,
            "process_model_path": process_model_path,
        }
        # Return updated status and processed response
        return status, response

    def _process_measurements(self, params: HyperoptIterationParams, status, evaluation_measurements):
        optimization_parameters = params.to_dict()
        optimization_parameters["status"] = status

        if status == STATUS_OK:
            for measurement in evaluation_measurements:
                values = {
                    "distance": measurement["distance"],
                    "metric": measurement["metric"],
                }
                values = values | optimization_parameters
                self.evaluation_measurements = pd.concat([self.evaluation_measurements, pd.DataFrame([values])])
        else:
            values = {
                "distance": 0,
                "metric": params.optimization_metric,
            }
            values = values | optimization_parameters
            self.evaluation_measurements = pd.concat([self.evaluation_measurements, pd.DataFrame([values])])

    def _discover_case_arrival_model(self, params: HyperoptIterationParams) -> CaseArrivalModel:
        print_step(f"Discovering Case Arrival Model with threshold {params.outlier_threshold}")
        return discover_case_arrival_model(
            self.event_log.train_partition,
            self.event_log.log_ids,
            outlier_threshold=params.outlier_threshold,
        )

    def _simulate_bps_model(self, bps_model: BPSModel, output_dir: Path) -> List[dict]:
        bps_model.replace_activity_names_with_ids()

        json_parameters_path = bps_model.to_json(output_dir, self.event_log.process_name)

        evaluation_measures = simulate_and_evaluate(
            process_model_path=bps_model.process_model,
            parameters_path=json_parameters_path,
            output_dir=output_dir,
            simulation_cases=self.event_log.validation_partition[self.event_log.log_ids.case].nunique(),
            simulation_start_time=self.event_log.validation_partition[self.event_log.log_ids.start_time].min(),
            validation_log=self.event_log.validation_partition,
            validation_log_ids=self.event_log.log_ids,
            metrics=[self.settings.optimization_metric],
            num_simulations=self.settings.num_evaluations_per_iteration,
        )

        return evaluation_measures
