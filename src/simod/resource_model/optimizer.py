import copy
import json
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import hyperopt
import numpy as np
import pandas as pd
from hyperopt import STATUS_FAIL, STATUS_OK, Trials, fmin, hp, tpe
from pix_framework.discovery.resource_calendar_and_performance.calendar_discovery_parameters import (
    CalendarDiscoveryParameters,
)
from pix_framework.discovery.resource_model import ResourceModel, discover_resource_model
from pix_framework.discovery.resource_profiles import discover_pool_resource_profiles
from pix_framework.filesystem.file_manager import create_folder, get_random_folder_id, remove_asset

from .repair import repair_with_missing_activities
from .settings import HyperoptIterationParams
from ..batching.discovery import discover_batching_rules
from ..cli_formatter import print_message, print_step, print_subsection
from ..event_log.event_log import EventLog
from ..prioritization.discovery import discover_prioritization_rules
from ..settings.resource_model_settings import CalendarType, ResourceModelSettings
from ..simulation.parameters.BPS_model import BPSModel
from ..simulation.prosimos import simulate_and_evaluate
from ..utilities import get_process_model_path, get_simulation_parameters_path, hyperopt_step


class ResourceModelOptimizer:
    # Event log with train/validation partitions
    event_log: EventLog
    # BPS model taken as starting point
    initial_bps_model: BPSModel
    # Configuration settings
    settings: ResourceModelSettings
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
        settings: ResourceModelSettings,
        base_directory: Path,
        model_activities: Optional[list[str]] = None,
    ):
        # Save event log, optimization settings, and output directory
        self.event_log = event_log
        self.initial_bps_model = bps_model.deep_copy()
        self.settings = settings
        self.base_directory = base_directory
        self.model_activities = model_activities
        # Initialize table to store quality measures of each iteration
        self.evaluation_measurements = pd.DataFrame(
            columns=[
                "distance",
                "metric",
                "status",
                "discovery_type",
                "granularity",
                "confidence",
                "support",
                "participation",
                "output_dir",
            ]
        )
        # Instantiate trials for hyper-optimization process
        self._bayes_trials = Trials()
        # Discover resource pools (performance purposes) if needed
        if self.settings.discovery_type is CalendarType.DIFFERENTIATED_BY_POOL:
            self._resource_pools = discover_pool_resource_profiles(
                self.event_log.train_partition, self.event_log.log_ids
            )
        else:
            self._resource_pools = None
        # Prioritization
        if self.settings.discover_prioritization_rules and len(self.initial_bps_model.case_attributes) > 0:
            print_subsection("Discovering prioritization rules")
            self._prioritization_rules = discover_prioritization_rules(
                self.event_log.train_partition,
                self.event_log.log_ids,
                self.initial_bps_model.case_attributes,
            )
        elif self.settings.discover_prioritization_rules:
            print_subsection("0 case attributes discovered, turning off prioritization discovery.")
            self._prioritization_rules = []
        else:
            self._prioritization_rules = None
        # Batching
        if self.settings.discover_batching_rules:
            print_subsection("Discovering batching rules")
            self._batching_rules = discover_batching_rules(self.event_log.train_partition, self.event_log.log_ids)
        else:
            self._batching_rules = None

    def _hyperopt_iteration(self, hyperopt_iteration_dict: dict):
        # Report new iteration
        print_subsection("Resource Model optimization iteration")

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
            discovery_type=self.settings.discovery_type,
            output_dir=output_dir,
            process_model_path=current_bps_model.process_model,
            project_name=self.event_log.process_name,
        )
        print_message(f"Parameters: {hyperopt_iteration_params}")

        # Discover resource model
        status, current_bps_model.resource_model = hyperopt_step(
            status, self._discover_resource_model, hyperopt_iteration_params.calendar_discovery_params
        )

        if self.model_activities is not None:
            repair_with_missing_activities(
                resource_model=current_bps_model.resource_model,
                model_activities=self.model_activities,
                event_log=self.event_log.train_validation_partition,
                log_ids=self.event_log.log_ids,
            )

        # Add prioritization if needed
        if hyperopt_iteration_params.discover_prioritization_rules:
            current_bps_model.prioritization_rules = self._prioritization_rules

        # Add batching rules if needed
        if hyperopt_iteration_params.discover_batching_rules:
            current_bps_model.batching_rules = self._batching_rules

        # Simulate candidate and evaluate its quality
        status, evaluation_measurements = hyperopt_step(
            status,
            self._simulate_bps_model,
            current_bps_model,
            hyperopt_iteration_params.output_dir,
            hyperopt_iteration_params.calendar_discovery_params.granularity,
        )

        # Define the response of this iteration
        status, response = self._define_response(
            status, evaluation_measurements, hyperopt_iteration_params.output_dir, current_bps_model.process_model
        )
        print(f"Resource Model optimization iteration response: {response}")

        # Save the quality of this evaluation
        self._process_measurements(hyperopt_iteration_params, status, evaluation_measurements)

        return response

    def run(self) -> HyperoptIterationParams:
        """
        Run Resource Model (resource profiles, resource calendars and activity-resource performance) discovery.
        :return: The parameters of the best iteration of the optimization process.
        """
        # Define search space
        search_space = self._define_search_space(settings=self.settings)

        # Launch optimization process
        best_hyperopt_params = fmin(
            fn=self._hyperopt_iteration,
            space=search_space,
            algo=tpe.suggest,
            max_evals=self.settings.num_iterations,
            trials=self._bayes_trials,
            show_progressbar=False,
        )
        best_hyperopt_params = hyperopt.space_eval(search_space, best_hyperopt_params)

        # Process best results
        results = pd.DataFrame(self._bayes_trials.results).sort_values("loss")
        best_result = results[results.status == STATUS_OK].iloc[0]

        # Re-build parameters of the best hyperopt iteration
        best_hyperopt_parameters = HyperoptIterationParams.from_hyperopt_dict(
            hyperopt_dict=best_hyperopt_params,
            optimization_metric=self.settings.optimization_metric,
            discovery_type=self.settings.discovery_type,
            output_dir=best_result["output_dir"],
            project_name=self.event_log.process_name,
            process_model_path=self.initial_bps_model.process_model,
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
        # Update resource model
        self.best_bps_model.resource_model = ResourceModel.from_dict(json.load(open(best_parameters_path, "r")))

        # Save evaluation measurements
        self.evaluation_measurements.sort_values("distance", ascending=True, inplace=True)
        self.evaluation_measurements.to_csv(self.base_directory / "evaluation_measures.csv", index=False)

        # Return settings of the best iteration
        return best_hyperopt_parameters

    def _discover_resource_model(self, params: CalendarDiscoveryParameters) -> ResourceModel:
        print_step(f"Discovering resource model with {params}")
        return discover_resource_model(
            event_log=self.event_log.train_partition,
            log_ids=self.event_log.log_ids,
            params=params,
            provided_profiles=copy.deepcopy(self._resource_pools),
        )

    def cleanup(self):
        print_step(f"Removing {self.base_directory}")
        remove_asset(self.base_directory)

    def _define_search_space(self, settings: ResourceModelSettings):
        space = {}

        # If discovery type requires discovery, create search space for parameters
        if settings.discovery_type in [
            CalendarType.UNDIFFERENTIATED,
            CalendarType.DIFFERENTIATED_BY_RESOURCE,
            CalendarType.DIFFERENTIATED_BY_POOL,
        ]:
            if isinstance(settings.granularity, tuple):
                space["granularity"] = hp.uniform("granularity", settings.granularity[0], settings.granularity[1])
            else:
                space["granularity"] = settings.granularity

            if isinstance(settings.confidence, tuple):
                space["confidence"] = hp.uniform("confidence", settings.confidence[0], settings.confidence[1])
            else:
                space["confidence"] = settings.confidence

            if isinstance(settings.support, tuple):
                space["support"] = hp.uniform("support", settings.support[0], settings.support[1])
            else:
                space["support"] = settings.support

            if isinstance(settings.participation, tuple):
                space["participation"] = hp.uniform(
                    "participation", settings.participation[0], settings.participation[1]
                )
            else:
                space["participation"] = settings.participation

            if settings.discover_prioritization_rules and len(self._prioritization_rules) > 0:
                space["discover_prioritization_rules"] = hp.choice("discover_prioritization_rules", [True, False])
            else:
                space["discover_prioritization_rules"] = False

            if settings.discover_batching_rules and len(self._batching_rules) > 0:
                space["discover_batching_rules"] = hp.choice("discover_batching_rules", [True, False])
            else:
                space["discover_batching_rules"] = False
        elif settings.discovery_type == CalendarType.DIFFERENTIATED_BY_RESOURCE_FUZZY:
            if isinstance(settings.granularity, tuple):
                space["granularity"] = hp.uniform("granularity", settings.granularity[0], settings.granularity[1])
            else:
                space["granularity"] = settings.granularity

            if isinstance(settings.fuzzy_angle, tuple):
                space["fuzzy_angle"] = hp.uniform("fuzzy_angle", settings.fuzzy_angle[0], settings.fuzzy_angle[1])
            else:
                space["fuzzy_angle"] = settings.fuzzy_angle

        return space

    def _process_measurements(self, params: HyperoptIterationParams, status: str, evaluation_measurements: list):
        data = {
            "output_dir": params.output_dir,
            "metric": params.optimization_metric,
            "discovery_type": params.calendar_discovery_params.discovery_type,
            "granularity": params.calendar_discovery_params.granularity,
            "confidence": params.calendar_discovery_params.confidence,
            "support": params.calendar_discovery_params.support,
            "participation": params.calendar_discovery_params.participation,
            "discover_prioritization_rules": params.discover_prioritization_rules,
            "discover_batching_rules": params.discover_batching_rules,
            "status": status,
        }
        if status == STATUS_OK:
            for measurement in evaluation_measurements:
                values = {
                    "distance": measurement["distance"],
                    "metric": measurement["metric"],
                }
                values = values | data
                self.evaluation_measurements = pd.concat([self.evaluation_measurements, pd.DataFrame([values])])
        else:
            values = {
                "distance": 0,
                "metric": params.optimization_metric,
            }
            values = values | data
            self.evaluation_measurements = pd.concat([self.evaluation_measurements, pd.DataFrame([values])])

    @staticmethod
    def _define_response(
        status: str, evaluation_measurements: list, output_dir: Path, process_model_path: Path
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

    def _simulate_bps_model(self, bps_model: BPSModel, output_dir: Path, granularity: int) -> List[dict]:
        bps_model.replace_activity_names_with_ids()

        json_parameters_path = bps_model.to_json(output_dir, self.event_log.process_name, granule_size=granularity)

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
