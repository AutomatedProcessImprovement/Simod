import json
import shutil
from pathlib import Path
from typing import List, Optional

import pandas as pd
from pix_framework.discovery.case_arrival import discover_case_arrival_model
from pix_framework.discovery.gateway_probabilities import compute_gateway_probabilities
from pix_framework.discovery.resource_calendar_and_performance.calendar_discovery_parameters import (
    CalendarDiscoveryParameters,
)
from pix_framework.discovery.resource_model import discover_resource_model
from pix_framework.filesystem.file_manager import create_folder, get_random_folder_id, remove_asset
from pix_framework.io.bpm_graph import BPMNGraph
from pix_framework.io.bpmn import get_activities_names_from_bpmn

from simod.batching.discovery import discover_batching_rules
from simod.case_attributes.discovery import discover_case_attributes
from simod.cli_formatter import print_section, print_subsection
from simod.control_flow.discovery import discover_process_model
from simod.control_flow.optimizer import ControlFlowOptimizer
from simod.control_flow.settings import HyperoptIterationParams as ControlFlowHyperoptIterationParams
from simod.event_log.event_log import EventLog
from simod.extraneous_delays.optimizer import ExtraneousDelaysOptimizer
from simod.extraneous_delays.types import ExtraneousDelay
from simod.extraneous_delays.utilities import add_timers_to_bpmn_model
from simod.prioritization.discovery import discover_prioritization_rules
from simod.resource_model.optimizer import ResourceModelOptimizer
from simod.resource_model.repair import repair_with_missing_activities
from simod.resource_model.settings import HyperoptIterationParams as ResourceModelHyperoptIterationParams
from simod.settings.simod_settings import SimodSettings
from simod.simulation.parameters.BPS_model import BPSModel
from simod.simulation.prosimos import simulate_and_evaluate
from simod.utilities import get_process_model_path, get_simulation_parameters_path


class Simod:
    """
    SIMOD optimization.
    """

    # Event log with the train, validation and test logs.
    _event_log: EventLog
    # Settings for all SIMOD optimization and discovery processes
    _settings: SimodSettings
    # Best BPS model obtained from the discovery processes
    _best_bps_model: BPSModel
    # Final BPS model discovered with the best hyperparams on the training+validation log
    final_bps_model: Optional[BPSModel]
    # Directory to write all the files
    _output_dir: Path

    # Optimizer for the Control-Flow and Gateway Probabilities
    _control_flow_optimizer: Optional[ControlFlowOptimizer]
    # Optimizer for the Resource Model
    _resource_model_optimizer: Optional[ResourceModelOptimizer]
    # Optimizer for the Extraneous Delay Timers
    _extraneous_delays_optimizer: Optional[ExtraneousDelaysOptimizer]

    def __init__(
        self,
        settings: SimodSettings,
        event_log: EventLog,
        output_dir: Optional[Path] = None,
    ):
        self._settings = settings
        self._event_log = event_log
        self._best_bps_model = BPSModel(process_model=self._settings.common.process_model_path)
        if output_dir is None:
            self._output_dir = Path(__file__).parent.parent.parent / "outputs" / get_random_folder_id()
            create_folder(self._output_dir)
        else:
            self._output_dir = output_dir
        self._control_flow_dir = self._output_dir / "control-flow"
        create_folder(self._control_flow_dir)
        self._resource_model_dir = self._output_dir / "resource_model"
        create_folder(self._resource_model_dir)
        if self._settings.extraneous_activity_delays is not None:
            self._extraneous_delays_dir = self._output_dir / "extraneous-delay-timers"
            create_folder(self._extraneous_delays_dir)
        self._best_result_dir = self._output_dir / "best_result"
        create_folder(self._best_result_dir)

    def run(self):
        """
        Optimizes the BPS model with the given event log and settings.
        """

        # Model activities might be different from event log activities if the model has been provided,
        # because we split the event log into train, test, and validation partitions.
        # We use model_activities to repair resource_model later after its discovery from a reduced event log.
        model_activities: Optional[list[str]] = None
        if self._settings.common.process_model_path is not None:
            model_activities = get_activities_names_from_bpmn(self._settings.common.process_model_path)

        # --- Discover Default Case Arrival and Resource Allocation models --- #
        print_section("Discovering initial BPS Model")
        self._best_bps_model.case_arrival_model = discover_case_arrival_model(
            self._event_log.train_validation_partition,  # No optimization process here, use train + validation
            self._event_log.log_ids,
            use_observed_arrival_distribution=self._settings.common.use_observed_arrival_distribution,
        )
        self._best_bps_model.resource_model = discover_resource_model(
            self._event_log.train_partition,  # Only train to not discover tasks that won't exist for control-flow opt.
            self._event_log.log_ids,
            CalendarDiscoveryParameters(),
        )
        if model_activities is not None:
            repair_with_missing_activities(
                resource_model=self._best_bps_model.resource_model,
                model_activities=model_activities,
                event_log=self._event_log.train_validation_partition,
                log_ids=self._event_log.log_ids,
            )

        # --- Control-Flow Optimization --- #
        print_section("Optimizing control-flow parameters")
        best_control_flow_params = self._optimize_control_flow()
        self._best_bps_model.process_model = self._control_flow_optimizer.best_bps_model.process_model
        self._best_bps_model.gateway_probabilities = self._control_flow_optimizer.best_bps_model.gateway_probabilities

        # --- Case Attributes --- #
        if (
            self._settings.common.discover_case_attributes
            or self._settings.resource_model.discover_prioritization_rules
        ):
            print_section("Discovering case attributes")
            case_attributes = discover_case_attributes(
                self._event_log.train_validation_partition,  # No optimization process here, use train + validation
                self._event_log.log_ids,
            )
            self._best_bps_model.case_attributes = case_attributes

        # --- Resource Model Discovery --- #
        print_section("Optimizing resource model parameters")
        best_resource_model_params = self._optimize_resource_model(model_activities)
        self._best_bps_model.resource_model = self._resource_model_optimizer.best_bps_model.resource_model
        self._best_bps_model.prioritization_rules = self._resource_model_optimizer.best_bps_model.prioritization_rules
        self._best_bps_model.batching_rules = self._resource_model_optimizer.best_bps_model.batching_rules

        # --- Extraneous Delays Discovery --- #
        if self._settings.extraneous_activity_delays is not None:
            print_section("Discovering extraneous delays")
            timers = self._optimize_extraneous_activity_delays()
            self._best_bps_model.extraneous_delays = timers
            add_timers_to_bpmn_model(self._best_bps_model.process_model, timers)  # Update BPMN model on disk

        # --- Discover final BPS model --- #
        print_section("Discovering final BPS model")
        self.final_bps_model = BPSModel(  # Bypass all models already discovered with train+validation
            process_model=get_process_model_path(self._best_result_dir, self._event_log.process_name),
            case_arrival_model=self._best_bps_model.case_arrival_model,
            case_attributes=self._best_bps_model.case_attributes,
        )
        # Process model
        if self._settings.common.process_model_path is None:
            # Discover process model with best control-flow parameters
            print_subsection(
                f"Discovering process model with best control-flow settings: {best_control_flow_params.to_dict()}"
            )
            # Instantiate event log to discover the process model with
            xes_log_path = self._best_result_dir / f"{self._event_log.process_name}_train_val.xes"
            self._event_log.train_validation_to_xes(xes_log_path)
            # Discover the process model
            discover_process_model(
                log_path=xes_log_path,
                output_model_path=self.final_bps_model.process_model,
                params=best_control_flow_params,
            )
        else:
            # Copy provided process model to best result folder
            print_subsection("Using provided process model")
            shutil.copy(self._settings.common.process_model_path, self.final_bps_model.process_model)
        # Gateway probabilities
        print_subsection("Discovering gateway probabilities")
        best_bpmn_graph = BPMNGraph.from_bpmn_path(self.final_bps_model.process_model)
        self.final_bps_model.gateway_probabilities = compute_gateway_probabilities(
            event_log=self._event_log.train_validation_partition,
            log_ids=self._event_log.log_ids,
            bpmn_graph=best_bpmn_graph,
            discovery_method=best_control_flow_params.gateway_probabilities_method,
        )
        # Resource model
        print_subsection("Discovering best resource model")
        self.final_bps_model.resource_model = discover_resource_model(
            event_log=self._event_log.train_validation_partition,
            log_ids=self._event_log.log_ids,
            params=best_resource_model_params.calendar_discovery_params,
        )
        if model_activities is not None:
            repair_with_missing_activities(
                resource_model=self.final_bps_model.resource_model,
                model_activities=model_activities,
                event_log=self._event_log.train_validation_partition,
                log_ids=self._event_log.log_ids,
            )
        # Prioritization
        if best_resource_model_params.discover_prioritization_rules:
            print_subsection("Discovering prioritization rules")
            self.final_bps_model.prioritization_rules = discover_prioritization_rules(
                self._event_log.train_validation_partition,
                self._event_log.log_ids,
                self._best_bps_model.case_attributes,
            )
        # Batching
        if best_resource_model_params.discover_batching_rules:
            print_subsection("Discovering batching rules")
            self.final_bps_model.batching_rules = discover_batching_rules(
                self._event_log.train_validation_partition, self._event_log.log_ids
            )
        # Extraneous delays
        if self._best_bps_model.extraneous_delays is not None:
            # Add discovered delays and update BPMN model on disk
            self.final_bps_model.extraneous_delays = self._best_bps_model.extraneous_delays
            add_timers_to_bpmn_model(self.final_bps_model.process_model, self._best_bps_model.extraneous_delays)
        self.final_bps_model.replace_activity_names_with_ids()
        # Write JSON parameters to file
        json_parameters_path = get_simulation_parameters_path(self._best_result_dir, self._event_log.process_name)
        with json_parameters_path.open("w") as f:
            json.dump(self.final_bps_model.to_prosimos_format(), f)

        # --- Evaluate final BPS model --- #
        if self._settings.common.perform_final_evaluation:
            print_subsection("Evaluate")
            simulation_dir = self._best_result_dir / "evaluation"
            simulation_dir.mkdir(parents=True, exist_ok=True)
            self._evaluate_model(self.final_bps_model.process_model, json_parameters_path, simulation_dir)

        # --- Export settings and clean temporal files --- #
        canonical_model_path = self._best_result_dir / "canonical_model.json"
        print_section(f"Exporting canonical model to {canonical_model_path}")
        _export_canonical_model(canonical_model_path, best_control_flow_params, best_resource_model_params)
        if self._settings.common.clean_intermediate_files:
            self._clean_up()
        self._settings.to_yaml(self._best_result_dir)

    def _optimize_control_flow(self) -> ControlFlowHyperoptIterationParams:
        """
        Control-flow and Gateway Probabilities discovery.
        """
        self._control_flow_optimizer = ControlFlowOptimizer(
            event_log=self._event_log,
            bps_model=self._best_bps_model,
            settings=self._settings.control_flow,
            base_directory=self._control_flow_dir,
        )
        best_control_flow_params = self._control_flow_optimizer.run()
        return best_control_flow_params

    def _optimize_resource_model(
        self, model_activities: Optional[list[str]] = None
    ) -> ResourceModelHyperoptIterationParams:
        """
        Resource Model (resource profiles, calendars an activity performances) discovery.
        """
        self._resource_model_optimizer = ResourceModelOptimizer(
            event_log=self._event_log,
            bps_model=self._best_bps_model,
            settings=self._settings.resource_model,
            base_directory=self._resource_model_dir,
            model_activities=model_activities,
        )
        best_resource_model_params = self._resource_model_optimizer.run()
        return best_resource_model_params

    def _optimize_extraneous_activity_delays(self) -> List[ExtraneousDelay]:
        settings = self._settings.extraneous_activity_delays
        self._extraneous_delays_optimizer = ExtraneousDelaysOptimizer(
            event_log=self._event_log,
            bps_model=self._best_bps_model,
            settings=settings,
            base_directory=self._extraneous_delays_dir,
        )
        timers = self._extraneous_delays_optimizer.run()
        return timers

    def _evaluate_model(self, process_model: Path, json_parameters: Path, output_dir: Path):
        simulation_cases = self._event_log.test_partition[self._settings.common.log_ids.case].nunique()
        simulation_start_time = self._event_log.test_partition[self._settings.common.log_ids.start_time].min()

        metrics = (
            self._settings.common.evaluation_metrics
            if isinstance(self._settings.common.evaluation_metrics, list)
            else [self._settings.common.evaluation_metrics]
        )

        self._event_log.test_partition.to_csv(output_dir / "test_log.csv", index=False)

        measurements = simulate_and_evaluate(
            process_model_path=process_model,
            parameters_path=json_parameters,
            output_dir=output_dir,
            simulation_cases=simulation_cases,
            simulation_start_time=simulation_start_time,
            validation_log=self._event_log.test_partition,
            validation_log_ids=self._event_log.log_ids,
            num_simulations=self._settings.common.num_final_evaluations,
            metrics=metrics,
        )

        measurements_path = output_dir / "evaluation_metrics.csv"
        measurements_df = pd.DataFrame.from_records(measurements)
        measurements_df.to_csv(measurements_path, index=False)

    def _clean_up(self):
        print_section("Removing intermediate files")
        self._control_flow_optimizer.cleanup()
        self._resource_model_optimizer.cleanup()
        if self._settings.extraneous_activity_delays is not None:
            self._extraneous_delays_optimizer.cleanup()
        if self._settings.common.process_model_path is None:
            final_xes_log_path = self._best_result_dir / f"{self._event_log.process_name}_train_val.xes"
            remove_asset(final_xes_log_path)


def _export_canonical_model(
    file_path: Path,
    control_flow_settings: ControlFlowHyperoptIterationParams,
    calendar_settings: ResourceModelHyperoptIterationParams,
):
    structure = control_flow_settings.to_dict()

    calendars = calendar_settings.to_dict()

    canon = {
        "control_flow": structure,
        "calendars": calendars,
    }

    with open(file_path, "w") as f:
        json.dump(canon, f)
