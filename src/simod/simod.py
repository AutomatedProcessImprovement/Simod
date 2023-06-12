import json
import shutil
from pathlib import Path
from typing import Optional, List

import pandas as pd
from extraneous_activity_delays.prosimos.simulation_model_enhancer import add_timers_to_simulation_model
from pix_framework.discovery.case_arrival import discover_case_arrival_model
from pix_framework.discovery.gateway_probabilities import compute_gateway_probabilities
from pix_framework.filesystem.file_manager import (
    get_random_folder_id,
    get_random_file_id,
    create_folder,
)
from pix_framework.io.bpm_graph import BPMNGraph

from simod.batching.discovery import discover_batching_rules
from simod.bpm.graph import get_activities_ids_by_name
from simod.bpm.reader_writer import BPMNReaderWriter
from simod.cli_formatter import print_section, print_subsection
from simod.control_flow.discovery import discover_process_model
from simod.control_flow.optimizer import ControlFlowOptimizer
from simod.control_flow.settings import (
    HyperoptIterationParams as ControlFlowHyperoptIterationParams,
)
from simod.event_log.event_log import EventLog
from simod.extraneous_delays.optimizer import ExtraneousDelayTimersOptimizer
from simod.extraneous_delays.utilities import make_simulation_model_from_bps_model
from simod.prioritization.discovery import discover_prioritization_rules
from simod.resource_model.optimizer import ResourceModelOptimizer
from simod.resource_model.settings import (
    HyperoptIterationParams as ResourceModelHyperoptIterationParams,
)
from simod.settings.resource_model_settings import CalendarDiscoveryParams
from simod.settings.simod_settings import SimodSettings, PROJECT_DIR
from simod.simulation.parameters.BPS_model import BPSModel
from simod.simulation.parameters.extraneous_delays import (
    ExtraneousDelay,
    convert_extraneous_delays_to_extraneous_package_format,
)
from simod.simulation.parameters.resource_model import discover_resource_model
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
    _extraneous_delay_timers_optimizer: Optional[ExtraneousDelayTimersOptimizer]

    def __init__(
        self,
        settings: SimodSettings,
        event_log: EventLog,
        output_dir: Optional[Path] = None,
    ):
        self._settings = settings
        self._event_log = event_log
        self._best_bps_model = BPSModel(process_model=self._settings.common.model_path)
        if output_dir is None:
            self._output_dir = PROJECT_DIR / "outputs" / get_random_folder_id()
            create_folder(self._output_dir)
        else:
            self._output_dir = output_dir
        self._control_flow_dir = self._output_dir / "control-flow"
        self._resource_model_dir = self._output_dir / "resource_model"
        self._extraneous_delay_timers_dir = self._output_dir / "extraneous-delay-timers"
        self._best_result_dir = self._output_dir / "best_result"
        create_folder(self._control_flow_dir)
        create_folder(self._resource_model_dir)
        create_folder(self._extraneous_delay_timers_dir)
        create_folder(self._best_result_dir)

    def run(self):
        """
        Optimizes the BPS model with the given event log and settings.
        """

        # --- Discover Default Case Arrival and Resource Allocation models --- #
        print_section("Discovering initial BPS Model")
        self._best_bps_model.case_arrival_model = discover_case_arrival_model(
            self._event_log.train_validation_partition,  # No optimization process here, use train + validation
            self._event_log.log_ids,
        )
        self._best_bps_model.resource_model = discover_resource_model(
            self._event_log.train_validation_partition,  # No optimization process here, use train + validation
            self._event_log.log_ids,
            CalendarDiscoveryParams(),
        )

        # --- Control-Flow Optimization --- #
        print_section("Optimizing control-flow parameters")
        best_control_flow_params = self._optimize_control_flow()
        self._best_bps_model.process_model = self._control_flow_optimizer.best_bps_model.process_model
        self._best_bps_model.gateway_probabilities = self._control_flow_optimizer.best_bps_model.gateway_probabilities

        self._add_prioritization_rules_if_needed()

        self._add_batching_rules_if_needed()

        # --- Congestion Model Discovery --- #
        print_section("Optimizing resource model parameters")
        best_resource_model_params = self._optimize_resource_model()
        self._best_bps_model.resource_model = self._resource_model_optimizer.best_bps_model.resource_model

        # --- Extraneous Delays Discovery --- #
        print_section("Optimizing extraneous delay timers")
        timers = self.optimize_extraneous_activity_delays()
        self._best_bps_model.extraneous_delays = timers

        # Update BPMN model on disk
        self._add_timers_to_bpmn(timers)

        # --- Final evaluation --- #
        print_section("Evaluating final BPS model")
        self.final_bps_model = BPSModel(
            process_model=get_process_model_path(self._best_result_dir, self._event_log.process_name),
            case_arrival_model=self._best_bps_model.case_arrival_model
            # Bypassing case arrival (already used train+valid)
        )
        # Process model
        if self._settings.common.model_path is None:
            print_subsection(f"Discovering process model with settings: {best_control_flow_params.to_dict()}")
            # Instantiate event log to discover the process model with
            xes_log_path = self.final_bps_model.process_model.with_suffix(".xes")
            self._event_log.train_validation_to_xes(xes_log_path)
            # Discover the process model
            discover_process_model(
                log_path=xes_log_path,
                output_model_path=self.final_bps_model.process_model,
                params=best_control_flow_params,
            )
        else:
            print_subsection("Using provided process model")
            shutil.copy(self._settings.common.model_path, self.final_bps_model.process_model)
        best_bpmn_graph = BPMNGraph.from_bpmn_path(self.final_bps_model.process_model)
        # Gateway probabilities
        print_subsection("Discovering gateway probabilities")
        self.final_bps_model.gateway_probabilities = compute_gateway_probabilities(
            event_log=self._event_log.train_validation_partition,
            log_ids=self._event_log.log_ids,
            bpmn_graph=best_bpmn_graph,
            discovery_method=best_control_flow_params.gateway_probabilities_method,
        )
        # Resource model
        print_subsection("Discovering resource model")
        self.final_bps_model.resource_model = discover_resource_model(
            event_log=self._event_log.train_validation_partition,
            log_ids=self._event_log.log_ids,
            params=best_resource_model_params.calendar_discovery_params,
        )
        # Evaluate
        print_subsection("Evaluate")
        simulation_dir = self._best_result_dir / "simulation"
        simulation_dir.mkdir(parents=True)
        self._evaluate_model(self.final_bps_model, simulation_dir)

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

    def _add_timers_to_bpmn(self, timers: Optional[List[ExtraneousDelay]]):
        """
        Rewrites the BPMN file by adding timers if there are any.
        """
        if timers is None or len(timers) == 0:
            return

        bps_model = self._best_bps_model

        simulation_model = make_simulation_model_from_bps_model(bps_model)
        timers_ = convert_extraneous_delays_to_extraneous_package_format(timers)
        enhanced_simulation_model = add_timers_to_simulation_model(
            simulation_model=simulation_model,
            timers=timers_,
        )

        enhanced_simulation_model.bpmn_document.write(bps_model.process_model, pretty_print=True)

    def _add_prioritization_rules_if_needed(self):
        """
        Adds prioritization _rules to the BPS model to pass them later to Primos during simulation.
        """
        if self._settings.resource_model.discover_prioritization_rules is False:
            return

        rules = discover_prioritization_rules(self._event_log.train_partition, self._event_log.log_ids)

        self._best_bps_model.prioritization_rules = rules

    def _add_batching_rules_if_needed(self):
        """
        Adds batching _rules to the BPS model to pass them later to Primos during simulation.
        """
        if self._settings.resource_model.discover_batching_rules is False:
            return

        rules = discover_batching_rules(self._event_log.train_partition, self._event_log.log_ids)
        self._best_bps_model.batching_rules = rules

    def _optimize_resource_model(self) -> ResourceModelHyperoptIterationParams:
        """
        Resource Model (resource profiles, calendars an activity performances) discovery.
        """
        self._resource_model_optimizer = ResourceModelOptimizer(
            event_log=self._event_log,
            bps_model=self._best_bps_model,
            settings=self._settings.resource_model,
            base_directory=self._resource_model_dir,
        )
        best_resource_model_params = self._resource_model_optimizer.run()
        return best_resource_model_params

    def optimize_extraneous_activity_delays(self) -> List[ExtraneousDelay]:
        if self._settings.extraneous_activity_delays is None:
            return []

        settings = self._settings.extraneous_activity_delays
        self._extraneous_delay_timers_optimizer = ExtraneousDelayTimersOptimizer(
            event_log=self._event_log,
            bps_model=self._best_bps_model,
            settings=settings,
            base_directory=self._extraneous_delay_timers_dir,
        )
        timers = self._extraneous_delay_timers_optimizer.run()
        return timers

    def _evaluate_model(self, bps_model: BPSModel, output_dir: Path):
        num_simulations = 10  # TODO: make this a parameter in configuration
        simulation_cases = self._event_log.test_partition[self._settings.common.log_ids.case].nunique()
        simulation_start_time = self._event_log.test_partition[self._settings.common.log_ids.start_time].min()

        metrics = (
            self._settings.common.evaluation_metrics
            if isinstance(self._settings.common.evaluation_metrics, list)
            else [self._settings.common.evaluation_metrics]
        )

        self._event_log.test_partition.to_csv(output_dir / "test_log.csv", index=False)

        # Update activity label -> activity ID mapping of current process model
        activity_label_to_id = get_activities_ids_by_name(BPMNReaderWriter(bps_model.process_model).as_graph())
        for resource_profile in bps_model.resource_model.resource_profiles:
            for resource in resource_profile.resources:
                resource.assigned_tasks = [
                    activity_label_to_id[activity_label] for activity_label in resource.assigned_tasks
                ]
        for activity_resource_distributions in bps_model.resource_model.activity_resource_distributions:
            activity_resource_distributions.activity_id = activity_label_to_id[
                activity_resource_distributions.activity_id
            ]
        # Write JSON parameters to file
        json_parameters_path = get_simulation_parameters_path(output_dir, self._event_log.process_name)
        with json_parameters_path.open("w") as f:
            json.dump(bps_model.to_prosimos(), f)

        measurements = simulate_and_evaluate(
            model_path=bps_model.process_model,
            parameters_path=json_parameters_path,
            output_dir=output_dir,
            simulation_cases=simulation_cases,
            simulation_start_time=simulation_start_time,
            validation_log=self._event_log.test_partition,
            validation_log_ids=self._event_log.log_ids,
            num_simulations=num_simulations,
            metrics=metrics,
        )

        measurements_path = output_dir.parent / get_random_file_id(extension="csv", prefix="evaluation_")
        measurements_df = pd.DataFrame.from_records(measurements)
        measurements_df.to_csv(measurements_path, index=False)

    def _clean_up(self):
        if not self._settings.common.clean_intermediate_files:
            return

        print_section("Removing intermediate files")
        self._control_flow_optimizer.cleanup()
        self._resource_model_optimizer.cleanup()


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
