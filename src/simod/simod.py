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
from simod.branch_rules.discovery import discover_branch_rules, map_branch_rules_to_flows
from simod.cli_formatter import print_section, print_subsection
from simod.control_flow.discovery import discover_process_model, add_bpmn_diagram_to_model
from simod.control_flow.optimizer import ControlFlowOptimizer
from simod.control_flow.settings import HyperoptIterationParams as ControlFlowHyperoptIterationParams
from simod.data_attributes.discovery import discover_data_attributes
from simod.event_log.event_log import EventLog
from simod.extraneous_delays.optimizer import ExtraneousDelaysOptimizer
from simod.extraneous_delays.types import ExtraneousDelay
from simod.extraneous_delays.utilities import add_timers_to_bpmn_model
from simod.prioritization.discovery import discover_prioritization_rules
from simod.resource_model.optimizer import ResourceModelOptimizer
from simod.resource_model.repair import repair_with_missing_activities
from simod.resource_model.settings import HyperoptIterationParams as ResourceModelHyperoptIterationParams
from simod.runtime_meter import RuntimeMeter
from simod.settings.simod_settings import SimodSettings
from simod.simulation.parameters.BPS_model import BPSModel
from simod.simulation.prosimos import simulate_and_evaluate
from simod.utilities import get_process_model_path, get_simulation_parameters_path


class Simod:
    """
    Class to run the full pipeline of SIMOD in order to discover a BPS model from an event log.

    Attributes
    ----------
        settings : :class:`~simod.settings.simod_settings.SimodSettings`
            Configuration to run SIMOD and all its stages.
        event_log : :class:`~simod.event_log.event_log.EventLog`
            EventLog class storing the preprocessed training, validation, and (optionally) test partitions.
        output_dir : :class:`~pathlib.Path`
            Path to the folder where to write all the SIMOD outputs.
        final_bps_model : :class:`~simod.simulation.parameters.BPS_model.BPSModel`
            Instance of the best BPS model discovered by SIMOD.
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

    def run(self, runtimes: Optional[RuntimeMeter] = None):
        """
        Executes the SIMOD pipeline to discover the BPS model that better reflects the behavior recorded in the input
        event log based on the specified configuration.

        Parameters
        ----------
        runtimes : :class:`~simod.runtime_meter.RuntimeMeter`, optional
            Instance for tracking the runtime of the different stages in the SIMOD pipeline. When provided, SIMOD
            pipeline stages will be tracked and reported along with stages previously tracked in the instance (e.g.,
            preprocessing). If not provided, the runtime tracking reported will only contain SIMOD stages.

        Returns
        -------
        None
            The method performs in-place execution of the pipeline and does not return a value.

        Notes
        -----
        - This method generates all output files under the folder ``[output_dir]/<latest_run>/best_result/``.
        - This method updates internal attributes of the class, such as `final_bps_model`, with the best BPS model found
          during the pipeline execution.
        """

        # Runtime object
        runtimes = RuntimeMeter() if runtimes is None else runtimes
        runtimes.start(RuntimeMeter.TOTAL)

        # Model activities might be different from event log activities if the model has been provided,
        # because we split the event log into train, test, and validation partitions.
        # We use model_activities to repair resource_model later after its discovery from a reduced event log.
        model_activities: Optional[list[str]] = None
        if self._settings.common.process_model_path is not None:
            model_activities = get_activities_names_from_bpmn(self._settings.common.process_model_path)

        # --- Discover Default Case Arrival and Resource Allocation models --- #
        print_section("Discovering initial BPS Model")
        runtimes.start(RuntimeMeter.INITIAL_MODEL)
        self._best_bps_model.case_arrival_model = discover_case_arrival_model(
            self._event_log.train_validation_partition,  # No optimization process here, use train + validation
            self._event_log.log_ids,
            use_observed_arrival_distribution=self._settings.common.use_observed_arrival_distribution,
        )
        calendar_discovery_parameters = CalendarDiscoveryParameters()
        self._best_bps_model.resource_model = discover_resource_model(
            self._event_log.train_partition,  # Only train to not discover tasks that won't exist for control-flow opt.
            self._event_log.log_ids,
            calendar_discovery_parameters,
        )
        self._best_bps_model.calendar_granularity = calendar_discovery_parameters.granularity
        if model_activities is not None:
            repair_with_missing_activities(
                resource_model=self._best_bps_model.resource_model,
                model_activities=model_activities,
                event_log=self._event_log.train_validation_partition,
                log_ids=self._event_log.log_ids,
            )
        runtimes.stop(RuntimeMeter.INITIAL_MODEL)

        # --- Control-Flow Optimization --- #
        print_section("Optimizing control-flow parameters")
        runtimes.start(RuntimeMeter.CONTROL_FLOW_MODEL)
        best_control_flow_params = self._optimize_control_flow()
        self._best_bps_model.process_model = self._control_flow_optimizer.best_bps_model.process_model
        self._best_bps_model.gateway_probabilities = self._control_flow_optimizer.best_bps_model.gateway_probabilities
        self._best_bps_model.branch_rules = self._control_flow_optimizer.best_bps_model.branch_rules
        runtimes.stop(RuntimeMeter.CONTROL_FLOW_MODEL)

        # --- Data Attributes --- #
        if (self._settings.common.discover_data_attributes or
                self._settings.resource_model.discover_prioritization_rules):
            print_section("Discovering data attributes")
            runtimes.start(RuntimeMeter.DATA_ATTRIBUTES_MODEL)
            global_attributes, case_attributes, event_attributes = discover_data_attributes(
                self._event_log.train_validation_partition,
                self._event_log.log_ids,
            )
            self._best_bps_model.global_attributes = global_attributes
            self._best_bps_model.case_attributes = case_attributes
            self._best_bps_model.event_attributes = event_attributes
            runtimes.stop(RuntimeMeter.DATA_ATTRIBUTES_MODEL)

        # --- Resource Model Discovery --- #
        print_section("Optimizing resource model parameters")
        runtimes.start(RuntimeMeter.RESOURCE_MODEL)
        best_resource_model_params = self._optimize_resource_model(model_activities)
        self._best_bps_model.resource_model = self._resource_model_optimizer.best_bps_model.resource_model
        self._best_bps_model.calendar_granularity = self._resource_model_optimizer.best_bps_model.calendar_granularity
        self._best_bps_model.prioritization_rules = self._resource_model_optimizer.best_bps_model.prioritization_rules
        self._best_bps_model.batching_rules = self._resource_model_optimizer.best_bps_model.batching_rules
        runtimes.stop(RuntimeMeter.RESOURCE_MODEL)

        # --- Extraneous Delays Discovery --- #
        if self._settings.extraneous_activity_delays is not None:
            print_section("Discovering extraneous delays")
            runtimes.start(RuntimeMeter.EXTRANEOUS_DELAYS)
            timers = self._optimize_extraneous_activity_delays()
            self._best_bps_model.extraneous_delays = timers
            add_timers_to_bpmn_model(self._best_bps_model.process_model, timers)  # Update BPMN model on disk
            runtimes.stop(RuntimeMeter.EXTRANEOUS_DELAYS)

        # --- Discover final BPS model --- #
        print_section("Discovering final BPS model")
        runtimes.start(RuntimeMeter.FINAL_MODEL)
        self.final_bps_model = BPSModel(  # Bypass all models already discovered with train+validation
            process_model=get_process_model_path(self._best_result_dir, self._event_log.process_name),
            case_arrival_model=self._best_bps_model.case_arrival_model,
            case_attributes=self._best_bps_model.case_attributes,
            global_attributes=self._best_bps_model.global_attributes,
            event_attributes=self._best_bps_model.event_attributes,
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
        #  Branch Rules
        if self._settings.control_flow.discover_branch_rules:
            print_section("Discovering branch conditions")
            self.final_bps_model.branch_rules = discover_branch_rules(
                best_bpmn_graph,
                self._event_log.train_validation_partition,
                self._event_log.log_ids,
                f_score=best_control_flow_params.f_score
            )
            self.final_bps_model.gateway_probabilities = \
                map_branch_rules_to_flows(self.final_bps_model.gateway_probabilities, self.final_bps_model.branch_rules)
        # Resource model
        print_subsection("Discovering best resource model")
        self.final_bps_model.resource_model = discover_resource_model(
            event_log=self._event_log.train_validation_partition,
            log_ids=self._event_log.log_ids,
            params=best_resource_model_params.calendar_discovery_params,
        )
        self.final_bps_model.calendar_granularity = best_resource_model_params.calendar_discovery_params.granularity
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
        runtimes.stop(RuntimeMeter.FINAL_MODEL)
        runtimes.stop(RuntimeMeter.TOTAL)

        # Write JSON parameters to file
        json_parameters_path = get_simulation_parameters_path(self._best_result_dir, self._event_log.process_name)
        with json_parameters_path.open("w") as f:
            json.dump(self.final_bps_model.to_prosimos_format(), f)

        # --- Evaluate final BPS model --- #
        if self._settings.common.perform_final_evaluation:
            print_subsection("Evaluate")
            runtimes.start(RuntimeMeter.EVALUATION)
            simulation_dir = self._best_result_dir / "evaluation"
            simulation_dir.mkdir(parents=True, exist_ok=True)
            self._evaluate_model(self.final_bps_model.process_model, json_parameters_path, simulation_dir)
            runtimes.stop(RuntimeMeter.EVALUATION)

        # --- Export settings and clean temporal files --- #
        print_section(f"Exporting canonical model, runtimes, settings and cleaning up intermediate files")
        canonical_model_path = self._best_result_dir / "canonical_model.json"
        _export_canonical_model(canonical_model_path, best_control_flow_params, best_resource_model_params)
        runtimes_model_path = self._best_result_dir / "runtimes.json"
        _export_runtimes(runtimes_model_path, runtimes)
        if self._settings.common.clean_intermediate_files:
            self._clean_up()
        self._settings.to_yaml(self._best_result_dir)

        # --- Add BPMN diagram to the model --- #
        add_bpmn_diagram_to_model(self.final_bps_model.process_model)

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
    canon = {
        "control_flow": control_flow_settings.to_dict(),
        "calendars": calendar_settings.to_dict(),
    }
    with open(file_path, "w") as f:
        json.dump(canon, f)


def _export_runtimes(
        file_path: Path,
        runtimes: RuntimeMeter
):
    with open(file_path, "w") as file:
        json.dump(
            runtimes.runtimes | {'explanation': f"Add '{RuntimeMeter.PREPROCESSING}' with '{RuntimeMeter.TOTAL}' "
                                                f"for the runtime of the entire SIMOD pipeline and preprocessing "
                                                f"stage. '{RuntimeMeter.EVALUATION}', if reported, should be left out "
                                                f"as it measures the quality assessment of the final BPS model (i.e., "
                                                f"it is not part of the discovery process."},
            file
        )
