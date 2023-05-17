import json
import shutil
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
from extraneous_activity_delays.config import SimulationModel
from pix_utils.filesystem.file_manager import get_random_folder_id, get_random_file_id, create_folder

from simod.bpm.reader_writer import BPMNReaderWriter
from simod.cli_formatter import print_section, print_message
from simod.control_flow.discovery import discover_process_model
from simod.control_flow.optimizer import ControlFlowOptimizer
from simod.control_flow.settings import HyperoptIterationParams as StructureHyperoptIterationParams
from simod.discovery.extraneous_delay_timers import discover_extraneous_delay_timers
from simod.event_log.event_log import EventLog
from simod.process_calendars.optimizer import CalendarOptimizer
from simod.process_calendars.settings import PipelineSettings as CalendarPipelineSettings, CalendarOptimizationSettings
from simod.settings.simod_settings import SimodSettings, PROJECT_DIR
from simod.settings.temporal_settings import CalendarSettings
from simod.simulation.parameters.BPS_model import BPSModel
from simod.simulation.parameters.case_arrival_model import discover_case_arrival_model
from simod.simulation.parameters.gateway_probabilities import GatewayProbabilities
from simod.simulation.parameters.miner import mine_parameters
from simod.simulation.parameters.resource_model import discover_resource_model
from simod.simulation.prosimos import simulate_and_evaluate


class Optimizer:
    """
    Structure and calendars optimization.
    """

    _settings: SimodSettings
    _event_log: EventLog
    _best_bps_model: BPSModel
    _output_dir: Path

    _control_flow_optimizer: Optional[ControlFlowOptimizer]
    _calendar_optimizer: Optional[CalendarOptimizer]

    def __init__(
            self,
            settings: SimodSettings,
            event_log: EventLog,
            output_dir: Optional[Path] = None
    ):
        # Save SIMOD settings
        self._settings = settings
        # Read event log from path if not provided
        self._event_log = event_log
        # Create empty BPS model
        self._best_bps_model = BPSModel(
            process_model=self._settings.common.model_path
        )
        # Create output directory if not provided
        if output_dir is None:
            self._output_dir = PROJECT_DIR / 'outputs' / get_random_folder_id()
            create_folder(self._output_dir)
        else:
            self._output_dir = output_dir
        # Create folders for the control-flow optimization, temporal optimization, and best result
        self._control_flow_dir = self._output_dir / "control-flow"
        self._temporal_opt_dir = self._output_dir / "temporal"
        self._best_result_dir = self._output_dir / 'best_result'
        create_folder(self._control_flow_dir)
        create_folder(self._temporal_opt_dir)
        create_folder(self._best_result_dir)

    def _optimize_structure(self) -> StructureHyperoptIterationParams:
        """Control-flow and Gateway Probabilities discovery."""
        # Instantiate class to perform the optimization of the control-flow discovery
        self._control_flow_optimizer = ControlFlowOptimizer(
            event_log=self._event_log,
            bps_model=self._best_bps_model,
            settings=self._settings.control_flow,
            base_directory=self._control_flow_dir,
        )
        # Run optimization process
        best_control_flow_settings = self._control_flow_optimizer.run()
        # Return results
        return best_control_flow_settings

    def _optimize_calendars(
            self,
            structure_settings: StructureHyperoptIterationParams,
            model_path: Path,
            gateway_probabilities: List[GatewayProbabilities],
            simulation_model: Optional[SimulationModel] = None,
    ) -> Tuple[CalendarOptimizationSettings, CalendarPipelineSettings]:
        calendar_settings = CalendarOptimizationSettings.from_configuration(self._settings, self._output_dir)

        event_distribution = simulation_model.simulation_parameters.get('event_distribution', None) \
            if simulation_model is not None \
            else None

        process_graph = BPMNReaderWriter(model_path).as_graph()

        optimizer = CalendarOptimizer(
            calendar_settings,
            self._event_log,
            train_model_path=model_path,
            base_directory=self._temporal_opt_dir,
            gateway_probabilities=gateway_probabilities,
            gateway_probabilities_method=structure_settings.gateway_probabilities_method,
            process_graph=process_graph,
            event_distribution=event_distribution,
        )

        best_pipeline_settings = optimizer.run()

        self._calendar_optimizer = optimizer

        return calendar_settings, best_pipeline_settings

    def _evaluate_model(
            self,
            model_path: Path,
            parameters_path: Path,
            simulation_dir: Path):
        assert parameters_path.exists(), f'Best calendar simulation parameters file does not exist'

        num_simulations = 10  # TODO: make this a parameter in configuration
        simulation_cases = self._event_log.test_partition[self._settings.common.log_ids.case].nunique()
        simulation_start_time = self._event_log.test_partition[self._settings.common.log_ids.start_time].min()

        metrics = self._settings.common.evaluation_metrics \
            if isinstance(self._settings.common.evaluation_metrics, list) \
            else [self._settings.common.evaluation_metrics]

        self._event_log.test_partition.to_csv(simulation_dir / 'test_log.csv', index=False)

        measurements = simulate_and_evaluate(
            model_path=model_path,
            parameters_path=parameters_path,
            output_dir=simulation_dir,
            simulation_cases=simulation_cases,
            simulation_start_time=simulation_start_time,
            validation_log=self._event_log.test_partition,
            validation_log_ids=self._event_log.log_ids,
            num_simulations=num_simulations,
            metrics=metrics,
        )

        measurements_path = simulation_dir.parent / get_random_file_id(extension="csv", prefix="evaluation_")
        measurements_df = pd.DataFrame.from_records(measurements)
        measurements_df.to_csv(measurements_path, index=False)

    def _clean_up(self):
        if not self._settings.common.clean_intermediate_files:
            return

        print_section('Removing intermediate files')
        self._control_flow_optimizer.cleanup()
        self._calendar_optimizer.cleanup()

    def _mine_structure(self, settings: StructureHyperoptIterationParams, output_dir: Path) \
            -> Path:
        print_message(f'Mining structure with settings {settings.to_dict()}')

        log_path = (output_dir / self._event_log.process_name).with_suffix('.xes')
        self._event_log.train_to_xes(log_path)  # TODO should be train+validation

        model_path = output_dir / (self._event_log.process_name + '.bpmn')

        discover_process_model(
            log_path,
            model_path,
            settings
        )

        return model_path

    def _mine_calendars(
            self,
            best_settings: CalendarPipelineSettings,
            model_path: Path,
            output_dir: Path,
            simulation_model: Optional[SimulationModel] = None,
    ) -> Tuple[Path, CalendarPipelineSettings]:
        settings = CalendarPipelineSettings(
            output_dir=output_dir,
            model_path=model_path,
            gateway_probabilities_method=best_settings.gateway_probabilities_method,
            case_arrival=best_settings.case_arrival,
            resource_profiles=best_settings.resource_profiles,
        )

        print_message(f'Mining calendars with settings {settings.to_dict()}')

        process_graph = BPMNReaderWriter(model_path).as_graph()
        parameters = mine_parameters(
            case_arrival_settings=settings.case_arrival,
            resource_profiles_settings=settings.resource_profiles,
            log=self._event_log.train_partition,
            log_ids=self._event_log.log_ids,
            model_path=model_path,
            gateways_probability_method=settings.gateway_probabilities_method,
            process_graph=process_graph,
        )

        if simulation_model is not None:
            parameters.event_distribution = simulation_model.simulation_parameters.get('event_distribution', None)

        json_path = settings.output_dir / 'simulation_parameters.json'

        parameters.to_json_file(json_path)

        return json_path, settings

    def run(self):
        """
        Run SIMOD main structure
        """

        # --- Discover Default Case Arrival and Resource Allocation models --- #
        self._best_bps_model.case_arrival_model = discover_case_arrival_model(
            self._event_log.train_validation_partition,  # No optimization process here, use train + validation
            self._event_log.log_ids
        )
        self._best_bps_model.resource_model = discover_resource_model(
            self._event_log.train_validation_partition,  # No optimization process here, use train + validation
            self._event_log.log_ids,
            CalendarSettings.default()
        )

        # --- Control-Flow Optimization --- #
        print_section('Control-flow optimization')
        best_control_flow_settings = self._optimize_structure()
        self._best_bps_model.process_model = self._control_flow_optimizer.best_bps_model.process_model
        self._best_bps_model.gateway_probabilities = self._control_flow_optimizer.best_bps_model.gateway_probabilities

        # --- Extraneous Delays Discovery --- #
        simulation_model = None
        if self._settings.extraneous_activity_delays is not None:
            print_section('Discovering extraneous delays')
            # Hardcoded from here for now, should work with the BPSModel in the future
            parameters_path = self._control_flow_optimizer.base_directory / f"{self._event_log.process_name}.json"
            with parameters_path.open() as f:
                parameters = json.load(f)
            simulation_model, model_path, parameters_path = discover_extraneous_delay_timers(
                self._event_log.train_validation_partition,
                self._event_log.log_ids,
                self._best_bps_model.process_model,
                parameters,
                self._settings.extraneous_activity_delays.optimization_metric,
                base_dir=self._best_result_dir,
                num_iterations=self._settings.extraneous_activity_delays.num_iterations,
                num_evaluation_simulations=self._settings.common.repetitions,
                max_alpha=50,
            )
        else:
            model_path = self._best_bps_model.process_model

        # --- Congestion Model Discovery --- #
        print_section('Calendars optimization')
        calendar_optimizer_settings, calendar_pipeline_settings = self._optimize_calendars(
            best_control_flow_settings, model_path, self._best_bps_model.gateway_probabilities, simulation_model)

        # --- Final evaluation of best BPS Model --- #
        if self._settings.common.model_path is None:
            print_section('Mining structure using the best hyperparameters')
            model_path = self._mine_structure(best_control_flow_settings, self._best_result_dir)
        else:
            try:
                shutil.copy(model_path, self._best_result_dir)
            except shutil.SameFileError:
                pass

        print_section('Mining calendars using the best hyperparameters')
        parameters_path, calendars_settings = self._mine_calendars(
            calendar_pipeline_settings, model_path, self._best_result_dir, simulation_model)

        print_section('Evaluation')
        simulation_dir = self._best_result_dir / 'simulation'
        simulation_dir.mkdir(parents=True)
        self._evaluate_model(model_path, parameters_path, simulation_dir)

        # --- Clean temporal files and export settings --- #
        self._clean_up()

        print_section('Exporting canonical model')
        _export_canonical_model(self._best_result_dir, best_control_flow_settings,
                                calendars_settings, calendar_optimizer_settings)

        self._settings.to_yaml(self._best_result_dir)


def _export_canonical_model(
        output_dir: Path,
        structure_settings: StructureHyperoptIterationParams,
        calendar_settings: CalendarPipelineSettings,
        calendar_optimizer_settings: CalendarOptimizationSettings,
):
    canon_path = output_dir / 'canonical_model.json'

    structure = structure_settings.to_dict()

    calendars = calendar_settings.to_dict() | {
        'optimization_metric': str(calendar_optimizer_settings.optimization_metric)
    }

    canon = {
        'control_flow': structure,
        'calendars': calendars,
    }

    with open(canon_path, 'w') as f:
        json.dump(canon, f)
