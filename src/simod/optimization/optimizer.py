import json
import shutil
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from networkx import DiGraph

from extraneous_activity_delays.config import SimulationModel
from simod.bpm.reader_writer import BPMNReaderWriter
from simod.cli_formatter import print_section, print_message
from simod.configuration import Configuration
from simod.discovery.extraneous_delay_timers import discover_extraneous_delay_timers
from simod.event_log.event_log import EventLog
from simod.process_calendars.optimizer import CalendarOptimizer
from simod.process_calendars.settings import PipelineSettings as CalendarPipelineSettings, CalendarOptimizationSettings
from simod.process_structure.miner import Settings as StructureMinerSettings, StructureMiner
from simod.process_structure.optimizer import StructureOptimizer
from simod.process_structure.settings import PipelineSettings as StructurePipelineSettings, \
    StructureOptimizationSettings
from simod.simulation.parameters.miner import mine_parameters
from simod.simulation.prosimos import simulate_and_evaluate
from simod.utilities import file_id, get_project_dir, folder_id


class Optimizer:
    """
    Structure and calendars optimization.
    """

    _settings: Configuration
    _event_log: EventLog
    _output_dir: Path
    _process_graph: Optional[DiGraph]

    _structure_optimizer: Optional[StructureOptimizer]
    _calendar_optimizer: Optional[CalendarOptimizer]

    def __init__(
            self,
            settings: Configuration,
            event_log: Optional[EventLog] = None,
            output_dir: Optional[Path] = None
    ):
        self._settings = settings

        if event_log is None:
            self._event_log = EventLog.from_path(
                path=settings.common.log_path,
                log_ids=settings.common.log_ids,
                process_name=settings.common.log_path.stem,
                test_path=settings.common.test_log_path,
            )
        else:
            self._event_log = event_log

        if output_dir is None:
            self._output_dir = get_project_dir() / 'outputs' / folder_id()
        else:
            self._output_dir = output_dir

        if self._settings.common.model_path is not None:
            self._process_graph = BPMNReaderWriter(self._settings.common.model_path).as_graph()
        else:
            self._process_graph = None

    def _optimize_structure(self) -> Tuple[StructureOptimizationSettings, StructurePipelineSettings, Path, list, Path]:
        settings = StructureOptimizationSettings.from_configuration(self._settings, self._output_dir)
        optimizer = StructureOptimizer(settings, self._event_log, process_graph=self._process_graph)
        self._structure_optimizer = optimizer
        best_pipeline_settings, model_path, gateway_probabilities, parameters_path = optimizer.run()

        return settings, best_pipeline_settings, model_path, gateway_probabilities, parameters_path

    def _optimize_calendars(
            self,
            structure_settings: StructureMinerSettings,
            model_path: Optional[Path],
            gateway_probabilities: list,
            simulation_model: Optional[SimulationModel] = None,
    ) -> Tuple[CalendarOptimizationSettings, CalendarPipelineSettings]:
        calendar_settings = CalendarOptimizationSettings.from_configuration(self._settings, self._output_dir)

        event_distribution = simulation_model.simulation_parameters.get('event_distribution', None) \
            if simulation_model is not None \
            else None

        optimizer = CalendarOptimizer(
            calendar_settings,
            self._event_log,
            train_model_path=model_path,
            gateway_probabilities=gateway_probabilities,
            gateway_probabilities_method=structure_settings.gateway_probabilities_method,
            process_graph=self._process_graph,
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

        num_simulations = self._settings.common.repetitions
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

        measurements_path = simulation_dir.parent / file_id(prefix='evaluation_')
        measurements_df = pd.DataFrame.from_records(measurements)
        measurements_df.to_csv(measurements_path, index=False)

    def _clean_up(self):
        if not self._settings.common.clean_intermediate_files:
            return

        print_section('Removing intermediate files')
        self._structure_optimizer.cleanup()
        self._calendar_optimizer.cleanup()

    def _mine_structure(self, settings: StructureMinerSettings, output_dir: Path) \
            -> Tuple[Path, StructureMinerSettings]:
        print_message(f'Mining structure with settings {settings.to_dict()}')

        log_path = (output_dir / self._event_log.process_name).with_suffix('.xes')
        self._event_log.train_to_xes(log_path)

        model_path = output_dir / (self._event_log.process_name + '.bpmn')

        StructureMiner(
            settings.mining_algorithm,
            log_path,
            model_path,
            concurrency=settings.concurrency,
            eta=settings.eta,
            epsilon=settings.epsilon,
            prioritize_parallelism=settings.prioritize_parallelism,
            replace_or_joins=settings.replace_or_joins,
        ).run()

        return model_path, settings

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

        parameters = mine_parameters(
            case_arrival_settings=settings.case_arrival,
            resource_profiles_settings=settings.resource_profiles,
            log=self._event_log.train_partition,
            log_ids=self._event_log.log_ids,
            model_path=model_path,
            gateways_probability_method=settings.gateway_probabilities_method,
            process_graph=self._process_graph,
        )

        if simulation_model is not None:
            parameters.event_distribution = simulation_model.simulation_parameters.get('event_distribution', None)

        json_path = settings.output_dir / 'simulation_parameters.json'

        parameters.to_json_file(json_path)

        return json_path, settings

    def run(self):
        """
        Runs the entire Simod optimization pipeline that consists of the structure and calendars optimization phases.
        :return: None
        """

        best_result_dir = self._output_dir / 'best_result'
        best_result_dir.mkdir(parents=True, exist_ok=True)

        print_section('Structure optimization')
        result = self._optimize_structure()
        structure_optimizer_settings, structure_pipeline_settings, model_path, gateway_probabilities, parameters_path = result

        simulation_model = None
        if self._settings.extraneous_activity_delays is not None:
            print_section('Mining extraneous delay timers')
            with parameters_path.open() as f:
                parameters = json.load(f)
            simulation_model, model_path, parameters_path = discover_extraneous_delay_timers(
                self._event_log.train_partition,
                self._event_log.log_ids,
                model_path,
                parameters,
                self._settings.extraneous_activity_delays.optimization_metric,
                base_dir=best_result_dir,
                num_iterations=self._settings.extraneous_activity_delays.num_iterations,
                max_alpha=50,
            )

        if not self._process_graph:
            self._process_graph = BPMNReaderWriter(model_path).as_graph()

        structure_miner_settings = StructureMinerSettings(
            gateway_probabilities_method=structure_pipeline_settings.gateway_probabilities_method,
            mining_algorithm=self._settings.structure.mining_algorithm,
            epsilon=structure_pipeline_settings.epsilon,
            eta=structure_pipeline_settings.eta,
            concurrency=structure_pipeline_settings.concurrency,
            prioritize_parallelism=structure_pipeline_settings.prioritize_parallelism,
            replace_or_joins=structure_pipeline_settings.replace_or_joins,
        )

        print_section('Calendars optimization')
        calendar_optimizer_settings, calendar_pipeline_settings = self._optimize_calendars(
            structure_miner_settings, model_path, gateway_probabilities, simulation_model)

        if model_path is None:
            print_section('Mining structure using the best hyperparameters')
            model_path, structure_miner_settings = self._mine_structure(structure_miner_settings, best_result_dir)
        else:
            try:
                shutil.copy(model_path, best_result_dir)
            except shutil.SameFileError:
                pass

        print_section('Mining calendars using the best hyperparameters')
        parameters_path, calendars_settings = self._mine_calendars(
            calendar_pipeline_settings, model_path, best_result_dir, simulation_model)

        # if self._settings.extraneous_activity_delays is not None:
        #     print_section('Mining extraneous delay timers')
        #     with parameters_path.open() as f:
        #         parameters = json.load(f)
        #     _, model_path, parameters_path = discover_extraneous_delay_timers(
        #         self._event_log.train_partition,
        #         self._event_log.log_ids,
        #         model_path,
        #         parameters,
        #         self._settings.extraneous_activity_delays.optimization_metric,
        #         base_dir=best_result_dir,
        #         num_iterations=self._settings.extraneous_activity_delays.num_iterations,
        #         max_alpha=50,
        #     )

        print_section('Evaluation')
        simulation_dir = best_result_dir / 'simulation'
        simulation_dir.mkdir(parents=True)
        self._evaluate_model(model_path, parameters_path, simulation_dir)

        self._clean_up()

        print_section('Exporting canonical model')
        _export_canonical_model(best_result_dir, structure_miner_settings, structure_optimizer_settings,
                                calendars_settings, calendar_optimizer_settings)

        self._settings.to_yaml(best_result_dir)


def _export_canonical_model(
        output_dir: Path,
        structure_settings: StructureMinerSettings,
        structure_optimizer_settings: StructureOptimizationSettings,
        calendar_settings: CalendarPipelineSettings,
        calendar_optimizer_settings: CalendarOptimizationSettings,
):
    canon_path = output_dir / 'canonical_model.json'

    structure = structure_settings.to_dict() | {
        'optimization_metric': str(structure_optimizer_settings.optimization_metric)
    }

    calendars = calendar_settings.to_dict() | {
        'optimization_metric': str(calendar_optimizer_settings.optimization_metric)
    }

    canon = {
        'structure': structure,
        'calendars': calendars,
    }

    with open(canon_path, 'w') as f:
        json.dump(canon, f)
