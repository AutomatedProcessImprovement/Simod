import json
import json
import shutil
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from simod.cli_formatter import print_section, print_message
from simod.configuration import Configuration
from simod.discovery.extraneous_delay_timers import discover_extraneous_delay_timers
from simod.event_log.reader_writer import LogReaderWriter
from simod.event_log.utilities import remove_outliers, read
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
    _output_dir: Path

    # Loaded and preprocessed event log
    _log_reader: LogReaderWriter

    # Event log split
    _log_train: LogReaderWriter
    _log_test: pd.DataFrame

    # Downstream executors
    _structure_optimizer: Optional[StructureOptimizer]
    _calendar_optimizer: Optional[CalendarOptimizer]

    def __init__(self, settings: Configuration, log: Optional[pd.DataFrame] = None, output_dir: Optional[Path] = None):
        self._settings = settings

        if output_dir is None:
            self._output_dir = get_project_dir() / 'outputs' / folder_id()
        else:
            self._output_dir = output_dir

        self._log_reader = LogReaderWriter(self._settings.common.log_path,
                                           self._settings.common.log_ids,
                                           log=log)

        if self._settings.common.test_log_path is not None:
            self._log_train = LogReaderWriter(self._settings.common.log_path, self._settings.common.log_ids)
            self._log_test, _ = read(self._settings.common.test_log_path, self._settings.common.log_ids)
        else:
            self._split_log(0.8, self._log_reader)

    def _split_log(self, train_ratio: float, log_reader: LogReaderWriter):
        train, test = log_reader.split_timeline(train_ratio)

        sort_key = self._settings.common.log_ids.start_time
        self._log_test = test.sort_values(by=[sort_key], ascending=True).reset_index(drop=True)

        train = train.sort_values(by=[sort_key], ascending=True).reset_index(drop=True)
        self._log_train = LogReaderWriter(self._settings.common.log_path, self._settings.common.log_ids, log=train,
                                          load=False)

    def _remove_outliers_from_train_data(self):
        df = self._log_train.get_traces_df()
        df = remove_outliers(df, self._settings.common.log_ids)
        sort_key = self._settings.common.log_ids.start_time
        self._log_train.set_data(df
                                 .sort_values(by=[sort_key], ascending=True)
                                 .reset_index(drop=True)
                                 .to_dict('records'))

    def _optimize_structure(self) -> Tuple[StructureOptimizationSettings, StructurePipelineSettings]:
        settings = StructureOptimizationSettings.from_configuration(self._settings, self._output_dir)
        optimizer = StructureOptimizer(settings, self._log_train, self._settings.common.log_ids)
        self._structure_optimizer = optimizer
        best_pipeline_settings = optimizer.run()

        return settings, best_pipeline_settings

    def _optimize_calendars(
            self,
            structure_settings: StructureMinerSettings,
            model_path: Optional[Path],
    ) -> Tuple[CalendarOptimizationSettings, CalendarPipelineSettings]:
        calendar_settings = CalendarOptimizationSettings.from_configuration(self._settings, self._output_dir)
        optimizer = CalendarOptimizer(calendar_settings, self._log_train, structure_settings, model_path,
                                      self._settings.common.log_ids)
        best_pipeline_settings = optimizer.run()
        self._calendar_optimizer = optimizer
        return calendar_settings, best_pipeline_settings

    def _mine_simulation_parameters(self, output_dir: Path, model_path: Path) -> Path:
        log = self._log_test
        log_ids = self._settings.common.log_ids
        profile_type = self._settings.calendars.resource_profiles.discovery_type

        parameters = mine_parameters(
            self._settings.calendars.case_arrival,
            self._settings.calendars.resource_profiles,
            log,
            log_ids,
            model_path,
            self._settings.structure.gateway_probabilities)

        json_path = output_dir / f'simulation_parameters_{profile_type.value}.json'
        parameters.to_json_file(json_path)

        return json_path

    def _evaluate_model(
            self,
            model_path: Path,
            parameters_path: Path,
            simulation_dir: Path):
        assert parameters_path.exists(), f'Best calendar simulation parameters file does not exist'

        self._log_test.to_csv(simulation_dir / 'test_log.csv', index=False)

        simulation_cases = self._log_test[self._settings.common.log_ids.case].nunique()

        num_simulations = self._settings.common.repetitions

        metrics = self._settings.common.evaluation_metrics \
            if isinstance(self._settings.common.evaluation_metrics, list) \
            else [self._settings.common.evaluation_metrics]

        simulation_start_time = self._log_test[self._settings.common.log_ids.start_time].min()

        measurements = simulate_and_evaluate(
            model_path=model_path,
            parameters_path=parameters_path,
            output_dir=simulation_dir,
            simulation_cases=simulation_cases,
            simulation_start_time=simulation_start_time,
            validation_log=self._log_test,
            validation_log_ids=self._settings.common.log_ids,
            num_simulations=num_simulations,
            metrics=metrics,
        )

        measurements_path = simulation_dir.parent / file_id(prefix='evaluation_')
        measurements_df = pd.DataFrame.from_records(measurements)
        measurements_df.to_csv(measurements_path, index=False)

    def _export_canonical_model(
            self, output_dir: Path,
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

    def _clean_up(self):
        if not self._settings.common.clean_intermediate_files:
            return

        print_section('Removing intermediate files')
        self._structure_optimizer.cleanup()
        self._calendar_optimizer.cleanup()

    def _mine_structure(self, settings: StructureMinerSettings, output_dir: Path) \
            -> Tuple[Path, StructureMinerSettings]:
        print_message(f'Mining structure with settings {settings.to_dict()}')

        # Saving the full pre-processed log to disk
        log_path = output_dir / self._settings.common.log_path.name
        log_path = log_path.with_suffix('.xes')
        self._log_train.write_xes(log_path)

        model_path = output_dir / (self._settings.common.log_path.stem + '.bpmn')

        StructureMiner(settings, log_path, model_path)

        return model_path, settings

    def _mine_calendars(
            self,
            best_settings: CalendarPipelineSettings,
            model_path: Path,
            output_dir: Path) -> Tuple[Path, CalendarPipelineSettings]:
        settings = CalendarPipelineSettings(
            output_dir=output_dir,
            model_path=model_path,
            gateway_probabilities_method=best_settings.gateway_probabilities_method,
            case_arrival=best_settings.case_arrival,
            resource_profiles=best_settings.resource_profiles,
        )

        print_message(f'Mining calendars with settings {settings.to_dict()}')

        # Taking the full pre-processed original log for extracting calendars
        log = self._log_train.get_traces_df()

        parameters = mine_parameters(
            settings.case_arrival, settings.resource_profiles, log, self._settings.common.log_ids, model_path,
            settings.gateway_probabilities_method)

        json_path = settings.output_dir / 'simulation_parameters.json'

        parameters.to_json_file(json_path)

        return json_path, settings

    def run(self):
        self._remove_outliers_from_train_data()

        best_result_dir = self._output_dir / 'best_result'
        best_result_dir.mkdir(parents=True, exist_ok=True)

        print_section('Structure optimization')
        structure_optimizer_settings, structure_pipeline_settings = self._optimize_structure()
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
        model_path = self._settings.common.model_path
        calendar_optimizer_settings, calendar_pipeline_settings = self._optimize_calendars(
            structure_miner_settings, model_path)

        if model_path is None:
            print_section('Mining structure using the best hyperparameters')
            model_path, structure_miner_settings = self._mine_structure(structure_miner_settings, best_result_dir)
        else:
            shutil.copy(model_path, best_result_dir)

        print_section('Mining calendars using the best hyperparameters')
        parameters_path, calendars_settings = self._mine_calendars(
            calendar_pipeline_settings, model_path, best_result_dir)

        print_section('Mining extraneous delay timers')
        with parameters_path.open() as f:
            parameters = json.load(f)
        _, model_path, parameters_path = discover_extraneous_delay_timers(
            self._log_train.get_traces_df(),
            self._settings.common.log_ids,
            model_path,
            parameters,
            base_dir=best_result_dir,
            num_iterations=50,
            max_alpha=50,
        )

        print_section('Evaluation')
        simulation_dir = best_result_dir / 'simulation'
        simulation_dir.mkdir(parents=True)
        self._evaluate_model(model_path, parameters_path, simulation_dir)

        self._clean_up()

        print_section('Exporting canonical model')
        self._export_canonical_model(best_result_dir, structure_miner_settings, structure_optimizer_settings,
                                     calendars_settings, calendar_optimizer_settings)

        self._settings.to_yaml(best_result_dir)
