import itertools
import json
import multiprocessing
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from tqdm import tqdm

from simod.cli_formatter import print_section, print_message
from simod.configuration import Configuration
from simod.evaluation_metrics import compute_metric
from simod.event_log.column_mapping import PROSIMOS_COLUMNS
from simod.event_log.preprocessor import Preprocessor
from simod.event_log.reader_writer import LogReaderWriter
from simod.event_log.utilities import remove_outliers
from simod.process_calendars.optimizer import CalendarOptimizer
from simod.process_calendars.settings import PipelineSettings as CalendarPipelineSettings, CalendarOptimizationSettings
from simod.process_structure.miner import Settings as StructureMinerSettings, StructureMiner
from simod.process_structure.optimizer import StructureOptimizer
from simod.process_structure.settings import PipelineSettings as StructurePipelineSettings, \
    StructureOptimizationSettings
from simod.simulation.parameters.miner import mine_parameters
from simod.simulation.prosimos import ProsimosSettings, simulate_with_prosimos, PROSIMOS_COLUMN_MAPPING
from simod.utilities import file_id, progress_bar_async, get_project_dir, folder_id


class Optimizer:
    """Structure and calendars optimization."""
    _settings: Configuration
    _output_dir: Path

    # Loaded and preprocessed event log
    _log_reader: LogReaderWriter

    # Event log split
    _log_train: LogReaderWriter
    _log_test: pd.DataFrame

    # Downstream executors
    _preprocessor: Optional[Preprocessor]
    _structure_optimizer: Optional[StructureOptimizer]
    _calendar_optimizer: Optional[CalendarOptimizer]

    def __init__(self, settings: Configuration):
        self._settings = settings

        self._output_dir = get_project_dir() / 'outputs' / folder_id()

        self._preprocessor = Preprocessor(settings, self._output_dir)
        self._settings = self._preprocessor.run()

        self._log_reader = LogReaderWriter(self._settings.common.log_path,
                                           self._settings.common.log_ids,
                                           log=self._preprocessor.log)

        self._split_log(0.8, self._log_reader)  # TODO: ratio can be an optimization parameter

    def _split_log(self, train_ratio: float, log_reader: LogReaderWriter):
        train, test = log_reader.split_timeline(train_ratio)

        sort_key = self._settings.common.log_ids.start_time
        self._log_test = test.sort_values(by=[sort_key], ascending=True).reset_index(drop=True)

        train = train.sort_values(by=[sort_key], ascending=True).reset_index(drop=True)
        self._log_train = LogReaderWriter(self._settings.common.log_path, self._settings.common.log_ids, log=train,
                                          load=False)

    def _remove_outliers_from_train_data(self):
        df = self._log_train.get_traces_df(include_start_end_events=True)
        df = remove_outliers(df, self._settings.common.log_ids)
        sort_key = self._settings.common.log_ids.start_time
        self._log_train.set_data(df
                                 .sort_values(by=[sort_key], ascending=True)
                                 .reset_index(drop=True)
                                 .to_dict('records'))

    def _optimize_structure(self) -> StructurePipelineSettings:
        settings = StructureOptimizationSettings.from_configuration(self._settings, self._output_dir)
        optimizer = StructureOptimizer(settings, self._log_train, self._settings.common.log_ids)
        self._structure_optimizer = optimizer
        return optimizer.run()

    def _optimize_calendars(self, structure_settings: StructureMinerSettings,
                            model_path: Path) -> CalendarPipelineSettings:
        calendar_settings = CalendarOptimizationSettings.from_configuration(self._settings, self._output_dir)
        optimizer = CalendarOptimizer(calendar_settings, self._log_train, structure_settings, model_path,
                                      self._settings.common.log_ids)
        result = optimizer.run()
        self._calendar_optimizer = optimizer
        return result

    def _read_simulated_log(self, arguments: Tuple):
        log_path, log_column_mapping, simulation_repetition_index = arguments

        reader = LogReaderWriter(log_path=log_path, log_ids=PROSIMOS_COLUMNS, column_names=log_column_mapping)

        reader.df['role'] = reader.df['resource']
        reader.df['source'] = 'simulation'
        reader.df['run_num'] = simulation_repetition_index
        reader.df = reader.df[
            ~reader.df[PROSIMOS_COLUMNS.activity].isin(
                ['Start', 'start', 'End', 'end'])]  # TODO: should we use EventLogIDs here?

        return reader.df

    def _evaluate_logs(self, arguments):
        settings: Configuration
        test_log: pd.DataFrame
        simulated_log: pd.DataFrame
        settings, test_log, simulated_log = arguments

        rep = simulated_log.iloc[0].run_num

        measurements = []
        for metric in settings.common.evaluation_metrics:
            value = compute_metric(metric, test_log, self._settings.common.log_ids, simulated_log, PROSIMOS_COLUMNS)
            measurements.append({'run_num': rep, 'metric': metric, 'similarity': value})

        return measurements

    def _simulate(
            self,
            settings: Configuration,
            bpmn_path: Path,
            json_path: Path,
            simulation_cases: int,
            output_dir: Path):
        assert bpmn_path.exists(), f'Process model {bpmn_path} does not exist.'
        assert json_path.exists(), f'Simulation parameters file {json_path} does not exist.'
        assert output_dir.exists(), f'Output folder {output_dir} does not exist.'

        num_simulations = settings.common.repetitions
        cpu_count = multiprocessing.cpu_count()
        w_count = num_simulations if num_simulations <= cpu_count else cpu_count
        pool = multiprocessing.Pool(processes=w_count)

        # Simulate
        simulation_arguments = [
            ProsimosSettings(
                bpmn_path=bpmn_path,
                parameters_path=json_path,
                output_log_path=output_dir / f'simulated_log_{rep}.csv',
                num_simulation_cases=simulation_cases,
                simulation_start=self._log_test[self._settings.common.log_ids.start_time].min(),
            )
            for rep in range(num_simulations)]
        p = pool.map_async(simulate_with_prosimos, simulation_arguments)
        progress_bar_async(p, 'simulating', num_simulations)

        # Read simulated logs
        read_arguments = [(simulation_arguments[index].output_log_path, PROSIMOS_COLUMN_MAPPING, index)
                          for index in range(num_simulations)]
        p = pool.map_async(self._read_simulated_log, read_arguments)
        progress_bar_async(p, 'reading simulated logs', num_simulations)

        # Evaluate
        evaluation_arguments = [(settings, self._log_test, log) for log in p.get()]
        if simulation_cases > 1000:
            pool.close()
            results = [self._evaluate_logs(arg) for arg in tqdm(evaluation_arguments, 'evaluating results')]
            evaluation_measurements = list(itertools.chain(*results))
        else:
            p = pool.map_async(self._evaluate_logs, evaluation_arguments)
            progress_bar_async(p, 'evaluating results', num_simulations)
            pool.close()
            evaluation_measurements = list(itertools.chain(*p.get()))

        return evaluation_measurements

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
            simulation_dir: Path,
            calendar_settings: CalendarPipelineSettings):
        self._log_test.to_csv(simulation_dir / 'test_log.csv', index=False)

        simulation_cases = self._log_test[self._settings.common.log_ids.case].nunique()

        parameters_path = calendar_settings.output_dir / 'simulation_parameters.json'
        assert parameters_path.exists(), f'Best calendar simulation parameters file does not exist'

        measurements = self._simulate(
            settings=self._settings,
            bpmn_path=model_path,
            json_path=parameters_path,
            simulation_cases=simulation_cases,
            output_dir=simulation_dir)

        measurements_path = simulation_dir.parent / file_id(prefix='evaluation_')
        measurements_df = pd.DataFrame.from_records(measurements)
        measurements_df.to_csv(measurements_path, index=False)

    def _export_canonical_model(
            self, output_dir: Path,
            structure_settings: StructureMinerSettings,
            calendar_settings: CalendarPipelineSettings):
        canon_path = output_dir / 'canonical_model.json'

        canon = {
            'structure': structure_settings.to_dict(),
            'calendars': calendar_settings.to_dict(),
            'resource_profile_type': self._settings.calendars.resource_profiles.discovery_type.name
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
            gateway_probabilities=best_settings.gateway_probabilities,
            case_arrival=best_settings.case_arrival,
            resource_profiles=best_settings.resource_profiles,
        )

        print_message(f'Mining calendars with settings {settings.to_dict()}')

        # Taking the full pre-processed original log for extracting calendars
        log = self._log_train.get_traces_df(include_start_end_events=True)

        parameters = mine_parameters(
            settings.case_arrival, settings.resource_profiles, log, self._settings.common.log_ids, model_path,
            settings.gateway_probabilities)

        json_path = settings.output_dir / 'simulation_parameters.json'

        parameters.to_json_file(json_path)

        return json_path, settings

    def run(self):
        self._remove_outliers_from_train_data()

        best_result_dir = self._output_dir / 'best_result'
        best_result_dir.mkdir(parents=True, exist_ok=True)

        structure_settings: Optional[StructureMinerSettings] = None

        model_path = None

        if self._settings.structure.disable_discovery is False:
            print_section('Structure optimization')
            structure_optimizer_settings = self._optimize_structure()

            structure_settings = StructureMinerSettings(
                mining_algorithm=self._settings.structure.mining_algorithm,
                epsilon=structure_optimizer_settings.epsilon,
                eta=structure_optimizer_settings.eta,
                concurrency=structure_optimizer_settings.concurrency,
                and_prior=structure_optimizer_settings.and_prior,
                or_rep=structure_optimizer_settings.or_rep,
            )
        else:
            print_section('No structure discovery needed, using the provided model')
            model_path = self._settings.common.model_path

        print_section('Calendars optimization')
        calendar_optimizer_settings = self._optimize_calendars(structure_settings, model_path)

        print_section('Mining structure using the best hyperparameters')
        model_path, structure_settings = self._mine_structure(structure_settings, best_result_dir)

        print_section('Mining calendars using the best hyperparameters')
        parameters_path, calendars_settings = self._mine_calendars(
            calendar_optimizer_settings, model_path, best_result_dir)

        print_section('Evaluation')

        simulation_dir = best_result_dir / 'simulation'
        simulation_dir.mkdir(parents=True)

        self._evaluate_model(model_path, simulation_dir, calendars_settings)

        self._clean_up()

        print_section('Exporting canonical model')
        self._export_canonical_model(best_result_dir, structure_settings, calendars_settings)
