import itertools
import multiprocessing
import shutil
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from tqdm import tqdm

from simod.analyzers.sim_evaluator import SimilarityEvaluator
from simod.bpm.reader_writer import BPMNReaderWriter
from simod.cli_formatter import print_section, print_message
from simod.configuration import ProjectSettings, GateManagement, PDFMethod, ResourceProfilesType
from simod.event_log.preprocessor import Preprocessor
from simod.event_log.reader_writer import LogReaderWriter
from simod.event_log.utilities import remove_outliers
from simod.optimization.settings import OptimizationSettings
from simod.process_calendars.optimizer import CalendarOptimizer
from simod.process_calendars.settings import PipelineSettings as CalendarPipelineSettings
from simod.process_structure.optimizer import StructureOptimizer
from simod.process_structure.settings import PipelineSettings as StructurePipelineSettings
from simod.simulation.parameters import miner as parameters_miner
from simod.simulation.prosimos import ProsimosSettings, simulate_with_prosimos, PROSIMOS_COLUMN_MAPPING
from simod.utilities import file_id, progress_bar_async


def prepare_project(project_settings: ProjectSettings) -> None:
    if not project_settings.output_dir.exists():
        project_settings.output_dir.mkdir(parents=True)


class Optimizer:
    """Structure and calendars optimization."""
    _settings: OptimizationSettings
    _preprocessor: Optional[Preprocessor] = None
    _log_train: LogReaderWriter
    _log_test: pd.DataFrame

    _structure_optimizer: Optional[StructureOptimizer] = None
    _calendar_optimizer: Optional[CalendarOptimizer] = None

    def __init__(self, settings: OptimizationSettings):
        self._settings = settings

        self._preprocessor = Preprocessor(settings)
        self._settings = self._preprocessor.run()

        self._split_log(0.8)  # TODO: ratio can be an optimization parameter

    def _split_log(self, train_ratio: float):
        log_reader = LogReaderWriter(self._settings.project_settings.log_path, log=self._preprocessor.log)

        train, test = log_reader.split_timeline(train_ratio)

        sort_key = self._settings.project_settings.log_ids.start_time
        self._log_test = test.sort_values(by=[sort_key], ascending=True).reset_index(drop=True)

        train = train.sort_values(by=[sort_key], ascending=True).reset_index(drop=True)
        self._log_train = LogReaderWriter(self._settings.project_settings.log_path, log=train, load=False)

    def _remove_outliers_from_train_data(self):
        df = self._log_train.get_traces_df(include_start_end_events=True)
        df = remove_outliers(df, self._settings.project_settings.log_ids)
        sort_key = self._settings.project_settings.log_ids.start_time
        self._log_train.set_data(df
                                 .sort_values(by=[sort_key], ascending=True)
                                 .reset_index(drop=True)
                                 .to_dict('records'))

    def _mine_and_optimize_structure(self) -> Tuple[StructurePipelineSettings, PDFMethod]:
        optimizer = StructureOptimizer(self._settings.structure_settings, self._log_train)
        self._structure_optimizer = optimizer
        return optimizer.run(), optimizer._settings.pdef_method

    def _optimize_calendars(self, model_path: Path) -> CalendarPipelineSettings:
        optimizer = CalendarOptimizer(
            self._settings.calendar_settings,
            self._log_train,
            model_path)
        result = optimizer.run()
        self._calendar_optimizer = optimizer
        return result

    @staticmethod
    def _read_simulated_log(arguments: Tuple):
        log_path, log_column_mapping, simulation_repetition_index = arguments

        reader = LogReaderWriter(log_path=log_path, column_names=log_column_mapping)

        reader.df.rename(columns={'user': 'resource'}, inplace=True)
        reader.df['role'] = reader.df['resource']
        reader.df['source'] = 'simulation'
        reader.df['run_num'] = simulation_repetition_index
        reader.df = reader.df[~reader.df.task.isin(['Start', 'End'])]  # TODO: should we use EventLogIDs here?

        return reader.df

    @staticmethod
    def _evaluate_logs(arguments):
        settings: OptimizationSettings
        test_log: pd.DataFrame
        simulated_log: pd.DataFrame
        settings, test_log, simulated_log = arguments

        rep = simulated_log.iloc[0].run_num

        evaluator = SimilarityEvaluator(test_log, simulated_log, max_cases=1000)

        measurements = []
        for metric in settings.evaluation_metrics:
            evaluator.measure_distance(metric)
            measurements.append({'run_num': rep, **evaluator.similarity})

        return measurements

    def _simulate(
            self,
            settings: OptimizationSettings,
            bpmn_path: Path,
            json_path: Path,
            simulation_cases: int,
            output_dir: Path):
        assert bpmn_path.exists(), f'Process model {bpmn_path} does not exist.'
        assert json_path.exists(), f'Simulation parameters file {json_path} does not exist.'
        assert output_dir.exists(), f'Output folder {output_dir} does not exist.'

        num_simulations = settings.num_simulations
        cpu_count = multiprocessing.cpu_count()
        w_count = num_simulations if num_simulations <= cpu_count else cpu_count
        pool = multiprocessing.Pool(processes=w_count)

        # Simulate
        simulation_arguments = [
            ProsimosSettings(
                bpmn_path=bpmn_path,
                parameters_path=json_path,
                output_log_path=output_dir / f'simulated_log_{rep}.csv',
                num_simulation_cases=simulation_cases)
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

    def _mine_simulation_parameters(self,
                                    output_dir: Path,
                                    model_path: Path,
                                    pdf_method: PDFMethod,
                                    gateway_probabilities: GateManagement) -> Path:
        bpmn_reader = BPMNReaderWriter(model_path)
        process_graph = bpmn_reader.as_graph()

        log = self._log_test
        log_ids = self._settings.project_settings.log_ids
        profile_type = self._settings.resource_profiles_type

        if profile_type == ResourceProfilesType.UNDIFFERENTIATED:
            simulation_parameters = parameters_miner.mine_for_undifferentiated_resources(
                log, log_ids, model_path, process_graph, pdf_method, bpmn_reader, gateway_probabilities)
        elif profile_type == ResourceProfilesType.POOLED:
            simulation_parameters = parameters_miner.mine_for_pooled_resources(
                log, log_ids, model_path, process_graph, pdf_method, bpmn_reader, gateway_probabilities)
        elif profile_type == ResourceProfilesType.DIFFERENTIATED:
            simulation_parameters = parameters_miner.mine_for_differentiated_resources(
                log, log_ids, model_path, process_graph, pdf_method, bpmn_reader, gateway_probabilities)
        elif profile_type == ResourceProfilesType.AROUND_THE_CLOCK:
            simulation_parameters = parameters_miner.mine_default_24_7(
                log, log_ids, model_path, process_graph, pdf_method, bpmn_reader, gateway_probabilities)
        elif profile_type == ResourceProfilesType.WORKING_HOURS:
            raise NotImplementedError('Working hours resource profiles are not implemented yet.')
        else:
            raise ValueError(f'Unknown resource profiles type {profile_type}')

        json_path = output_dir / f'simulation_parameters_{profile_type.value}.json'
        simulation_parameters.to_json_file(json_path)

        return json_path

    def _evaluate_model(
            self,
            model_path: Path,
            simulation_dir: Path,
            pdf_method: PDFMethod,
            gateway_probabilities_type: GateManagement):
        simulation_cases = self._log_test[self._settings.project_settings.log_ids.case].nunique()

        parameters_path = self._mine_simulation_parameters(
            simulation_dir, model_path, pdf_method, gateway_probabilities_type)

        measurements = self._simulate(
            settings=self._settings,
            bpmn_path=model_path,
            json_path=parameters_path,
            simulation_cases=simulation_cases,
            output_dir=simulation_dir)

        measurements_path = simulation_dir.parent / file_id(prefix='evaluation_')
        measurements_df = pd.DataFrame.from_records(measurements)
        measurements_df.to_csv(measurements_path, index=False)

    def _save_results(
            self,
            output_dir: Path,
            calendar_settings: CalendarPipelineSettings,
            structure_settings: Optional[StructurePipelineSettings] = None):

        print_message(f'Copying calendar results from {calendar_settings.output_dir} to {output_dir}')

        copy_fn = shutil.move if self._settings.remove_intermediate_files else shutil.copytree

        copy_fn(calendar_settings.output_dir, output_dir / calendar_settings.output_dir.name)
        if structure_settings is not None:
            print_message(f'Copying structure results from {structure_settings.output_dir} to {output_dir}')
            copy_fn(structure_settings.output_dir, output_dir / structure_settings.output_dir.name)

        if self._settings.remove_intermediate_files:
            self._structure_optimizer.cleanup()
            self._calendar_optimizer.cleanup()

    def run(self):
        self._remove_outliers_from_train_data()

        structure_settings = None
        pdf_method = PDFMethod.DEFAULT

        if self._settings.discover_structure:
            print_section('Structure optimization')
            structure_settings, pdf_method = self._mine_and_optimize_structure()

            # Taking the best model from the structure optimization
            model_path = structure_settings.model_path
        else:
            print_section('No structure discovery needed, using the provided model')
            model_path = self._settings.project_settings.model_path

        assert model_path.exists(), 'Model does not exist'

        print_section('Calendars optimization')
        calendars_settings = self._optimize_calendars(model_path)

        print_section('Evaluation')
        best_result_dir = self._settings.project_settings.output_dir / 'best_result'
        simulation_dir = best_result_dir / 'simulation'
        simulation_dir.mkdir(parents=True)

        gateway_probabilities_type = structure_settings.gateway_probabilities
        self._evaluate_model(model_path, simulation_dir, pdf_method, gateway_probabilities_type)

        print_section('Saving results')
        self._save_results(best_result_dir, calendars_settings, structure_settings)

        # TODO: export model
        # export_canonical_model ?

        # TODO: track all evaluation metrics across all optimization steps and unite into one table, don't discard intermediate results
