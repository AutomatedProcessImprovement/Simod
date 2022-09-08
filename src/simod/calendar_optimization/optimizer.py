import copy
import itertools
import multiprocessing
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, Union, List

import numpy as np
import pandas as pd
import yaml
from hyperopt import Trials, hp, fmin, STATUS_OK, STATUS_FAIL
from hyperopt import tpe
from tqdm import tqdm

from simod import support_utils as sup
from simod.analyzers import sim_evaluator as sim
from simod.cli_formatter import print_subsection, print_message, print_warning
from simod.configuration import Configuration, Metric, PDFMethod, DataType, \
    GateManagement
from simod.event_log_processing.event_log_ids import EventLogIDs, SIMOD_DEFAULT_COLUMNS
from simod.event_log_processing.reader import EventLogReader
from simod.hyperopt_pipeline import HyperoptPipeline
from simod.process_model.bpmn import BPMNReaderWriter
from simod.process_structure.simulation import ProsimosSettings, simulate_with_prosimos, PROSIMOS_COLUMN_MAPPING, \
    undifferentiated_resources_parameters
from simod.simulator import simulate
from simod.support_utils import get_project_dir, remove_asset, progress_bar_async


@dataclass
class ProjectSettings:
    project_name: str
    output_dir: Optional[Path]
    log_path: Path
    log_ids: Optional[EventLogIDs]
    model_path: Optional[Path]

    @staticmethod
    def from_dict(data: dict) -> 'ProjectSettings':
        project_name = data.get('project_name', None)
        assert project_name is not None, 'Project name is not specified'

        output_dir = data.get('output_dir', None)

        log_path = data.get('log_path', None)
        assert log_path is not None, 'Log path is not specified'

        log_ids = data.get('log_ids', None)

        model_path = data.get('model_path', None)

        return ProjectSettings(
            project_name=project_name,
            log_path=log_path,
            log_ids=log_ids,
            model_path=model_path,
            output_dir=output_dir)

    @staticmethod
    def from_stream(stream: Union[str, bytes]) -> 'ProjectSettings':
        settings = yaml.load(stream, Loader=yaml.FullLoader)

        log_path = settings.get('log_path', None)
        assert log_path is not None, 'Log path is not specified'
        log_path = Path(log_path)

        project_name = os.path.splitext(os.path.basename(log_path))[0]

        output_dir = settings.get('output_dir', None)

        # TODO: log_ids
        log_ids = settings.get('log_ids', None)

        model_path = settings.get('model_path', None)

        return ProjectSettings(
            project_name=project_name,
            log_path=log_path,
            model_path=model_path,
            log_ids=log_ids,
            output_dir=output_dir)


@dataclass
class Settings:
    """Settings for resources' and arrival calendars optimizer."""
    max_evaluations: int = 1
    simulation_repetitions: int = 1
    pdef_method: Optional[PDFMethod] = PDFMethod.AUTOMATIC
    gateway_probabilities: Optional[Union[GateManagement, List[GateManagement]]] = GateManagement.DISCOVERY

    rp_similarity: Union[float, list[float], None] = None
    res_sup_dis: Optional[list[float]] = None
    res_con_dis: Optional[list[float]] = None
    res_dtype: Union[DataType, list[DataType], None] = None
    arr_support: Union[float, list[float], None] = None
    arr_confidence: Union[float, list[float], None] = None
    arr_dtype: Union[DataType, list[DataType], None] = None

    @staticmethod
    def from_stream(stream: Union[str, bytes]) -> 'Settings':
        settings = yaml.load(stream, Loader=yaml.FullLoader)

        v = settings.get('time_optimizer', None)
        if v is not None:
            settings = v

        max_evaluations = settings.get('max_evaluations', None)
        if max_evaluations is None:
            max_evaluations = settings.get('max_eval_t', 1)  # legacy support

        simulation_repetitions = settings.get('simulation_repetitions', 1)

        pdef_method = settings.get('pdef_method', PDFMethod.AUTOMATIC)

        gateway_probabilities = settings.get('gateway_probabilities', None)
        if gateway_probabilities is None:
            gateway_probabilities = settings.get('gate_management', None)  # legacy key support
        if gateway_probabilities is not None:
            if isinstance(gateway_probabilities, list):
                gateway_probabilities = [GateManagement.from_str(g) for g in gateway_probabilities]
            elif isinstance(gateway_probabilities, str):
                gateway_probabilities = GateManagement.from_str(gateway_probabilities)
            else:
                raise ValueError('Gateway probabilities must be a list or a string.')

        rp_similarity = settings.get('rp_similarity', None)
        res_sup_dis = settings.get('res_sup_dis', None)
        res_con_dis = settings.get('res_con_dis', None)
        res_dtype = settings.get('res_dtype', None)
        if res_dtype is not None:
            res_dtype = DataType.from_str(res_dtype)  # TODO: this should change with new calendars
        arr_support = settings.get('arr_support', None)
        arr_confidence = settings.get('arr_confidence', None)
        arr_dtype = settings.get('arr_dtype', None)  # TODO: this should change with new calendars
        if arr_dtype is not None:
            arr_dtype = DataType.from_str(arr_dtype)

        return Settings(
            max_evaluations=max_evaluations,
            simulation_repetitions=simulation_repetitions,
            pdef_method=pdef_method,
            gateway_probabilities=gateway_probabilities,
            rp_similarity=rp_similarity,
            res_sup_dis=res_sup_dis,
            res_con_dis=res_con_dis,
            res_dtype=res_dtype,
            arr_support=arr_support,
            arr_confidence=arr_confidence,
            arr_dtype=arr_dtype)


@dataclass
class ResourceSettings:
    # in case of "discovered"
    res_confidence: Optional[float] = None
    res_support: Optional[float] = None

    # in case of "default"
    res_dtype: Optional[DataType] = None

    def __post_init__(self):
        assert (self.res_confidence is not None and self.res_support is not None) or (self.res_dtype is not None), \
            'Either resource confidence and support or calendar type should be specified'


@dataclass
class ArrivalSettings:
    # in case of "discovered"
    arr_confidence: Optional[float] = None
    arr_support: Optional[float] = None

    # in case of "default"
    arr_dtype: Optional[DataType] = None

    def __post_init__(self):
        assert (self.arr_confidence is not None and self.arr_support is not None) or (self.arr_dtype is not None), \
            'Either arrival confidence and support or calendar type should be specified'


class OptimizationType(Enum):
    """Type of optimization."""
    DISCOVERED = 1
    DEFAULT = 2

    @staticmethod
    def from_str(s: str) -> 'OptimizationType':
        if s.lower() == 'discovered':
            return OptimizationType.DISCOVERED
        elif s.lower() == 'default':
            return OptimizationType.DEFAULT
        else:
            raise ValueError(f'Unknown optimization type: {s}')

    def __str__(self):
        return self.name.lower()


@dataclass
class PipelineSettings(ProjectSettings):
    """Settings for the calendars optimizer pipeline."""
    gateway_probabilities: Optional[GateManagement]
    rp_similarity: float
    res_cal_met: Tuple[OptimizationType, ResourceSettings]
    arr_cal_met: Tuple[OptimizationType, ArrivalSettings]

    @staticmethod
    def from_dict(data: dict) -> 'PipelineSettings':
        project_settings = ProjectSettings.from_dict(data)

        rp_similarity = data.get('rp_similarity', None)
        assert rp_similarity is not None, 'rp_similarity is not specified'

        res_cal_met = data.get('res_cal_met', None)
        assert res_cal_met is not None, 'res_cal_met is not specified'

        arr_cal_met = data.get('arr_cal_met', None)
        assert arr_cal_met is not None, 'arr_cal_met is not specified'

        resource_optimization_type = OptimizationType.from_str(res_cal_met[0])
        if resource_optimization_type == OptimizationType.DISCOVERED:
            res_confidence = res_cal_met[1].get('res_confidence', None)
            assert res_confidence is not None, 'res_confidence is not specified'

            res_support = res_cal_met[1].get('res_support', None)
            assert res_support is not None, 'res_support is not specified'

            resource_settings = ResourceSettings(res_confidence, res_support)
        elif resource_optimization_type == OptimizationType.DEFAULT:
            res_dtype = res_cal_met[1].get('res_dtype', None)
            assert res_dtype is not None, 'res_dtype is not specified'

            resource_settings = ResourceSettings(res_dtype=res_dtype)
        else:
            raise ValueError(f'Unknown optimization type: {resource_optimization_type}')

        arrival_optimization_type = OptimizationType.from_str(arr_cal_met[0])
        if arrival_optimization_type == OptimizationType.DISCOVERED:
            arr_confidence = arr_cal_met[1].get('arr_confidence', None)
            assert arr_confidence is not None, 'arr_confidence is not specified'

            arr_support = arr_cal_met[1].get('arr_support', None)
            assert arr_support is not None, 'arr_support is not specified'

            arrival_settings = ArrivalSettings(arr_confidence, arr_support)
        elif arrival_optimization_type == OptimizationType.DEFAULT:
            arr_dtype = arr_cal_met[1].get('arr_dtype', None)
            assert arr_dtype is not None, 'arr_dtype is not specified'

            arrival_settings = ArrivalSettings(arr_dtype=arr_dtype)
        else:
            raise ValueError(f'Unknown optimization type: {arrival_optimization_type}')

        gateway_probabilities = data.get('gateway_probabilities', None)

        return PipelineSettings(
            **project_settings.__dict__,
            gateway_probabilities=gateway_probabilities,
            rp_similarity=rp_similarity,
            res_cal_met=(resource_optimization_type, resource_settings),
            arr_cal_met=(arrival_optimization_type, arrival_settings)
        )


class CalendarOptimizer(HyperoptPipeline):
    best_output: Optional[Path]
    best_parameters: dict
    _measurements_file_name: Path
    _temp_output: Path

    _settings_global: Configuration
    _settings_time: Configuration

    _log: EventLogReader
    _log_ids: EventLogIDs
    _log_train: EventLogReader
    _log_validation: pd.DataFrame
    _original_log: EventLogReader
    _original_log_train: EventLogReader
    _original_log_validation: pd.DataFrame

    _project_settings: ProjectSettings
    _calendar_optimizer_settings: Settings

    _bayes_trials: Trials = Trials()

    def __init__(
            self,
            project_settings: ProjectSettings,
            calendar_optimizer_settings: Settings,
            log: EventLogReader,
            model_path: Path,
            log_ids: Optional[EventLogIDs] = None):
        self._project_settings = project_settings

        if model_path is not None and project_settings.model_path != model_path:
            print_warning(f'Overriding model path from {project_settings.model_path} to {model_path}')
            self._project_settings.model_path = model_path

        self._calendar_optimizer_settings = calendar_optimizer_settings
        self._log = log
        self._log_ids = log_ids if log_ids is not None else SIMOD_DEFAULT_COLUMNS

        # hyperopt search space
        self._space = self._define_search_space(project_settings, calendar_optimizer_settings)

        # setting train and validation log data
        train, validation = self._split_timeline(0.8)
        self._log_train = EventLogReader.copy_without_data(self._log)
        self._log_train.set_data(train
                                 .sort_values(self._log_ids.start_time, ascending=True)
                                 .reset_index(drop=True)
                                 .to_dict('records'))
        self._log_validation = validation

        log_df = pd.DataFrame(self._log_train.data)
        self._conformant_traces = log_df
        self._process_stats = log_df

        # setting original log data
        # TODO: deepcopy is expensive, can we do better?
        self._original_log = copy.deepcopy(log)
        self._original_log_train = copy.deepcopy(self._log_train)
        self._original_log_validation = copy.deepcopy(self._log_validation)

        # creating files and folders
        self._temp_output = get_project_dir() / 'outputs' / sup.folder_id()
        self._temp_output.mkdir(parents=True, exist_ok=True)
        self._measurements_file_name = self._temp_output / sup.file_id(prefix='OP_')
        with self._measurements_file_name.open('w') as _:
            pass

    def run(self):
        def pipeline(trial_stg: Union[dict, PipelineSettings]):
            print_subsection('Trial')
            print_message(f'train split: {len(pd.DataFrame(self._log_train.data).caseid.unique())}, '
                          f'valdn split: {len(pd.DataFrame(self._log_validation).caseid.unique())}')

            if isinstance(trial_stg, dict):
                trial_stg = PipelineSettings.from_dict(trial_stg)

            status = STATUS_OK

            status, result = self.step(status, self._create_folder, trial_stg)
            if status == STATUS_OK:
                trial_stg = result
            # TODO: should we continue if status is not OK without folders?

            status, result = self.step(status, self._extract_parameters_undifferentiated, trial_stg)
            bpmn_path, json_path, simulation_cases = result

            # TODO: in simulation, the old parameters aren't used: rp_similarity, res_cal_met, arr_cal_met -- how can I integrate them?
            # TODO: redefine pipeline settings for calendars optimization, Prosimos uses confidence and support
            status, result = self.step(status, self._simulate_undifferentiated, trial_stg, json_path, simulation_cases)
            evaluation_measurements = result if status == STATUS_OK else []

            response = self._define_response(trial_stg, status, evaluation_measurements)

            # reinstate log
            self._log = self._original_log  # TODO: no need
            self._log_train = copy.deepcopy(self._original_log_train)
            self._log_validation = copy.deepcopy(self._original_log_validation)

            return response

        # Optimize
        best = fmin(fn=pipeline,
                    space=self._space,
                    algo=tpe.suggest,
                    max_evals=self._calendar_optimizer_settings.max_evaluations,
                    trials=self._bayes_trials,
                    show_progressbar=False)
        # Save results
        self.best_parameters = best
        results = pd.DataFrame(self._bayes_trials.results).sort_values('loss')
        results_ok = results[results.status == STATUS_OK]
        try:
            self.best_output = results_ok.iloc[0].output
        except Exception as e:
            raise e

    def cleanup(self):
        remove_asset(self._temp_output)

    @staticmethod
    def _define_search_space(project_settings: ProjectSettings, optimizer_settings: Settings):
        assert len(optimizer_settings.rp_similarity) == 2, 'rp_similarity must have 2 values: low and high'
        assert len(optimizer_settings.res_sup_dis) == 2, 'res_sup_dis must have 2 values: low and high'
        assert len(optimizer_settings.res_con_dis) == 2, 'res_con_dis must have 2 values: low and high'
        assert len(optimizer_settings.arr_support) == 2, 'arr_support must have 2 values: low and high'
        assert len(optimizer_settings.arr_confidence) == 2, 'arr_confidence must have 2 values: low and high'

        # TODO: decrypt the names

        rp_similarity = {'rp_similarity': hp.uniform('rp_similarity', *optimizer_settings.rp_similarity)}

        resource_calendars = {'res_cal_met': hp.choice(
            'res_cal_met',
            [
                ('discovered', {'res_support': hp.uniform('res_support', *optimizer_settings.res_sup_dis),
                                'res_confidence': hp.uniform('res_confidence', *optimizer_settings.res_con_dis)}),
                ('default', {'res_dtype': hp.choice('res_dtype', optimizer_settings.res_dtype)})
            ]
        )}

        arrival_calendar = {'arr_cal_met': hp.choice(
            'arr_cal_met',
            [
                ('discovered', {'arr_support': hp.uniform('arr_support', *optimizer_settings.arr_support),
                                'arr_confidence': hp.uniform('arr_confidence', *optimizer_settings.arr_confidence)}),
                ('default', {'arr_dtype': hp.choice('arr_dtype', optimizer_settings.arr_dtype)})
            ]
        )}

        gateway_probabilities = {
            'gateway_probabilities': hp.choice('gateway_probabilities', optimizer_settings.gateway_probabilities)
        }

        space = project_settings.__dict__ | rp_similarity | resource_calendars | arrival_calendar | gateway_probabilities

        return space

    def _create_folder(self, settings: PipelineSettings) -> PipelineSettings:
        settings.output_dir = self._temp_output / sup.folder_id()

        simulation_data_path = settings.output_dir / 'sim_data'
        simulation_data_path.mkdir(parents=True, exist_ok=True)

        return settings

    # def _extract_parameters(self, settings: PipelineSettings):
    #     parameters = self._extract_time_parameters(settings)
    #
    #     self._xml_print(parameters._asdict(), os.path.join(settings.output_dir, settings.project_name + '.bpmn'))
    #     self._log_validation.rename(columns={'user': 'resource'}, inplace=True)
    #     self._log_validation['source'] = 'log'
    #     self._log_validation['run_num'] = 0
    #     self._log_validation = self._log_validation.merge(parameters.resource_table[['resource', 'role']],
    #                                                       on='resource', how='left')
    #     self._log_validation = self._log_validation[~self._log_validation.task.isin(['Start', 'End'])]
    #     parameters.resource_table.to_pickle(os.path.join(settings.output_dir, 'resource_table.pkl'))

    def _extract_parameters_undifferentiated(self, settings: PipelineSettings) -> Tuple:
        bpmn_path = settings.model_path
        bpmn_reader = BPMNReaderWriter(bpmn_path)
        process_graph = bpmn_reader.as_graph()

        log = self._log_train.get_traces_df(include_start_end_events=True)
        pdf_method = self._calendar_optimizer_settings.pdef_method

        simulation_parameters = undifferentiated_resources_parameters(
            log, self._log_ids, bpmn_path, process_graph, pdf_method, bpmn_reader, settings.gateway_probabilities)

        json_path = bpmn_path.with_suffix('.json')
        simulation_parameters.to_json_file(json_path)

        simulation_cases = log[self._log_ids.case].nunique()

        return bpmn_path, json_path, simulation_cases

    @staticmethod
    def _read_simulated_log(arguments: Tuple):
        log_path, log_column_mapping, simulation_repetition_index = arguments

        reader = EventLogReader(log_path=log_path, column_names=log_column_mapping)

        reader.df.rename(columns={'user': 'resource'}, inplace=True)
        reader.df['role'] = reader.df['resource']
        reader.df['source'] = 'simulation'
        reader.df['run_num'] = simulation_repetition_index
        reader.df = reader.df[~reader.df.task.isin(['Start', 'End'])]

        return reader.df

    def _simulate_undifferentiated(self, settings: PipelineSettings, json_path: Path, simulation_cases: int):
        num_simulations = self._calendar_optimizer_settings.simulation_repetitions
        bpmn_path = self._project_settings.model_path
        output_dir = settings.output_dir

        cpu_count = multiprocessing.cpu_count()
        w_count = num_simulations if num_simulations <= cpu_count else cpu_count
        pool = multiprocessing.Pool(processes=w_count)

        # Simulate
        simulation_arguments = [
            ProsimosSettings(
                bpmn_path=bpmn_path,
                parameters_path=json_path,
                output_log_path=output_dir / 'sim_data' / f'{settings.project_name}_{rep}.csv',
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
        evaluation_arguments = [(settings, self._log_validation, log) for log in p.get()]
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

    def _simulate(self, trial_stg: Configuration):
        return simulate(trial_stg, self._log_validation, evaluate_fn=self._evaluate_logs)

    def _define_response(self, settings: PipelineSettings, status: str, evaluation_measurements: list) -> dict:
        data = {
            'rp_similarity': settings.rp_similarity,
            'gate_management': settings.gateway_probabilities,
            'output': str(settings.output_dir.absolute())
        }

        response = {
            'output': str(settings.output_dir.absolute()),
        }

        measurements = []

        if status == STATUS_OK:
            similarity = np.mean([x['sim_val'] for x in evaluation_measurements])
            loss = (1 - similarity)  # TODO: should it be just 'similarity'?

            response['loss'] = loss
            response['status'] = status if loss > 0 else STATUS_FAIL

            for sim_val in evaluation_measurements:
                values = {
                    'similarity': sim_val['sim_val'],
                    'sim_metric': sim_val['metric'],
                    'status': response['status'],
                }
                values = values | data
                measurements.append(values)
        else:
            response['status'] = status
            values = {
                'similarity': 0,
                'sim_metric': Metric.DAY_HOUR_EMD,
                'status': response['status'], **data
            }
            values = values | data
            measurements.append(values)

        if os.path.getsize(self._measurements_file_name) > 0:
            sup.create_csv_file(measurements, self._measurements_file_name, mode='a')
        else:
            sup.create_csv_file_header(measurements, self._measurements_file_name)

        return response

    @staticmethod
    def _evaluate_logs(args) -> Optional[list]:
        settings: PipelineSettings = args[0]
        data: pd.DataFrame = args[1]
        sim_log = args[2]

        if sim_log is None:
            return None

        rep = sim_log.iloc[0].run_num
        sim_values = []
        evaluator = sim.SimilarityEvaluator(data, sim_log, max_cases=1000)
        evaluator.measure_distance(Metric.DAY_HOUR_EMD)
        sim_values.append({**{'run_num': rep}, **evaluator.similarity})

        return sim_values

    def _split_timeline(self, size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train, validation = self._log.split_timeline(size)
        key = self._log_ids.start_time
        validation = validation.sort_values(key, ascending=True).reset_index(drop=True)
        train = train.sort_values(key, ascending=True).reset_index(drop=True)
        return train, validation
