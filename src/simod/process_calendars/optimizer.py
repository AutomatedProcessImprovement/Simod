import copy
import multiprocessing
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from hyperopt import Trials, hp, fmin, STATUS_OK, STATUS_FAIL
from hyperopt import tpe
from tqdm import tqdm

from simod.analyzers import sim_evaluator as sim
from simod.bpm.reader_writer import BPMNReaderWriter
from simod.cli_formatter import print_subsection, print_message
from simod.configuration import Configuration, Metric
from simod.event_log.column_mapping import EventLogIDs, SIMOD_DEFAULT_COLUMNS
from simod.event_log.reader_writer import LogReaderWriter
from simod.hyperopt_pipeline import HyperoptPipeline
from simod.process_calendars.settings import CalendarOptimizationSettings, PipelineSettings
from simod.simulation.parameters.miner import mine_simulation_parameters_default_24_7
from simod.simulation.prosimos import PROSIMOS_COLUMN_MAPPING, ProsimosSettings, simulate_with_prosimos
from simod.utilities import remove_asset, progress_bar_async, folder_id, file_id


# Calendar optimization with:
# - predefined calendars:
#   - 24/7
#   - 9-5
# - discovered
#   - undifferentiated, one calendar for all resources
#   - pools
#   - differentiated

# Prosimos accepts the following parameters:
# - confidence
# - support
# - granularity
# - resource participation

# Does Prosimos discover arrival calendars?


class CalendarOptimizer(HyperoptPipeline):
    best_output: Optional[Path]
    best_parameters: PipelineSettings
    evaluation_measurements: pd.DataFrame
    _output_dir: Path
    _model_path: Path

    _settings_global: Configuration
    _settings_time: Configuration

    _log: LogReaderWriter
    _log_ids: EventLogIDs
    _log_train: LogReaderWriter
    _log_validation: pd.DataFrame
    _original_log: LogReaderWriter
    _original_log_train: LogReaderWriter
    _original_log_validation: pd.DataFrame

    _calendar_optimizer_settings: CalendarOptimizationSettings

    _bayes_trials: Trials = Trials()

    def __init__(
            self,
            calendar_optimizer_settings: CalendarOptimizationSettings,
            log: LogReaderWriter,
            model_path: Path,
            log_ids: Optional[EventLogIDs] = None):
        self._model_path = model_path

        self._calendar_optimizer_settings = calendar_optimizer_settings
        self._log = log
        self._log_ids = log_ids if log_ids is not None else SIMOD_DEFAULT_COLUMNS

        # hyperopt search space
        self._space = self._define_search_space(calendar_optimizer_settings)

        # setting train and validation log data
        train, validation = self._split_timeline(0.8)
        self._log_train = LogReaderWriter.copy_without_data(self._log)
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
        self._output_dir = self._calendar_optimizer_settings.base_dir / folder_id(prefix='calendars_')
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self.evaluation_measurements = pd.DataFrame(
            columns=['similarity', 'metric', 'rp_similarity', 'gateway_probabilities', 'status', 'output_dir'])

    def run(self) -> PipelineSettings:
        def pipeline(trial_stg: Union[dict, PipelineSettings]):
            print_subsection('Trial')
            print_message(f'train split: {len(pd.DataFrame(self._log_train.data).caseid.unique())}, '
                          f'validation split: {len(pd.DataFrame(self._log_validation).caseid.unique())}')

            # casting a dictionary provided by hyperopt to PipelineSettings for convenience
            if isinstance(trial_stg, dict):
                trial_stg = PipelineSettings.from_dict(
                    trial_stg,
                    # TODO: output_dir is rewritten below anyway, but it's needed for the constructor
                    output_dir=self._output_dir,
                    model_path=self._model_path,
                )

            # initializing status
            status = STATUS_OK

            # creating and defining folders and paths
            output_dir = self._output_dir / folder_id(prefix='calendars_trial_')
            output_dir.mkdir(parents=True, exist_ok=True)
            trial_stg.output_dir = output_dir

            status, result = self.step(status, self._extract_parameters_undifferentiated, trial_stg)
            bpmn_path, json_path, simulation_cases = result

            # TODO: in simulation, the old parameters aren't used: rp_similarity, res_cal_met, arr_cal_met -- how can I integrate them?
            # TODO: redefine pipeline settings for calendars optimization, Prosimos uses confidence and support
            status, result = self.step(status, self._simulate_undifferentiated, trial_stg, json_path, simulation_cases)
            evaluation_measurements = result if status == STATUS_OK else []

            response, status = self._define_response(trial_stg, status, evaluation_measurements)

            self._process_measurements(trial_stg, status, evaluation_measurements)

            self._reset_log_buckets()

            return response

        # Optimization
        best = fmin(fn=pipeline,
                    space=self._space,
                    algo=tpe.suggest,
                    max_evals=self._calendar_optimizer_settings.max_evaluations,
                    trials=self._bayes_trials,
                    show_progressbar=False)

        # Best results

        results = pd.DataFrame(self._bayes_trials.results).sort_values('loss')
        results_ok = results[results.status == STATUS_OK]
        self.best_output = results_ok.iloc[0].output_dir

        best_settings = PipelineSettings.from_hyperopt_response(
            data=best,
            initial_settings=self._calendar_optimizer_settings,
            output_dir=Path(self.best_output),
            model_path=self._model_path)
        self.best_parameters = best_settings

        # Save evaluation measurements
        self.evaluation_measurements.sort_values('similarity', ascending=False, inplace=True)
        self.evaluation_measurements.to_csv(self._output_dir / file_id(prefix='evaluation_'), index=False)

        return best_settings

    def cleanup(self):
        remove_asset(self._output_dir)

    @staticmethod
    def _define_search_space(optimizer_settings: CalendarOptimizationSettings):
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

        space = rp_similarity | resource_calendars | arrival_calendar | gateway_probabilities

        return space

    def _process_measurements(self, settings: PipelineSettings, status: str, evaluation_measurements: list):
        data = {
            'rp_similarity': settings.rp_similarity,
            'gateway_probabilities': settings.gateway_probabilities,
            'output_dir': settings.output_dir,
            'status': status,
        }  # TODO: arr_support, arr_confidence, res_support, res_confidence and other metrics can be exposed too

        if status == STATUS_OK:
            for sim_val in evaluation_measurements:
                values = {
                    'similarity': sim_val['sim_val'],
                    'metric': sim_val['metric'],
                }
                values = values | data
                self.evaluation_measurements = pd.concat([self.evaluation_measurements, pd.DataFrame([values])])
        else:
            values = {
                'similarity': 0,
                'metric': Metric.DAY_HOUR_EMD,
            }
            values = values | data
            self.evaluation_measurements = pd.concat([self.evaluation_measurements, pd.DataFrame([values])])

    @staticmethod
    def _define_response(
            settings: PipelineSettings,
            status: str,
            evaluation_measurements: list) -> Tuple[dict, str]:
        response = {
            'output_dir': settings.output_dir,
            'status': status,
            'loss': None,
        }

        if status == STATUS_OK:
            similarity = np.mean([x['sim_val'] for x in evaluation_measurements])
            loss = 1 - similarity  # TODO: should it be just 'similarity'?
            response['loss'] = loss

            status = status if loss > 0 else STATUS_FAIL
            response['status'] = status if loss > 0 else STATUS_FAIL

        return response, status

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
        bpmn_path = self._model_path
        bpmn_reader = BPMNReaderWriter(bpmn_path)
        process_graph = bpmn_reader.as_graph()

        log = self._log_train.get_traces_df(include_start_end_events=True)
        pdf_method = self._calendar_optimizer_settings.pdef_method

        simulation_parameters = mine_simulation_parameters_default_24_7(
            log, self._log_ids, bpmn_path, process_graph, pdf_method, bpmn_reader,
            settings.gateway_probabilities)

        json_path = settings.output_dir / 'simulation_parameters.json'
        simulation_parameters.to_json_file(json_path)

        simulation_cases = log[self._log_ids.case].nunique()

        return bpmn_path, json_path, simulation_cases

    @staticmethod
    def _read_simulated_log(arguments: Tuple):
        log_path, log_column_mapping, simulation_repetition_index = arguments

        reader = LogReaderWriter(log_path=log_path, column_names=log_column_mapping)

        reader.df.rename(columns={'user': 'resource'}, inplace=True)
        reader.df['role'] = reader.df['resource']
        reader.df['source'] = 'simulation'
        reader.df['run_num'] = simulation_repetition_index
        reader.df = reader.df[~reader.df.task.isin(['Start', 'End'])]

        return reader.df

    def _simulate_undifferentiated(self, settings: PipelineSettings, json_path: Path, simulation_cases: int):
        num_simulations = self._calendar_optimizer_settings.simulation_repetitions
        bpmn_path = self._model_path
        cpu_count = multiprocessing.cpu_count()
        w_count = num_simulations if num_simulations <= cpu_count else cpu_count
        pool = multiprocessing.Pool(processes=w_count)

        # Simulate
        simulation_arguments = [
            ProsimosSettings(
                bpmn_path=bpmn_path,
                parameters_path=json_path,
                output_log_path=settings.output_dir / f'simulation_log_{rep}.csv',
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
        evaluation_arguments = [(self._log_validation, log) for log in p.get()]
        if simulation_cases > 1000:
            pool.close()
            evaluation_measurements = [self._evaluate_logs(args)
                                       for args in tqdm(evaluation_arguments, 'evaluating results')]
        else:
            p = pool.map_async(self._evaluate_logs, evaluation_arguments)
            progress_bar_async(p, 'evaluating results', num_simulations)
            pool.close()
            evaluation_measurements = p.get()

        return evaluation_measurements

    @staticmethod
    def _evaluate_logs(args) -> dict:
        validation_log, log = args

        evaluator = sim.SimilarityEvaluator(validation_log, log, max_cases=1000)
        evaluator.measure_distance(Metric.DAY_HOUR_EMD)

        result = {
            'run_num': log.iloc[0].run_num,
            **evaluator.similarity
        }

        return result

    def _split_timeline(self, size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train, validation = self._log.split_timeline(size)
        key = self._log_ids.start_time
        validation = validation.sort_values(key, ascending=True).reset_index(drop=True)
        train = train.sort_values(key, ascending=True).reset_index(drop=True)
        return train, validation

    def _reset_log_buckets(self):
        self._log = self._original_log  # TODO: no need
        self._log_train = copy.deepcopy(self._original_log_train)
        self._log_validation = copy.deepcopy(self._original_log_validation)
