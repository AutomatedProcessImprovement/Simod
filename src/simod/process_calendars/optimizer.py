import copy
import multiprocessing
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from hyperopt import Trials, hp, fmin, STATUS_OK, STATUS_FAIL
from hyperopt import tpe
from lxml import etree
from tqdm import tqdm

from extraneous_activity_delays.bpmn_enhancer import set_number_instances_to_simulate, set_start_datetime_to_simulate
from extraneous_activity_delays.config import Configuration as ExtraneousActivityDelaysConfiguration
from extraneous_activity_delays.enhance_with_delays import HyperOptEnhancer
from simod.cli_formatter import print_subsection, print_message
from simod.configuration import GatewayProbabilitiesDiscoveryMethod
from simod.event_log.column_mapping import EventLogIDs, PROSIMOS_COLUMNS
from simod.event_log.reader_writer import LogReaderWriter
from simod.hyperopt_pipeline import HyperoptPipeline
from simod.metrics.metrics import compute_metric
from simod.process_calendars.settings import CalendarOptimizationSettings, PipelineSettings
from simod.process_structure.miner import Settings as StructureMinerSettings, StructureMiner
from simod.simulation.parameters.miner import mine_parameters
from simod.simulation.prosimos import PROSIMOS_COLUMN_MAPPING, ProsimosSettings, simulate_with_prosimos
from simod.utilities import remove_asset, progress_bar_async, folder_id, file_id


class CalendarOptimizer(HyperoptPipeline):
    def __init__(
            self,
            calendar_optimizer_settings: CalendarOptimizationSettings,
            log: LogReaderWriter,
            structure_settings: Optional[StructureMinerSettings] = None,
            model_path: Optional[Path] = None,
            log_ids: Optional[EventLogIDs] = None):

        self._calendar_optimizer_settings = calendar_optimizer_settings
        self._log = log
        self._log_ids = log_ids

        # setting train and validation log data
        train, validation = self._split_timeline(0.8)
        self._log_train = LogReaderWriter.copy_without_data(self._log, self._log_ids)
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

        if structure_settings is not None:
            self._gateway_probabilities_method = structure_settings.gateway_probabilities_method
        else:
            self._gateway_probabilities_method = GatewayProbabilitiesDiscoveryMethod.DISCOVERY

        # creating files and folders
        self._output_dir = self._calendar_optimizer_settings.base_dir / folder_id(prefix='calendars_')
        self._output_dir.mkdir(parents=True, exist_ok=True)

        if model_path is not None:
            self._train_model_path = model_path
        else:
            self._train_model_path = self._mine_structure(structure_settings, self._output_dir)

        self.evaluation_measurements = pd.DataFrame(
            columns=['similarity', 'metric', 'gateway_probabilities', 'status', 'output_dir'])

        self._bayes_trials = Trials()

    def _mine_structure(self, settings: StructureMinerSettings, output_dir: Path) -> Path:
        print_message(f'Mining structure with settings {settings.to_dict()}')

        # Saving the full pre-processed log to disk
        log_path = output_dir / 'train_log.xes'
        self._log_train.write_xes(log_path)

        model_path = output_dir / 'train.bpmn'

        StructureMiner(settings, log_path, model_path)

        return model_path

    def run(self) -> PipelineSettings:
        def pipeline(trial_stg: Union[dict, PipelineSettings]):
            print_subsection('Trial')
            print_message(f'train split: {len(pd.DataFrame(self._log_train.data))}, '
                          f'validation split: {len(self._log_validation)}')

            # casting a dictionary provided by hyperopt to PipelineSettings for convenience
            if isinstance(trial_stg, dict):
                trial_stg = PipelineSettings.from_hyperopt_option_dict(
                    trial_stg,
                    output_dir=self._output_dir,
                    model_path=self._train_model_path,
                    gateway_probabilities_method=self._gateway_probabilities_method
                )

            # initializing status
            status = STATUS_OK

            # creating and defining folders and paths
            output_dir = self._output_dir / folder_id(prefix='calendars_trial_')
            output_dir.mkdir(parents=True, exist_ok=True)
            trial_stg.output_dir = output_dir
            assert trial_stg.output_dir.exists(), 'Output directory does not exist'

            # simulation parameters extraction
            status, result = self.step(status, self._extract_parameters, trial_stg)
            if result is None:
                status = STATUS_FAIL
                json_path, simulation_cases = None, None
            else:
                json_path, simulation_cases = result

            # simulation and evaluation
            status, result = self.step(status, self._simulate_with_prosimos, trial_stg, json_path, simulation_cases)
            evaluation_measurements = result if status == STATUS_OK else []

            # response for hyperopt
            response, status = self._define_response(trial_stg, status, evaluation_measurements)

            # recording measurements internally
            self._process_measurements(trial_stg, status, evaluation_measurements)

            self._reset_log_buckets()

            return response

        # Adding timers to the model
        # log = self._log_train.get_traces_df(include_start_end_events=True)
        # modified_model_path = CalendarOptimizer.add_extraneous_delay_timers(log, self._log_ids, self._train_model_path)
        # self._train_model_path = modified_model_path

        # Optimization
        space = self._define_search_space(self._calendar_optimizer_settings)
        best = fmin(fn=pipeline,
                    space=space,
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
            model_path=self._train_model_path,
            gateway_probabilities_method=self._gateway_probabilities_method)
        self.best_parameters = best_settings

        # Save evaluation measurements
        assert len(self.evaluation_measurements) > 0, 'No evaluation measurements were collected'
        self.evaluation_measurements.sort_values('similarity', ascending=False, inplace=True)
        self.evaluation_measurements.to_csv(self._output_dir / file_id(prefix='evaluation_'), index=False)

        return best_settings

    def cleanup(self):
        remove_asset(self._output_dir)

    @staticmethod
    def _define_search_space(optimizer_settings: CalendarOptimizationSettings):
        resource_calendars = {
            'resource_profiles': hp.choice(
                'resource_profiles',
                # NOTE: 'prefix' is used later in PipelineSettings.from_hyperopt_response
                optimizer_settings.resource_profiles.to_hyperopt_options(prefix='resource_profile')
            )
        }

        arrival_calendar = {
            'case_arrival': hp.choice(
                'case_arrival',
                # NOTE: 'prefix' is used later in PipelineSettings.from_hyperopt_response
                optimizer_settings.case_arrival.to_hyperopt_options(prefix='case_arrival')
            )}

        space = resource_calendars | arrival_calendar

        return space

    def _process_measurements(self, settings: PipelineSettings, status: str, evaluation_measurements: list):
        data = {
            'gateway_probabilities': settings.gateway_probabilities_method,
            'case_arrival_granularity': settings.case_arrival.granularity,
            'case_arrival_confidence': settings.case_arrival.confidence,
            'case_arrival_participation': settings.case_arrival.participation,
            'case_arrival_support': settings.case_arrival.support,
            'case_arrival_discovery_type': settings.case_arrival.discovery_type.name,
            'resource_profile_granularity': settings.resource_profiles.granularity,
            'resource_profile_confidence': settings.resource_profiles.confidence,
            'resource_profile_participation': settings.resource_profiles.participation,
            'resource_profile_support': settings.resource_profiles.support,
            'resource_profile_discovery_type': settings.resource_profiles.discovery_type.name,
            'output_dir': settings.output_dir,
            'status': status,
        }

        if status == STATUS_OK:
            for sim_val in evaluation_measurements:
                values = {
                    'similarity': sim_val['similarity'],
                    'metric': sim_val['metric'],
                }
                values = values | data
                self.evaluation_measurements = pd.concat([self.evaluation_measurements, pd.DataFrame([values])])
        else:
            values = {
                'similarity': 0,
                'metric': self._calendar_optimizer_settings.optimization_metric,
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
            distance = np.mean([x['similarity'] for x in evaluation_measurements])
            loss = distance
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

    def _extract_parameters(self, settings: PipelineSettings) -> Tuple:
        log = self._log_train.get_traces_df(include_start_end_events=True)

        parameters = mine_parameters(
            settings.case_arrival, settings.resource_profiles, log, self._log_ids, settings.model_path,
            settings.gateway_probabilities_method)

        json_path = settings.output_dir / 'simulation_parameters.json'

        parameters.to_json_file(json_path)

        simulation_cases = log[self._log_ids.case].nunique()

        return json_path, simulation_cases

    def _read_simulated_log(self, arguments: Tuple):
        log_path, log_column_mapping, simulation_repetition_index = arguments
        assert log_path.exists(), f'Simulated log file {log_path} does not exist'

        reader = LogReaderWriter(log_path=log_path, log_ids=PROSIMOS_COLUMNS)

        reader.df['role'] = reader.df['resource']
        reader.df['source'] = 'simulation'
        reader.df['run_num'] = simulation_repetition_index
        reader.df = reader.df[~reader.df[PROSIMOS_COLUMNS.activity].isin(['Start', 'End', 'start', 'end'])]

        return reader.df

    def _simulate_with_prosimos(self, settings: PipelineSettings, json_path: Path, simulation_cases: int):
        num_simulations = self._calendar_optimizer_settings.simulation_repetitions
        bpmn_path = settings.model_path
        cpu_count = multiprocessing.cpu_count()
        w_count = num_simulations if num_simulations <= cpu_count else cpu_count
        pool = multiprocessing.Pool(processes=w_count)

        # Simulate
        simulation_arguments = [
            ProsimosSettings(
                bpmn_path=bpmn_path,
                parameters_path=json_path,
                output_log_path=settings.output_dir / f'simulation_log_{rep}.csv',
                num_simulation_cases=simulation_cases,
                simulation_start=self._log_validation[self._log_ids.start_time].min(),
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

    def _evaluate_logs(self, args) -> dict:
        validation_log, log = args
        metric = self._calendar_optimizer_settings.optimization_metric

        value = compute_metric(metric, validation_log, self._log_ids, log, PROSIMOS_COLUMNS)

        return {
            'run_num': log.iloc[0].run_num,
            'metric': metric,
            'similarity': value,
        }

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

    @staticmethod
    def add_extraneous_delay_timers(event_log: pd.DataFrame, log_ids: EventLogIDs, model_path: Path) -> Path:
        """
        Adds extraneous delay timers to the BPMN model.
        See https://github.com/AutomatedProcessImprovement/extraneous-activity-delays.

        :param event_log: The event log.
        :param log_ids: The event log IDs.
        :param model_path: The BPMN model path.
        :return: The path to the BPMN model with extraneous delay timers.
        """

        # log_path = entry_point / test_data['log_name']
        # model_path = entry_point / test_data['model_name']

        # log_ids = STANDARD_COLUMNS
        configuration = ExtraneousActivityDelaysConfiguration(
            log_ids=log_ids,
            num_evaluations=1,
            num_evaluation_simulations=1,
        )

        # event_log = pd.read_csv(log_path)
        # event_log[log_ids.start_time] = pd.to_datetime(event_log[log_ids.start_time], utc=True)
        # event_log[log_ids.end_time] = pd.to_datetime(event_log[log_ids.end_time], utc=True)
        # event_log[log_ids.resource].fillna("NOT_SET", inplace=True)
        # event_log[log_ids.resource] = event_log[log_ids.resource].astype("string")

        parser = etree.XMLParser(remove_blank_text=True)
        bpmn_model = etree.parse(model_path, parser)
        set_number_instances_to_simulate(bpmn_model, len(event_log[configuration.log_ids.case].unique()))
        set_start_datetime_to_simulate(bpmn_model, min(event_log[configuration.log_ids.start_time]))

        enhancer = HyperOptEnhancer(event_log, bpmn_model, configuration)
        enhanced_bpmn_model = enhancer.enhance_bpmn_model_with_delays()

        output_path = model_path.with_stem(model_path.stem + '_timers')
        enhanced_bpmn_model.write(output_path, pretty_print=True)

        return output_path
