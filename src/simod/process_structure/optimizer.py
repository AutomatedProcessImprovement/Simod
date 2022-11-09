import copy
import multiprocessing
from pathlib import Path
from typing import Union, Tuple

import numpy as np
import pandas as pd
from hyperopt import Trials, hp, fmin, STATUS_OK, STATUS_FAIL
from hyperopt import tpe
from tqdm import tqdm

from simod.analyzers.sim_evaluator import SimilarityEvaluator
from simod.cli_formatter import print_message, print_subsection
from simod.event_log.reader_writer import LogReaderWriter
from simod.hyperopt_pipeline import HyperoptPipeline
from simod.simulation.parameters.miner import mine_default_24_7
from simod.utilities import remove_asset, progress_bar_async, file_id, folder_id
from .miner import StructureMiner, Settings as StructureMinerSettings
from .settings import StructureOptimizationSettings, PipelineSettings
from ..bpm.reader_writer import BPMNReaderWriter
from ..configuration import StructureMiningAlgorithm, Metric
from ..event_log.column_mapping import EventLogIDs, PROSIMOS_COLUMNS
from ..event_log.utilities import sample_log
from ..simulation.prosimos import PROSIMOS_COLUMN_MAPPING, ProsimosSettings, simulate_with_prosimos


class StructureOptimizer(HyperoptPipeline):
    def __init__(
            self,
            settings: StructureOptimizationSettings,
            log: LogReaderWriter,
            log_ids: EventLogIDs,
    ):
        assert log_ids is not None, 'Event log IDs must be provided'

        self._log_ids = log_ids

        self._settings = settings
        self._log_reader = log

        train, validation = self._log_reader.split_timeline(0.8)
        train = sample_log(train, log_ids)

        self._log_validation = validation.sort_values(log_ids.start_time, ascending=True).reset_index(drop=True)
        self._log_train = LogReaderWriter.copy_without_data(self._log_reader, self._log_ids)
        self._log_train.set_data(
            train.sort_values(log_ids.start_time, ascending=True).reset_index(drop=True).to_dict('records'))

        # TODO: ensure we need to copy all the logs, it's an expensive operation
        self._original_log = copy.deepcopy(log)
        self._original_log_train = copy.deepcopy(self._log_train)
        self._original_log_validation = copy.deepcopy(self._log_validation)

        self._output_dir = self._settings.base_dir / folder_id(prefix='structure_')
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self.evaluation_measurements = pd.DataFrame(
            columns=['similarity', 'metric', 'status', 'gateway_probabilities', 'epsilon', 'eta', 'and_prior', 'or_rep',
                     'output_dir'])

        self._bayes_trials = Trials()

    def run(self) -> PipelineSettings:
        self._log_train = copy.deepcopy(self._original_log_train)

        def pipeline(trial_stage_settings: Union[PipelineSettings, dict]):
            print_subsection("Trial")
            print_message(f'train split: {len(pd.DataFrame(self._log_train.data))}, '
                          f'validation split: {len(self._log_validation)}')

            # casting a dictionary provided by hyperopt to PipelineSettings for convenience
            if isinstance(trial_stage_settings, dict):
                trial_stage_settings = PipelineSettings(
                    model_path=None,
                    output_dir=None,
                    project_name=self._settings.project_name,
                    **trial_stage_settings)
            print_message(f'Parameters: {trial_stage_settings}', capitalize=False)

            # initializing status
            status = STATUS_OK

            # current trial folder
            output_dir = self._output_dir / folder_id(prefix='structure_trial_')
            output_dir.mkdir(parents=True, exist_ok=True)

            # saving customized event-log for the external tools
            log_path = output_dir / (trial_stage_settings.project_name + '.xes')
            self._log_train.write_xes(log_path)

            # redefining pipeline settings
            trial_stage_settings.output_dir = output_dir
            trial_stage_settings.model_path = output_dir / (trial_stage_settings.project_name + '.bpmn')

            # structure mining
            try:
                status, result = self.step(
                    status,
                    self._mine_structure,
                    trial_stage_settings,
                    log_path,
                    self._settings.mining_algorithm)

                bpmn_reader, process_graph = result
            except Exception as e:
                print_message(f'Mining failed: {e}')
                status = STATUS_FAIL
                bpmn_reader, process_graph = None, None

            # simulation parameters mining
            status, result = self.step(
                status,
                self._extract_parameters_undifferentiated,
                trial_stage_settings,
                bpmn_reader,
                process_graph)
            if status == STATUS_FAIL:
                json_path, simulation_cases = None, None
            else:
                json_path, simulation_cases = result

            # simulation
            status, result = self.step(status, self._simulate_undifferentiated,
                                       trial_stage_settings,
                                       self._settings.simulation_repetitions,
                                       json_path,
                                       simulation_cases)
            evaluation_measurements = result if status == STATUS_OK else []

            # loss
            response, status = self._define_response(status, evaluation_measurements, trial_stage_settings)
            print(f'StructureOptimizer pipeline response: {response}')

            # saving results
            self._process_measurements(trial_stage_settings, status, evaluation_measurements)

            self._reset_log_buckets()

            return response

        space = self._define_search_space(self._settings)

        # Optimization
        best = fmin(fn=pipeline,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=self._settings.max_evaluations,
                    trials=self._bayes_trials,
                    show_progressbar=False)

        # Best results

        results = pd.DataFrame(self._bayes_trials.results).sort_values('loss')
        results_ok = results[results.status == STATUS_OK]
        self.best_output = results_ok.iloc[0].output_dir

        best_model_path = results_ok.iloc[0].model_path
        assert best_model_path.exists(), f'Best model path {best_model_path} does not exist'
        best_settings = PipelineSettings.from_hyperopt_dict(
            data=best,
            initial_settings=self._settings,
            model_path=best_model_path,
            project_name=self._settings.project_name,
        )
        self.best_parameters = best_settings

        # Save evaluation measurements
        self.evaluation_measurements.sort_values('similarity', ascending=False, inplace=True)
        self.evaluation_measurements.to_csv(self._output_dir / file_id(prefix='evaluation_'), index=False)

        return best_settings

    def cleanup(self):
        remove_asset(self._output_dir)

    @staticmethod
    def _define_search_space(settings: StructureOptimizationSettings) -> dict:
        space = {
            'gateway_probabilities': hp.choice('gateway_probabilities', settings.gateway_probabilities),
        }

        if settings.mining_algorithm in [StructureMiningAlgorithm.SPLIT_MINER_1,
                                         StructureMiningAlgorithm.SPLIT_MINER_3]:
            space |= {
                'epsilon': hp.uniform('epsilon', *settings.epsilon),
                'eta': hp.uniform('eta', *settings.eta),
            }

            if settings.mining_algorithm is StructureMiningAlgorithm.SPLIT_MINER_3:
                space |= {
                    'and_prior': hp.choice('and_prior', list(map(lambda v: str(v).lower(), settings.and_prior))),
                    'or_rep': hp.choice('or_rep', list(map(lambda v: str(v).lower(), settings.or_rep))),
                }
        elif settings.mining_algorithm is StructureMiningAlgorithm.SPLIT_MINER_2:
            space |= {
                'concurrency': hp.uniform('concurrency', *settings.concurrency)
            }

        return space

    @staticmethod
    def _define_response(
            status: str,
            evaluation_measurements: list,
            pipeline_settings: PipelineSettings) -> Tuple[dict, str]:
        similarity = np.mean([x['similarity'] for x in evaluation_measurements])
        loss = 1 - similarity
        status = status if loss > 0 else STATUS_FAIL

        response = {
            'loss': loss,
            'status': status,
            'output_dir': pipeline_settings.output_dir,
            'model_path': pipeline_settings.model_path,
        }

        return response, status

    def _process_measurements(
            self,
            settings: PipelineSettings,
            status,
            evaluation_measurements):
        optimization_parameters = settings.optimization_parameters_as_dict(self._settings.mining_algorithm)
        optimization_parameters['status'] = status

        if status == STATUS_OK:
            for sim_val in evaluation_measurements:
                values = {
                    'similarity': sim_val['similarity'],
                    'metric': sim_val['metric'],
                }
                values = values | optimization_parameters
                self.evaluation_measurements = pd.concat([self.evaluation_measurements, pd.DataFrame([values])])
        else:
            values = {
                'similarity': 0,
                'metric': Metric.DL,
            }
            values = values | optimization_parameters
            self.evaluation_measurements = pd.concat([self.evaluation_measurements, pd.DataFrame([values])])

    @staticmethod
    def _mine_structure(
            settings: PipelineSettings,
            log_path: Path,
            mining_algorithm: StructureMiningAlgorithm) -> Tuple:

        miner_settings = StructureMinerSettings(
            mining_algorithm=mining_algorithm,
            epsilon=settings.epsilon,
            eta=settings.eta,
            concurrency=settings.concurrency,
            and_prior=settings.and_prior,
            or_rep=settings.or_rep,
        )

        _ = StructureMiner(miner_settings, xes_path=log_path, output_model_path=settings.model_path)

        bpmn_reader = BPMNReaderWriter(settings.model_path)
        process_graph = bpmn_reader.as_graph()

        return bpmn_reader, process_graph

    def _extract_parameters_undifferentiated(self, settings: PipelineSettings, bpmn_reader, process_graph) -> Tuple:
        log = self._log_train.get_traces_df(include_start_end_events=True)
        pdf_method = self._settings.pdef_method

        simulation_parameters = mine_default_24_7(
            log,
            self._log_ids,
            settings.model_path,
            process_graph,
            pdf_method,
            bpmn_reader,
            settings.gateway_probabilities)

        json_path = settings.model_path.with_suffix('.json')
        simulation_parameters.to_json_file(json_path)

        simulation_cases = log[self._log_ids.case].nunique()

        return json_path, simulation_cases

    def _simulate_undifferentiated(
            self,
            settings: PipelineSettings,
            simulation_repetitions: int,
            json_path: Path,
            simulation_cases: int):
        self._log_validation['source'] = 'log'
        self._log_validation['run_num'] = 0
        self._log_validation['role'] = 'SYSTEM'
        self._log_validation = self._log_validation[
            ~self._log_validation[self._log_ids.activity].isin(['Start', 'start', 'End', 'end'])]

        num_simulations = simulation_repetitions
        cpu_count = multiprocessing.cpu_count()
        w_count = num_simulations if num_simulations <= cpu_count else cpu_count
        pool = multiprocessing.Pool(processes=w_count)

        # Simulate
        simulation_arguments = [
            ProsimosSettings(
                bpmn_path=settings.model_path,
                parameters_path=json_path,
                output_log_path=settings.output_dir / f'simulated_log_{rep}.csv',
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

    def _evaluate_logs(self, arguments) -> dict:
        data: pd.DataFrame
        sim_log: pd.DataFrame
        data, sim_log = arguments

        evaluator = SimilarityEvaluator(data, self._log_ids, sim_log, PROSIMOS_COLUMNS, max_cases=1000)
        evaluator.measure_distance(Metric.DL)

        result = {
            'run_num': sim_log.iloc[0].run_num,
            **evaluator.similarity
        }

        return result

    def _read_simulated_log(self, arguments: Tuple):
        log_path, log_column_mapping, simulation_repetition_index = arguments

        reader = LogReaderWriter(log_path=log_path, log_ids=PROSIMOS_COLUMNS, column_names=log_column_mapping)

        reader.df['role'] = reader.df['resource']
        reader.df['source'] = 'simulation'
        reader.df['run_num'] = simulation_repetition_index
        reader.df = reader.df[~reader.df[PROSIMOS_COLUMNS.activity].isin(['Start', 'start', 'End', 'end'])]

        return reader.df

    def _reset_log_buckets(self):
        self._log_reader = copy.deepcopy(self._original_log)
        self._log_train = copy.deepcopy(self._original_log_train)
        self._log_validation = copy.deepcopy(self._original_log_validation)
