import copy
import shutil
from pathlib import Path
from typing import Union, Tuple

import numpy as np
import pandas as pd
from hyperopt import Trials, hp, fmin, STATUS_OK, STATUS_FAIL
from hyperopt import tpe

from simod.cli_formatter import print_message, print_subsection, print_step
from simod.event_log.reader_writer import LogReaderWriter
from simod.hyperopt_pipeline import HyperoptPipeline
from simod.simulation.parameters.miner import mine_default_24_7
from simod.utilities import remove_asset, file_id, folder_id
from .miner import StructureMiner, Settings as StructureMinerSettings
from .settings import StructureOptimizationSettings, PipelineSettings
from ..bpm.reader_writer import BPMNReaderWriter
from ..configuration import StructureMiningAlgorithm, Metric
from ..event_log.column_mapping import EventLogIDs
from ..event_log.utilities import sample_log
from ..simulation.prosimos import simulate_and_evaluate


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
        train = sample_log(train, log_ids)  # TODO: remove this in future

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
            columns=['value', 'metric', 'status', 'gateway_probabilities', 'epsilon', 'eta', 'prioritize_parallelism',
                     'replace_or_joins', 'output_dir'])

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
                    model_path=self._settings.model_path,
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

            # structure mining
            trial_stage_settings.output_dir = output_dir
            model_path = output_dir / (trial_stage_settings.project_name + '.bpmn')
            bpmn_reader, process_graph = None, None
            try:
                if trial_stage_settings.model_path is None:
                    trial_stage_settings.model_path = model_path
                    print_step('Executing SplitMiner')
                    status, result = self.step(status, self._mine_structure, trial_stage_settings, log_path,
                                               self._settings.mining_algorithm)
                else:
                    if self._settings.model_path is not None:
                        shutil.copy(self._settings.model_path, model_path)
                    else:
                        raise ValueError('Model path is not provided')

                    print_step('Model is provided, skipping SplitMiner execution')

                bpmn_reader = BPMNReaderWriter(trial_stage_settings.model_path)
                process_graph = bpmn_reader.as_graph()
            except Exception as e:
                print_message(f'Mining failed: {e}')
                status = STATUS_FAIL

            # simulation parameters mining
            status, result = self.step(
                status,
                self._extract_parameters_undifferentiated,
                trial_stage_settings,
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
        self.evaluation_measurements.sort_values('value', ascending=False, inplace=True)
        self.evaluation_measurements.to_csv(self._output_dir / file_id(prefix='evaluation_'), index=False)

        return best_settings

    def cleanup(self):
        remove_asset(self._output_dir)

    @staticmethod
    def _define_search_space(settings: StructureOptimizationSettings) -> dict:
        space = {
            'gateway_probabilities_method':
                hp.choice('gateway_probabilities_method',
                          settings.gateway_probabilities_method
                          if isinstance(settings.gateway_probabilities_method, list)
                          else [settings.gateway_probabilities_method]),
        }

        # When a BPMN model is not provided, we call SplitMiner and optimize the SplitMiner input parameters
        if settings.model_path is None:
            if settings.mining_algorithm in [StructureMiningAlgorithm.SPLIT_MINER_1,
                                             StructureMiningAlgorithm.SPLIT_MINER_3]:
                space |= {
                    'epsilon': hp.uniform('epsilon', *settings.epsilon),
                    'eta': hp.uniform('eta', *settings.eta),
                }

                if settings.mining_algorithm is StructureMiningAlgorithm.SPLIT_MINER_3:
                    space |= {
                        'prioritize_parallelism':
                            hp.choice('prioritize_parallelism',
                                      list(map(lambda v: str(v).lower(), settings.prioritize_parallelism))),
                        'replace_or_joins':
                            hp.choice('replace_or_joins',
                                      list(map(lambda v: str(v).lower(), settings.replace_or_joins))),
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
            pipeline_settings: PipelineSettings,
    ) -> Tuple[dict, str]:
        similarity = np.mean([x['value'] for x in evaluation_measurements])
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
            for measurement in evaluation_measurements:
                values = {
                    'value': measurement['value'],
                    'metric': measurement['metric'],
                }
                values = values | optimization_parameters
                self.evaluation_measurements = pd.concat([self.evaluation_measurements, pd.DataFrame([values])])
        else:
            values = {
                'value': 0,
                'metric': Metric.DL,
            }
            values = values | optimization_parameters
            self.evaluation_measurements = pd.concat([self.evaluation_measurements, pd.DataFrame([values])])

    @staticmethod
    def _mine_structure(
            settings: PipelineSettings,
            log_path: Path,
            mining_algorithm: StructureMiningAlgorithm) -> None:
        miner_settings = StructureMinerSettings(
            gateway_probabilities_method=settings.gateway_probabilities_method,
            mining_algorithm=mining_algorithm,
            epsilon=settings.epsilon,
            eta=settings.eta,
            concurrency=settings.concurrency,
            prioritize_parallelism=settings.prioritize_parallelism,
            replace_or_joins=settings.replace_or_joins,
        )

        _ = StructureMiner(miner_settings, xes_path=log_path, output_model_path=settings.model_path)

    def _extract_parameters_undifferentiated(self, settings: PipelineSettings, process_graph) -> Tuple:
        log = self._log_train.get_traces_df()

        simulation_parameters = mine_default_24_7(
            log,
            self._log_ids,
            settings.model_path,
            process_graph,
            settings.gateway_probabilities_method)

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

        return simulate_and_evaluate(
            model_path=settings.model_path,
            parameters_path=json_path,
            output_dir=settings.output_dir,
            simulation_cases=simulation_cases,
            simulation_start_time=self._log_validation[self._log_ids.start_time].min(),
            validation_log=self._log_validation,
            validation_log_ids=self._log_ids,
            metrics=[self._settings.optimization_metric],
            num_simulations=simulation_repetitions,
        )

    def _reset_log_buckets(self):
        self._log_reader = copy.deepcopy(self._original_log)
        self._log_train = copy.deepcopy(self._original_log_train)
        self._log_validation = copy.deepcopy(self._original_log_validation)
