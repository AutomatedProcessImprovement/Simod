import json
import shutil
from pathlib import Path
from typing import Union, Tuple, Optional

import numpy as np
import pandas as pd
from hyperopt import Trials, hp, fmin, STATUS_OK, STATUS_FAIL
from hyperopt import tpe
from networkx import DiGraph

from simod.cli_formatter import print_message, print_subsection, print_step
from simod.hyperopt_pipeline import HyperoptPipeline
from simod.simulation.parameters.miner import mine_default_24_7
from simod.utilities import remove_asset, file_id, folder_id
from .miner import StructureMiner
from .settings import StructureOptimizationSettings, PipelineSettings
from ..bpm.reader_writer import BPMNReaderWriter
from ..configuration import StructureMiningAlgorithm, Metric
from ..event_log.column_mapping import EventLogIDs
from ..event_log.event_log import EventLog
from ..simulation.prosimos import simulate_and_evaluate


class StructureOptimizer(HyperoptPipeline):
    _event_log: EventLog
    _log_train: pd.DataFrame
    _log_validation: pd.DataFrame
    _log_ids: EventLogIDs
    _train_log_path: Path
    _output_dir: Path
    _process_graph: Optional[DiGraph]

    evaluation_measurements: pd.DataFrame

    def __init__(
            self,
            settings: StructureOptimizationSettings,
            event_log: EventLog,
            process_graph: Optional[DiGraph] = None,
    ):
        self._event_log = event_log
        self._settings = settings
        self._process_graph = process_graph
        self._log_ids = event_log.log_ids

        self._log_train = event_log.train_partition.sort_values(by=event_log.log_ids.start_time)
        self._log_validation = event_log.validation_partition.sort_values(event_log.log_ids.start_time, ascending=True)

        self._output_dir = self._settings.base_dir / folder_id(prefix='structure_')
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._train_log_path = self._output_dir / (event_log.process_name + '.xes')

        self.evaluation_measurements = pd.DataFrame(
            columns=['value', 'metric', 'status', 'gateway_probabilities', 'epsilon', 'eta', 'prioritize_parallelism',
                     'replace_or_joins', 'output_dir'])

        self._bayes_trials = Trials()

        if self._settings.model_path is not None and self._process_graph is None:
            self._process_graph = BPMNReaderWriter(self._settings.model_path).as_graph()

    def _optimization_objective(self, trial_stage_settings: Union[PipelineSettings, dict]):
        print_subsection("Structure Optimization Trial")

        # casting a dictionary provided by hyperopt to PipelineSettings for convenience
        if isinstance(trial_stage_settings, dict):
            trial_stage_settings = PipelineSettings(
                model_path=self._settings.model_path,
                output_dir=None,
                project_name=self._event_log.process_name,
                **trial_stage_settings)
        print_message(f'Parameters: {trial_stage_settings}', capitalize=False)

        # initializing status
        status = STATUS_OK

        # current trial folder
        output_dir = self._output_dir / folder_id(prefix='structure_trial_')
        output_dir.mkdir(parents=True, exist_ok=True)

        # structure mining
        trial_stage_settings.output_dir = output_dir
        model_path = output_dir / (trial_stage_settings.project_name + '.bpmn')
        bpmn_reader, process_graph = None, None
        try:
            if trial_stage_settings.model_path is None:
                trial_stage_settings.model_path = model_path
                print_step('Executing SplitMiner')
                status, result = self.step(status, self._mine_structure,
                                           trial_stage_settings, self._train_log_path, self._settings.mining_algorithm)

                bpmn_reader = BPMNReaderWriter(trial_stage_settings.model_path)
                process_graph = bpmn_reader.as_graph()
            else:
                print_step('Model is provided, skipping SplitMiner execution')

                process_graph = self._process_graph

                if self._settings.model_path is not None:
                    # We copy the model mostly for debugging purposes, so we have the model always in the output folder
                    shutil.copy(self._settings.model_path, model_path)
                else:
                    raise ValueError('Model path is not provided')
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

        return response

    def run(self) -> Tuple[PipelineSettings, Path, list, Path]:
        """
        Runs the structure optimization pipeline.
        :return: Tuple of the best settings, the path to the best model and the list of evaluation measurements.
        """

        self._event_log.train_to_xes(self._train_log_path)

        space = self._define_search_space(self._settings)

        # Optimization
        best = fmin(fn=self._optimization_objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=self._settings.max_evaluations,
                    trials=self._bayes_trials,
                    show_progressbar=False)

        # Best results

        results = pd.DataFrame(self._bayes_trials.results).sort_values('loss')
        results_ok = results[results.status == STATUS_OK]

        best_model_path = results_ok.iloc[0].model_path
        assert best_model_path.exists(), f'Best model path {best_model_path} does not exist'
        best_settings = PipelineSettings.from_hyperopt_dict(
            data=best,
            initial_settings=self._settings,
            model_path=best_model_path,
            project_name=self._settings.project_name,
        )

        best_parameters_path = best_model_path.parent / 'simulation_parameters.json'
        best_gateway_probabilities = json.load(open(best_parameters_path, 'r'))['gateway_branching_probabilities']

        # Save evaluation measurements
        self.evaluation_measurements.sort_values('value', ascending=False, inplace=True)
        self.evaluation_measurements.to_csv(self._output_dir / file_id(prefix='evaluation_'), index=False)

        return best_settings, best_model_path, best_gateway_probabilities, best_parameters_path

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
        StructureMiner(
            mining_algorithm,
            log_path,
            settings.model_path,
            concurrency=settings.concurrency,
            eta=settings.eta,
            epsilon=settings.epsilon,
            prioritize_parallelism=settings.prioritize_parallelism,
            replace_or_joins=settings.replace_or_joins,
        ).run()

    def _extract_parameters_undifferentiated(self, settings: PipelineSettings, process_graph) -> Tuple:
        # Below, we mine simulation parameters with undifferentiated resources, because we optimize the structure,
        # not calendars. So, we do not need to differentiate resources.
        simulation_parameters = mine_default_24_7(
            self._log_train,
            self._log_ids,
            settings.model_path,
            process_graph,
            settings.gateway_probabilities_method)

        json_path = settings.model_path.parent / 'simulation_parameters.json'
        simulation_parameters.to_json_file(json_path)

        simulation_cases = self._log_train[self._log_ids.case].nunique()

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
