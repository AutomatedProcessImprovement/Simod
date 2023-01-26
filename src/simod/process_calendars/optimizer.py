from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from hyperopt import Trials, hp, fmin, STATUS_OK, STATUS_FAIL
from hyperopt import tpe
from networkx import DiGraph

from simod.bpm.reader_writer import BPMNReaderWriter
from simod.cli_formatter import print_subsection
from simod.configuration import GatewayProbabilitiesDiscoveryMethod
from simod.event_log.column_mapping import EventLogIDs
from simod.event_log.event_log import EventLog
from simod.hyperopt_pipeline import HyperoptPipeline
from simod.process_calendars.settings import CalendarOptimizationSettings, PipelineSettings
from simod.simulation.parameters.miner import mine_parameters
from simod.simulation.prosimos import simulate_and_evaluate
from simod.utilities import remove_asset, folder_id, file_id, nearest_divisor_for_granularity


class CalendarOptimizer(HyperoptPipeline):
    _event_log: EventLog
    _log_train: pd.DataFrame
    _log_validation: pd.DataFrame
    _log_ids: EventLogIDs
    _gateway_probabilities_method: GatewayProbabilitiesDiscoveryMethod
    _gateway_probabilities: Optional[dict]
    _train_model_path: Path
    _output_dir: Path
    _bayes_trials: Trials
    _process_graph: Optional[DiGraph]

    evaluation_measurements: pd.DataFrame

    def __init__(
            self,
            calendar_optimizer_settings: CalendarOptimizationSettings,
            event_log: EventLog,
            train_model_path: Path,
            gateway_probabilities_method: GatewayProbabilitiesDiscoveryMethod,
            gateway_probabilities: Optional[list] = None,
            process_graph: Optional[DiGraph] = None,
            event_distribution: Optional[list[dict]] = None,
    ):
        self._calendar_optimizer_settings = calendar_optimizer_settings
        self._event_log = event_log
        self._log_ids = event_log.log_ids
        self._train_model_path = train_model_path
        self._gateway_probabilities_method = gateway_probabilities_method
        self._gateway_probabilities = gateway_probabilities
        self._process_graph = process_graph
        self._event_distribution = event_distribution

        self._log_train = event_log.train_partition.sort_values(by=event_log.log_ids.start_time)
        self._log_validation = event_log.validation_partition.sort_values(event_log.log_ids.start_time, ascending=True)

        # Calendar optimization base folder
        self._output_dir = self._calendar_optimizer_settings.base_dir / folder_id(prefix='calendars_')
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self.evaluation_measurements = pd.DataFrame(
            columns=['value', 'metric', 'gateway_probabilities', 'status', 'output_dir'])

        self._bayes_trials = Trials()

        if self._process_graph is None:
            bpmn_reader = BPMNReaderWriter(train_model_path)
            self._process_graph = bpmn_reader.as_graph()

    def _optimization_objective(self, trial_stg: Union[dict, PipelineSettings]):
        print_subsection('Calendar Optimization Trial')

        # casting a dictionary provided by hyperopt to PipelineSettings for convenience
        if isinstance(trial_stg, dict):
            trial_stg = PipelineSettings.from_hyperopt_option_dict(
                trial_stg,
                output_dir=self._output_dir,
                model_path=self._train_model_path,
                gateway_probabilities_method=self._gateway_probabilities_method
            )

        # update granularity
        if 1440 % trial_stg.case_arrival.granularity != 0:
            trial_stg.case_arrival.granularity = nearest_divisor_for_granularity(
                trial_stg.case_arrival.granularity)
        if 1440 % trial_stg.resource_profiles.granularity != 0:
            trial_stg.resource_profiles.granularity = nearest_divisor_for_granularity(
                trial_stg.resource_profiles.granularity)

        # initializing status
        status = STATUS_OK

        # creating and defining folders and paths
        output_dir = self._output_dir / folder_id(prefix='calendars_trial_')
        output_dir.mkdir(parents=True, exist_ok=True)
        trial_stg.output_dir = output_dir

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
        response, status = self._define_response(
            trial_stg,
            status,
            evaluation_measurements,
        )

        # recording measurements internally
        self._process_measurements(trial_stg, status, evaluation_measurements)

        return response

    def run(self) -> PipelineSettings:

        # Optimization
        space = self._define_search_space(self._calendar_optimizer_settings)
        best = fmin(fn=self._optimization_objective,
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
        self.evaluation_measurements.sort_values('value', ascending=False, inplace=True)
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
            for measurement in evaluation_measurements:
                values = {
                    'value': measurement['value'],
                    'metric': measurement['metric'],
                }
                values = values | data
                self.evaluation_measurements = pd.concat([self.evaluation_measurements, pd.DataFrame([values])])
        else:
            values = {
                'value': 0,
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
            distance = np.mean([x['value'] for x in evaluation_measurements])
            loss = distance
            response['loss'] = loss

            status = status if loss > 0 else STATUS_FAIL
            response['status'] = status if loss > 0 else STATUS_FAIL

        return response, status

    def _extract_parameters(self, settings: PipelineSettings) -> Tuple:
        parameters = mine_parameters(
            settings.case_arrival,
            settings.resource_profiles,
            self._log_train,
            self._log_ids,
            settings.model_path,
            gateways_probability_method=self._gateway_probabilities_method,
            gateway_probabilities=self._gateway_probabilities,
            process_graph=self._process_graph,
        )

        json_path = settings.output_dir / 'simulation_parameters.json'

        if self._event_distribution is not None:
            parameters.event_distribution = self._event_distribution

        parameters.to_json_file(json_path)

        simulation_cases = self._log_train[self._log_ids.case].nunique()

        return json_path, simulation_cases

    def _simulate_with_prosimos(self, settings: PipelineSettings, json_path: Path, simulation_cases: int):
        num_simulations = self._calendar_optimizer_settings.simulation_repetitions
        bpmn_path = settings.model_path

        return simulate_and_evaluate(
            model_path=bpmn_path,
            parameters_path=json_path,
            output_dir=settings.output_dir,
            simulation_cases=simulation_cases,
            simulation_start_time=self._log_validation[self._log_ids.start_time].min(),
            validation_log=self._log_validation,
            validation_log_ids=self._log_ids,
            metrics=[self._calendar_optimizer_settings.optimization_metric],
            num_simulations=num_simulations,
        )
