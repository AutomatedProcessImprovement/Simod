import copy
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from hyperopt import Trials, hp, fmin, STATUS_OK, STATUS_FAIL
from hyperopt import tpe
from lxml import etree
from lxml.builder import ElementMaker

from simod.common_routines import extract_times_parameters, split_timeline, hyperopt_pipeline_step, remove_asset
from . import support_utils as sup
from .analyzers import sim_evaluator as sim
from .cli_formatter import print_subsection, print_message, print_step
from .configuration import Configuration, MiningAlgorithm, Metric, QBP_NAMESPACE_URI
from .readers import bpmn_reader as br
from .readers import process_structure as gph
from .event_log import LogReader
from .simulator import simulate
from .support_utils import get_project_dir


class TimesOptimizer:
    """Hyperparameter-optimizer class"""
    best_output: Optional[Path]
    best_parameters: dict
    measurements_file_name: Path
    _temp_output: Path

    _bayes_trials: Trials
    _settings_global: Configuration
    _settings_time: Configuration
    _log_train: LogReader
    _log_validation: pd.DataFrame
    _original_log: LogReader
    _original_log_train: LogReader
    _original_log_validation: pd.DataFrame

    def __init__(self, settings_global: Configuration, settings_time: Configuration, log: LogReader, model_path):
        self._settings_global = settings_global
        self._settings_time = settings_time
        self._log = log

        self._space = self._define_search_space(settings_global, settings_time)

        train, validation = self._split_timeline(0.8)
        self._log_train = LogReader.copy_without_data(self._log)
        self._log_train.set_data(
            train.sort_values('start_timestamp', ascending=True).reset_index(drop=True).to_dict('records'))
        self._log_validation = validation

        self._original_log = copy.deepcopy(log)
        self._original_log_train = copy.deepcopy(self._log_train)
        self._original_log_validation = copy.deepcopy(self._log_validation)

        self._load_sim_model(model_path)

        log_df = pd.DataFrame(self._log_train.data)
        self._conformant_traces = log_df
        self._process_stats = log_df

        self._temp_output = get_project_dir() / 'outputs' / sup.folder_id()
        self._temp_output.mkdir(parents=True, exist_ok=True)

        self.measurements_file_name = self._temp_output / sup.file_id(prefix='OP_')
        with self.measurements_file_name.open('w') as _:
            pass

        self._bayes_trials = Trials()

    def run(self):
        def pipeline(trial_stg):
            print_subsection('Trial')
            print_message(f'train split: {len(pd.DataFrame(self._log_train.data).caseid.unique())}, '
                          f'valdn split: {len(pd.DataFrame(self._log_validation).caseid.unique())}')

            if isinstance(trial_stg, dict):
                trial_stg = Configuration(**trial_stg)

            trial_stg = self._filter_parms(trial_stg)

            status = STATUS_OK

            status, result = hyperopt_pipeline_step(status, self._temp_path_redef, trial_stg)
            if status == STATUS_OK:
                trial_stg = result

            status, _ = hyperopt_pipeline_step(status, self._extract_parameters, trial_stg)

            status, result = hyperopt_pipeline_step(status, self._simulate, trial_stg)
            sim_values = result if status == STATUS_OK else []

            response = self._define_response(trial_stg, status, sim_values)

            # reinstate log
            self._log = self._original_log  # TODO: no need
            self._log_train = copy.deepcopy(self._original_log_train)
            self._log_validation = copy.deepcopy(self._original_log_validation)

            return response

        # Optimize
        best = fmin(fn=pipeline,
                    space=self._space,
                    algo=tpe.suggest,
                    max_evals=self._settings_time.max_eval_t,
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
    def _define_search_space(settings_global: Configuration, settings_time: Configuration):
        space = {
            **settings_global.__dict__,
            'rp_similarity': hp.uniform('rp_similarity', settings_time.rp_similarity[0],
                                        settings_time.rp_similarity[1]),
            'res_cal_met': hp.choice(
                'res_cal_met',
                [('discovered', {
                    'res_support':
                        hp.uniform('res_support', settings_time.res_sup_dis[0], settings_time.res_sup_dis[1]),
                    'res_confidence':
                        hp.uniform('res_confidence', settings_time.res_con_dis[0], settings_time.res_con_dis[1])}),
                 ('default', {'res_dtype': hp.choice('res_dtype', settings_time.res_dtype)})]),
            'arr_cal_met': hp.choice(
                'arr_cal_met',
                [('discovered', {
                    'arr_support':
                        hp.uniform('arr_support', settings_time.arr_support[0], settings_time.arr_support[1]),
                    'arr_confidence':
                        hp.uniform('arr_confidence', settings_time.arr_confidence[0],
                                   settings_time.arr_confidence[1])}),
                 ('default', {'arr_dtype': hp.choice('arr_dtype', settings_time.arr_dtype)})])}
        return space

    def _temp_path_redef(self, settings: Configuration):
        settings.output = self._temp_output / sup.folder_id()
        simulation_data_path = settings.output / 'sim_data'
        simulation_data_path.mkdir(parents=True, exist_ok=True)
        return settings

    def _extract_parameters(self, settings: Configuration):
        num_inst = len(self._log_validation.caseid.unique())
        start_time = self._log_validation.start_timestamp.min().strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")

        parameters = extract_times_parameters(
            settings, self.process_graph, self._log_train, self._conformant_traces, self._process_stats)

        parameters.instances = num_inst
        parameters.start_time = start_time

        self._xml_print(parameters.__dict__, os.path.join(settings.output, settings.project_name + '.bpmn'))
        self._log_validation.rename(columns={'user': 'resource'}, inplace=True)
        self._log_validation['source'] = 'log'
        self._log_validation['run_num'] = 0
        self._log_validation = self._log_validation.merge(parameters.resource_table[['resource', 'role']],
                                                          on='resource', how='left')
        self._log_validation = self._log_validation[~self._log_validation.task.isin(['Start', 'End'])]
        parameters.resource_table.to_pickle(os.path.join(settings.output, 'resource_table.pkl'))

    def _simulate(self, trial_stg: Configuration):
        return simulate(trial_stg, self._log_validation, evaluate_fn=self._evaluate_logs)

    def _define_response(self, settings: Configuration, status: str, sim_values: list) -> dict:
        data = {'rp_similarity': settings.rp_similarity,
                'gate_management': settings.gate_management,
                'output': settings.output}

        # Miner parameters
        if settings.mining_alg in [MiningAlgorithm.SM1, MiningAlgorithm.SM3]:
            data['epsilon'] = settings.epsilon
            data['eta'] = settings.eta
            data['and_prior'] = settings.and_prior
            data['or_rep'] = settings.or_rep
        elif settings.mining_alg is MiningAlgorithm.SM2:
            data['concurrency'] = settings.concurrency
        else:
            raise ValueError(settings.mining_alg)

        response = {}
        measurements = []
        response['output'] = settings.output
        if status == STATUS_OK:
            similarity = np.mean([x['sim_val'] for x in sim_values])
            loss = (1 - similarity)
            response['loss'] = loss
            response['status'] = status if loss > 0 else STATUS_FAIL
            for sim_val in sim_values:
                measurements.append(
                    {'similarity': sim_val['sim_val'], 'sim_metric': sim_val['metric'], 'status': response['status'],
                     **data})
        else:
            response['status'] = status
            measurements.append(
                {'similarity': 0, 'sim_metric': Metric.DAY_HOUR_EMD, 'status': response['status'], **data})

        if os.path.getsize(self.measurements_file_name) > 0:
            sup.create_csv_file(measurements, self.measurements_file_name, mode='a')
        else:
            sup.create_csv_file_header(measurements, self.measurements_file_name)

        return response

    @staticmethod
    def _evaluate_logs(args) -> Optional[list]:
        settings: Configuration = args[0]
        if isinstance(settings, dict):
            settings = Configuration(**settings)
        data: pd.DataFrame = args[1]
        sim_log = args[2]
        if sim_log is None:
            return None

        rep = sim_log.iloc[0].run_num - 1
        sim_values = []
        evaluator = sim.SimilarityEvaluator(data, sim_log, settings, max_cases=1000)
        evaluator.measure_distance(Metric.DAY_HOUR_EMD)
        sim_values.append({**{'run_num': rep}, **evaluator.similarity})

        return sim_values

    def _xml_print(self, params, path) -> None:
        ns = {'qbp': QBP_NAMESPACE_URI}

        def print_xml_resources(parms) -> str:
            E = ElementMaker(namespace=ns['qbp'], nsmap=ns)
            PROCESSSIMULATIONINFO = E.processSimulationInfo
            TIMEUNIT = E.timeUnit
            RESOURCES = E.resources
            RESOURCE = E.resource
            ELEMENTS = E.elements
            ELEMENT = E.element
            DURATION = E.durationDistribution
            RESOURCESIDS = E.resourceIds
            RESOURCESID = E.resourceId
            my_doc = PROCESSSIMULATIONINFO(
                RESOURCES(
                    *[
                        RESOURCE(
                            id=res['id'],
                            name=res['name'],
                            totalAmount=res['total_amount'],
                            costPerHour=res['costxhour'],
                            timetableId=res['timetable_id']
                        ) for res in parms['resource_pool']
                    ]
                ),
                ELEMENTS(
                    *[
                        ELEMENT(
                            DURATION(
                                TIMEUNIT("seconds"),
                                type=e['type'],
                                mean=e['mean'],
                                arg1=e['arg1'],
                                arg2=e['arg2']
                            ),
                            RESOURCESIDS(
                                RESOURCESID(str(e['resource']))
                            ),
                            id=e['id'], elementId=e['elementid']
                        ) for e in parms['elements_data']
                    ]
                )
            )
            return my_doc

        def replace_parm(element, tag, xml_sim_model):
            childs = element.findall('qbp:' + tag, namespaces=ns)
            # Transform model from Etree to lxml
            node = xml_sim_model.find('qbp:' + tag + 's', namespaces=ns)
            # Clear existing elements
            for table in node.findall('qbp:' + tag, namespaces=ns):
                table.getparent().remove(table)
            # Insert new elements
            for i, child in enumerate(childs):
                node.insert((i + 1), child)
            return xml_sim_model

        self.xml_sim_model = replace_parm(params['time_table'], 'timetable', self.xml_sim_model)
        xml_new_elements = print_xml_resources(params)
        self.xml_sim_model = replace_parm(
            xml_new_elements.find('qbp:resources', namespaces=ns),
            'resource',
            self.xml_sim_model)
        self.xml_sim_model = replace_parm(
            xml_new_elements.find('qbp:elements', namespaces=ns),
            'element',
            self.xml_sim_model)
        arrival = self.xml_sim_model.find('qbp:arrivalRateDistribution',
                                          namespaces=ns)
        arrival.attrib['type'] = params['arrival_rate']['dname']
        arrival.attrib['mean'] = str(params['arrival_rate']['dparams']['mean'])
        arrival.attrib['arg1'] = str(params['arrival_rate']['dparams']['arg1'])
        arrival.attrib['arg2'] = str(params['arrival_rate']['dparams']['arg2'])

        self.xml_bpmn.getroot().append(self.xml_sim_model)

        def create_file(output_file, element):
            with open(output_file, 'wb') as f:
                f.write(element)

        # Print model
        create_file(path, etree.tostring(self.xml_bpmn, pretty_print=True))

    def _load_sim_model(self, model_path) -> None:
        tree = ET.parse(model_path)
        root = tree.getroot()
        ns = {'qbp': QBP_NAMESPACE_URI}
        parser = etree.XMLParser(remove_blank_text=True, resolve_entities=False, no_network=True)
        self.xml_bpmn = etree.parse(model_path, parser)
        process_info = self.xml_bpmn.find('qbp:processSimulationInfo', namespaces=ns)
        process_info.getparent().remove(process_info)

        ET.register_namespace('qbp', QBP_NAMESPACE_URI)
        self.xml_sim_model = etree.fromstring(
            ET.tostring(root.find('qbp:processSimulationInfo', ns)), parser)
        # load bpmn model
        self.bpmn = br.BpmnReader(model_path)
        self.process_graph = gph.create_process_structure(self.bpmn)

    @staticmethod
    def _filter_parms(settings: Configuration):
        # resources discovery
        method, values = settings.res_cal_met
        if method in 'discovered':
            settings.res_confidence = values['res_confidence']
            settings.res_support = values['res_support']
        else:
            settings.res_dtype = values['res_dtype']
        settings.res_cal_met = method
        # arrivals calendar
        method, values = settings.arr_cal_met
        if method in 'discovered':
            settings.arr_confidence = values['arr_confidence']
            settings.arr_support = values['arr_support']
        else:
            settings.arr_dtype = values['arr_dtype']
        settings.arr_cal_met = method
        return settings

    def _split_timeline(self, size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train, validation, key = split_timeline(self._log, size)
        validation = validation.sort_values(key, ascending=True).reset_index(drop=True)
        train = train.sort_values(key, ascending=True).reset_index(drop=True)
        return train, validation
