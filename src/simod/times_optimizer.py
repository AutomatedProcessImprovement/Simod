import copy
import itertools
import multiprocessing
import os
import subprocess
import time
import xml.etree.ElementTree as ET
from multiprocessing import Pool

import numpy as np
import pandas as pd
from hyperopt import Trials, hp, fmin, STATUS_OK, STATUS_FAIL
from hyperopt import tpe
from lxml import etree
from lxml.builder import ElementMaker
from tqdm import tqdm

from simod.common_routines import extract_times_parameters, split_timeline, save_times
from . import support_utils as sup
from .analyzers import sim_evaluator as sim
from .cli_formatter import print_subsection, print_message, print_notice
from .configuration import Configuration, MiningAlgorithm, ReadOptions, Metric, QBP_NAMESPACE_URI
from .decorators import timeit, safe_exec_with_values_and_status
from .readers import bpmn_reader as br
from .readers import log_reader as lr
from .readers import process_structure as gph
from .readers.log_reader import LogReader
from .support_utils import get_project_dir


class TimesOptimizer:
    """Hyperparameter-optimizer class"""

    # @profile(stream=open('logs/memprof_TimesOptimizer.log', 'a+'))
    def __init__(self, settings: Configuration, args: Configuration, log, model_path):
        self.space = self.define_search_space(settings, args)
        # read inputs
        self.log = log
        self._split_timeline(0.8, settings.read_options.one_timestamp)
        self.org_log = log
        self.org_log_train = copy.deepcopy(self.log_train)
        self.org_log_valdn = copy.deepcopy(self.log_valdn)
        # Load settings
        self.settings: Configuration = settings
        self._load_sim_model(model_path)

        # NOTE: with new replayer, we don't need conformant_traces or process_stats
        # self._replay_process()
        log_df = pd.DataFrame(self.log_train.data)
        self.conformant_traces = log_df
        self.process_stats = log_df

        # Temp folder
        self.temp_output = get_project_dir() / 'outputs' / sup.folder_id()
        if not os.path.exists(self.temp_output):
            os.makedirs(self.temp_output)
        self.file_name = os.path.join(self.temp_output, sup.file_id(prefix='OP_'))
        # Results file
        if not os.path.exists(self.file_name):
            open(self.file_name, 'w').close()
        self.args = args
        # Trials object to track progress
        self.bayes_trials = Trials()
        self.best_output = None
        self.best_parms = dict()

    @staticmethod
    def define_search_space(settings: Configuration, args: Configuration):
        space = {
            **settings.__dict__,
            'rp_similarity': hp.uniform('rp_similarity', args.rp_similarity[0], args.rp_similarity[1]),
            'res_cal_met': hp.choice(
                'res_cal_met',
                [('discovered', {
                    'res_support': hp.uniform('res_support', args.res_sup_dis[0], args.res_sup_dis[1]),
                    'res_confidence': hp.uniform('res_confidence', args.res_con_dis[0], args.res_con_dis[1])}),
                 ('default', {'res_dtype': hp.choice('res_dtype', args.res_dtype)})]),
            'arr_cal_met': hp.choice(
                'arr_cal_met',
                [('discovered', {
                    'arr_support': hp.uniform('arr_support', args.arr_support[0], args.arr_support[1]),
                    'arr_confidence': hp.uniform('arr_confidence', args.arr_confidence[0], args.arr_confidence[1])}),
                 ('default', {'arr_dtype': hp.choice('arr_dtype', args.arr_dtype)})])}
        return space

    # @profile(stream=open('logs/memprof_TimesOptimizer.execute_trials.log', 'a+'))
    def execute_trials(self):
        # create a new instance of Simod
        # @profile(stream=open('logs/memprof_TimesOptimizer.execute_trials.exec_pipeline.log', 'a+'))
        def exec_pipeline(trial_stg):
            print_subsection('Trial')
            print_message(f'train split: {len(pd.DataFrame(self.log_train.data).caseid.unique())}, '
                          f'valdn split: {len(pd.DataFrame(self.log_valdn).caseid.unique())}')

            status = STATUS_OK
            exec_times = dict()
            sim_values = []
            trial_stg = self._filter_parms(trial_stg)

            # Path redefinition
            rsp = self._temp_path_redef(trial_stg, status=status, log_time=exec_times)
            status = rsp['status']
            trial_stg = rsp['values'] if status == STATUS_OK else trial_stg

            # Parameters extraction
            rsp = self._extract_parameters(Configuration(**trial_stg), status=status, log_time=exec_times)
            status = rsp['status']

            # Simulate model
            rsp = self._simulate(trial_stg, self.log_valdn, status=status, log_time=exec_times)
            status = rsp['status']
            sim_values = rsp['values'] if status == STATUS_OK else sim_values

            # Save times
            save_times(exec_times, trial_stg, self.temp_output)

            # Optimizer results
            rsp = self._define_response(trial_stg, status, sim_values)

            # reinstate log
            self.log = self.org_log  # TODO: no need
            self.log_train = copy.deepcopy(self.org_log_train)
            self.log_valdn = copy.deepcopy(self.org_log_valdn)

            return rsp

        # Optimize
        best = fmin(fn=exec_pipeline, space=self.space, algo=tpe.suggest, max_evals=self.args.max_eval_t,
                    trials=self.bayes_trials, show_progressbar=False)
        # Save results
        try:
            results = (pd.DataFrame(self.bayes_trials.results).sort_values('loss', ascending=bool))
            self.best_output = results[results.status == 'ok'].head(1).iloc[0].output
            self.best_parms = best
        except Exception as e:
            print(e)
            pass

    @timeit(rec_name='PATH_DEF')
    @safe_exec_with_values_and_status
    def _temp_path_redef(self, settings, **kwargs) -> None:
        # Paths redefinition
        settings['output'] = os.path.join(self.temp_output, sup.folder_id())
        # Output folder creation
        if not os.path.exists(settings['output']):
            os.makedirs(settings['output'])
            os.makedirs(os.path.join(settings['output'], 'sim_data'))
        return settings

    # @profile(stream=open('logs/memprof_TimesOptimizer._extract_parameters.log', 'a+'))
    @timeit(rec_name='EXTRACTING_PARAMS')
    @safe_exec_with_values_and_status
    def _extract_parameters(self, settings: Configuration, **kwargs) -> None:
        num_inst = len(self.log_valdn.caseid.unique())
        start_time = self.log_valdn.start_timestamp.min().strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")

        parameters = extract_times_parameters(
            settings, self.process_graph, self.log_train, self.conformant_traces, self.process_stats)

        parameters.instances = num_inst
        parameters.start_time = start_time

        self._xml_print(parameters.__dict__, os.path.join(settings.output, settings.project_name + '.bpmn'))
        self.log_valdn.rename(columns={'user': 'resource'}, inplace=True)
        self.log_valdn['source'] = 'log'
        self.log_valdn['run_num'] = 0
        self.log_valdn = self.log_valdn.merge(parameters.resource_table[['resource', 'role']],
                                              on='resource', how='left')
        self.log_valdn = self.log_valdn[~self.log_valdn.task.isin(['Start', 'End'])]
        parameters.resource_table.to_pickle(os.path.join(settings.output, 'resource_table.pkl'))

    @timeit(rec_name='SIMULATION_EVAL')
    @safe_exec_with_values_and_status
    def _simulate(self, settings, data, **kwargs) -> list:
        def pbar_async(p, msg):
            pbar = tqdm(total=reps, desc=msg)
            processed = 0
            while not p.ready():
                cprocesed = (reps - p._number_left)
                if processed < cprocesed:
                    increment = cprocesed - processed
                    pbar.update(n=increment)
                    processed = cprocesed
            time.sleep(1)
            pbar.update(n=(reps - processed))
            p.wait()
            pbar.close()

        reps = settings['repetitions']
        cpu_count = multiprocessing.cpu_count()
        w_count = reps if reps <= cpu_count else cpu_count
        pool = Pool(processes=w_count)
        # Simulate
        args = [(settings, rep) for rep in range(reps)]
        p = pool.map_async(self.execute_simulator, args)
        pbar_async(p, 'simulating:')
        # Read simulated logs
        p = pool.map_async(self.read_stats, args)
        pbar_async(p, 'reading simulated logs:')
        # Evaluate
        args = [(settings, data, log) for log in p.get()]
        if len(self.log_valdn.caseid.unique()) > 1000:
            pool.close()
            results = [self.evaluate_logs(arg) for arg in tqdm(args, 'evaluating results:')]
            # Save results
            sim_values = list(itertools.chain(*results))
        else:
            p = pool.map_async(self.evaluate_logs, args)
            pbar_async(p, 'evaluating results:')
            pool.close()
            # Save results
            sim_values = list(itertools.chain(*p.get()))
        return sim_values

    def _define_response(self, settings, status, sim_values, **kwargs) -> None:
        response = dict()
        measurements = list()
        data = {'rp_similarity': settings['rp_similarity'],
                'gate_management': settings['gate_management'],
                'output': settings['output']}
        # Miner parms
        if settings['mining_alg'] in [MiningAlgorithm.SM1, MiningAlgorithm.SM3]:
            data['epsilon'] = settings['epsilon']
            data['eta'] = settings['eta']
            data['and_prior'] = settings['and_prior']
            data['or_rep'] = settings['or_rep']
        elif settings['mining_alg'] is MiningAlgorithm.SM2:
            data['concurrency'] = settings['concurrency']
        else:
            raise ValueError(settings['mining_alg'])
        similarity = 0
        # response['params'] = settings
        response['output'] = settings['output']
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
        if os.path.getsize(self.file_name) > 0:
            sup.create_csv_file(measurements, self.file_name, mode='a')
        else:
            sup.create_csv_file_header(measurements, self.file_name)
        return response

    @staticmethod
    def read_stats(args):
        def read(settings, rep):
            """Reads the simulation results stats
            Args:
                settings (dict): Path to jar and file names
                rep (int): repetition number
            """
            m_settings = dict()
            m_settings['output'] = settings['output']
            column_names = {'resource': 'user'}
            read_options: ReadOptions = settings['read_options']
            read_options.timeformat = '%Y-%m-%d %H:%M:%S.%f'
            read_options.column_names = column_names
            m_settings['read_options'] = read_options
            m_settings['project_name'] = settings['project_name']
            file_path = os.path.join(m_settings['output'], 'sim_data',
                                     m_settings['project_name'] + '_' + str(rep + 1) + '.csv')
            if not os.path.exists(file_path):
                print_notice(f'File does not exist at {file_path}')
                return
            temp = lr.LogReader(file_path, m_settings['read_options'], verbose=False)
            temp = pd.DataFrame(temp.data)
            temp.rename(columns={'user': 'resource'}, inplace=True)
            temp['role'] = temp['resource']
            temp['source'] = 'simulation'
            temp['run_num'] = rep + 1
            temp = temp[~temp.task.isin(['Start', 'End'])]
            return temp

        return read(*args)

    @staticmethod
    def evaluate_logs(args):
        # settings, bpmn, rep = args
        def evaluate(settings: Configuration, data, sim_log):
            """Reads the simulation results stats
            Args:
                settings (dict): Path to jar and file names
                rep (int): repetition number
            """
            if sim_log is None:
                return

            if isinstance(settings, dict):
                settings = Configuration(**settings)

            rep = sim_log.iloc[0].run_num - 1
            sim_values = list()
            evaluator = sim.SimilarityEvaluator(data, sim_log, settings, max_cases=1000)
            evaluator.measure_distance(Metric.DAY_HOUR_EMD)
            sim_values.append({**{'run_num': rep}, **evaluator.similarity})
            return sim_values

        return evaluate(*args)

    @staticmethod
    def execute_simulator(args):
        def sim_call(settings, rep):
            """Executes BIMP Simulations.
            Args:
                settings (dict): Path to jar and file names
                rep (int): repetition number
            """
            args = ['java', '-jar', settings['bimp_path'],
                    os.path.join(settings['output'],
                                 settings['project_name'] + '.bpmn'),
                    '-csv',
                    os.path.join(settings['output'], 'sim_data',
                                 settings['project_name'] + '_' + str(rep + 1) + '.csv')]
            subprocess.run(args, check=True, stdout=subprocess.PIPE)

        sim_call(*args)

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

        # lxml.etree._Element
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

        self.xml_sim_model = replace_parm(params['time_table'],
                                          'timetable',
                                          self.xml_sim_model)
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
            f.close()

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
    def _filter_parms(parms):
        # resources discovery
        method, values = parms['res_cal_met']
        if method in 'discovered':
            parms['res_confidence'] = values['res_confidence']
            parms['res_support'] = values['res_support']
        else:
            parms['res_dtype'] = values['res_dtype']
        parms['res_cal_met'] = method
        # arrivals calendar
        method, values = parms['arr_cal_met']
        if method in 'discovered':
            parms['arr_confidence'] = values['arr_confidence']
            parms['arr_support'] = values['arr_support']
        else:
            parms['arr_dtype'] = values['arr_dtype']
        parms['arr_cal_met'] = method
        return parms

    def _split_timeline(self, size: float, one_ts: bool) -> None:
        train, valdn, key = split_timeline(self.log, size, one_ts)
        self.log_valdn = valdn.sort_values(key, ascending=True).reset_index(drop=True)
        # self.log_train = copy.deepcopy(self.log)
        self.log_train = LogReader.copy_without_data(self.log)
        self.log_train.set_data(train.sort_values(key, ascending=True).reset_index(drop=True).to_dict('records'))
