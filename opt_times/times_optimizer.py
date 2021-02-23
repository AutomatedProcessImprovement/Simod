# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 11:47:47 2020

@author: Manuel Camargo
"""
import os
import copy
import subprocess
import multiprocessing
from multiprocessing import Pool
import itertools

import xml.etree.ElementTree as ET
from lxml import etree
from lxml.builder import ElementMaker
import traceback


import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from hyperopt import tpe
from hyperopt import Trials, hp, fmin, STATUS_OK, STATUS_FAIL


import readers.log_reader as lr
import readers.bpmn_reader as br
import readers.process_structure as gph
import readers.log_splitter as ls
import analyzers.sim_evaluator as sim

from extraction import log_replayer as rpl
import opt_times.times_params_miner as tpm

import utils.support as sup
from utils.support import timeit

class TimesOptimizer():
    """
    Hyperparameter-optimizer class
    """
    class Decorators(object):

        @classmethod
        def safe_exec(cls, method):
            """
            Decorator to safe execute methods and return the state
            ----------
            method : Any method.
            Returns
            -------
            dict : execution status
            """
            def safety_check(*args, **kw):
                status = kw.get('status', method.__name__.upper())
                response = {'values': [], 'status': status}
                if status == STATUS_OK:
                    try:
                        response['values'] = method(*args)
                    except Exception as e:
                        print(e)
                        traceback.print_exc()
                        response['status'] = STATUS_FAIL
                return response
            return safety_check

    def __init__(self, settings, args, log, struc_model):
        """constructor"""
        self.space = self.define_search_space(settings, args)
        # read inputs
        self.log = log
        self._split_timeline(0.8, settings['read_options']['one_timestamp'])
        self.org_log = copy.deepcopy(log)
        self.org_log_train = copy.deepcopy(self.log_train)
        self.org_log_valdn = copy.deepcopy(self.log_valdn)
        # Load settings
        self.settings = settings
        self._load_sim_model(struc_model)
        self._replay_process()
        # Temp folder
        self.temp_output = os.path.join('outputs', sup.folder_id())
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
    def define_search_space(settings, args):
        space = {**{'rp_similarity': hp.uniform('rp_similarity',
                                                args['rp_similarity'][0],
                                                args['rp_similarity'][1]),
                    'res_cal_met': hp.choice('res_cal_met',
                        [
                            ('discovered',{
                            'res_support': hp.uniform('res_support',
                                                      args['res_sup_dis'][0],
                                                      args['res_sup_dis'][1]),
                            'res_confidence': hp.uniform('res_confidence',
                                                          args['res_con_dis'][0],
                                                          args['res_con_dis'][1])}),
                          ('default', {
                              'res_dtype': hp.choice('res_dtype',
                                                      args['res_dtype'])})
                          ]),
                    'arr_cal_met': hp.choice('arr_cal_met',
                        [('discovered',{
                            'arr_support': hp.uniform('arr_support',
                                                      args['arr_support'][0],
                                                      args['arr_support'][1]),
                            'arr_confidence': hp.uniform('arr_confidence',
                                                          args['arr_confidence'][0],
                                                          args['arr_confidence'][1])}),
                          ('default', {
                              'arr_dtype': hp.choice('arr_dtype',
                                                      args['arr_dtype'])})
                          ])},
                  **settings}
        return space

    def execute_trials(self):
        # create a new instance of Simod
        def exec_pipeline(trial_stg):
            print('train split:', 
                  len(pd.DataFrame(self.log_train.data).caseid.unique()), 
                  ', valdn split:', 
                  len(pd.DataFrame(self.log_valdn).caseid.unique()),
                  sep=' ')
            # Vars initialization
            status = STATUS_OK
            exec_times = dict()
            # print(len(data))
            sim_values = []
            trial_stg = self._filter_parms(trial_stg)
            # Path redefinition
            rsp = self._temp_path_redef(trial_stg,
                                        status=status,
                                        log_time=exec_times)
            status = rsp['status']
            trial_stg = rsp['values'] if status == STATUS_OK else trial_stg
            # # Parameters extraction
            rsp = self._extract_parameters(trial_stg,
                                           status=status,
                                           log_time=exec_times)
            status = rsp['status']
            # Simulate model
            rsp = self._simulate(trial_stg,
                                 self.log_valdn,
                                 status=status,
                                 log_time=exec_times)
            status = rsp['status']
            sim_values = rsp['values'] if status == STATUS_OK else sim_values
            # Save times
            self._save_times(exec_times, trial_stg, self.temp_output)
            # Optimizer results
            rsp = self._define_response(trial_stg, status, sim_values)
            # reinstate log
            self.log = copy.deepcopy(self.org_log)
            self.log_train = copy.deepcopy(self.org_log_train)
            self.log_valdn = copy.deepcopy(self.org_log_valdn)
            print("-- End of trial --")
            return rsp

        # Optimize
        best = fmin(fn=exec_pipeline,
                    space=self.space,
                    algo=tpe.suggest,
                    max_evals=self.args['max_eval_t'],
                    trials=self.bayes_trials,
                    show_progressbar=False)
        # Save results
        try:
            results = (pd.DataFrame(self.bayes_trials.results)
                        .sort_values('loss', ascending=bool))
            self.best_output = (results[results.status=='ok']
                                .head(1).iloc[0].output)
            self.best_parms = best
        except Exception as e:
            print(e)
            pass

    @timeit(rec_name='PATH_DEF')
    @Decorators.safe_exec
    def _temp_path_redef(self, settings, **kwargs) -> None:
        # Paths redefinition
        settings['output'] = os.path.join(self.temp_output, sup.folder_id())
        # Output folder creation
        if not os.path.exists(settings['output']):
            os.makedirs(settings['output'])
            os.makedirs(os.path.join(settings['output'], 'sim_data'))
        return settings


    @timeit(rec_name='EXTRACTING_PARAMS')
    @Decorators.safe_exec
    def _extract_parameters(self, settings, **kwargs) -> None:
        p_extractor = tpm.TimesParametersMiner(self.log_train,
                                               self.bpmn,
                                               self.process_graph,
                                               self.conformant_traces,
                                               self.process_stats,
                                               settings)
        num_inst = len(self.log_valdn.caseid.unique())
        # Get minimum date
        start_time = (self.log_valdn
                      .start_timestamp
                      .min().strftime("%Y-%m-%dT%H:%M:%S.%f+00:00"))
        p_extractor.extract_parameters(num_inst, start_time)
        if p_extractor.is_safe:
            self._xml_print(p_extractor.parameters,
                            os.path.join(
                                settings['output'],
                                settings['file'].split('.')[0]+'.bpmn'))

            self.log_valdn.rename(columns={'user': 'resource'}, inplace=True)
            self.log_valdn['source'] = 'log'
            self.log_valdn['run_num'] = 0
            self.log_valdn = self.log_valdn.merge(
                p_extractor.resource_table[['resource', 'role']],
                on='resource',
                how='left')
            self.log_valdn = self.log_valdn[
                ~self.log_valdn.task.isin(['Start', 'End'])]
            p_extractor.resource_table.to_pickle(
                os.path.join(settings['output'], 'resource_table.pkl'))
        else:
            raise RuntimeError('Parameters extraction error')

    @timeit(rec_name='SIMULATION_EVAL')
    @Decorators.safe_exec
    def _simulate(self, settings, data,**kwargs) -> list:
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
        w_count =  reps if reps <= cpu_count else cpu_count
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
            results = [self.evaluate_logs(arg) 
                       for arg in tqdm(args, 'evaluating results:')]
            # Save results
            sim_values = list(itertools.chain(*results))
        else:
            p = pool.map_async(self.evaluate_logs, args)
            pbar_async(p, 'evaluating results:')
            pool.close()
            # Save results
            sim_values = list(itertools.chain(*p.get()))
        return sim_values

    @staticmethod
    def _save_times(times, settings, temp_output):
        if times:
            times = [{**{'output': settings['output']}, **times}]
            log_file = os.path.join(temp_output, 'execution_times.csv')
            if not os.path.exists(log_file):
                    open(log_file, 'w').close()
            if os.path.getsize(log_file) > 0:
                sup.create_csv_file(times, log_file, mode='a')
            else:
                sup.create_csv_file_header(times, log_file)

    def _define_response(self, settings, status, sim_values, **kwargs) -> None:
        response = dict()
        measurements = list()
        data = {'alg_manag': settings['alg_manag'],
                'rp_similarity': settings['rp_similarity'],
                'gate_management': settings['gate_management'],
                'output': settings['output']}
        # Miner parms
        if settings['mining_alg'] in ['sm1', 'sm3']:
            data['epsilon'] = settings['epsilon']
            data['eta'] = settings['eta']
        elif settings['mining_alg'] == 'sm2':
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
                measurements.append({
                    **{'similarity': sim_val['sim_val'],
                        'sim_metric': sim_val['metric'],
                        'status': response['status']},
                    **data})
        else:
            response['status'] = status
            measurements.append({**{'similarity': 0,
                                    'sim_metric': 'day_hour_emd',
                                    'status': response['status']},
                                  **data})
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
            m_settings['file'] = settings['file']
            column_names = {'resource': 'user'}
            m_settings['read_options'] = settings['read_options']
            m_settings['read_options']['timeformat'] = '%Y-%m-%d %H:%M:%S.%f'
            m_settings['read_options']['column_names'] = column_names
            temp = lr.LogReader(os.path.join(
                m_settings['output'], 'sim_data',
                m_settings['file'].split('.')[0] + '_'+str(rep + 1)+'.csv'),
                m_settings['read_options'],
                verbose=False)
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
        def evaluate(settings, data, sim_log):
            """Reads the simulation results stats
            Args:
                settings (dict): Path to jar and file names
                rep (int): repetition number
            """
            rep = (sim_log.iloc[0].run_num) - 1
            sim_values = list()
            evaluator = sim.SimilarityEvaluator(
                data,
                sim_log,
                settings,
                max_cases=1000)
            evaluator.measure_distance('day_hour_emd')
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
                                  settings['file'].split('.')[0]+'.bpmn'),
                    '-csv',
                    os.path.join(settings['output'], 'sim_data',
                                  settings['file']
                                  .split('.')[0]+'_'+str(rep+1)+'.csv')]
            subprocess.run(args, check=True, stdout=subprocess.PIPE)
        sim_call(*args)

    def _xml_print(self, params, path) -> None:
        ns = {'qbp': "http://www.qbp-simulator.com/Schema201212"}
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
            childs = element.findall('qbp:'+tag, namespaces=ns)
            # Transform model from Etree to lxml
            node = xml_sim_model.find('qbp:'+tag+'s', namespaces=ns)
            # Clear existing elements
            for table in node.findall('qbp:'+tag, namespaces=ns):
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

    def _load_sim_model(self, struc_model) -> None:
        bpmn_file = os.path.join(struc_model,
                                  self.settings['file'].split('.')[0]+'.bpmn')
        tree = ET.parse(bpmn_file)
        root = tree.getroot()
        ns = {'qbp': "http://www.qbp-simulator.com/Schema201212"}
        parser = etree.XMLParser(remove_blank_text=True)
        self.xml_bpmn = etree.parse(bpmn_file, parser)
        process_info = self.xml_bpmn.find('qbp:processSimulationInfo',
                                          namespaces=ns)
        process_info.getparent().remove(process_info)

        ET.register_namespace('qbp', "http://www.qbp-simulator.com/Schema201212")
        self.xml_sim_model = etree.fromstring(
            ET.tostring(root.find('qbp:processSimulationInfo', ns)), parser)
        # load bpmn model
        self.bpmn = br.BpmnReader(bpmn_file)
        self.process_graph = gph.create_process_structure(self.bpmn)

    def _replay_process(self) -> None:
        """
        Process replaying
        """
        replayer = rpl.LogReplayer(self.process_graph, 
                                   self.log_train.get_traces(),
                                   self.settings, 
                                   msg='reading conformant training traces')
        self.process_stats = replayer.process_stats
        self.conformant_traces = replayer.conformant_traces

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
        """
        Split an event log dataframe by time to peform split-validation.
        prefered method time splitting removing incomplete traces.
        If the testing set is smaller than the 10% of the log size
        the second method is sort by traces start and split taking the whole
        traces no matter if they are contained in the timeframe or not

        Parameters
        ----------
        size : float, validation percentage.
        one_ts : bool, Support only one timestamp.
        """
        # Split log data
        splitter = ls.LogSplitter(self.log.data)
        train, valdn = splitter.split_log('timeline_contained', size, one_ts)
        total_events = len(self.log.data)
        # Check size and change time splitting method if necesary
        if len(valdn) < int(total_events*0.1):
            train, valdn = splitter.split_log('timeline_trace', size, one_ts)
        # Set splits
        key = 'end_timestamp' if one_ts else 'start_timestamp'
        valdn = pd.DataFrame(valdn)
        train = pd.DataFrame(train)
        self.log_valdn = (valdn.sort_values(key, ascending=True)
                         .reset_index(drop=True))
        self.log_train = copy.deepcopy(self.log)
        self.log_train.set_data(train.sort_values(key, ascending=True)
                                .reset_index(drop=True).to_dict('records'))
