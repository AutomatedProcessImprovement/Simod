# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 10:56:25 2019

@author: Manuel Camargo
"""
import os
# import subprocess
import types
import itertools
import platform as pl
import copy
import multiprocessing
from multiprocessing import Pool
from xml.dom import minidom


import pandas as pd
import numpy as np
from operator import itemgetter
from tqdm import tqdm


# import utils.support as sup
from utils.support import timeit
import readers.log_reader as lr
import readers.bpmn_reader as br
import readers.process_structure as gph
import readers.log_splitter as ls


from support_modules.writers import xes_writer as xes
from support_modules.writers import xml_writer as xml
from support_modules.analyzers import sim_evaluator as sim
from support_modules.log_repairing import conformance_checking as chk

# from extraction import parameter_extraction as par
# from extraction import log_replayer as rpl

import opt_times.times_optimizer as to
import opt_structure.structure_optimizer as so

import opt_times.times_optimizer as to
import opt_structure.structure_optimizer as so


class Simod():
    """
    Main class of the Simulation Models Discoverer
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
                is_safe = kw.get('is_safe', method.__name__.upper())
                if is_safe:
                    try:
                        method(*args)
                    except Exception as e:
                        print(e)
                        is_safe = False
                return is_safe
            return safety_check

    def __init__(self, settings, args):
        """constructor"""
        self.settings = settings
        self.args = args

        self.log = types.SimpleNamespace()
        self.log_train = types.SimpleNamespace()
        self.log_test = types.SimpleNamespace()
        # self.bpmn = types.SimpleNamespace()
        # self.process_graph = types.SimpleNamespace()

        self.sim_values = list()
        self.response = dict()
        self.parameters = dict()
        self.is_safe = True

    def execute_pipeline(self, can=False) -> None:
        exec_times = dict()
        self.is_safe = self.read_inputs(
            log_time=exec_times, is_safe=self.is_safe)
        self.is_safe = self.evaluate_alignment(
            log_time=exec_times, is_safe=self.is_safe)
        self.is_safe = self.extract_parameters(
            log_time=exec_times, is_safe=self.is_safe)
        self.is_safe = self.simulate(
            log_time=exec_times, is_safe=self.is_safe)
        self.mannage_results()
        self.save_times(exec_times, self.settings)
        # self.is_safe = self.export_canonical_model(is_safe=self.is_safe)
        print("-- End of trial --")

    @timeit(rec_name='READ_INPUTS')
    @Decorators.safe_exec
    def read_inputs(self, **kwargs) -> None:
        # Output folder creation
        # if not os.path.exists(self.settings['output']):
        #     os.makedirs(self.settings['output'])
        #     os.makedirs(os.path.join(self.settings['output'], 'sim_data'))
        # Event log reading
        self.log = lr.LogReader(os.path.join(self.settings['input'],
                                             self.settings['file']),
                                self.settings['read_options'])
        # Time splitting 80-20
        self.split_timeline(0.8,
                            self.settings['read_options']['one_timestamp'])
        # Create customized event-log for the external tools
        xes.XesWriter(self.log, self.settings)
        # Execution steps
        self.mining_structure(self.settings)
        self.bpmn = br.BpmnReader(os.path.join(
            self.settings['output'],
            self.settings['file'].split('.')[0]+'.bpmn'))
        self.process_graph = gph.create_process_structure(self.bpmn)
        # Replaying test partition
        print("-- Reading test partition --")
        test_replayer = rpl.LogReplayer(
            self.process_graph,
            self.get_traces(self.log_test,
                            self.settings['read_options']['one_timestamp']),
            self.settings)
        self.process_stats = test_replayer.process_stats
        self.process_stats = pd.DataFrame.from_records(self.process_stats)
        self.log_test = test_replayer.conformant_traces
        print("-- End of trial --")


    @timeit(rec_name='ALIGNMENT')
    @Decorators.safe_exec
    def evaluate_alignment(self, **kwargs) -> None:
        """
        Evaluate alignment
        """
        # Evaluate alignment
        chk.evaluate_alignment(self.process_graph,
                               self.log_train,
                               self.settings)

    @timeit(rec_name='EXTRACTION')
    @Decorators.safe_exec
    def extract_parameters(self, **kwargs) -> None:
        print("-- Mining Simulation Parameters --")
        p_extractor = par.ParameterMiner(self.log_train,
                                         self.bpmn,
                                         self.process_graph,
                                         self.settings)
        num_inst = len(pd.DataFrame(self.log_test).caseid.unique())
        start_time = (pd.DataFrame(self.log_test)
                      .start_timestamp.min().strftime("%Y-%m-%dT%H:%M:%S.%f+00:00"))
        p_extractor.extract_parameters(num_inst, start_time)
        if p_extractor.is_safe:
            self.process_stats = self.process_stats.merge(
                p_extractor.resource_table[['resource', 'role']],
                on='resource',
                how='left')
            # save parameters
            self.parameters = copy.deepcopy(p_extractor.parameters)
            # print parameters in xml bimp format
            xml.print_parameters(os.path.join(
                self.settings['output'],
                self.settings['file'].split('.')[0]+'.bpmn'),
                os.path.join(self.settings['output'],
                             self.settings['file'].split('.')[0]+'.bpmn'),
                p_extractor.parameters)
        else:
            raise RuntimeError('Parameters extraction error')

    @timeit(rec_name='SIMULATION_EVAL')
    @Decorators.safe_exec
    def simulate(self, **kwargs) -> None:
        reps = self.settings['repetitions']
        cpu_count = multiprocessing.cpu_count()
        w_count =  reps if reps <= cpu_count else cpu_count
        pool = Pool(processes=w_count)
        # Simulate
        args = [(self.settings, rep) for rep in range(reps)]
        p = pool.map_async(self.execute_simulator, args)
        p.wait()
        # Read simulated logs
        args = [(self.settings, self.bpmn, rep) for rep in range(reps)]
        p = pool.map_async(self.read_stats, args)
        p.wait()
        # Evaluate
        args = [(self.settings, self.process_stats, log) for log in p.get()]
        p = pool.map_async(self.evaluate_logs, args)
        p.wait()
        pool.close()
        # Save results
        self.sim_values = list(itertools.chain(*p.get()))
        print(self.sim_values)

    @staticmethod
    def read_stats(args):
        def read(settings, bpmn, rep):
            """Reads the simulation results stats
            Args:
                settings (dict): Path to jar and file names
                rep (int): repetition number
            """
            message = 'Reading log repetition: ' + str(rep+1)
            print(message)
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
            process_graph = gph.create_process_structure(bpmn, verbose=False)
            results_replayer = rpl.LogReplayer(process_graph,
                                               temp.get_traces(),
                                               settings,
                                               source='simulation',
                                               run_num=rep + 1,
                                               verbose=False)
            temp_stats = results_replayer.process_stats
            temp_stats['role'] = temp_stats['resource']
            return temp_stats
        return read(*args)

    @staticmethod
    def evaluate_logs(args):
        def evaluate(settings, process_stats, sim_log):
            """Reads the simulation results stats
            Args:
                settings (dict): Path to jar and file names
                rep (int): repetition number
            """
            # print('Reading repetition:', (rep+1), sep=' ')
            rep = (sim_log.iloc[0].run_num) - 1
            sim_values = list()
            message = 'Evaluating repetition: ' + str(rep+1)
            print(message)
            process_stats = process_stats.append(
                sim_log,
                ignore_index=True,
                sort=False)
            evaluator = sim.SimilarityEvaluator(
                process_stats,
                settings,
                rep)
            metrics = [settings['sim_metric']]
            if 'add_metrics' in settings.keys():
                metrics = list(set(list(settings['add_metrics']) +
                                    metrics))
            for metric in metrics:
                evaluator.measure_distance(metric)
                sim_values.append(evaluator.similarity)
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
            message = 'Executing BIMP Simulations Repetition: ' + str(rep+1)
            print(message)
            args = ['java', '-jar', settings['bimp_path'],
                    os.path.join(settings['output'],
                                  settings['file'].split('.')[0]+'.bpmn'),
                    '-csv',
                    os.path.join(settings['output'], 'sim_data',
                                  settings['file']
                                  .split('.')[0]+'_'+str(rep+1)+'.csv')]
            subprocess.run(args, check=True, stdout=subprocess.PIPE)
        sim_call(*args)

    def mannage_results(self) -> None:
        self.response, measurements = self.define_response(self.is_safe,
                                                           self.sim_values,
                                                           self.settings)
        if self.settings['exec_mode'] in ['optimizer'] and measurements:
            if os.path.getsize(os.path.join('outputs',
                                            self.settings['temp_file'])) > 0:
                sup.create_csv_file(measurements,
                                    os.path.join('outputs',
                                                 self.settings['temp_file']),
                                    mode='a')
            else:
                sup.create_csv_file_header(measurements,
                                           os.path.join(
                                               'outputs',
                                               self.settings['temp_file']))
        elif self.settings['exec_mode'] == 'single':
            print('------ Final results ------')
            [print(k, v, sep=': ') for k, v in self.response.items()
             if k != 'params']
            self.response.pop('params', None)
            if measurements:
                sup.create_csv_file_header(measurements,
                                           os.path.join(
                                               'outputs',
                                               self.settings['temp_file']))

    @staticmethod
    def define_response(is_safe, sim_values, settings):
        response = dict()
        measurements = list()
        data = {'alg_manag': settings['alg_manag'],
                'concurrency': settings['concurrency'],
                'rp_similarity': settings['rp_similarity'],
                'gate_management': settings['gate_management'],
                'output': settings['output']}
        if settings['res_cal_met'] in ['discovered' ,'pool']:
            data['res_cal_met'] = settings['res_cal_met']
            data['res_support'] = settings['res_support']
            data['res_confidence'] = settings['res_confidence']
        if settings['arr_cal_met'] == 'discovered':
            data['arr_support'] = settings['arr_support']
            data['arr_confidence'] = settings['arr_confidence']
        if is_safe:
            similarity = np.mean([x['sim_val'] for x in sim_values
                                  if x['metric'] == settings['sim_metric']])
            response['similarity'] = similarity
            response['params'] = settings
            response['status'] = is_safe if similarity > 0 else False
            response = {**response, **data}
            for sim_val in sim_values:
                measurements.append({
                    **{'similarity': sim_val['sim_val'],
                       'sim_metric': sim_val['metric'],
                       'status': response['status']},
                    **data})
        else:
            response['similarity'] = 0
            response['sim_metric'] = settings['sim_metric']
            response['status'] = is_safe
            response = {**response, **data}
        return response, measurements

    @staticmethod
    def save_times(times, settings):
        times = [{**{'output': settings['output']}, **times}]
        log_file = os.path.join('outputs', 'execution_times.csv')
        if not os.path.exists(log_file):
                open(log_file, 'w').close()
        if os.path.getsize(log_file) > 0:
            sup.create_csv_file(times, log_file, mode='a')
        else:
            sup.create_csv_file_header(times, log_file)

    # def export_canonical_model(self):
    #     ns = {'qbp': "http://www.qbp-simulator.com/Schema201212"}
    #     time_table = etree.tostring(self.parameters['time_table'], pretty_print=True)
    #     time_table = xtd.parse(time_table, process_namespaces=True, namespaces=ns)
    #     self.parameters['time_table'] = time_table
    #     sup.create_json(self.parameters, os.path.join(
    #         self.settings['output'],
    #         self.settings['file'].split('.')[0]+'_canon.json'))


# =============================================================================
# Support methods
# =============================================================================

    def split_timeline(self, size: float, one_ts: bool) -> None:
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
        train, test = splitter.split_log('timeline_contained', size, one_ts)
        total_events = len(self.log.data)
        # Check size and change time splitting method if necesary
        if len(test) < int(total_events*0.1):
            train, test = splitter.split_log('timeline_trace', size, one_ts)
        # Set splits
        key = 'end_timestamp' if one_ts else 'start_timestamp'
        test = pd.DataFrame(test)
        train = pd.DataFrame(train)
        self.log_test = (test.sort_values(key, ascending=True)
                         .reset_index(drop=True).to_dict('records'))
        self.log_train = copy.deepcopy(self.log)
        self.log_train.set_data(train.sort_values(key, ascending=True)
                                .reset_index(drop=True).to_dict('records'))


    @staticmethod
    def get_traces(data, one_timestamp):
        """
        returns the data splitted by caseid and ordered by start_timestamp
        """
        cases = list(set([x['caseid'] for x in data]))
        traces = list()
        for case in cases:
            order_key = 'end_timestamp' if one_timestamp else 'start_timestamp'
            trace = sorted(
                list(filter(lambda x: (x['caseid'] == case), data)),
                key=itemgetter(order_key))
            traces.append(trace)
        return traces

# =============================================================================
# External tools calling
# =============================================================================
    @staticmethod
    def mining_structure(settings):
        """
        Executes splitminer for bpmn structure mining.

        Returns
        -------
        None
            DESCRIPTION.
        """
        print(" -- Mining Process Structure --")
        # Event log file_name
        file_name = settings['file'].split('.')[0]
        input_route = os.path.join(settings['output'], file_name+'.xes')
        sep = ';' if pl.system().lower() == 'windows' else ':'
        # Mining structure definition
        args = ['java', '-cp',
                (settings['miner_path']+sep+os.path.join(
                    'external_tools','splitminer','lib','*')),
                'au.edu.unimelb.services.ServiceProvider',
                'SM2',
                input_route,
                os.path.join(settings['output'], file_name),
                str(settings['concurrency'])]
        subprocess.call(args)

# =============================================================================
# Hyperparameter-optimizer
# =============================================================================


class DiscoveryOptimizer():
    """
    Hyperparameter-optimizer class
    """

    def __init__(self, settings, args):
        """constructor"""
        self.settings = settings
        self.args = args
        self.log = types.SimpleNamespace()
        self.log_train = types.SimpleNamespace()
        self.log_test = types.SimpleNamespace()

    def execute_pipeline(self) -> None:
        exec_times = dict()
        self.read_inputs(log_time=exec_times)
        output_file = sup.file_id(prefix='SE_')
        print('############ Structure optimization ############')
        # Structure optimization
        structure_optimizer = so.StructureOptimizer(
            self.settings,
            self.args,
            copy.deepcopy(self.log_train))
        structure_optimizer.execute_trials()
        struc_model = structure_optimizer.best_output
        best_parms = structure_optimizer.best_parms
        self.settings['alg_manag'] = (
            self.args['alg_manag'][best_parms['alg_manag']])
        self.settings['gate_management'] = (
            self.args['gate_management'][best_parms['gate_management']])
        self.settings['concurrency'] = best_parms['concurrency']
        for key in ['rp_similarity', 'res_dtype', 'arr_dtype', 'res_sup_dis',
                    'res_con_dis', 'arr_support', 'arr_confidence',
                    'res_cal_met', 'arr_cal_met']:
            self.settings.pop(key, None)
        self._test_model(struc_model, output_file)
        print('############ Times optimization ############')
        times_optimizer = to.TimesOptimizer(
            self.settings,
            self.args,
            copy.deepcopy(self.log_train),
            struc_model)
        times_optimizer.execute_trials()
        print('############ Final comparison ############')
        self._test_model(times_optimizer.best_output, output_file, mode='a')
        print("-- End of trial --")

    def _test_model(self, best_output, output_file, mode='w'):
        output_path = os.path.join('outputs', sup.folder_id())
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            os.makedirs(os.path.join(output_path, 'sim_data'))
        self.settings.pop('output', None)
        self.settings['output'] = output_path
        self._modify_simulation_model(
            os.path.join(best_output,
                          self.settings['file'].split('.')[0]+'.bpmn'))
        # self._modify_simulation_model(best_output)
        self._load_model_and_measures()
        self._simulate()
        self.sim_values = pd.DataFrame.from_records(self.sim_values)
        self.sim_values['output'] = output_path
        self.sim_values = pd.pivot_table(self.sim_values, values='sim_val', index=['output'],
                    columns=['metric'], aggfunc=np.mean).reset_index()
        if mode == 'w':
            self.sim_values.to_csv(os.path.join('outputs', output_file),
                                   index=False)
        elif mode == 'a':
            self.sim_values.to_csv(os.path.join('outputs', output_file),
                                   index=False, mode=mode, header=False)
        else:
            raise ValueError(mode)


    @timeit
    def read_inputs(self, **kwargs) -> None:
        # Event log reading
        self.log = lr.LogReader(os.path.join(self.settings['input'],
                                             self.settings['file']),
                                self.settings['read_options'])
        # Time splitting 80-20
        self.split_timeline(0.8,
                            self.settings['read_options']['one_timestamp'])

    def split_timeline(self, size: float, one_ts: bool) -> None:
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
        train, test = splitter.split_log('timeline_contained', size, one_ts)
        total_events = len(self.log.data)
        # Check size and change time splitting method if necesary
        if len(test) < int(total_events*0.1):
            train, test = splitter.split_log('timeline_trace', size, one_ts)
        # Set splits
        key = 'end_timestamp' if one_ts else 'start_timestamp'
        test = pd.DataFrame(test)
        train = pd.DataFrame(train)
        self.log_test = (test.sort_values(key, ascending=True)
                         .reset_index(drop=True).to_dict('records'))
        self.log_train = copy.deepcopy(self.log)
        self.log_train.set_data(train.sort_values(key, ascending=True)
                                .reset_index(drop=True).to_dict('records'))

    def _modify_simulation_model(self, model):
        """Modifies the number of instances of the BIMP simulation model
        to be equal to the number of instances in the testing log"""
        test_log = pd.DataFrame(self.log_test)
        num_inst = len(test_log.caseid.unique())
        # Get minimum date
        start_time = (test_log
                      .start_timestamp
                      .min().strftime("%Y-%m-%dT%H:%M:%S.%f+00:00"))
        mydoc = minidom.parse(model)
        items = mydoc.getElementsByTagName('qbp:processSimulationInfo')
        items[0].attributes['processInstances'].value = str(num_inst)
        items[0].attributes['startDateTime'].value = start_time
        new_model_path = os.path.join(self.settings['output'],
                                      os.path.split(model)[1])
        with open(new_model_path, 'wb') as f:
            f.write(mydoc.toxml().encode('utf-8'))
        f.close()
        return new_model_path

    def _load_model_and_measures(self):
        self.process_stats = pd.DataFrame.from_records(self.log_test)
        self.process_stats['source'] = 'log'
        self.process_stats['run_num'] = 0

    # @timeit(rec_name='SIMULATION_EVAL')
    # @Decorators.safe_exec
    def _simulate(self, **kwargs) -> None:
        reps = self.settings['repetitions']
        cpu_count = multiprocessing.cpu_count()
        w_count =  reps if reps <= cpu_count else cpu_count
        pool = Pool(processes=w_count)
        # Simulate
        args = [(self.settings, rep) for rep in range(reps)]
        p = pool.map_async(self.execute_simulator, args)
        p.wait()
        # Read simulated logs
        args = [(self.settings, rep) for rep in range(reps)]
        p = pool.map_async(self.read_stats, args)
        p.wait()
        # Evaluate
        args = [(self.settings, self.process_stats, log) for log in p.get()]
        p = pool.map_async(self.evaluate_logs, args)
        p.wait()
        pool.close()
        # Save results
        self.sim_values = list(itertools.chain(*p.get()))

    @staticmethod
    def read_stats(args):
        def read(settings, rep):
            """Reads the simulation results stats
            Args:
                settings (dict): Path to jar and file names
                rep (int): repetition number
            """
            message = 'Reading log repetition: ' + str(rep+1)
            print(message)
            path = os.path.join(settings['output'], 'sim_data')
            log_name = settings['file'].split('.')[0]+'_'+str(rep+1)+'.csv'
            rep_results = pd.read_csv(os.path.join(path, log_name),
                                      dtype={'caseid': object})
            rep_results['caseid'] = 'Case' + rep_results['caseid']
            rep_results['run_num'] = rep
            rep_results['source'] = 'simulation'
            rep_results.rename(columns={'resource': 'user'}, inplace=True)
            rep_results['start_timestamp'] =  pd.to_datetime(
                rep_results['start_timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
            rep_results['end_timestamp'] =  pd.to_datetime(
                rep_results['end_timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
            return rep_results
        return read(*args)

    @staticmethod
    def evaluate_logs(args):
        def evaluate(settings, process_stats, sim_log):
            """Reads the simulation results stats
            Args:
                settings (dict): Path to jar and file names
                rep (int): repetition number
            """
            # print('Reading repetition:', (rep+1), sep=' ')
            rep = (sim_log.iloc[0].run_num) - 1
            sim_values = list()
            message = 'Evaluating repetition: ' + str(rep+2)
            print(message)
            process_stats = process_stats.append(
                sim_log,
                ignore_index=True,
                sort=False)
            evaluator = sim.SimilarityEvaluator(
                process_stats,
                settings,
                rep)
            metrics = [settings['sim_metric']]
            if 'add_metrics' in settings.keys():
                metrics = list(set(list(settings['add_metrics']) +
                                    metrics))
            for metric in metrics:
                evaluator.measure_distance(metric)
                sim_values.append(evaluator.similarity)
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
            message = 'Executing BIMP Simulations Repetition: ' + str(rep+1)
            print(message)
            args = ['java', '-jar', settings['bimp_path'],
                    os.path.join(settings['output'],
                                  settings['file'].split('.')[0]+'.bpmn'),
                    '-csv',
                    os.path.join(settings['output'], 'sim_data',
                                  settings['file']
                                  .split('.')[0]+'_'+str(rep+1)+'.csv')]
            subprocess.run(args, check=True, stdout=subprocess.PIPE)
        sim_call(*args)


    @staticmethod
    def get_traces(data, one_timestamp):
        """
        returns the data splitted by caseid and ordered by start_timestamp
        """
        cases = list(set([x['caseid'] for x in data]))
        traces = list()
        for case in cases:
            order_key = 'end_timestamp' if one_timestamp else 'start_timestamp'
            trace = sorted(
                list(filter(lambda x: (x['caseid'] == case), data)),
                key=itemgetter(order_key))
            traces.append(trace)
        return traces
