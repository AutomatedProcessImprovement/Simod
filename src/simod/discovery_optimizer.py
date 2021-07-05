import copy
import itertools
import multiprocessing
import os
import shutil
import subprocess
import time
import types
from multiprocessing import Pool
from operator import itemgetter
from xml.dom import minidom

import analyzers.sim_evaluator as sim
import pandas as pd
import readers.log_reader as lr
import readers.log_splitter as ls
import utils.support as sup
from tqdm import tqdm
from utils.support import timeit

from .structure_optimizer import StructureOptimizer
from .times_optimizer import TimesOptimizer
from .writers.model_serialization import serialize_model


class DiscoveryOptimizer:
    """
    Hyperparameter-optimizer class
    """

    def __init__(self, settings):
        """constructor"""
        self.settings = settings
        self.best_params = dict()
        self.log = types.SimpleNamespace()
        self.log_train = types.SimpleNamespace()
        self.log_test = types.SimpleNamespace()
        if not os.path.exists('outputs'):
            os.makedirs('outputs')

    def execute_pipeline(self) -> None:
        exec_times = dict()
        self.read_inputs(log_time=exec_times)
        output_file = sup.file_id(prefix='SE_')
        print('############ Structure optimization ############')
        # Structure optimization
        structure_optimizer = StructureOptimizer(
            {**self.settings['gl'], **self.settings['strc']},
            copy.deepcopy(self.log_train))
        structure_optimizer.execute_trials()
        struc_model = structure_optimizer.best_output
        best_parms = structure_optimizer.best_parms
        self.settings['gl']['alg_manag'] = (
            self.settings['strc']['alg_manag'][best_parms['alg_manag']])
        self.best_params['alg_manag'] = self.settings['gl']['alg_manag']
        self.settings['gl']['gate_management'] = (
            self.settings['strc']['gate_management'][best_parms['gate_management']])
        self.best_params['gate_management'] = self.settings['gl']['gate_management']
        # best structure mining parameters
        if self.settings['gl']['mining_alg'] in ['sm1', 'sm3']:
            self.settings['gl']['epsilon'] = best_parms['epsilon']
            self.settings['gl']['eta'] = best_parms['eta']
            self.best_params['epsilon'] = best_parms['epsilon']
            self.best_params['eta'] = best_parms['eta']
        elif self.settings['gl']['mining_alg'] == 'sm2':
            self.settings['gl']['concurrency'] = best_parms['concurrency']
            self.best_params['concurrency'] = best_parms['concurrency']
        for key in ['rp_similarity', 'res_dtype', 'arr_dtype', 'res_sup_dis',
                    'res_con_dis', 'arr_support', 'arr_confidence',
                    'res_cal_met', 'arr_cal_met']:
            self.settings.pop(key, None)
        # self._test_model(struc_model, output_file)
        print('############ Times optimization ############')
        times_optimizer = TimesOptimizer(
            self.settings['gl'],
            self.settings['tm'],
            copy.deepcopy(self.log_train),
            struc_model)
        times_optimizer.execute_trials()
        # Discovery parameters
        if times_optimizer.best_parms['res_cal_met'] == 1:
            self.best_params['res_dtype'] = (
                self.settings['tm']['res_dtype']
                [times_optimizer.best_parms['res_dtype']])
        else:
            self.best_params['res_support'] = (
                times_optimizer.best_parms['res_support'])
            self.best_params['res_confidence'] = (
                times_optimizer.best_parms['res_confidence'])
        if times_optimizer.best_parms['arr_cal_met'] == 1:
            self.best_params['arr_dtype'] = (
                self.settings['tm']['res_dtype']
                [times_optimizer.best_parms['arr_dtype']])
        else:
            self.best_params['arr_support'] = (
                times_optimizer.best_parms['arr_support'])
            self.best_params['arr_confidence'] = (
                times_optimizer.best_parms['arr_confidence'])

        print('############ Final comparison ############')
        self._test_model(times_optimizer.best_output,
                         output_file,
                         structure_optimizer.file_name,
                         times_optimizer.file_name)
        self._export_canonical_model(times_optimizer.best_output)
        shutil.rmtree(structure_optimizer.temp_output)
        shutil.rmtree(times_optimizer.temp_output)
        print("-- End of trial --")
        print(f"Output folder is at {self.settings['gl']['output']}")

    def _test_model(self, best_output, output_file, opt_strf, opt_timf):
        output_path = os.path.join('outputs', sup.folder_id())
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            os.makedirs(os.path.join(output_path, 'sim_data'))
        self.settings['gl'].pop('output', None)
        self.settings['gl']['output'] = output_path
        self._modify_simulation_model(
            os.path.join(best_output,
                         self.settings['gl']['file'].split('.')[0] + '.bpmn'))
        self._load_model_and_measures()
        self._simulate()
        self.sim_values = pd.DataFrame.from_records(self.sim_values)
        self.sim_values['output'] = output_path
        self.sim_values.to_csv(os.path.join(output_path, output_file),
                               index=False)
        shutil.move(opt_strf, output_path)
        shutil.move(opt_timf, output_path)

    def _export_canonical_model(self, best_output):
        print("Model file location:", os.path.join(
            self.settings['gl']['output'],
            self.settings['gl']['file'].split('.')[0] + '.bpmn'))
        canonical_model = serialize_model(os.path.join(
            self.settings['gl']['output'],
            self.settings['gl']['file'].split('.')[0] + '.bpmn'))
        # Users in rol data
        resource_table = pd.read_pickle(
            os.path.join(best_output, 'resource_table.pkl'))
        user_rol = dict()
        for key, group in resource_table.groupby('role'):
            user_rol[key] = list(group.resource)
        canonical_model['rol_user'] = user_rol
        # Json creation
        self.best_params = {k: str(v) for k, v in self.best_params.items()}
        canonical_model['discovery_parameters'] = self.best_params
        sup.create_json(canonical_model, os.path.join(
            self.settings['gl']['output'],
            self.settings['gl']['file'].split('.')[0] + '_canon.json'))

    @timeit
    def read_inputs(self, **kwargs) -> None:
        # Event log reading
        self.log = lr.LogReader(os.path.join(self.settings['gl']['input'],
                                             self.settings['gl']['file']),
                                self.settings['gl']['read_options'])
        # Time splitting 80-20
        self.split_timeline(0.8,
                            self.settings['gl']['read_options']['one_timestamp'])

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
        if len(test) < int(total_events * 0.1):
            train, test = splitter.split_log('timeline_trace', size, one_ts)
        # Set splits
        key = 'end_timestamp' if one_ts else 'start_timestamp'
        test = pd.DataFrame(test)
        train = pd.DataFrame(train)
        self.log_test = (test.sort_values(key, ascending=True)
                         .reset_index(drop=True))
        self.log_train = copy.deepcopy(self.log)
        self.log_train.set_data(train.sort_values(key, ascending=True)
                                .reset_index(drop=True).to_dict('records'))

    def _modify_simulation_model(self, model):
        """Modifies the number of instances of the BIMP simulation model
        to be equal to the number of instances in the testing log"""
        num_inst = len(self.log_test.caseid.unique())
        # Get minimum date
        start_time = (self.log_test
                      .start_timestamp
                      .min().strftime("%Y-%m-%dT%H:%M:%S.%f+00:00"))
        mydoc = minidom.parse(model)
        items = mydoc.getElementsByTagName('qbp:processSimulationInfo')
        items[0].attributes['processInstances'].value = str(num_inst)
        items[0].attributes['startDateTime'].value = start_time
        new_model_path = os.path.join(self.settings['gl']['output'],
                                      os.path.split(model)[1])
        with open(new_model_path, 'wb') as f:
            f.write(mydoc.toxml().encode('utf-8'))
        f.close()
        return new_model_path

    def _load_model_and_measures(self):
        self.process_stats = self.log_test
        self.process_stats['source'] = 'log'
        self.process_stats['run_num'] = 0

    def _simulate(self, **kwargs) -> None:

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

        reps = self.settings['gl']['repetitions']
        cpu_count = multiprocessing.cpu_count()
        w_count = reps if reps <= cpu_count else cpu_count
        pool = Pool(processes=w_count)
        # Simulate
        args = [(self.settings['gl'], rep) for rep in range(reps)]
        p = pool.map_async(self.execute_simulator, args)
        pbar_async(p, 'simulating:')
        # Read simulated logs
        args = [(self.settings['gl'], rep) for rep in range(reps)]
        p = pool.map_async(self.read_stats, args)
        pbar_async(p, 'reading simulated logs:')
        # Evaluate
        args = [(self.settings['gl'], self.process_stats, log) for log in p.get()]
        if len(self.log_test.caseid.unique()) > 1000:
            pool.close()
            results = [self.evaluate_logs(arg) for arg in tqdm(args, 'evaluating results:')]
            # Save results
            self.sim_values = list(itertools.chain(*results))
        else:
            p = pool.map_async(self.evaluate_logs, args)
            pbar_async(p, 'evaluating results:')
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
            # message = 'Reading log repetition: ' + str(rep+1)
            # print(message)
            path = os.path.join(settings['output'], 'sim_data')
            log_name = settings['file'].split('.')[0] + '_' + str(rep + 1) + '.csv'
            rep_results = pd.read_csv(os.path.join(path, log_name),
                                      dtype={'caseid': object})
            rep_results['caseid'] = 'Case' + rep_results['caseid']
            rep_results['run_num'] = rep
            rep_results['source'] = 'simulation'
            rep_results.rename(columns={'resource': 'user'}, inplace=True)
            rep_results['start_timestamp'] = pd.to_datetime(
                rep_results['start_timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
            rep_results['end_timestamp'] = pd.to_datetime(
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
            rep = (sim_log.iloc[0].run_num)
            sim_values = list()
            evaluator = sim.SimilarityEvaluator(
                process_stats,
                sim_log,
                settings,
                max_cases=1000)
            metrics = [settings['sim_metric']]
            if 'add_metrics' in settings.keys():
                metrics = list(set(list(settings['add_metrics']) +
                                   metrics))
            for metric in metrics:
                evaluator.measure_distance(metric)
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
            # message = 'Executing BIMP Simulations Repetition: ' + str(rep+1)
            # print(message)
            args = ['java', '-jar', settings['bimp_path'],
                    os.path.join(settings['output'],
                                 settings['file'].split('.')[0] + '.bpmn'),
                    '-csv',
                    os.path.join(settings['output'], 'sim_data',
                                 settings['file']
                                 .split('.')[0] + '_' + str(rep + 1) + '.csv')]
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
