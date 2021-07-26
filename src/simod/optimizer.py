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
import utils.support as sup
from pm4py.objects.log.importer.xes import importer as xes_importer
from tqdm import tqdm

from .cli_formatter import *
from .configuration import Configuration, MiningAlgorithm
from .decorators import timeit
from .readers import log_reader as lr
from .readers import log_splitter as ls
from .structure_optimizer import StructureOptimizer
from .times_optimizer import TimesOptimizer
from .writers.model_serialization import serialize_model


class Optimizer:
    """ Hyperparameter-optimizer class"""

    def __init__(self, settings):
        self.settings = settings
        self.settings_global: Configuration = settings['gl']
        self.settings_structure: Configuration = settings['strc']
        self.settings_time: Configuration = settings['tm']
        self.best_params = dict()
        self.log = types.SimpleNamespace()
        self.log_train = types.SimpleNamespace()
        self.log_test = types.SimpleNamespace()
        if not os.path.exists('outputs'):
            os.makedirs('outputs')

    def execute_pipeline(self, structure_optimizer=StructureOptimizer) -> None:
        print_section('Log Parsing')
        exec_times = dict()

        print_section('Structure Optimization')

        print_step('Preparing log buckets')
        self.set_log(log_time=exec_times)
        self.split_and_set_log_buckets(0.8, self.settings['gl'].read_options.one_timestamp)

        strctr_optimizer = structure_optimizer(self.settings_structure, copy.deepcopy(self.log_train))
        strctr_optimizer.execute_trials()
        struc_model = strctr_optimizer.best_output
        best_parms = strctr_optimizer.best_parms
        self.settings_global.alg_manag = self.settings_structure.alg_manag[best_parms['alg_manag']]
        self.best_params['alg_manag'] = self.settings_global.alg_manag
        self.settings_global.gate_management = self.settings_structure.gate_management[best_parms['gate_management']]
        self.best_params['gate_management'] = self.settings_global.gate_management
        # best structure mining parameters
        if self.settings_global.mining_alg in [MiningAlgorithm.SM1, MiningAlgorithm.SM3]:
            self.settings_global.epsilon = best_parms['epsilon']
            self.settings_global.eta = best_parms['eta']
            self.best_params['epsilon'] = best_parms['epsilon']
            self.best_params['eta'] = best_parms['eta']
        elif self.settings_global.mining_alg is MiningAlgorithm.SM2:
            self.settings_global.concurrency = best_parms['concurrency']
            self.best_params['concurrency'] = best_parms['concurrency']
        for key in ['rp_similarity', 'res_dtype', 'arr_dtype', 'res_sup_dis', 'res_con_dis', 'arr_support',
                    'arr_confidence', 'res_cal_met', 'arr_cal_met']:
            self.settings.pop(key, None)

        print_section('Times Optimization')
        times_optimizer = TimesOptimizer(self.settings_global, self.settings_time, copy.deepcopy(self.log_train),
                                         struc_model)
        times_optimizer.execute_trials()
        # Discovery parameters
        if times_optimizer.best_parms['res_cal_met'] == 1:
            self.best_params['res_dtype'] = (self.settings_time.res_dtype[times_optimizer.best_parms['res_dtype']])
        else:
            self.best_params['res_support'] = (times_optimizer.best_parms['res_support'])
            self.best_params['res_confidence'] = (times_optimizer.best_parms['res_confidence'])
        if times_optimizer.best_parms['arr_cal_met'] == 1:
            self.best_params['arr_dtype'] = (self.settings_time.res_dtype[times_optimizer.best_parms['arr_dtype']])
        else:
            self.best_params['arr_support'] = (times_optimizer.best_parms['arr_support'])
            self.best_params['arr_confidence'] = (times_optimizer.best_parms['arr_confidence'])

        print_section('Final Comparison')
        output_file = sup.file_id(prefix='SE_')
        self._test_model(times_optimizer.best_output, output_file, strctr_optimizer.file_name,
                         times_optimizer.file_name)
        self._export_canonical_model(times_optimizer.best_output)
        shutil.rmtree(strctr_optimizer.temp_output)
        shutil.rmtree(times_optimizer.temp_output)
        print_asset(f"Output folder is at {self.settings_global.output}")

    def _test_model(self, best_output, output_file, opt_strf, opt_timf):
        output_path = os.path.join('outputs', sup.folder_id())
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            os.makedirs(os.path.join(output_path, 'sim_data'))
        # self.settings_global.__dict__.pop('output', None)
        self.settings_global.output = output_path
        self._modify_simulation_model(os.path.join(best_output, self.settings_global.project_name + '.bpmn'))
        self._load_model_and_measures()
        self._simulate()
        self.sim_values = pd.DataFrame.from_records(self.sim_values)
        self.sim_values['output'] = output_path
        self.sim_values.to_csv(os.path.join(output_path, output_file), index=False)
        shutil.move(opt_strf, output_path)
        shutil.move(opt_timf, output_path)

    def _export_canonical_model(self, best_output):
        print_asset(f"Model file location: "
                    f"{os.path.join(self.settings_global.output, self.settings_global.project_name + '.bpmn')}")
        canonical_model = serialize_model(
            os.path.join(self.settings_global.output, self.settings_global.project_name + '.bpmn'))
        # Users in rol data
        resource_table = pd.read_pickle(os.path.join(best_output, 'resource_table.pkl'))
        user_rol = dict()
        for key, group in resource_table.groupby('role'):
            user_rol[key] = list(group.resource)
        canonical_model['rol_user'] = user_rol
        # Json creation
        self.best_params = {k: str(v) for k, v in self.best_params.items()}
        canonical_model['discovery_parameters'] = self.best_params
        sup.create_json(canonical_model, os.path.join(
            self.settings_global.output, self.settings_global.project_name + '_canon.json'))

    @timeit
    def set_log(self, **kwargs) -> None:
        # Event log reading
        global_config: Configuration = self.settings['gl']
        self.log = lr.LogReader(os.path.join(global_config.input, global_config.file), global_config.read_options)

    def split_and_set_log_buckets(self, size: float, one_ts: bool) -> None:
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
        self.log_test = (test.sort_values(key, ascending=True).reset_index(drop=True))
        self.log_train = copy.deepcopy(self.log)
        self.log_train.set_data(train.sort_values(key, ascending=True).reset_index(drop=True).to_dict('records'))

    def _modify_simulation_model(self, model):
        """Modifies the number of instances of the BIMP simulation model
        to be equal to the number of instances in the testing log"""
        num_inst = len(self.log_test.caseid.unique())
        # Get minimum date
        start_time = self.log_test.start_timestamp.min().strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
        mydoc = minidom.parse(model)
        items = mydoc.getElementsByTagName('qbp:processSimulationInfo')
        items[0].attributes['processInstances'].value = str(num_inst)
        items[0].attributes['startDateTime'].value = start_time
        new_model_path = os.path.join(self.settings_global.output, os.path.split(model)[1])
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

        reps = self.settings_global.repetitions
        cpu_count = multiprocessing.cpu_count()
        w_count = reps if reps <= cpu_count else cpu_count
        pool = Pool(processes=w_count)
        # Simulate
        args = [(self.settings_global, rep) for rep in range(reps)]
        p = pool.map_async(self.execute_simulator, args)
        pbar_async(p, 'simulating:')
        # Read simulated logs
        args = [(self.settings_global, rep) for rep in range(reps)]
        p = pool.map_async(self.read_stats, args)
        pbar_async(p, 'reading simulated logs:')
        # Evaluate
        args = [(self.settings_global, self.process_stats, log) for log in p.get()]
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
        def read(settings: Configuration, rep):
            """Reads the simulation results stats
            Args:
                settings (dict): Path to jar and file names
                rep (int): repetition number
            """
            # message = 'Reading log repetition: ' + str(rep+1)
            # print(message)
            path = os.path.join(settings.output, 'sim_data')
            log_name = settings.project_name + '_' + str(rep + 1) + '.csv'
            rep_results = pd.read_csv(os.path.join(path, log_name), dtype={'caseid': object})
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
        def evaluate(settings: Configuration, process_stats, sim_log):
            """Reads the simulation results stats
            Args:
                settings (dict): Path to jar and file names
                rep (int): repetition number
            """
            # print('Reading repetition:', (rep+1), sep=' ')
            rep = (sim_log.iloc[0].run_num)
            sim_values = list()
            evaluator = sim.SimilarityEvaluator(process_stats, sim_log, settings, max_cases=1000)
            metrics = [settings.sim_metric]
            if 'add_metrics' in settings.__dict__.keys():
                metrics = list(set(list(settings.add_metrics) + metrics))
            for metric in metrics:
                evaluator.measure_distance(metric)
                sim_values.append({**{'run_num': rep}, **evaluator.similarity})
            return sim_values

        return evaluate(*args)

    @staticmethod
    def execute_simulator(args):
        def sim_call(settings: Configuration, rep):
            """Executes BIMP Simulations.
            Args:
                settings (dict): Path to jar and file names
                rep (int): repetition number
            """
            # message = 'Executing BIMP Simulations Repetition: ' + str(rep+1)
            # print(message)
            args = ['java', '-jar', settings.bimp_path,
                    os.path.join(settings.output, settings.project_name + '.bpmn'),
                    '-csv',
                    os.path.join(settings.output, 'sim_data', settings.project_name + '_' + str(rep + 1) + '.csv')]
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
            trace = sorted(list(filter(lambda x: (x['caseid'] == case), data)), key=itemgetter(order_key))
            traces.append(trace)
        return traces
