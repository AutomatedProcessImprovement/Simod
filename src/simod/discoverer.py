import copy
import itertools
import multiprocessing
import os
import subprocess
import time
import traceback
import types
from multiprocessing import Pool
from operator import itemgetter

import analyzers.sim_evaluator as sim
import pandas as pd
import readers.log_reader as lr
import readers.log_splitter as ls
import utils.support as sup
import xmltodict as xtd
from lxml import etree
from tqdm import tqdm
from utils.support import timeit

from .extraction.log_replayer import LogReplayer
from .extraction.parameter_extraction import ParameterMiner
from .structure_miner import StructureMiner
from .writers import xml_writer as xml, xes_writer as xes


class Discoverer:
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
                        traceback.print_exc()
                        is_safe = False
                return is_safe

            return safety_check

    def __init__(self, settings):
        """constructor"""
        self.settings = settings

        self.log = types.SimpleNamespace()
        self.log_train = types.SimpleNamespace()
        self.log_test = types.SimpleNamespace()

        self.sim_values = list()
        self.response = dict()
        # self.parameters = dict()
        self.is_safe = True
        self.output_file = sup.file_id(prefix='SE_')

    def execute_pipeline(self, can=False) -> None:
        exec_times = dict()
        self.is_safe = self.read_inputs(
            log_time=exec_times, is_safe=self.is_safe)
        self.is_safe = self.temp_path_creation(
            log_time=exec_times, is_safe=self.is_safe)
        self.is_safe = self.mine_structure(
            log_time=exec_times, is_safe=self.is_safe)
        self.is_safe = self.replay_process(
            log_time=exec_times, is_safe=self.is_safe)
        self.is_safe = self.extract_parameters(
            log_time=exec_times, is_safe=self.is_safe)
        self.is_safe = self.simulate(
            log_time=exec_times, is_safe=self.is_safe)
        self.mannage_results()
        self.save_times(exec_times, self.settings)
        self.is_safe = self.export_canonical_model(is_safe=self.is_safe)
        print("-- End of trial --")
        print(f"Output folder is at {self.settings['output']}")

    @timeit(rec_name='READ_INPUTS')
    @Decorators.safe_exec
    def read_inputs(self, **kwargs) -> None:
        # Event log reading
        self.log = lr.LogReader(os.path.join(self.settings['logfile']),
                                self.settings['read_options'])
        # Time splitting 80-20
        self.split_timeline(0.8,
                            self.settings['read_options']['one_timestamp'])

    @timeit(rec_name='PATH_DEF')
    @Decorators.safe_exec
    def temp_path_creation(self, **kwargs) -> None:
        # Output folder creation
        if not os.path.exists(self.settings['output']):
            os.makedirs(self.settings['output'])
            os.makedirs(os.path.join(self.settings['output'], 'sim_data'))
        # Create customized event-log for the external tools
        xes.XesWriter(self.log_train, self.settings)

    @timeit(rec_name='MINING_STRUCTURE')
    @Decorators.safe_exec
    def mine_structure(self, **kwargs) -> None:
        print(self.settings)
        structure_miner = StructureMiner(self.settings, self.log_train)
        structure_miner.execute_pipeline()
        if structure_miner.is_safe:
            self.bpmn = structure_miner.bpmn
            self.process_graph = structure_miner.process_graph
        else:
            raise RuntimeError('Mining Structure error')

    @timeit(rec_name='REPLAY_PROCESS')
    @Decorators.safe_exec
    def replay_process(self) -> None:
        """
        Process replaying
        """
        replayer = LogReplayer(self.process_graph,
                               self.log_train.get_traces(),
                               self.settings,
                               msg='reading conformant training traces')
        self.process_stats = replayer.process_stats
        self.conformant_traces = replayer.conformant_traces

    @timeit(rec_name='EXTRACTION')
    @Decorators.safe_exec
    def extract_parameters(self, **kwargs) -> None:
        print("-- Mining Simulation Parameters --")
        p_extractor = ParameterMiner(self.log_train,
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
                self.settings['file'].split('.')[0] + '.bpmn'),
                os.path.join(self.settings['output'],
                             self.settings['file'].split('.')[0] + '.bpmn'),
                p_extractor.parameters)
        else:
            raise RuntimeError('Parameters extraction error')

    @timeit(rec_name='SIMULATION_EVAL')
    @Decorators.safe_exec
    def simulate(self, **kwargs) -> None:

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

        reps = self.settings['repetitions']
        cpu_count = multiprocessing.cpu_count()
        w_count = reps if reps <= cpu_count else cpu_count
        pool = Pool(processes=w_count)
        # Simulate
        args = [(self.settings, rep) for rep in range(reps)]
        p = pool.map_async(self.execute_simulator, args)
        pbar_async(p, 'simulating:')
        # Read simulated logs
        args = [(self.settings, rep) for rep in range(reps)]
        p = pool.map_async(self.read_stats, args)
        pbar_async(p, 'reading simulated logs:')
        # Evaluate
        args = [(self.settings, self.process_stats, log) for log in p.get()]
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

    def mannage_results(self) -> None:
        self.sim_values = pd.DataFrame.from_records(self.sim_values)
        self.sim_values['output'] = self.settings['output']
        self.sim_values.to_csv(os.path.join(self.settings['output'],
                                            self.output_file),
                               index=False)

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

    @Decorators.safe_exec
    def export_canonical_model(self, **kwargs):
        ns = {'qbp': "http://www.qbp-simulator.com/Schema201212"}
        time_table = etree.tostring(self.parameters['time_table'],
                                    pretty_print=True)
        time_table = xtd.parse(time_table,
                               process_namespaces=True,
                               namespaces=ns)
        self.parameters['time_table'] = time_table
        self.parameters['discovery_parameters'] = self.filter_dic_params(
            self.settings)
        sup.create_json(self.parameters, os.path.join(
            self.settings['output'],
            self.settings['file'].split('.')[0] + '_canon.json'))

    @staticmethod
    def filter_dic_params(settings):
        best_params = dict()
        best_params['alg_manag'] = settings['alg_manag']
        best_params['gate_management'] = settings['gate_management']
        best_params['rp_similarity'] = str(settings['rp_similarity'])
        # best structure mining parameters
        if settings['mining_alg'] in ['sm1', 'sm3']:
            best_params['epsilon'] = str(settings['epsilon'])
            best_params['eta'] = str(settings['eta'])
        elif settings['mining_alg'] == 'sm2':
            best_params['concurrency'] = str(settings['concurrency'])
        if settings['res_cal_met'] == 'default':
            best_params['res_dtype'] = settings['res_dtype']
        else:
            best_params['res_support'] = str(settings['res_support'])
            best_params['res_confidence'] = str(settings['res_confidence'])
        if settings['arr_cal_met'] == 'default':
            best_params['arr_dtype'] = settings['res_dtype']
        else:
            best_params['arr_support'] = str(settings['arr_support'])
            best_params['arr_confidence'] = str(settings['arr_confidence'])
        return best_params

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
