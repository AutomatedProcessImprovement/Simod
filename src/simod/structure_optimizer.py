import copy
import itertools
import math
import multiprocessing
import os
import random
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from hyperopt import Trials, hp, fmin, STATUS_OK, STATUS_FAIL
from hyperopt import tpe
from tqdm import tqdm

from . import support_utils as sup
from .analyzers import sim_evaluator as sim
from .cli_formatter import print_message, print_subsection
from .common_routines import execute_simulator, mine_resources, extract_structure_parameters, split_timeline
from .configuration import Configuration, MiningAlgorithm, Metric
from .decorators import timeit, safe_exec_with_values_and_status
from .readers.log_reader import LogReader
from .structure_miner import StructureMiner
from .writers import xml_writer as xml, xes_writer as xes


class StructureOptimizer:
    """Hyperparameter-optimizer class"""

    def __init__(self, settings: Configuration, log: LogReader, **kwargs):
        self.space = self.define_search_space(settings)

        self.log = log
        self._split_timeline(0.8, settings.read_options.one_timestamp)

        self.org_log = copy.deepcopy(log)
        self.org_log_train = copy.deepcopy(self.log_train)
        self.org_log_valdn = copy.deepcopy(self.log_valdn)
        # Load settings
        self.settings = settings
        self.temp_output = os.path.join(os.path.dirname(__file__), '../../', 'outputs', sup.folder_id())
        if not os.path.exists(self.temp_output):
            os.makedirs(self.temp_output)
        self.file_name = os.path.join(self.temp_output, sup.file_id(prefix='OP_'))
        # Results file
        if not os.path.exists(self.file_name):
            open(self.file_name, 'w').close()
        # Trials object to track progress
        self.bayes_trials = Trials()
        self.best_output = None
        self.best_parms = dict()
        self.best_similarity = 0

    @staticmethod
    def define_search_space(settings: Configuration):
        var_dim = {'gate_management': hp.choice('gate_management', settings.gate_management)}
        if settings.mining_alg in [MiningAlgorithm.SM1, MiningAlgorithm.SM3]:
            var_dim['epsilon'] = hp.uniform('epsilon', settings.epsilon[0], settings.epsilon[1])
            var_dim['eta'] = hp.uniform('eta', settings.eta[0], settings.eta[1])
            var_dim['and_prior'] = hp.choice('and_prior', ['true', 'false'])
            var_dim['or_rep'] = hp.choice('or_rep', ['true', 'false'])
        elif settings.mining_alg is MiningAlgorithm.SM2:
            var_dim['concurrency'] = hp.uniform('concurrency', settings.concurrency[0], settings.concurrency[1])
        csettings = copy.deepcopy(settings.__dict__)
        for key in var_dim.keys():
            csettings.pop(key, None)
        space = {**var_dim, **csettings}
        return space

    def execute_trials(self):
        parameters = mine_resources(self.settings)
        self.log_train = copy.deepcopy(self.org_log_train)

        def exec_pipeline(trial_stg: Configuration):
            print_subsection("Trial")
            print_message(f'train split: {len(pd.DataFrame(self.log_train.data).caseid.unique())}, '
                          f'validation split: {len(self.log_valdn.caseid.unique())}')
            # Vars initialization
            status = STATUS_OK
            exec_times = dict()
            sim_values = []
            # Path redefinition
            rsp = self._temp_path_redef(trial_stg, status=status, log_time=exec_times)
            status = rsp['status']
            trial_stg = rsp['values'] if status == STATUS_OK else trial_stg
            # Structure mining
            rsp = self._mine_structure(Configuration(**trial_stg), status=status, log_time=exec_times)
            status = rsp['status']
            # Parameters extraction
            rsp = self._extract_parameters(trial_stg, rsp['values'], copy.deepcopy(parameters), status=status,
                                           log_time=exec_times)
            status = rsp['status']
            # Simulate model
            rsp = self._simulate(trial_stg, self.log_valdn, status=status, log_time=exec_times)
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
            return rsp

        # Optimize
        best = fmin(fn=exec_pipeline,
                    space=self.space,
                    algo=tpe.suggest,
                    max_evals=self.settings.max_eval_s,
                    trials=self.bayes_trials,
                    show_progressbar=False)
        # Save results
        try:
            results = pd.DataFrame(self.bayes_trials.results).sort_values('loss', ascending=bool)
            self.best_output = results[results.status == 'ok'].head(1).iloc[0].output
            self.best_parms = best
            self.best_similarity = results[results.status == 'ok'].head(1).iloc[0].loss
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
        # Create customized event-log for the external tools
        xes.XesWriter(self.log_train, Configuration(**settings))
        return settings

    @timeit(rec_name='MINING_STRUCTURE')
    @safe_exec_with_values_and_status
    def _mine_structure(self, settings: Configuration, **kwargs) -> None:
        structure_miner = StructureMiner(settings, self.log_train)
        structure_miner.execute_pipeline()
        if structure_miner.is_safe:
            return [structure_miner.bpmn, structure_miner.process_graph]
        else:
            raise RuntimeError('Mining Structure error')

    @timeit(rec_name='EXTRACTING_PARAMS')
    @safe_exec_with_values_and_status
    def _extract_parameters(self, settings: Configuration, structure, parameters, **kwargs) -> None:
        if isinstance(settings, dict):
            settings = Configuration(**settings)

        _, process_graph = structure
        num_inst = len(self.log_valdn.caseid.unique())  # TODO: why do we use log_valdn instead of log_train?
        start_time = self.log_valdn.start_timestamp.min().strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")  # getting minimum date

        model_path = Path(os.path.join(settings.output, settings.project_name + '.bpmn'))
        structure_parameters = extract_structure_parameters(
            settings=settings, process_graph=process_graph, log=self.log_train, model_path=model_path)

        parameters = {**parameters, **{'resource_pool': structure_parameters.resource_pool,
                                       'time_table': structure_parameters.time_table,
                                       'arrival_rate': structure_parameters.arrival_rate,
                                       'sequences': structure_parameters.sequences,
                                       'elements_data': structure_parameters.elements_data,
                                       'instances': num_inst,
                                       'start_time': start_time}}
        bpmn_path = os.path.join(settings.output, settings.project_name + '.bpmn')
        xml.print_parameters(bpmn_path, bpmn_path, parameters)

        self.log_valdn.rename(columns={'user': 'resource'}, inplace=True)
        self.log_valdn['source'] = 'log'
        self.log_valdn['run_num'] = 0
        self.log_valdn['role'] = 'SYSTEM'
        self.log_valdn = self.log_valdn[~self.log_valdn.task.isin(['Start', 'End'])]

    @timeit(rec_name='SIMULATION_EVAL')
    @safe_exec_with_values_and_status
    def _simulate(self, settings: Configuration, data, **kwargs) -> list:
        if isinstance(settings, dict):
            settings = Configuration(**settings)

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

        reps = settings.repetitions
        cpu_count = multiprocessing.cpu_count()
        w_count = reps if reps <= cpu_count else cpu_count
        pool = Pool(processes=w_count)
        # Simulate
        args = [(settings, rep) for rep in range(reps)]
        p = pool.map_async(execute_simulator, args)
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
    def read_stats(args):
        def read(settings: Configuration, rep):
            """Reads the simulation results stats
            Args:
                settings (dict): Path to jar and file names
                rep (int): repetition number
            """
            m_settings = dict()
            m_settings['output'] = settings.output
            column_names = {'resource': 'user'}
            m_settings['read_options'] = settings.read_options
            m_settings['read_options'].timeformat = '%Y-%m-%d %H:%M:%S.%f'
            m_settings['read_options'].column_names = column_names
            m_settings['project_name'] = settings.project_name
            temp = LogReader(os.path.join(m_settings['output'], 'sim_data',
                                          m_settings['project_name'] + '_' + str(rep + 1) + '.csv'),
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
        def evaluate(settings: Configuration, data, sim_log):
            """Reads the simulation results stats
            Args:
                settings (dict): Path to jar and file names
                rep (int): repetition number
            """
            rep = (sim_log.iloc[0].run_num)
            sim_values = list()
            evaluator = sim.SimilarityEvaluator(data, sim_log, settings, max_cases=1000)
            evaluator.measure_distance(Metric.DL)
            sim_values.append({**{'run_num': rep}, **evaluator.similarity})
            return sim_values

        return evaluate(*args)

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

    def _define_response(self, settings, status, sim_values, **kwargs) -> dict:
        response = dict()
        measurements = list()
        data = {'gate_management': settings['gate_management'], 'output': settings['output']}
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
                measurements.append({
                    **{'similarity': sim_val['sim_val'],
                       'sim_metric': sim_val['metric'],
                       'status': response['status']},
                    **data})
        else:
            response['status'] = status
            measurements.append({**{'similarity': 0,
                                    'sim_metric': Metric.DL,
                                    'status': response['status']},
                                 **data})
        if os.path.getsize(self.file_name) > 0:
            sup.create_csv_file(measurements, self.file_name, mode='a')
        else:
            sup.create_csv_file_header(measurements, self.file_name)
        return response

    def _split_timeline(self, size: float, one_ts: bool) -> None:
        """
        Split an event log dataframe by time to perform split-validation. preferred method time splitting removing
        incomplete traces. If the testing set is smaller than the 10% of the log size the second method is sort by
        traces start and split taking the whole traces no matter if they are contained in the timeframe or not

        Parameters
        ----------
        size: float, validation percentage.
        one_ts: bool, Support only one timestamp.
        """
        # Split log data
        # splitter = ls.LogSplitter(self.log)
        # train, validation = splitter.split_log('timeline_contained', size, one_ts)
        # total_events = len(self.log)
        # # Check size and change time splitting method if necessary
        # if len(validation) < int(total_events * 0.1):
        #     train, validation = splitter.split_log('timeline_trace', size, one_ts)
        # # Set splits
        # key = 'end_timestamp' if one_ts else 'start_timestamp'
        # validation = pd.DataFrame(validation)
        # train = pd.DataFrame(train)
        # # If the log is big sample train partition
        # train = self._sample_log(train)

        train, validation, key = split_timeline(self.log, size, one_ts)
        train = self._sample_log(train)

        # Save partitions
        self.log_valdn = validation.sort_values(key, ascending=True).reset_index(drop=True)
        self.log_train = copy.deepcopy(self.log)
        self.log_train.set_data(train.sort_values(key, ascending=True).reset_index(drop=True).to_dict('records'))

    @staticmethod
    def _sample_log(train):

        def sample_size(p_size, c_level, c_interval):
            """
            p_size : population size.
            c_level : confidence level.
            c_interval : confidence interval.
            """
            c_level_constant = {50: .67, 68: .99, 90: 1.64, 95: 1.96, 99: 2.57}
            Z = 0.0
            p = 0.5
            e = c_interval / 100.0
            N = p_size
            n_0 = 0.0
            n = 0.0
            # DEVIATIONS FOR THAT CONFIDENCE LEVEL
            Z = c_level_constant[c_level]
            # CALC SAMPLE SIZE
            n_0 = ((Z ** 2) * p * (1 - p)) / (e ** 2)
            # ADJUST SAMPLE SIZE FOR FINITE POPULATION
            n = n_0 / (1 + ((n_0 - 1) / float(N)))
            return int(math.ceil(n))  # THE SAMPLE SIZE

        cases = list(train.caseid.unique())
        if len(cases) > 1000:
            sample_sz = sample_size(len(cases), 95.0, 3.0)
            scases = random.sample(cases, sample_sz)
            train = train[train.caseid.isin(scases)]
        return train
