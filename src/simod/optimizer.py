import copy
import os
import shutil
from xml.dom import minidom

import pandas as pd

from . import support_utils as sup
from .analyzers import sim_evaluator as sim
from .cli_formatter import print_section, print_asset, print_subsection, print_notice, print_step
from .common_routines import extract_structure_parameters, extract_process_graph, simulate, \
    evaluate_logs_with_add_metrics, remove_outliers
from .configuration import Configuration, MiningAlgorithm
from .readers import log_splitter as ls
from .readers.log_reader import LogReader
from .structure_optimizer import StructureOptimizer
from .times_optimizer import TimesOptimizer
from .writers import xml_writer
from .writers.model_serialization import serialize_model


class Optimizer:
    """Hyper-parameter Optimizer class"""
    log_train: LogReader
    log_test: LogReader

    def __init__(self, settings):
        self.settings = settings
        self.settings_global: Configuration = settings['gl']
        self.settings_structure: Configuration = settings['strc']
        self.settings_time: Configuration = settings['tm']
        self.best_params = dict()
        self.log = LogReader(self.settings_global.log_path, self.settings_global.read_options)
        if not os.path.exists(self.settings_global.output.parent):
            os.makedirs(self.settings_global.output.parent)

    def execute_pipeline(self, discover_model: bool = True) -> None:
        print_notice(f'Log path: {self.settings_global.log_path}')
        self.split_and_set_log_buckets(0.8, self.settings['gl'].read_options.one_timestamp)

        print_step('Removing outliers from the training partition')
        self._remove_outliers()

        strctr_optimizer_file_name = None
        strctr_optimizer_temp_output = None

        # optional model discovery if model is not provided
        if discover_model:
            print_section('Model Discovery and Parameters Extraction')
            # mining the structure
            strctr_optimizer = StructureOptimizer(
                self.settings_structure, copy.deepcopy(self.log_train), discover_model=discover_model)
            strctr_optimizer.execute_trials()
            # redefining local variables
            self._redefine_best_params_after_structure_optimization(strctr_optimizer)
            strctr_optimizer_file_name = strctr_optimizer.file_name
            strctr_optimizer_temp_output = strctr_optimizer.temp_output
            model_path = os.path.join(strctr_optimizer.best_output, self.settings_global.project_name + '.bpmn')
        else:
            print_section('Parameters Extraction')
            model_path = self._extract_parameters_and_rewrite_model()

        print_section('Times Optimization')
        # mining times
        times_optimizer = TimesOptimizer(
            self.settings_global, self.settings_time, copy.deepcopy(self.log_train), model_path)
        times_optimizer.execute_trials()
        # redefining local variables
        self._redefine_best_params_after_times_optimization(times_optimizer)

        print_section('Final Comparison')
        output_file = sup.file_id(prefix='SE_')
        self._test_model(
            times_optimizer.best_output, output_file, strctr_optimizer_file_name, times_optimizer.file_name)
        self._export_canonical_model(times_optimizer.best_output)
        if strctr_optimizer_temp_output:
            shutil.rmtree(strctr_optimizer_temp_output)
        shutil.rmtree(times_optimizer.temp_output)
        print_asset(f"Output folder is at {self.settings_global.output}")

    def _remove_outliers(self):
        # removing outliers
        log_train_df = pd.DataFrame(self.log_train.data)
        log_train_df = remove_outliers(log_train_df)
        # converting data back to LogReader format
        key = 'end_timestamp' if self.settings['gl'].read_options.one_timestamp else 'start_timestamp'
        self.log_train.set_data(log_train_df.sort_values(key, ascending=True).reset_index(drop=True).to_dict('records'))

    def _redefine_best_params_after_structure_optimization(self, strctr_optimizer):
        # receiving mined results and redefining local variables

        best_parms = strctr_optimizer.best_parms

        self.settings_global.gate_management = self.settings_structure.gate_management[
            best_parms['gate_management']]
        self.best_params['gate_management'] = self.settings_global.gate_management
        # best structure mining parameters
        if self.settings_global.mining_alg in [MiningAlgorithm.SM1, MiningAlgorithm.SM3]:
            self.settings_global.epsilon = best_parms['epsilon']
            self.settings_global.eta = best_parms['eta']
            self.settings_global.and_prior = self.settings_structure.and_prior[best_parms['and_prior']]
            self.settings_global.or_rep = self.settings_structure.or_rep[best_parms['or_rep']]
            self.best_params['epsilon'] = best_parms['epsilon']
            self.best_params['eta'] = best_parms['eta']
            self.best_params['and_prior'] = self.settings_global.and_prior
            self.best_params['or_rep'] = self.settings_global.or_rep
        elif self.settings_global.mining_alg is MiningAlgorithm.SM2:
            self.settings_global.concurrency = best_parms['concurrency']
            self.best_params['concurrency'] = best_parms['concurrency']
        for key in ['rp_similarity', 'res_dtype', 'arr_dtype', 'res_sup_dis', 'res_con_dis', 'arr_support',
                    'arr_confidence', 'res_cal_met', 'arr_cal_met']:  # TODO: seems like this is unnecessary
            self.settings.pop(key, None)
        return

    def _redefine_best_params_after_times_optimization(self, times_optimizer):
        # redefining parameters after times optimizer
        if times_optimizer.best_parms['res_cal_met'] == 1:
            self.best_params['res_dtype'] = self.settings_time.res_dtype[times_optimizer.best_parms['res_dtype']]
        else:
            self.best_params['res_support'] = times_optimizer.best_parms['res_support']
            self.best_params['res_confidence'] = times_optimizer.best_parms['res_confidence']
        if times_optimizer.best_parms['arr_cal_met'] == 1:
            self.best_params['arr_dtype'] = self.settings_time.res_dtype[times_optimizer.best_parms['arr_dtype']]
        else:
            self.best_params['arr_support'] = (times_optimizer.best_parms['arr_support'])
            self.best_params['arr_confidence'] = (times_optimizer.best_parms['arr_confidence'])

    def _extract_parameters_and_rewrite_model(self):
        # extracting parameters
        model_path = self.settings_global.model_path
        process_graph = extract_process_graph(model_path)
        parameters = extract_structure_parameters(
            settings=self.settings_structure, process_graph=process_graph, log=self.log_train, model_path=model_path)

        # TODO: usually, self.log_valdn is used, but we don't have it here, in Discoverer,
        #  self.log_test is used instead. What whould be used here?
        num_inst = len(self.log_test.caseid.unique())
        start_time = self.log_test.start_timestamp.min().strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
        parameters.instances = num_inst
        parameters.start_time = start_time

        # copying the original model and rewriting it with extracted parameters
        if not os.path.exists(self.settings_global.output):
            os.makedirs(self.settings_global.output)
        bpmn_path = os.path.join(self.settings_global.output, os.path.basename(model_path))
        shutil.copy(model_path, bpmn_path)
        xml_writer.print_parameters(bpmn_path, bpmn_path, parameters.__dict__)
        model_path = bpmn_path

        # simulation
        sim_data_path = os.path.join(self.settings_global.output, 'sim_data')
        if not os.path.exists(sim_data_path):
            os.makedirs(sim_data_path)
        _ = simulate(self.settings_global, parameters.process_stats, self.log_test)
        return model_path

    def _test_model(self, best_output, output_file, opt_strf, opt_timf):
        output_path = os.path.join(self.settings_global.output.parent, sup.folder_id())
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            os.makedirs(os.path.join(output_path, 'sim_data'))
        # self.settings_global.__dict__.pop('output', None)
        self.settings_global.output = output_path
        self._modify_simulation_model(os.path.join(best_output, self.settings_global.project_name + '.bpmn'))
        self._load_model_and_measures()
        print_subsection("Simulation")
        self.sim_values = simulate(self.settings_global, self.process_stats, self.log_test,
                                   evaluate_fn=evaluate_logs_with_add_metrics)
        self.sim_values = pd.DataFrame.from_records(self.sim_values)
        self.sim_values['output'] = output_path
        self.sim_values.to_csv(os.path.join(output_path, output_file), index=False)
        if opt_strf:
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
        splitter = ls.LogSplitter(pd.DataFrame(self.log.data))
        train, test = splitter.split_log('timeline_contained', size, one_ts)
        total_events = len(self.log.data)
        # Check size and change time splitting method if necesary
        if len(test) < int(total_events * 0.1):
            train, test = splitter.split_log('timeline_trace', size, one_ts)
        # Set splits
        key = 'end_timestamp' if one_ts else 'start_timestamp'
        test = pd.DataFrame(test)
        train = pd.DataFrame(train)
        self.log_test = test.sort_values(key, ascending=True).reset_index(drop=True)
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

    @staticmethod
    def evaluate_logs(args):
        def evaluate(settings: Configuration, process_stats, sim_log):
            """Reads the simulation results stats
            Args:
                settings (dict): Path to jar and file names
                rep (int): repetition number
            """
            # print('Reading repetition:', (rep+1), sep=' ')
            rep = sim_log.iloc[0].run_num
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
