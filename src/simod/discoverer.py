# class Discoverer:
#     _settings: Configuration
#     _output_file: str
#     _log: LogReaderWriter
#     _log_train: LogReaderWriter
#     _log_test: LogReaderWriter
#     _sim_values: list = []
#
#     def __init__(self, settings: Configuration):
#         self._settings = settings
#         self._output_file = sup.file_id(prefix='SE_')
#
#     def run(self):
#         print_notice(f'Log path: {self._settings.log_path}')
#         self._read_inputs()
#         self._temp_path_creation()
#         self._preprocess()
#         self._mine_structure()
#         self._extract_parameters()
#         self._simulate()
#         self._manage_results()
#         self._export_canonical_model()
#         print_asset(f"Output folder is at {self._settings.output}")
#
#     def _preprocess(self):
#         processor = Preprocessor(self._settings)
#         self._settings = processor.run()
#
#     def _read_inputs(self):
#         print_section("Log Parsing")
#         # Event log reading
#         self._log = LogReaderWriter(self._settings.log_path, column_names=DEFAULT_XES_COLUMNS)
#         # Time splitting 80-20
#         self._split_timeline(0.8)
#
#     def _temp_path_creation(self):
#         print_section("Log Customization")
#         # Output folder creation
#         if not os.path.exists(self._settings.output):
#             os.makedirs(self._settings.output)
#             os.makedirs(os.path.join(self._settings.output, 'sim_data'))
#         # Create customized event-log for the external tools
#         output_path = self._settings.output / (self._settings.project_name + '.xes')
#         self._settings.log_path = output_path
#         self._log_train.write_xes(output_path)
#         reformat_timestamps(output_path, output_path)
#
#     def _mine_structure(self):
#         """Mines structure from the event log, saves it as a BPMN file and updates the process graph."""
#         print_section("Process Structure Mining")
#
#         if self._settings.model_path is None:
#             # Structure mining
#             print_step("Mining the model structure")
#
#             model_path = (self._settings.output / (self._settings.project_name + '.bpmn')).absolute()
#
#             settings = StructureMinerSettings(
#                 mining_algorithm=self._settings.structure_mining_algorithm,
#                 epsilon=self._settings.epsilon,
#                 eta=self._settings.eta,
#                 concurrency=self._settings.concurrency,
#                 and_prior=self._settings.and_prior,
#                 or_rep=self._settings.or_rep,
#             )
#
#             _ = StructureMiner(settings, xes_path=self._settings.log_path, output_model_path=model_path)
#         else:
#             # Taking provided structure
#             print_step("Copying the model")
#
#             shutil.copy(self._settings.model_path.absolute(), self._settings.output.absolute())
#             model_path = (self._settings.output / self._settings.model_path.name).absolute()
#
#         bpmn_reader = BPMNReaderWriter(model_path)
#         self.process_graph = bpmn_reader.as_graph()
#
#     def _extract_parameters(self):
#         print_section("Simulation Parameters Mining")
#
#         time_table, resource_pool, resource_table = \
#             discover_timetables_with_resource_pools(self._log_train, self._settings)
#
#         log_train_df = pd.DataFrame(self._log_train.data)
#         arrival_rate = inter_arrival_distribution.discover(self.process_graph, log_train_df, self._settings.pdef_method)
#
#         bpmn_path = os.path.join(self._settings.output, self._settings.project_name + '.bpmn')
#         bpmn_graph = BPMNGraph.from_bpmn_path(Path(bpmn_path))
#         traces = self._log_train.get_traces()
#         sequences = gateway_probabilities.discover(traces, bpmn_graph)
#
#         self.process_stats = log_train_df.merge(resource_table[['resource', 'role']],
#                                                 left_on='user', right_on='resource', how='left')
#         elements_data = TaskEvaluator(
#             self.process_graph, self.process_stats, resource_pool, self._settings.pdef_method).elements_data
#
#         # rewriting the model file
#         num_inst = len(log_train_df.caseid.unique())
#         start_time = log_train_df.start_timestamp.min().strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
#         parameters = {
#             'instances': num_inst,
#             'start_time': start_time,
#             'resource_pool': resource_pool,
#             'time_table': time_table,
#             'arrival_rate': arrival_rate,
#             'sequences': sequences,
#             'elements_data': elements_data
#         }
#         self.parameters = copy.deepcopy(parameters)
#
#         xml.print_parameters(bpmn_path, bpmn_path, parameters)
#
#     def _simulate(self):
#         print_section("Simulation")
#         # TODO: change this to the new simulation function
#         # self._sim_values = simulate(self._settings, self.process_stats, evaluate_fn=evaluate_logs_with_add_metrics)
#         raise NotImplementedError
#
#     def _manage_results(self):
#         self._sim_values = pd.DataFrame.from_records(self._sim_values)
#         self._sim_values['output'] = self._settings.output
#         self._sim_values.to_csv(os.path.join(self._settings.output, self._output_file), index=False)
#
#     def _export_canonical_model(self):
#         ns = {'qbp': QBP_NAMESPACE_URI}
#         time_table = etree.tostring(self.parameters['time_table'], pretty_print=True)
#         time_table = xtd.parse(time_table, process_namespaces=True, namespaces=ns)
#         self.parameters['time_table'] = time_table
#         self.parameters['discovery_parameters'] = self._filter_dic_params(self._settings)
#         sup.create_json(self.parameters,
#                         os.path.join(self._settings.output, self._settings.project_name + '_canon.json'))
#
#     @staticmethod
#     def _filter_dic_params(settings: Configuration) -> dict:
#         best_params = dict()
#         best_params['gate_management'] = str(settings.gate_management)
#         best_params['rp_similarity'] = str(settings.rp_similarity)
#         # best structure mining parameters
#         if settings.structure_mining_algorithm in [StructureMiningAlgorithm.SPLIT_MINER_1,
#                                                    StructureMiningAlgorithm.SPLIT_MINER_3]:
#             best_params['epsilon'] = str(settings.epsilon)
#             best_params['eta'] = str(settings.eta)
#         elif settings.structure_mining_algorithm == StructureMiningAlgorithm.SPLIT_MINER_2:
#             best_params['concurrency'] = str(settings.concurrency)
#         if settings.res_cal_met == CalendarType.DEFAULT:  # TODO: do we need this?
#             best_params['res_dtype'] = settings.res_dtype.__str__().split('.')[1]
#         else:
#             best_params['res_support'] = str(settings.res_support)
#             best_params['res_confidence'] = str(settings.res_confidence)
#         if settings.arr_cal_met == CalendarType.DEFAULT:  # TODO: do we need this?
#             best_params['arr_dtype'] = settings.res_dtype.__str__().split('.')[1]
#         else:
#             best_params['arr_support'] = str(settings.arr_support)
#             best_params['arr_confidence'] = str(settings.arr_confidence)
#         return best_params
#
#     def _split_timeline(self, size: float):
#         train, test = self._log.split_timeline(size)
#         key = 'start_timestamp'
#         self._log_test = test.sort_values(key, ascending=True).reset_index(drop=True)
#         self._log_train = copy.deepcopy(self._log)
#         self._log_train.set_data(train.sort_values(key, ascending=True).reset_index(drop=True).to_dict('records'))
