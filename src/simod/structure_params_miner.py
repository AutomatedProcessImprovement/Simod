from .configuration import Configuration, PDFMethod, CalculationMethod, DataType
from .decorators import safe_exec
from .extraction.gateways_probabilities import GatewaysEvaluator
from .extraction.interarrival_definition import InterArrivalEvaluator
from .extraction.log_replayer import LogReplayer
from .extraction.schedule_tables import TimeTablesCreator
from .extraction.tasks_evaluator import TaskEvaluator


class StructureParametersMiner():
    """
    This class extracts all the BPS parameters
    """
    def __init__(self, log, bpmn, process_graph, settings: Configuration):
        self.log = log
        self.bpmn = bpmn
        self.process_graph = process_graph
        self.settings = settings
        # inter-arrival times and durations by default mean an exponential
        # 'manual', 'automatic', 'semi-automatic', 'default'
        self.settings.pdef_method = PDFMethod.DEFAULT
        # self.settings['rp_similarity'] = 0.5
        self.process_stats = list()
        self.parameters = dict()
        self.conformant_traces = list()
        self.is_safe = True

    def extract_parameters(self, num_inst, start_time, resource_pool) -> None:
        """
        main method for parameters extraction
        """
        self.is_safe = self._replay_process(is_safe=self.is_safe)
        self.is_safe = self._mine_interarrival(is_safe=self.is_safe)
        self.is_safe = self._mine_gateways_probabilities(is_safe=self.is_safe)
        self.is_safe = self._process_tasks(resource_pool, is_safe=self.is_safe)

        # TODO: Num of test partition
        self.parameters['instances'] = num_inst
        self.parameters['start_time'] = start_time

    @safe_exec
    def _replay_process(self, **kwargs) -> None:
        """
        Process replaying
        """
        replayer = LogReplayer(self.process_graph, self.log.get_traces(), self.settings,
                               msg='reading conformant training traces:')
        self.process_stats = replayer.process_stats
        self.process_stats['role'] = 'SYSTEM'
        self.conformant_traces = replayer.conformant_traces

    # @safe_exec
    @staticmethod
    def mine_resources(settings, log) -> None:
        """
        Analysing resource pool LV917 or 247
        """
        parameters = dict()
        settings.res_cal_met = CalculationMethod.DEFAULT
        settings.res_dtype = DataType.DT247
        settings.arr_cal_met = CalculationMethod.DEFAULT
        settings.arr_dtype = DataType.DT247
        ttcreator = TimeTablesCreator(settings)
        args = {'res_cal_met': settings.res_cal_met, 'arr_cal_met': settings.arr_cal_met}

        if not isinstance(args['res_cal_met'], CalculationMethod):
            args['res_cal_met'] = CalculationMethod.from_str(args['res_cal_met'])
        if not isinstance(args['arr_cal_met'], CalculationMethod):
            args['arr_cal_met'] = CalculationMethod.from_str(args['arr_cal_met'])

        ttcreator.create_timetables(args)
        resource_pool = [{'id': 'QBP_DEFAULT_RESOURCE', 'name': 'SYSTEM',
                          'total_amount': '100000', 'costxhour': '20',
                          'timetable_id': ttcreator.res_ttable_name['arrival']}]

        parameters['resource_pool'] = resource_pool
        parameters['time_table'] = ttcreator.time_table
        return parameters

    @safe_exec
    def _mine_interarrival(self, **kwargs) -> None:
        """
        Calculates the inter-arrival rate
        """
        inter_evaluator = InterArrivalEvaluator(self.process_graph, self.conformant_traces, self.settings)
        self.parameters['arrival_rate'] = inter_evaluator.dist

    @safe_exec
    def _mine_gateways_probabilities(self, **kwargs) -> None:
        """
        Gateways probabilities 1=Historical, 2=Random, 3=Equiprobable
        """
        gevaluator = GatewaysEvaluator(self.process_graph, self.settings.gate_management)
        sequences = gevaluator.probabilities
        for seq in sequences:
            seq['elementid'] = self.bpmn.find_sequence_id(seq['gatewayid'],
                                                          seq['out_path_id'])
        self.parameters['sequences'] = sequences

    @safe_exec
    def _process_tasks(self, resource_pool, **kwargs) -> None:
        """
        Tasks id information
        """
        tevaluator = TaskEvaluator(self.process_graph, self.process_stats, resource_pool, self.settings)
        self.parameters['elements_data'] = tevaluator.elements_data
