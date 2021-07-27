import os
from dataclasses import dataclass

from .cli_formatter import print_step
from .configuration import Configuration, PDFMethod
from .decorators import safe_exec_with_values_and_status, timeit
from .parameter_extraction import Operator, TasksProcessor, InterArrivalMiner, \
    ParameterExtractionOutput, ParameterExtractionInput
from .parameter_extraction import Pipeline
from .stochastic_miner_datatypes import BPMNGraph
from .structure_optimizer import StructureOptimizer, LogReplayerForStructureOptimizer, \
    ResourceMinerForStructureOptimizer
from .writers import xml_writer


class StructureOptimizerForStochasticProcessMiner(StructureOptimizer):
    bpmn_graph: BPMNGraph

    def __init__(self, settings: Configuration, log):
        super().__init__(settings, log)

        print_step('Parsing the given model')
        self.bpmn_graph = BPMNGraph.from_bpmn_path(self.settings.model_path)

    @timeit(rec_name='EXTRACTING_PARAMS')
    @safe_exec_with_values_and_status
    def _extract_parameters(self, settings: dict, structure, parameters, **kwargs) -> None:
        if isinstance(settings, dict):
            settings = Configuration(**settings)

        bpmn, process_graph = structure
        num_inst = len(self.log_valdn.caseid.unique())
        start_time = self.log_valdn.start_timestamp.min().strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")  # getting minimum date

        settings.pdef_method = PDFMethod.DEFAULT
        input = ParameterExtractionInputForStochasticMiner(
            log_traces=self.log_train.get_traces(), log_traces_raw=self.log_train.get_raw_traces(), bpmn=bpmn,
            bpmn_graph=self.bpmn_graph, process_graph=process_graph, settings=settings)
        output = ParameterExtractionOutput()
        output.process_stats['role'] = 'SYSTEM'  # TODO: what is this for?
        structure_mining_pipeline = Pipeline(input=input, output=output)
        structure_mining_pipeline.set_pipeline([
            LogReplayerForStructureOptimizer,
            ResourceMinerForStructureOptimizer,
            InterArrivalMiner,
            GatewayProbabilitiesMinerForStochasticMiner,
            TasksProcessor
        ])
        structure_mining_pipeline.execute()

        parameters = {**parameters, **{'resource_pool': output.resource_pool,
                                       'time_table': output.time_table,
                                       'arrival_rate': output.arrival_rate,
                                       'sequences': output.sequences,
                                       'elements_data': output.elements_data,
                                       'instances': num_inst,
                                       'start_time': start_time}}
        bpmn_path = os.path.join(settings.output, settings.project_name + '.bpmn')
        xml_writer.print_parameters(bpmn_path, bpmn_path, parameters)

        self.log_valdn.rename(columns={'user': 'resource'}, inplace=True)
        self.log_valdn['source'] = 'log'
        self.log_valdn['run_num'] = 0
        self.log_valdn['role'] = 'SYSTEM'
        self.log_valdn = self.log_valdn[~self.log_valdn.task.isin(['Start', 'End'])]


# Parameters Extraction Implementations: For Stochastic Process Miner

@dataclass
class ParameterExtractionInputForStochasticMiner(ParameterExtractionInput):
    log_traces_raw: list = None
    bpmn_graph: BPMNGraph = None


class GatewayProbabilitiesMinerForStochasticMiner(Operator):
    input: ParameterExtractionInputForStochasticMiner
    output: ParameterExtractionOutput

    def __init__(self, input: ParameterExtractionInputForStochasticMiner,
                 output: ParameterExtractionOutput):
        self.input = input
        self.output = output
        self._execute()

    def _execute(self):
        print_step('Mining gateway probabilities')

        arcs_frequencies = self._compute_sequence_flow_frequencies()
        gateways_branching = self.input.bpmn_graph.compute_branching_probability(arcs_frequencies)

        sequences = []
        for gateway_id in gateways_branching:
            for seqflow_id in gateways_branching[gateway_id]:
                probability = gateways_branching[gateway_id][seqflow_id]
                sequences.append({'elementid': seqflow_id, 'prob': probability})

        self.output.sequences = sequences

    def _compute_sequence_flow_frequencies(self):
        flow_arcs_frequency = dict()
        for trace in self.input.log_traces_raw:
            task_sequence = list()
            for event in trace:
                task_name = event['task']  # original: concept:name
                state = event['event_type'].lower()  # original: lifecycle:transition
                if state in ["start", "assign"]:
                    task_sequence.append(task_name)
            self.input.bpmn_graph.replay_trace(task_sequence, flow_arcs_frequency)
        return flow_arcs_frequency
