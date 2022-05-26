from simod.cli_formatter import print_step, print_notice
from simod.configuration import GateManagement
from simod.replayer_datatypes import BPMNGraph


def __compute_sequence_flow_frequencies(log_traces: list, bpmn_graph: BPMNGraph):
    flow_arcs_frequency = dict()
    for trace in log_traces:
        task_sequence = [event['task'] for event in trace]
        bpmn_graph.replay_trace(task_sequence, flow_arcs_frequency)
    return flow_arcs_frequency


def discover(log_traces: list, bpmn_graph: BPMNGraph) -> list:
    print_step('Mining gateway probabilities')
    arcs_frequencies = __compute_sequence_flow_frequencies(log_traces, bpmn_graph)
    gateways_branching = bpmn_graph.compute_branching_probability(arcs_frequencies)

    sequences = []
    for gateway_id in gateways_branching:
        for seqflow_id in gateways_branching[gateway_id]:
            probability = gateways_branching[gateway_id][seqflow_id]
            sequences.append({'elementid': seqflow_id, 'prob': probability})

    return sequences


def discover_with_gateway_management(
        log_traces: list, bpmn_graph: BPMNGraph, gate_management: GateManagement) -> list:
    if isinstance(gate_management, list) and len(gate_management) >= 1:
        print_notice(
            f'A list of gateway management options was provided: {gate_management}, taking the first option: {gate_management[0]}')
        gate_management = gate_management[0]

    print_step(f'Mining gateway probabilities with {gate_management}')
    if gate_management is GateManagement.EQUIPROBABLE:
        gateways_branching = bpmn_graph.compute_branching_probability_alternative_equiprobable()
    elif gate_management is GateManagement.DISCOVERY:
        arcs_frequencies = __compute_sequence_flow_frequencies(log_traces, bpmn_graph)
        gateways_branching = bpmn_graph.compute_branching_probability_alternative_discovery(arcs_frequencies)
    else:
        raise Exception(
            f'Only GatewayManagement.DISCOVERY and .EQUIPROBABLE are supported, got {gate_management}, {type(gate_management)}')

    sequences = []
    for gateway_id in gateways_branching:
        for seqflow_id in gateways_branching[gateway_id]:
            probability = gateways_branching[gateway_id][seqflow_id]
            sequences.append({'elementid': seqflow_id, 'prob': probability})

    return sequences
