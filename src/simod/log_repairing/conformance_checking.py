from operator import itemgetter

import utils.support as sup

from .traces_alignment import TracesAligner
from .traces_replacement import replacement
from ..extraction.log_replayer import LogReplayer


def evaluate_alignment(process_graph, log, settings):
    traces = log.get_traces()
    test_replayer = LogReplayer(
        process_graph,
        traces,
        settings,
        msg='evaluating train partition conformance:')
    conformant = get_traces(test_replayer.conformant_traces, False)
    not_conformant = get_traces(test_replayer.not_conformant_traces, False)
    # ------conformance percentage before repair------------------
    print_stats(log, conformant, traces)
    if settings['alg_manag'] == 'replacement':
        log.set_data(replacement(conformant,
                                 not_conformant,
                                 log,
                                 settings))
    elif settings['alg_manag'] == 'repair':
        repaired_event_log = list()
        [repaired_event_log.extend(x) for x in conformant]
        trace_aligner = TracesAligner(log, not_conformant, settings)
        repaired_event_log.extend(trace_aligner.aligned_traces)
        log.set_data(repaired_event_log)

    elif settings['alg_manag'] == 'removal':
        ref_conformant = list()
        for trace in conformant:
            ref_conformant.extend(trace)
        log.set_data(ref_conformant)
    # ------conformance percentage after repair------------------
    aligned_traces = log.get_traces()
    test_replayer = LogReplayer(
        process_graph,
        aligned_traces,
        settings,
        msg='evaluating conformance after ' + settings['alg_manag'] + ':')
    conformant = get_traces(test_replayer.conformant_traces, False)
    print_stats(log, conformant, aligned_traces)


def print_stats(log, conformant, traces):
    print('complete traces:', str(len(traces)),
          ', events:', str(len(log.data)), sep=' ')
    print('conformance percentage:',
          str(sup.ffloat((len(conformant) / len(traces)) * 100, 2)) + '%', sep=' ')


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
