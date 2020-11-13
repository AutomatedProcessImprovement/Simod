# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 14:38:41 2019

@author: Manuel Camargo
"""

from support_modules.log_repairing import log_replayer as rpl
from support_modules.log_repairing import traces_replacement as rep
from support_modules.log_repairing import traces_alignment as tal

import utils.support as sup


def evaluate_alignment(process_graph, log, settings):
    traces = log.get_traces()
    conformant, not_conformant = rpl.replay(process_graph, traces)
    # ------conformance percentage before repair------------------
    print_stats(log, conformant, traces)
    if settings['alg_manag'] == 'replacement':
        log.set_data(rep.replacement(conformant,
                                     not_conformant,
                                     log,
                                     settings))
    elif settings['alg_manag'] == 'repair':
        repaired_event_log = list()
        [repaired_event_log.extend(x) for x in conformant]
        trace_aligner = tal.TracesAligner(log, not_conformant, settings)
        repaired_event_log.extend(trace_aligner.aligned_traces)
        log.set_data(repaired_event_log)

    elif settings['alg_manag'] == 'removal':
        ref_conformant = list()
        for trace in conformant:
            ref_conformant.extend(trace)
        log.set_data(ref_conformant)
    # ------conformance percentage after repair------------------
    aligned_traces = log.get_traces()
    conformant, not_conformant = rpl.replay(process_graph, aligned_traces)
    print_stats(log, conformant, traces)


def print_stats(log, conformant, traces):
    print('Num. traces:', str(len(traces)), sep=' ')
    print('Num. events:', str(len(log.data)), sep=' ')
    print('Conformance percentage:',
          str(sup.ffloat((len(conformant)/len(traces)) * 100, 2))+'%', sep=' ')
