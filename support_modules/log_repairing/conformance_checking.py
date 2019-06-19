# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 14:38:41 2019

@author: Manuel Camargo
"""

from support_modules.log_repairing import log_replayer as rpl
from support_modules.log_repairing import traces_replacement as rep
from support_modules.log_repairing import traces_alignment as tal

from support_modules import support as sup


def evaluate_alignment(process_graph, log, settings):
    conformant, not_conformant = rpl.replay(process_graph, log.get_traces())
    #------conformance percentage before repair------------------
    print_stats(log, conformant)
    if settings['alg_manag'] == 'replacement':
        log.set_data(rep.replacement(conformant, not_conformant, log))
    elif settings['alg_manag'] == 'repairment':
        log.set_data(tal.align_traces(log, settings))
    elif settings['alg_manag'] == 'removal':
        ref_conformant = list()
        for trace in conformant:
            ref_conformant.extend(trace)
        log.set_data(ref_conformant)
    #------conformance percentage after repair------------------
    conformant, not_conformant = rpl.replay(process_graph, log.get_traces())
    print_stats(log, conformant)
        
def print_stats(log, conformant):
    traces = log.get_traces()
    print('Num. traces:', str(len(traces)), sep=' ')
    print('Num. events:', str(len(log.data)), sep=' ')
    print('Conformance percentage:', str(sup.ffloat((len(conformant)/len(traces)) * 100,2)) + '%', sep=' ')
   
