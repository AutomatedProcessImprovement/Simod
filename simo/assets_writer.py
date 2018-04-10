# -*- coding: utf-8 -*-
import support as sup
import networkx as nx
import numpy as np

def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)

def export_not_conformed_traces(not_conformed_traces,assets_output ):
    not_conformed_traces_output = assets_output + chr(92) + 'log_not_conformed.csv'
    index=list()
    for trace in not_conformed_traces:
        [index.append(x) for x in trace]
    sup.create_csv_file_header(index, not_conformed_traces_output)

def export_tasks_statistics(process_graph,assets_output):
    log_stats_output = assets_output + chr(92) + 'log_tasks_stats_output.csv'
    tasks = list(filter(lambda x: process_graph.node[x]['type'] =='task',nx.nodes(process_graph)))
    index=list()
    for task in tasks:
        index.append(dict(task_name=process_graph.node[task]['name'],avg_proc=mean(process_graph.node[task]['processing_times']),
        max_proc=max(process_graph.node[task]['processing_times'], default=0),min_proc=min(process_graph.node[task]['processing_times'], default=0),
        avg_wait=mean(process_graph.node[task]['waiting_times']),max_wait=max(process_graph.node[task]['waiting_times'], default=0),
        min_wait=min(process_graph.node[task]['multi_tasking'], default=0), avg_multi=mean(process_graph.node[task]['multi_tasking']),
        max_multi=max(process_graph.node[task]['multi_tasking'], default=0), min_multi=min(process_graph.node[task]['multi_tasking'], default=0)))
    sup.create_csv_file_header(index, log_stats_output)

def export_process_statistics(process_stats, process_graph, assets_output):
    log_stats_output = assets_output + chr(92) + 'log_process_stats_output.csv'
    total_processing, total_waiting, total_multitasking = list(), list(), list()
    for x in process_stats:
        total_processing.append(x['total_processing'])
        total_waiting.append(x['total_waiting'])
        total_multitasking.append(x['total_multitasking'])
    index=list()
    index.append(dict(
        total_proc=sum(total_processing),avg_proc=mean(total_processing), max_proc=max(total_processing, default=0),min_proc=min(total_processing, default=0),
        total_wait=sum(total_waiting),avg_wait=mean(total_waiting),max_wait=max(total_waiting, default=0), min_wait=min(total_waiting, default=0),
        total_multitasking=sum(total_multitasking), avg_multitasking=mean(total_multitasking),max_multitasking=max(total_multitasking, default=0), min_multitasking=min(total_multitasking, default=0)))
    sup.create_csv_file_header(index, log_stats_output)
