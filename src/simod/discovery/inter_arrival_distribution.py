import itertools

import numpy as np
import pandas as pd
from networkx import DiGraph
from tqdm import tqdm

from simod.discovery.pdf_finder import DistributionFinder
from simod.configuration import PDFMethod


def discover(
        process_graph: DiGraph,
        log: pd.DataFrame,
        pdef_method: PDFMethod = PDFMethod.AUTOMATIC) -> dict:
    tasks = __analyze_first_tasks(process_graph)
    inter_arrival_times = __mine_interarrival_time(log, tasks)
    return __define_interarrival_distribution(inter_arrival_times, pdef_method)


def __analyze_first_tasks(process_graph: DiGraph) -> list:
    """
    Extracts the first tasks of the process

    Parameters
    ----------
    process_graph : Networkx di-graph

    Returns
    -------
    list of tasks
    """
    temp_process_graph = process_graph.copy()
    for node in list(temp_process_graph.nodes):
        if process_graph.nodes[node]['type'] not in ['start', 'end', 'task']:
            preds = list(temp_process_graph.predecessors(node))
            succs = list(temp_process_graph.successors(node))
            temp_process_graph.add_edges_from(list(itertools.product(preds, succs)))
            temp_process_graph.remove_node(node)
    graph_data = pd.DataFrame.from_dict(dict(temp_process_graph.nodes.data()), orient='index')
    start = graph_data[graph_data.type.isin(['start'])]
    start = start.index.tolist()[0]  # start node id
    in_tasks = [temp_process_graph.nodes[x]['name'] for x in temp_process_graph.successors(start)]
    return in_tasks


def __mine_interarrival_time(log: pd.DataFrame, tasks: list) -> list:
    """
    Extracts the interarrival distribution from data

    Returns
    -------
    inter_arrival_times : list
    """
    # Analysis of start tasks
    ordering_field = 'start_timestamp'
    # Find the initial activity
    log = log[log.task.isin(tasks)]
    arrival_timestamps = (pd.DataFrame(
        log.groupby('caseid')[ordering_field].min())
                          .reset_index()
                          .rename(columns={ordering_field: 'times'}))
    # group by day and calculate inter-arrival
    arrival_timestamps['date'] = arrival_timestamps['times'].dt.floor('d')
    inter_arrival_times = []
    for key, group in tqdm(arrival_timestamps.groupby('date'), desc='extracting inter-arrivals'):
        daily_times = sorted(list(group.times))
        for i in range(1, len(daily_times)):
            delta = (daily_times[i] - daily_times[i - 1]).total_seconds()
            inter_arrival_times.append(delta)
    return inter_arrival_times


def __define_interarrival_distribution(inter_arrival_times: list, pdf_method: PDFMethod = PDFMethod.AUTOMATIC) -> dict:
    """
    Process the interarrival distribution

    Returns
    -------
    elements_data : Dataframe

    """
    # processing time discovery method
    if pdf_method is PDFMethod.AUTOMATIC:
        return DistributionFinder(inter_arrival_times).distribution
    elif pdf_method is PDFMethod.DEFAULT:
        return {
            'dname': 'EXPONENTIAL',
            'dparams': {
                'mean': 0,
                'arg1': np.round(np.mean(inter_arrival_times), 2),
                'arg2': 0
            }
        }
    raise ValueError(f'PDF method not supported: {pdf_method}')
