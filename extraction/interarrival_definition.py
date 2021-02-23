# -*- coding: utf-8 -*-
import tkinter as tk
import pandas as pd
from tqdm import tqdm
import itertools
import numpy as np


from extraction import pdf_finder as pdf
from extraction.user_interface import dist_manual_edition_ui as me


class InterArrivalEvaluator():
    """
    This class evaluates the inter-arrival times
    """

    def __init__(self, process_graph, log, settings):
        """constructor"""
        self.log = pd.DataFrame.from_records(log)
        self.tasks = self._analize_first_tasks(process_graph)
        self.one_timestamp = settings['read_options']['one_timestamp']
        self.pdef_method = settings['pdef_method']

        self.inter_arrival_times = self._mine_interarrival_time()
        self.dist = dict()
        self.define_interarrival_distribution()

    def define_interarrival_distribution(self):
        """
        Process the interarrival distribution

        Returns
        -------
        elements_data : Dataframe

        """
        # processing time discovery method
        if self.pdef_method == 'automatic':
            self.dist = pdf.DistributionFinder(
                self.inter_arrival_times).distribution
        elif self.pdef_method in ['manual', 'semi-automatic']:
            self.dist = self._define_distributions_manually(
                pdf.DistributionFinder(self.inter_arrival_times).distribution)
        elif self.pdef_method == 'default':
            self.dist = {'dname': 'EXPONENTIAL', 
                         'dparams': {'mean': 0, 
                                     'arg1': np.round(np.mean(self.inter_arrival_times), 2), 
                                     'arg2': 0}}
        else:
            raise ValueError(self.pdef_method)
            
            
    def _mine_interarrival_time(self):
        """
        Extracts the interarrival distribution from data

        Returns
        -------
        inter_arrival_times : list

        """
        # Analysis of start tasks
        ordering_field = ('end_timestamp'
                          if self.one_timestamp else 'start_timestamp')
        # Find the initial activity
        log = self.log[self.log.task.isin(self.tasks)]
        arrival_timestamps = (pd.DataFrame(
            log.groupby('caseid')[ordering_field].min())
            .reset_index()
            .rename(columns={ordering_field:'times'}))
        # group by day and calculate inter-arrival
        arrival_timestamps['date'] = arrival_timestamps['times'].dt.floor('d')
        inter_arrival_times = list()
        for key, group in tqdm(arrival_timestamps.groupby('date'),
                               desc='extracting interarrivals:'):
            daily_times = sorted(list(group.times))
            for i in range(1, len(daily_times)):
                delta = (daily_times[i] - daily_times[i-1]).total_seconds()
                # TODO: Check this condition,
                # if interarrival is 0 what does it means?
                # if delta > 0:
                inter_arrival_times.append(delta)
        return inter_arrival_times

    def _analize_first_tasks(self, process_graph) -> list():
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
                temp_process_graph.add_edges_from(
                    list(itertools.product(preds, succs)))
                temp_process_graph.remove_node(node)    
        graph_data = (pd.DataFrame.from_dict(
            dict(temp_process_graph.nodes.data()), orient='index'))
        start = graph_data[graph_data.type.isin(['start'])]
        start = start.index.tolist()[0]  # start node id 
        in_tasks = [temp_process_graph.nodes[x]['name']
                    for x in temp_process_graph.successors(start)]
        return in_tasks

    def _define_distributions_manually(self, dist):
        """
        Enable the manual edition of tasks duration

        Returns
        -------
        elements_data : Dataframe

        """
        root = tk.Tk()
        window = me.MainWindow(root, dist)
        root.mainloop()
        new_elements = window.new_elements
        if new_elements:
            dist['dname'] = new_elements[0]['type']
            dist['dparams']['mean'] = new_elements[0]['mean']
            dist['dparams']['arg1'] = new_elements[0]['arg1']
            dist['dparams']['arg2'] = new_elements[0]['arg2']
        return dist
