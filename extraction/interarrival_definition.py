# -*- coding: utf-8 -*-
import tkinter as tk
import pandas as pd
from support_modules import support as sup

from extraction import pdf_finder as pdf
from extraction.user_interface import dist_manual_edition_ui as me


class InterArrivalEvaluator():
    """
    This class evaluates the inter-arrival times
    """

    def __init__(self, process_graph, log, settings):
        """constructor"""
        self.log = pd.DataFrame.from_records(log)
        self.tasks = self.analize_first_tasks(process_graph)
        self.one_timestamp = settings['read_options']['one_timestamp']
        self.pdef_method = settings['pdef_method']

        self.inter_arrival_times = self.mine_interarrival_time()
        self.dist = dict()
        self.define_interarrival_distribution()

    def define_interarrival_distribution(self):
        """
        Process the interarrival distribution

        Returns
        -------
        elements_data : Dataframe

        """
        dist = pdf.DistributionFinder(self.inter_arrival_times).distribution
        # processing time discovery method
        if self.pdef_method == 'automatic':
            self.dist = dist
        if self.pdef_method in ['manual', 'semi-automatic']:
            self.dist = self.define_distributions_manually(dist)

    def mine_interarrival_time(self):
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
        for key, group in arrival_timestamps.groupby('date'):
            daily_times = sorted(list(group.times))
            for i in range(1, len(daily_times)):
                delta = (daily_times[i] - daily_times[i-1]).total_seconds()
                # TODO: Check this condition,
                # if interarrival is 0 what does it means?
                # if delta > 0:
                inter_arrival_times.append(delta)
        return inter_arrival_times

    def analize_first_tasks(self, process_graph):
        """
        Extracts the first tasks of the process

        Parameters
        ----------
        process_graph : Networkx di-graph

        Returns
        -------
        list of tasks

        """
        tasks_list = list()
        for node in process_graph.nodes:
            if process_graph.node[node]['type']=='task':
                tasks_list.append(
                    self.find_tasks_predecesors(process_graph,node))
        in_tasks = list()
        i=0
        for task in tasks_list:
            sup.print_progress(((i / (len(tasks_list)-1))* 100),
                               'Defining inter-arrival rate ')
            for path in task['sources']:
                for in_task in path['in_tasks']:
                    if process_graph.node[in_task]['type']=='start':
                        in_tasks.append(
                            process_graph.node[task['task']]['name'])
            i+=1
        return list(set(in_tasks))


    def find_tasks_predecesors(self, process_graph, num):
        """
        Support method for finding task predecesors

        Parameters
        ----------
        process_graph : Networkx di-graph
        num : num node int

        Returns
        -------
        dict of task and predecesors

        """
        # Sources
        r = process_graph.reverse(copy=True)
        paths = list(r.neighbors(num))
        task_paths = self.extract_target_tasks(r, num)
        in_paths = [sup.reduce_list(path) for path in task_paths]
        ins = [dict(in_tasks=y, in_node= x) for x,y in zip(paths, in_paths)]
        return dict(task=num,sources=ins)


    def extract_target_tasks(self, process_graph, num):
        """
        Support method for extract target tasks

        Parameters
        ----------
        process_graph : Networkx di-graph
        num : num node int

        Returns
        -------
        tasks_list : list of tasks 

        """
        tasks_list=list()
        for node in process_graph.neighbors(num):
            if process_graph.node[node]['type'] in ['task', 'start', 'end']:
                tasks_list.append([node])
            else:
                tasks_list.append(
                    self.extract_target_tasks(process_graph, node))
        return tasks_list

    def define_distributions_manually(self, dist):
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
