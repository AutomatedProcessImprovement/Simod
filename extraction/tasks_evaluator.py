# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 17:28:46 2020

@author: manuel.chavez
"""
import pandas as pd
import tkinter as tk
import networkx as nx

from extraction import pdf_definition as pdf
from extraction import manual_edition_ui as me

from support_modules import support as sup


class TaskEvaluator(object):
    """
        This class evaluates the tasks durations and associates resources to it
     """

    def __init__(self, process_graph, process_stats, resource_pool, settings):
        """constructor"""
        self.tasks = self.get_task_list(process_graph)
        self.model_data = self.get_model_data(process_graph)
        self.process_stats = process_stats
        self.resource_pool = resource_pool
        # self.settings = settings
        self.pdef_method = settings['pdef_method']
        self.pdef_values = (settings['tasks']
                            if settings['pdef_method'] == 'apx' else dict())
        self.one_timestamp = settings['read_options']['one_timestamp']

        self.elements_data = self.evaluate_tasks()

    def evaluate_tasks(self):
        elements_data = list()
        # processing time discovery method
        if self.pdef_method == 'automatic':
            elements_data = self.mine_processing_time()
        if self.pdef_method in ['manual', 'semi-automatic']:
            elements_data = self.define_distributions_manually()
        if self.pdef_method == 'apx':
            elements_data = self.match_predefined_time()
        # Resource association
        elements_data = self.associate_resource(elements_data)
        elements_data = elements_data.to_dict('records')
        return elements_data

    def mine_processing_time(self):
        elements_data = list()
        for task in self.tasks:
            if self.one_timestamp:
                task_processing = (
                    self.process_stats[
                        self.process_stats.task == task]['duration'].tolist())
            else:
                task_processing = (
                    self.process_stats[
                        self.process_stats.task == task]['processing_time']
                    .tolist())
            dist = pdf.get_task_distribution(task_processing)
            elements_data.append(
                dict(id=sup.gen_id(),
                     type=dist['dname'],
                     name=task,
                     mean=str(dist['dparams']['mean']),
                     arg1=str(dist['dparams']['arg1']),
                     arg2=str(dist['dparams']['arg2'])))
        elements_data = pd.DataFrame(elements_data)
        elements_data = elements_data.merge(
            self.model_data[['name', 'elementid']], on='name', how='left')
        return elements_data

    def match_predefined_time(self):
        elements_data = list()
        # Predefined tasks records creation
        default_record = {'type': 'EXPONENTIAL', 'mean': '0', 'arg2': '0'}
        for task, value in self.pdef_values.items():
            record = {
                **{'id': sup.gen_id(), 'name': str(task), 'arg1': str(value)},
                **default_record}
            elements_data.append(record)

        # Check If there is tasks with not predefined time
        pdef_tasks = list(self.pdef_values.keys())
        not_included = [task for task in self.tasks if task not in pdef_tasks]
        default_record = {'type': 'EXPONENTIAL', 'mean': '0',
                          'arg1': '3600', 'arg2': '0'}
        for task in not_included:
            elements_data.append({**{'id': sup.gen_id(), 'name': task},
                                  **default_record})
        elements_data = pd.DataFrame(elements_data)
        # Matching with model info
        elements_data = elements_data.merge(self.model_data[['name', 'elementid']],
                                            on='name',
                                            how='left').sort_values(by='name')
        return elements_data

    def define_distributions_manually(self):
        if self.pdef_method == 'semi-automatic':
            elements_data = self.mine_processing_time().sort_values(by='name')
            elements_data = elements_data.to_dict('records')
        else:
            elements_data = self.default_values()
        root = tk.Tk()
        a = me.MainWindow(root, elements_data)
        root.mainloop()
        new_elements = pd.DataFrame(a.new_elements)
        elements_data = pd.DataFrame(elements_data)

        elements_data = new_elements.merge(
            elements_data[['id', 'name', 'elementid']], on='id', how='left')
        return elements_data

    def default_values(self):
        elements_data = list()
        default_record = {'type': 'EXPONENTIAL',
                          'mean': '0', 'arg1': '3600', 'arg2': '0'}
        for task in self.tasks:
            elements_data.append({**{'id': sup.gen_id(), 'name': task},
                                  **default_record})
        elements_data = pd.DataFrame(elements_data)
        
        elements_data = elements_data.merge(self.model_data[['name', 'elementid']],
                                            on='name',
                                            how='left').sort_values(by='name')
        return elements_data.to_dict('records')

    def associate_resource(self, elements_data):
        roles_table = (self.process_stats[['caseid', 'role', 'task']]
                       .groupby(['task', 'role']).count()
                       .sort_values(by=['caseid'])
                       .groupby(level=0)
                       .tail(1)
                       .reset_index())
        resource_id = (pd.DataFrame.from_dict(self.resource_pool)[['id', 'name']]
                       .rename(columns={'id': 'resource', 'name': 'r_name'}))
        roles_table = (roles_table.merge(resource_id,
                                         left_on='role',
                                         right_on='r_name',
                                         how='left')
                       .drop(columns=['role', 'r_name', 'caseid']))
        elements_data = elements_data.merge(roles_table,
                                            left_on='name',
                                            right_on='task',
                                            how='left').drop(columns=['task'])
        return elements_data

    def get_task_list(self, process_graph):
        tasks = list(filter(
            lambda x: process_graph.node[x]['type'] == 'task',
            list(nx.nodes(process_graph))))
        tasks = [process_graph.node[x]['name'] for x in tasks]
        return tasks

    def get_model_data(self, process_graph):
        model_data = pd.DataFrame.from_dict(
            dict(process_graph.nodes.data()), orient='index')
        model_data = model_data[model_data.type == 'task'].rename(
            columns={'id': 'elementid'})
        return model_data