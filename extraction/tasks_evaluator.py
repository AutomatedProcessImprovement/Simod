# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 17:28:46 2020

@author: manuel.chavez
"""
import tkinter as tk
import pandas as pd
import networkx as nx
import numpy as np

from extraction import pdf_finder as pdf
from extraction.user_interface import manual_edition_ui as me

import utils.support as sup
from tqdm import tqdm



class TaskEvaluator():
    """
        This class evaluates the tasks durations and associates resources to it
     """

    def __init__(self, process_graph, process_stats, resource_pool, settings):
        """constructor"""
        self.tasks = self.get_task_list(process_graph)
        self.model_data = self.get_model_data(process_graph)
        self.process_stats = process_stats
        self.resource_pool = resource_pool

        self.pdef_method = settings['pdef_method']
        self.pdef_values = dict()

        self.one_timestamp = settings['read_options']['one_timestamp']
        self.elements_data = self.evaluate_tasks()


    def evaluate_tasks(self):
        """
        Process the task data and association of resources

        Returns
        -------
        elements_data : Dataframe

        """
        elements_data = list()
        # processing time discovery method
        if self.pdef_method == 'automatic':
            elements_data = self.mine_processing_time()
        elif self.pdef_method in ['manual', 'semi-automatic']:
            elements_data = self.define_distributions_manually()
        elif self.pdef_method == 'default':
            elements_data = self.default_processing_time()
        else:
            raise ValueError(self.pdef_method)
        # Resource association
        elements_data = self.associate_resource(elements_data)
        elements_data = elements_data.to_dict('records')
        elements_data = self.add_start_end_info(elements_data)
        return elements_data

    def default_processing_time(self):
        elements_data = self.default_values()
        elements_data = pd.DataFrame(elements_data)
        return elements_data


    def mine_processing_time(self):
        """
        Performs the mining of activities durations from data

        Returns
        -------
        elements_data : Dataframe

        """
        elements_data = list()
        for task in tqdm(self.tasks, 
                         desc='mining tasks distributions:'):
            s_key = 'duration' if self.one_timestamp else 'processing_time'
            task_processing = (
                self.process_stats[
                    self.process_stats.task == task][s_key].tolist())
            dist = pdf.DistributionFinder(task_processing).distribution
            elements_data.append({'id': sup.gen_id(),
                                  'type': dist['dname'],
                                  'name': task,
                                  'mean': str(dist['dparams']['mean']),
                                  'arg1': str(dist['dparams']['arg1']),
                                  'arg2': str(dist['dparams']['arg2'])})
        elements_data = pd.DataFrame(elements_data)
        elements_data = elements_data.merge(
            self.model_data[['name', 'elementid']], on='name', how='left')
        return elements_data


    def define_distributions_manually(self):
        """
        Enable the manual edition of tasks duration

        Returns
        -------
        elements_data : Dataframe

        """
        if self.pdef_method == 'semi-automatic':
            elements_data = self.mine_processing_time().sort_values(by='name')
            elements_data = elements_data.to_dict('records')
        else:
            elements_data = self.default_values()
        root = tk.Tk()
        window = me.MainWindow(root, elements_data)
        root.mainloop()
        new_elements = pd.DataFrame(window.new_elements)
        elements_data = pd.DataFrame(elements_data)

        elements_data = new_elements.merge(
            elements_data[['id', 'name', 'elementid']], on='id', how='left')
        return elements_data

    def default_values(self):
        """
        Performs the mining of activities durations from data
        Returns
        -------
        elements_data : Dataframe
        """
        elements_data = list()
        for task in tqdm(self.tasks, 
                         desc='mining tasks distributions:'):
            s_key = 'duration' if self.one_timestamp else 'processing_time'
            task_processing = (
                self.process_stats[
                    self.process_stats.task == task][s_key].tolist())
            try:
                mean_time = np.mean(task_processing) if task_processing else 0
            except:
                mean_time = 0
            elements_data.append({'id': sup.gen_id(),
                                  'type': 'EXPONENTIAL',
                                  'name': task,
                                  'mean': str(0),
                                  'arg1': str(np.round(mean_time, 2)),
                                  'arg2': str(0)})
        elements_data = pd.DataFrame(elements_data)
        elements_data = elements_data.merge(
            self.model_data[['name', 'elementid']], on='name', how='left')
        return elements_data.to_dict('records')

    def add_start_end_info(self, elements_data):
        # records creation
        temp_elements_data = list()
        default_record = {'type': 'FIXED',
                          'mean': '0', 'arg1': '0', 'arg2': '0'}
        for task in ['Start', 'End']:
            temp_elements_data.append({**{'id': sup.gen_id(), 'name': task},
                                       **default_record})
        temp_elements_data = pd.DataFrame(temp_elements_data)

        temp_elements_data = temp_elements_data.merge(
            self.model_data[['name', 'elementid']],
            on='name',
            how='left').sort_values(by='name')
        temp_elements_data['r_name'] = 'SYSTEM'
        # resource id addition
        resource_id = (pd.DataFrame.from_dict(self.resource_pool)[['id', 'name']]
                       .rename(columns={'id': 'resource', 'name': 'r_name'}))
        temp_elements_data = (temp_elements_data.merge(
            resource_id, on='r_name', how='left').drop(columns=['r_name']))
        # Appening to the elements data
        temp_elements_data = temp_elements_data.to_dict('records')
        elements_data.extend(temp_elements_data)
        return elements_data

    def associate_resource(self, elements_data):
        """
        Merge the resource information with the task data

        Parameters
        ----------
        elements_data : Dataframe

        Returns
        -------
        elements_data : Dataframe

        """
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
        elements_data['resource'].fillna('QBP_DEFAULT_RESOURCE', inplace=True)
        return elements_data

    def get_task_list(self, process_graph):
        """
        Extracts the tasks list from the BPM model

        Parameters
        ----------
        process_graph : Networkx Digraph

        Returns
        -------
        tasks : List

        """
        tasks = list(filter(
            lambda x: process_graph.nodes[x]['type'] == 'task',
            list(nx.nodes(process_graph))))
        tasks = [process_graph.nodes[x]['name'] for x in tasks]
        return tasks

    def get_model_data(self, process_graph):
        """
        Extracts the tasks data from the BPM model

        Parameters
        ----------
        process_graph : Networkx Digraph

        Returns
        -------
        model_data : Dataframe
        """
        model_data = pd.DataFrame.from_dict(
            dict(process_graph.nodes.data()), orient='index')
        model_data = (model_data[model_data.type.isin(['task', 'start', 'end'])]
                      .rename(columns={'id': 'elementid'}))
        return model_data
