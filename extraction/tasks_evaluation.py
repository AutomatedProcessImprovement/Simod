# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 11:17:33 2020

@author: Manuel Camargo
"""
import pandas as pd
from tkinter import *

from extraction import pdf_definition as pdf
from extraction import manual_edition_ui as me

from support_modules import support as sup

def evaluate_tasks(process_graph, process_stats, resource_pool, settings):
    elements_data = list()
    # processing time discovery method
    if settings['pdef_method'] == 'automatic':
        elements_data = mine_processing_time(process_stats, process_graph, settings)
    if settings['pdef_method'] == 'manual':
        define_distributions_manually(process_stats, process_graph)
    print(elements_data)
    print(elements_data.dtypes)

    # Resource association
    elements_data = associate_resource(elements_data, process_stats, resource_pool)
    elements_data = elements_data.to_dict('records')

    return elements_data

def mine_processing_time(process_stats, process_graph, settings):
    """
    This method extracts the activities processing time from the statistics

    Parameters
    ----------
    process_stats : Dataframe
    settings : dict

    Returns
    -------
    Dataframe
        Activities processing time.
    """
    # TODO: check if all the tasks have distribution and time associated
    elements_data = list()
    tasks = process_stats.task.unique()
    for task in tasks:
        if settings['read_options']['one_timestamp']:
            task_processing = process_stats[process_stats.task==task]['duration'].tolist() 
        else:
            task_processing = process_stats[process_stats.task==task]['processing_time'].tolist() 
        dist = pdf.get_task_distribution(task_processing)
        elements_data.append(
            dict(id=sup.gen_id(),
                 type=dist['dname'],
                 name=task,
                 mean=str(dist['dparams']['mean']),
                 arg1=str(dist['dparams']['arg1']),
                 arg2=str(dist['dparams']['arg2'])))
    elements_data = pd.DataFrame(elements_data)
    model_data = pd.DataFrame.from_dict(dict(process_graph.nodes.data()), orient='index')
    model_data = model_data[model_data.type=='task'].rename(columns={'id': 'elementid'})
    
    elements_data = elements_data.merge(model_data[['name','elementid']],
                                        on='name',
                                        how='left')
    return elements_data



   
def define_distributions_manually(process_stats, process_graph):
    elements_data = default_values(process_stats, process_graph)
    root = Tk()
    a = me.MainWindow(root, elements_data)
    root.mainloop()

def default_values(process_stats, process_graph):
    elements_data = list()
    tasks = process_stats.task.unique()
    default_record = {'type':'FIXED', 'mean':3600,'arg1':0, 'arg2':0}
    for task in tasks:
        elements_data.append({**{'id':sup.gen_id(),'name':task},**default_record})
    elements_data = pd.DataFrame(elements_data)
    
    model_data = pd.DataFrame.from_dict(dict(process_graph.nodes.data()), orient='index')
    model_data = model_data[model_data.type=='task'].rename(columns={'id': 'elementid'})
    elements_data = elements_data.merge(model_data[['name','elementid']],
                                        on='name',
                                        how='left')
    return elements_data.to_dict('records')


def associate_resource(elements_data, process_stats, resource_pool):
    roles_table = (process_stats[['caseid', 'role','task']]
                    .groupby(['task','role']).count()
                    .sort_values(by=['caseid'])
                    .groupby(level=0)
                    .tail(1)
                    .reset_index())
    resource_id = (pd.DataFrame.from_dict(resource_pool)[['id','name']]
                    .rename(columns={'id':'resource', 'name':'r_name'}))
    roles_table = (roles_table.merge(resource_id,
                                              left_on='role',
                                              right_on='r_name',
                                              how='left')
                        .drop(columns=['role','r_name', 'caseid']))
    elements_data = elements_data.merge(roles_table,
                                        left_on='name',
                                        right_on='task',
                                        how='left').drop(columns=['task'])
    return elements_data