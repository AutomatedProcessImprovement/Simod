# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 11:17:33 2020

@author: Manuel Camargo
"""
import pandas as pd
from extraction import pdf_definition as pdf
from support_modules import support as sup


def evaluate_tasks(process_graph, process_stats, resource_pool, settings):
    elements_data = list()
    tasks = process_stats.task.unique()
    for task in tasks:
        if settings['read_options']['one_timestamp']:
            task_processing = process_stats[process_stats.task==task]['duration'].tolist() 
        else:
            task_processing = process_stats[process_stats.task==task]['processing_time'].tolist() 
        dist = pdf.get_task_distribution(task_processing)
        print(dist)
        elements_data.append(
            dict(id=sup.gen_id(),
                 type=dist['dname'],
                 name=task,
                 mean=str(dist['dparams']['mean']),
                 arg1=str(dist['dparams']['arg1']),
                 arg2=str(dist['dparams']['arg2'])))
    elements_data = pd.DataFrame(elements_data)
    activities_table = (process_stats[['caseid', 'role','task']]
                        .groupby(['task','role']).count()
                        .sort_values(by=['caseid'])
                        .groupby(level=0)
                        .tail(1)
                        .reset_index())
    model_data = pd.DataFrame.from_dict(dict(process_graph.nodes.data()), orient='index')
    model_data = model_data[model_data.type=='task']
    activities_table = (activities_table.merge(model_data[['name','id']],
                                              left_on='task',
                                              right_on='name',
                                              how='left')
                        .drop(columns=['task','caseid'])
                        .rename(columns={'id': 'elementid'}))
    resource_id = (pd.DataFrame.from_dict(resource_pool)[['id','name']]
                   .rename(columns={'id':'resource', 'name':'r_name'}))
    activities_table = (activities_table.merge(resource_id,
                                              left_on='role',
                                              right_on='r_name',
                                              how='left')
                        .drop(columns=['role','r_name']))
    elements_data = elements_data.merge(activities_table, on='name', how='left')
    elements_data = elements_data.to_dict('records')
    return elements_data

