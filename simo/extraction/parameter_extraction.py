# -*- coding: utf-8 -*-
import support as sup
from extraction import traces_alignment as tal
from extraction import process_structure as gph
from extraction import log_replayer2 as rpl
from extraction import task_duration_distribution as td
from extraction import interarrival_definition as arr
from extraction import gateways_probabilities as gt
from extraction import role_discovery as rl
from extraction import schedule_tables as sch

import networkx as nx
import itertools


# --support --
def find_resource_id(resource_pool, resource_name):
    id = 0
    for resource in resource_pool:
        # print(resource)
        if resource['name'] == resource_name:
            id = resource['id']
            break
    return id

# -- Extract parameters --
def extract_parameters(log, bpmn, bimp, scylla, align_info_file=None, align_type_file=None):
    if bpmn != None and log != None:
        bpmnId = bpmn.getProcessId()
        startEventId = bpmn.getStartEventId()
        # Creation of process graph
        process_graph = gph.create_process_structure(bpmn)
        #-------------------------------------------------------------------
        # Analysing resource pool LV917 or 247
        roles, resource_table = rl.read_resource_pool(log, drawing=False, sim_percentage=0.5)
#        [print(x) for x in resource_table]
        resource_pool, time_table, resource_table = sch.analize_schedules(resource_table, log, True, '247')
        #-------------------------------------------------------------------
        # Aligning the process traces
        if align_info_file != None:
            tal.align_traces(log, align_info_file, align_type_file)
        #-------------------------------------------------------------------
        # Process replaying
        conformed_traces, not_conformed_traces, process_stats = rpl.replay(process_graph, log)
        # -------------------------------------------------------------------
        # Adding role to process stats
        for stat in process_stats:
            role = list(filter(lambda x: x['resource']==stat['resource'],resource_table))[0]['role']
            stat['role'] = role
        #-------------------------------------------------------------------
        # Determination of first tasks for calculate the arrival rate
        inter_arrival_times = arr.define_interarrival_tasks(process_graph, conformed_traces)
        if scylla:
            arrival_rate = (td.get_task_distribution(inter_arrival_times, False, False, 50))
            arrival_rate['startEventId']=startEventId
        if bimp:
            arrival_rate_bimp = (td.get_task_distribution(inter_arrival_times, bimp, False, 50))
            arrival_rate_bimp['startEventId'] = startEventId
        #-------------------------------------------------------------------
        # Gateways probabilities 1=Historycal, 2=Random, 3=Equiprobable
        sequences = gt.define_probabilities(process_graph, bpmn, log, 1)
        #-------------------------------------------------------------------
        # Tasks id information
        elements_data = list()
        elements_data_bimp = list()
        i = 0
        task_list = list(filter(lambda x: process_graph.node[x]['type']=='task' , list(nx.nodes(process_graph))))
        for task in task_list:
            task_name = process_graph.node[task]['name']
            task_id = process_graph.node[task]['id']
            values = list(filter(lambda x: x['task'] == task_name, process_stats))
            task_processing = [x['processing_time'] for x in values]
            if scylla:
                dist = td.get_task_distribution(task_processing,False)
            if bimp:
                dist_bimp = td.get_task_distribution(task_processing, bimp)
            max_role, max_count = '', 0
            role_sorted = sorted(values, key=lambda x:x['role'])
            for key2, group2 in itertools.groupby(role_sorted, key=lambda x:x['role']):
                group_count = list(group2)
                if len(group_count)>max_count:
                    max_count = len(group_count)
                    max_role = key2
            if scylla:
                elements_data.append(dict(id=sup.gen_id(), elementid=task_id, type=dist['dname'],name = task_name,
                                          mean=str(dist['dparams']['mean']), arg1=str(dist['dparams']['arg1']),
                                          arg2=str(dist['dparams']['arg2']), resource=find_resource_id(resource_pool, max_role)))
            if bimp:
                elements_data_bimp.append(dict(id=sup.gen_id(), elementid=task_id, type=dist_bimp['dname'],name = task_name,
                         mean=str(dist_bimp['dparams']['mean']), arg1=str(dist_bimp['dparams']['arg1']),
                         arg2=str(dist_bimp['dparams']['arg2']), resource=find_resource_id(resource_pool, max_role)))
            sup.print_progress(((i / (len(task_list) - 1)) * 100), 'Analysing tasks data ')
            i += 1
        sup.print_done_task()
#        [print(x['name'],x['type'],x['mean'],x['arg1'],x['arg2'],sep=',') for x in elements_data_bimp]
        parameters_scylla = dict()
        parameters = dict()
        if scylla:
            parameters_scylla = dict(arrival_rate=arrival_rate, time_table=time_table, resource_pool=resource_pool, 
                                     elements_data=elements_data, sequences=sequences, instances=len(conformed_traces), bpmnId=bpmnId)
        if bimp:
            parameters = dict(arrival_rate=arrival_rate_bimp, time_table=time_table, resource_pool=resource_pool,
                              elements_data=elements_data_bimp, sequences=sequences, instances=len(conformed_traces),
                              bpmnId=bpmnId)
        # parameters, process_stats, parameters_scylla = dict(),list(),dict()
        return parameters, process_stats, parameters_scylla
