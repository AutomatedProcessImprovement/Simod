# -*- coding: utf-8 -*-
import support as sup
import networkx as nx
import numpy as np
import xml.etree.ElementTree as ET
import os
import sys, getopt
from readers import readers as rd
import itertools


def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)

# # Event log mesures processing
# def export_not_conformed_traces(not_conformed_traces,assets_output ):
#     #not_conformed_traces_output = assets_output + chr(92) + 'log_not_conformed.csv'
#     not_conformed_traces_output = assets_output + '/' + 'log_not_conformed.csv'
#     index=list()
#     for trace in not_conformed_traces:
#         [index.append(x) for x in trace]
#     sup.create_csv_file_header(index, not_conformed_traces_output)
#
# def export_tasks_statistics(process_graph,assets_output):
#     #log_stats_output = assets_output + chr(92) + 'log_tasks_stats_output.csv'
#     log_stats_output = assets_output + '/' + 'log_tasks_stats_output.csv'
#     tasks = list(filter(lambda x: process_graph.node[x]['type'] =='task',nx.nodes(process_graph)))
#     index=list()
#     for task in tasks:
#         index.append(dict(task_name=process_graph.node[task]['name'],avg_proc=mean(process_graph.node[task]['processing_times']),
#         max_proc=max(process_graph.node[task]['processing_times'], default=0),min_proc=min(process_graph.node[task]['processing_times'], default=0),
#         avg_wait=mean(process_graph.node[task]['waiting_times']),max_wait=max(process_graph.node[task]['waiting_times'], default=0),
#         min_wait=min(process_graph.node[task]['waiting_times'], default=0)))
#     sup.create_csv_file_header(index, log_stats_output)
#
# def export_process_statistics(process_stats, process_graph, assets_output):
#     #log_stats_output = assets_output + chr(92) + 'log_process_stats_output.csv'
#     log_stats_output = assets_output + '/' + 'log_process_stats_output.csv'
#     total_processing, total_waiting, total_multitasking = list(), list(), list()
#     for x in process_stats:
#         total_processing.append(x['total_processing'])
#         total_waiting.append(x['total_waiting'])
#         total_multitasking.append(x['total_multitasking'])
#     index=list()
#     index.append(dict(
#         total_proc=sum(total_processing),avg_proc=mean(total_processing), max_proc=max(total_processing, default=0),min_proc=min(total_processing, default=0),
#         total_wait=sum(total_waiting),avg_wait=mean(total_waiting),max_wait=max(total_waiting, default=0), min_wait=min(total_waiting, default=0),
#         total_multitasking=sum(total_multitasking), avg_multitasking=mean(total_multitasking),max_multitasking=max(total_multitasking, default=0), min_multitasking=min(total_multitasking, default=0)))
#     sup.create_csv_file_header(index, log_stats_output)
#
#
# # Scyla output processing
# def readResourcesUtilization(filename,outputPath):
#     tree = ET.parse(filename)
#     root = tree.getroot()
#     index = list()
#
#     for resource in root.iter('resource'):
#         utilization = dict(
#             type=resource.find('type').text
#         )
#         for cost in resource.findall('cost'):
#             cost_data = dict(cost_min=cost.find('min').text,cost_max=cost.find('max').text,
#                           cost_median = cost.find('median').text, cost_Q1 = cost.find('Q1').text,
#                           cost_Q3=cost.find('Q3').text,cost_avg = cost.find('avg').text,cost_total=cost.find('total').text)
#             utilization.update(cost_data)
#
#         for time in resource.findall('time'):
#             for in_use in time.findall('in_use'):
#                 in_use_data = dict(time_inUse_min=in_use.find('min').text,time_inUse_max=in_use.find('max').text,
#                           time_inUse_median = in_use.find('median').text, time_inUse_Q1 = in_use.find('Q1').text,
#                           time_inUse_Q3=in_use.find('Q3').text,time_inUse_avg = in_use.find('avg').text,time_inUse_total=in_use.find('total').text)
#                 utilization.update(in_use_data)
#
#             for available in time.findall('available'):
#                 available_data = dict(time_available_min=available.find('min').text,time_available_max=available.find('max').text,
#                           time_available_median = available.find('median').text, time_available_Q1 = available.find('Q1').text,
#                           time_available_Q3=available.find('Q3').text,time_available_avg = available.find('avg').text,time_available_total=available.find('total').text)
#                 utilization.update(available_data)
#
#             for workload in time.findall('workload'):
#                 workload_data = dict(time_workload_min=workload.find('min').text,time_workload_max=workload.find('max').text,
#                           time_workload_median = workload.find('median').text, time_workload_Q1 = workload.find('Q1').text,
#                           time_workload_Q3=workload.find('Q3').text,time_workload_avg = workload.find('avg').text,time_workload_total=workload.find('total').text)
#                 utilization.update(workload_data)
#
#         for instances in resource.findall('instances'):
#             i = 0
#             for instance in instances.findall('instance'):
#                 instanceId = 'instances_instance_'+str(i)+'_id'
#                 instanceCost = 'instances_instance_' + str(i) + '_cost'
#                 instance_data = {instanceId:instance.find('id').text,
#                                  instanceCost:instance.find('cost').text}
#                 utilization.update(instance_data)
#                 for time in instance.findall('time'):
#                     instanceTimeInUse = 'instances_instance_'+str(i)+'_time_inUse'
#                     instanceTimeAvailable = 'instances_instance_' + str(i) + '_time_available'
#                     instanceTimeWorkload = 'instances_instance_' + str(i) + '_time_workload'
#                     instance_time_data = {instanceTimeInUse:time.find('in_use').text,
#                                           instanceTimeAvailable:time.find('available').text,
#                                           instanceTimeWorkload:time.find('workload').text}
#                     utilization.update(instance_time_data)
#                 i+=1
#         index.append(utilization)
#     sup.create_csv_file_header(index, outputPath+'_resourceUtilization.csv')
#
# def processMetadata(filename,outputPath):
#     tree = ET.parse(filename)
#     root = tree.getroot()
#     index = list()
#     for process in root.iter('process'):
#         metadata = dict(idProcess=process.find('id').text)
#         for cost in process.findall('cost'):
#             cost_data = dict(cost_min=cost.find('min').text,cost_max=cost.find('max').text,
#                           cost_median = cost.find('median').text, cost_Q1 = cost.find('Q1').text,
#                           cost_Q3=cost.find('Q3').text,cost_avg = cost.find('avg').text,cost_total=cost.find('total').text)
#             metadata.update(cost_data)
#         for time in process.findall('time'):
#             for flowTime in time.findall('flow_time'):
#                 flowTime_data = dict(flowTime_min=flowTime.find('min').text, flowTime_max=flowTime.find('max').text,
#                                      flowTime_median=flowTime.find('median').text, flowTime_Q1=flowTime.find('Q1').text,
#                                      flowTime_Q3=flowTime.find('Q3').text, flowTime_avg=flowTime.find('avg').text,
#                                      flowTime_total=flowTime.find('total').text)
#                 metadata.update(flowTime_data)
#             for effective in time.findall('effective'):
#                 effective_data = dict(effective_min=effective.find('min').text, effective_max=effective.find('max').text,
#                                       effective_median=effective.find('median').text, effective_Q1=effective.find('Q1').text,
#                                       effective_Q3=effective.find('Q3').text, effective_avg=effective.find('avg').text,
#                                       effective_total=effective.find('total').text)
#                 metadata.update(effective_data)
#             for waiting in time.findall('waiting'):
#                 waiting_data = dict(waiting_min=waiting.find('min').text, waiting_max=waiting.find('max').text,
#                                     waiting_median=waiting.find('median').text, waiting_Q1=waiting.find('Q1').text,
#                                     waiting_Q3=waiting.find('Q3').text, waiting_avg=waiting.find('avg').text,
#                                     waiting_total=waiting.find('total').text)
#                 metadata.update(waiting_data)
#             for off_timetable in time.findall('off_timetable'):
#                 off_timetable_data = dict(off_timetable_min=off_timetable.find('min').text, off_timetable_max=off_timetable.find('max').text,
#                                           off_timetable_median=off_timetable.find('median').text,off_timetable_Q1=off_timetable.find('Q1').text,
#                                           off_timetable_Q3=off_timetable.find('Q3').text, off_timetable_avg=off_timetable.find('avg').text,
#                                           off_timetable_total=off_timetable.find('total').text)
#                 metadata.update(off_timetable_data)
#         index.append(metadata)
#     sup.create_csv_file_header(index, outputPath + '_processMetadata.csv')
#
# def instancesData(filename,outputPath):
#     tree = ET.parse(filename)
#     root = tree.getroot()
#     index = list()
#     for process in root.iter('process'):
#         for instances in process.findall('instances'):
#             id = 0
#             for instance in instances.findall('instance'):
#                 instanceData=(dict(instanceId=id, costs=instance.find('costs').text))
#                 for time in instance.findall('time'):
#                     instanceData.update(dict(duration=time.find('duration').text,
#                                              effective=time.find('effective').text,
#                                              waiting = time.find('waiting').text,
#                                              offTime = time.find('offTime').text))
#                 id+=1
#                 index.append(instanceData)
#
#     sup.create_csv_file_header(index, outputPath + '_instancesData.csv')
