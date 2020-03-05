# -*- coding: utf-8 -*-
import networkx as nx
import pandas as pd

# from support_modules import support as sup
import support as sup
import log_reader as lr
import json
import os
from networkx.readwrite import json_graph

from collections import OrderedDict
#%%
class LogReplayer():
    """
    """

    def __init__(self, process_graph, log, settings, source='log', run_num=0):
        """constructor"""
        self.source = source
        self.run_num = run_num
        self.one_timestamp = settings['read_options']['one_timestamp']
        self.process_graph = process_graph
        self.g_data = pd.DataFrame.from_dict(dict(process_graph.nodes.data()),
                                             orient='index')
        self.start_tasks_list = list()
        self.end_tasks_list = list()
        self.find_start_finish_tasks()

        self.subsec_set = set()
        self.create_subsec_set()

        self.parallel_gt_exec = list()
        self.parallel_execution_list()

        self.not_conformant_traces = list()
        self.conformant_traces = list()
        self.process_stats = list()
        self.traces = log.get_traces()

        self.replay()

    def replay(self):
        for index in range(0, len(self.traces)):
            trace_times = list()
            trace = self.traces[index][1:-1]  # remove start and end event
            current_node = self.find_task_node(self.process_graph,
                                               trace[0]['task'])
            last_node = self.find_task_node(self.process_graph,
                                            trace[-1]['task'])
            if current_node not in self.start_tasks_list:
                self.not_conformant_traces.append(trace)
                next
            if last_node not in self.end_tasks_list:
                self.not_conformant_traces.append(trace)
                next
            temp_gt_exec = self.parallel_gt_exec
            cursor = [current_node]
            removal_allowed = True
            is_conformant = True
            # ----time recording------
            trace_times = self.save_record(trace_times, trace, 0)
            # ------------------------
            for i in range(1, len(trace)):
                next_node = self.find_task_node(self.process_graph,
                                                trace[i]['task'])
                # If loop management
                if next_node == cursor[-1]:
                    prev_record = self.find_previous_record(
                        trace_times,
                        self.process_graph.node[next_node]['name'])
                    trace_times = self.save_record(trace_times,
                                                   trace,
                                                   i,
                                                   prev_record)
                    self.process_graph.node[next_node]['executions'] += 1
                else:
                    try:
                        cursor, prev_node = self.update_cursor(
                            next_node, process_graph, cursor)
                        # ----time recording------
                        prev_record = self.find_previous_record(
                            trace_times,
                            self.process_graph.node[prev_node]['name'])
                        trace_times = self.save_record(trace_times,
                                                       trace,
                                                       i,
                                                       prev_record)
                        self.process_graph.node[next_node]['executions'] += 1
                        # ------------------------
                    except:
                        is_conformant = False
                        break
                    for element in reversed(cursor[:-1]):
                        # TODO ejecutar una sola vez self.process_graph.node[element]['type']
                        # Process AND
                        if self.process_graph.node[element]['type'] == 'gate3':
                            gate = [d for d in temp_gt_exec if d['nod_num'] == element][0]
                            gate.update({'executed': gate['executed'] + 1})
                            if gate['executed'] < gate['num_paths']:
                                removal_allowed = False
                            else:
                                removal_allowed = True
                                cursor.remove(element)
                        # Process Task
                        elif self.process_graph.node[element]['type'] == 'task':
                            if (element, next_node) in self.subsec_set:
                                if removal_allowed:
                                    cursor.remove(element)
                        # Process other
                        else:
                            if removal_allowed:
                                cursor.remove(element)
            if not is_conformant:
                self.not_conformant_traces.extend(trace)
            else:
                # Append the original one
                self.conformant_traces.extend(self.traces[index])
                self.process_stats.extend(trace_times)
            sup.print_progress(((index / (len(self.traces) - 1)) * 100),
                               'Replaying process traces ')
        # ------Filtering records and calculate stats---
        self.process_stats = list(filter(lambda x: x['task'] not in ['Start', 'End'] and x['resource'] != 'AUTO', self.process_stats))
        self.process_stats = self.calculate_process_metrics(
            self.process_stats, self.one_timestamp)
        [x.update(dict(source=self.source, run_num=self.run_num)) for x in self.process_stats]
        # ----------------------------------------------
        sup.print_done_task()

    @staticmethod
    def update_cursor(nnode, process_graph, cursor):
        tasks = list(filter(
            lambda x: process_graph.node[x]['type'] == 'task', cursor))
        shortest_path = list()
        prev_node = 0
        for pnode in reversed(tasks):
            try:
                shortest_path = list(nx.shortest_path(process_graph,
                                                      pnode,
                                                      nnode))[1:]
                prev_node = pnode
                break
            except nx.NetworkXNoPath:
                pass
        if len(list(filter(lambda x: process_graph.node[x]['type'] == 'task',
                           shortest_path))) > 1:
            raise Exception('Incoherent path')
        ap_list = cursor + shortest_path
        # Preserve order and leave only new
        cursor = list(OrderedDict.fromkeys(ap_list))
        return cursor, prev_node

    # =============================================================================
    # Time recording
    # =============================================================================

    def save_record(self, trace_times, trace, i, prev_record=dict()):
        record = self.create_record(trace, i, self.one_timestamp, prev_record)
        if record['resource'] != 'AUTO':
            trace_times.append(record)
        return trace_times

    @staticmethod
    def create_record(trace, index, one_timestamp, last_event=dict()):
        if not bool(last_event):
            enabling_time = trace[index]['end_timestamp']
        else:
            enabling_time = last_event['end_timestamp']
        record = {'caseid': trace[index]['caseid'],
                  'task': trace[index]['task'],
                  'end_timestamp': trace[index]['end_timestamp'],
                  'enable_timestamp': enabling_time,
                  'resource': trace[index]['user']}
        if not one_timestamp:
            record['start_timestamp'] = trace[index]['start_timestamp']
        return record

    @staticmethod
    def find_previous_record(trace_times, task):
        event = dict()
        for x in trace_times[::-1]:
            if task == x['task']:
                event = x
                break
        return event

    @staticmethod
    def calculate_process_metrics(process_stats, one_timestamp):
        for record in process_stats:
            if one_timestamp:
                record['duration']=(record['end_timestamp']-record['enable_timestamp']).total_seconds()
            else:
                duration=(record['end_timestamp']-record['start_timestamp']).total_seconds()
                waiting=(record['start_timestamp']-record['enable_timestamp']).total_seconds()
                multitasking=0
                #TODO check resourse for multi_tasking
                if waiting<0:
                    waiting=0
                    if record['end_timestamp'] > record['enable_timestamp']:
                        duration=(record['end_timestamp']-record['enable_timestamp']).total_seconds()
                        multitasking=(record['enable_timestamp']-record['start_timestamp']).total_seconds()
                    else:
                        multitasking = duration
                record['processing_time'] = duration
                record['waiting_time'] = waiting
                record['multitasking'] = multitasking
        return process_stats

    def find_start_finish_tasks(self) -> None:
        g_data = self.g_data.copy()
        start_node = g_data[g_data.type == 'start'].index.tolist()[0]
        end_node = g_data[g_data.type == 'end'].index.tolist()[0]
        self.start_tasks_list = sup.reduce_list(
            self.find_next_tasks(self.process_graph,
                                 start_node))
        self.end_tasks_list = sup.reduce_list(
            self.find_next_tasks(self.process_graph.reverse(copy=True),
                                 end_node))

    def create_subsec_set(self) -> None:
        g_data = self.g_data.copy()
        task_list = g_data[g_data.type == 'task'].index.tolist()

        for task in task_list:
            next_tasks = sup.reduce_list(
                self.find_next_tasks(self.process_graph, task))
            for n_task in next_tasks:
                self.subsec_set.add((task, n_task))

    def parallel_execution_list(self) -> None:
        g_data = self.g_data.copy()
        para_gates = g_data[g_data.type == 'gate3'].index.tolist()
        for x in para_gates:
            self.parallel_gt_exec.append(
                {'nod_num': x,
                 'num_paths': len(list(self.process_graph.neighbors(x))),
                 'executed': 0})

    @staticmethod
    def find_next_tasks(process_graph, num):
        tasks_list = list()
        for node in process_graph.neighbors(num):
            if process_graph.node[node]['type'] in ['task', 'start', 'end']:
                tasks_list.append([node])
            else:
                tasks_list.append(
                    LogReplayer.find_next_tasks(process_graph, node))
        return tasks_list

    @staticmethod
    def find_task_node(process_graph, task_name):
        resp = list(filter(
            lambda x: process_graph.node[x]['name'] == task_name,
            process_graph.nodes))
        if len(resp) > 0:
            resp = resp[0]
        else:
            raise Exception('Task not found on bpmn structure...')
        return resp



#%%
route = 'C:/Users/Manuel Camargo/Documents/GitHub/SiMo-Discoverer/'

with open(route + 'process_graph.json') as file:
    gdata = json.load(file)
    file.close()

process_graph = json_graph.node_link_graph(gdata)

with open(route + 'id.json') as file:
    id_data = json.load(file)
    file.close()
id_data = {int(k): v for k, v in id_data.items()}
nx.set_node_attributes(process_graph, id_data)

with open(route + 'settings.json') as file:
    settings = json.load(file)
    file.close()

log = lr.LogReader(os.path.join(route, settings['input'], settings['file']),
                   settings['read_options'])

replayer = LogReplayer(process_graph, log, settings)
print(replayer.process_stats)