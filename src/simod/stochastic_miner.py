import copy
import datetime
import json
import os
import random
import shutil
import sys
import warnings
import xml.etree.ElementTree as ET
from collections import deque
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import List

import numpy as np
import pytz
import scipy.stats as st
from pm4py.objects.log.importer.xes import importer as xes_importer

from .cli_formatter import print_section, print_asset, print_step
from .configuration import Configuration
from .writers import xml_writer


# BPMN Graph

class BPMNNodeType(Enum):
    TASK = 'TASK'
    START_EVENT = 'START-EVENT'
    END_EVENT = 'END-EVENT',
    EXCLUSIVE_GATEWAY = 'EXCLUSIVE-GATEWAY'
    INCLUSIVE_GATEWAY = 'INCLUSIVE-GATEWAY'
    PARALLEL_GATEWAY = 'PARALLEL-GATEWAY'
    UNDEFINED = 'UNDEFINED'


class ElementInfo:
    def __init__(self, element_type: BPMNNodeType, element_id: str, element_name: str):
        self.id = element_id
        self.name = element_name
        self.type = element_type
        self.incoming_flows = list()
        self.outgoing_flows = list()

    def is_split(self):
        return len(self.outgoing_flows) > 1

    def is_join(self):
        return len(self.incoming_flows) > 1

    def is_gateway(self):
        return self.type in [BPMNNodeType.EXCLUSIVE_GATEWAY, BPMNNodeType.PARALLEL_GATEWAY,
                             BPMNNodeType.INCLUSIVE_GATEWAY]


class ProcessInfo:
    def __init__(self):
        self.traces = dict()
        self.resource_profiles = dict()


class TaskEvent:
    def __init__(self, p_case, task_id, task_name, enabled_at, enabled_by):
        self.p_case = p_case
        self.task_id = task_id
        self.task_name = task_name
        self.enabled_at = enabled_at
        self.enabled_by = enabled_by
        self.started_at = None
        self.completed_at = None
        self.idle_time = None
        self.resource_id = None

    def print_event_info(self):
        print("Task: %s(%s)" % (self.task_name, str(self.p_case)))

    def start_event(self, started_at, resource_id):
        self.started_at = started_at
        self.resource_id = resource_id

    def complete_event(self, ended_at, idle_time):
        self.completed_at = ended_at
        self.idle_time = idle_time

    def waiting_time(self):
        return (self.started_at - self.enabled_at).total_seconds()

    def idle_processing_time(self):
        return (self.completed_at - self.started_at).total_seconds()

    def processing_time(self):
        return self.idle_processing_time() - self.idle_time

    def idle_cycle_time(self):
        return (self.completed_at - self.enabled_at).total_seconds()

    def cycle_time(self):
        return self.idle_cycle_time() - self.idle_time


class Trace:
    def __init__(self, p_case, started_at=datetime.datetime(9999, 12, 31, 23, 59, 59, 999999, pytz.utc)):
        self.p_case = p_case
        self.started_at = started_at
        self.completed_at = started_at
        self.event_list = list()
        self.next_parallel_tasks = list()

        self.cycle_time = None
        self.idle_cycle_time = None
        self.processing_time = None
        self.idle_processing_time = None
        self.waiting_time = None
        self.idle_time = None

    def start_event(self, task_id, task_name, started_at, started_by, enabled_at, enabled_by):
        event_info = TaskEvent(self.p_case, task_id, task_name, enabled_at, enabled_by)
        event_index = len(self.event_list)
        self.event_list.append(event_info)
        self.started_at = min(self.started_at, enabled_at)
        self.next_parallel_tasks.append(list())
        if enabled_by is not None:
            self.next_parallel_tasks[enabled_by].append(event_index)
        self.event_list[event_index].start_event(started_at, started_by)
        return event_index

    def complete_event(self, event_index, completed_at, idle_time=0):
        self.event_list[event_index].complete_event(completed_at, idle_time)
        self.completed_at = max(self.completed_at, self.event_list[event_index].completed_at)
        return self.event_list[event_index]


class ProcessState:
    def __init__(self, bpmn_graph):
        self.arcs_bitset = bpmn_graph.arcs_bitset
        self.tokens = dict()
        self.state_mask = 0
        for flow_arc in bpmn_graph.flow_arcs:
            self.tokens[flow_arc] = 0

    def add_token(self, flow_id):
        if flow_id in self.tokens:
            self.tokens[flow_id] += 1
            self.state_mask |= self.arcs_bitset[flow_id]

    def remove_token(self, flow_id):
        if self.has_token(flow_id):
            self.tokens[flow_id] = 0
            self.state_mask &= ~self.arcs_bitset[flow_id]

    def has_token(self, flow_id):
        return flow_id in self.tokens and self.tokens[flow_id] > 0

    def pending_tokens(self):
        marked_flows = list()
        for flow_id in self.tokens:
            if self.tokens[flow_id] > 0:
                marked_flows.append(flow_id)
        return marked_flows


class BPMNGraph:
    def __init__(self):
        self.starting_event = None
        self.end_event = None
        self.element_info = dict()
        self.from_name = dict()
        self.flow_arcs = dict()
        self.concurrent_enablers = dict()
        self.nodes_bitset = dict()
        self.arcs_bitset = dict()
        self.or_join_pred = dict()  # or_id -> [0 = node predecesors bitset, 1 = predecesors flow arcs]
        self.or_join_conflicting_pred = dict()
        self.decision_successors = dict()
        self.element_probability = None
        self.task_resource_probability = None

    def set_element_probabilities(self, element_probability, task_resource_probability):
        self.element_probability = element_probability
        self.task_resource_probability = task_resource_probability

    def add_bpmn_element(self, element_id, element_info):
        if element_info.type == BPMNNodeType.START_EVENT:
            self.starting_event = element_id
        if element_info.type == BPMNNodeType.END_EVENT:
            self.end_event = element_id
        self.element_info[element_id] = element_info
        self.from_name[element_info.name] = element_id
        self.nodes_bitset[element_id] = (1 << len(self.element_info))

    def add_flow_arc(self, flow_id, source_id, target_id):
        for node_id in [source_id, target_id]:
            if node_id not in self.element_info:
                self.element_info[node_id] = ElementInfo(BPMNNodeType.UNDEFINED, node_id, node_id)
        self.element_info[source_id].outgoing_flows.append(flow_id)
        self.element_info[target_id].incoming_flows.append(flow_id)
        self.flow_arcs[flow_id] = [source_id, target_id]
        self.arcs_bitset[flow_id] = (1 << len(self.flow_arcs))

    def encode_or_join_predecessors(self):
        for e_id in self.element_info:
            element = self.element_info[e_id]
            if element.type is BPMNNodeType.INCLUSIVE_GATEWAY and element.is_join():
                self.or_join_pred[e_id] = [0, 0]
                self._find_or_conflicting_predecessors(e_id)
                pred_queue = deque([e_id])
                while len(pred_queue) > 0:
                    element = self.element_info[pred_queue.popleft()]
                    for flow_id in element.incoming_flows:
                        prev_id = self.flow_arcs[flow_id][0]
                        if self.or_join_pred[e_id][0] & self.nodes_bitset[prev_id] == 0:
                            pred_queue.append(prev_id)
                        self.or_join_pred[e_id][0] |= self.nodes_bitset[prev_id]
                        if self.flow_arcs[flow_id][1] != e_id:
                            self.or_join_pred[e_id][1] |= self.arcs_bitset[flow_id]
            if element.type in [BPMNNodeType.EXCLUSIVE_GATEWAY, BPMNNodeType.INCLUSIVE_GATEWAY] and element.is_split():
                self._find_decision_successors(element)

    def _find_decision_successors(self, split_info):
        self.decision_successors[split_info.id] = set()
        visited = {split_info.id}
        suc_queue = deque([split_info])
        while suc_queue:
            e_info = suc_queue.popleft()
            for out_flow in e_info.outgoing_flows:
                next_info = self._get_successor(out_flow)
                if next_info.id not in visited:
                    visited.add(next_info.id)
                    next_info = self.element_info[next_info.id]
                    if next_info.type is BPMNNodeType.TASK:
                        self.decision_successors[split_info.id].add(next_info.id)
                    elif next_info.is_gateway():
                        suc_queue.append(next_info)

    def _find_or_conflicting_predecessors(self, or_join_id):
        visited = {or_join_id}
        self.or_join_conflicting_pred[or_join_id] = set()
        for in_flow in self.element_info[or_join_id].incoming_flows:
            self._dfs_from_or_join(or_join_id, in_flow, self._get_predecessor(in_flow), visited)

    def _dfs_from_or_join(self, or_id, flow_id, e_info, visited):
        visited.add(e_info.id)
        if e_info.type in [BPMNNodeType.INCLUSIVE_GATEWAY, BPMNNodeType.EXCLUSIVE_GATEWAY] and e_info.is_split():
            self.or_join_conflicting_pred[or_id].add(e_info.id)
        for in_flow in e_info.incoming_flows:
            prev_info = self._get_predecessor(in_flow)
            if prev_info.id not in visited and prev_info.is_gateway():
                self._dfs_from_or_join(or_id, flow_id, prev_info, visited)

    # def discover_path(self, from_e_id, to_e_id):
    #     if from_e_id not in self.element_info or to_e_id not in self.element_info:
    #         return None
    #     visited_elements = dict()
    #     visited_elements[from_e_id] = None
    #     elements_queue = deque()
    #     elements_queue.append(from_e_id)
    #     while len(elements_queue) > 0:
    #         from_e_id = elements_queue.popleft()
    #         out_flows = self.element_info[from_e_id].outgoing_flows
    #         for flow_id in out_flows:
    #             next_e = self.element_info[self.flow_arcs[flow_id][1]]
    #             if next_e.id in visited_elements:
    #                 continue
    #             visited_elements[next_e.id] = flow_id
    #             if next_e == to_e_id:
    #                 return visited_elements
    #             if next_e.type in [BPMN.EXCLUSIVE_GATEWAY, BPMN.INCLUSIVE_GATEWAY, BPMN.PARALLEL_GATEWAY]:
    #                 elements_queue.append(next_e.id)
    #     return None

    def _is_enabled(self, e_id, p_state):
        if e_id not in self.element_info:
            return False
        if e_id == self.starting_event:
            return True
        e_info = self.element_info[e_id]
        if e_info.type in [BPMNNodeType.TASK, BPMNNodeType.END_EVENT, BPMNNodeType.PARALLEL_GATEWAY]:
            for f_arc in e_info.incoming_flows:
                if p_state.tokens[f_arc] < 1:
                    return False
            return True
        elif e_info.type == BPMNNodeType.EXCLUSIVE_GATEWAY:
            for f_arc in e_info.incoming_flows:
                if p_state.tokens[f_arc] > 0:
                    return True
            return False
        elif e_info.type == BPMNNodeType.INCLUSIVE_GATEWAY:
            if e_info.is_split():
                if p_state.has_token(e_info.incoming_flows[0]):
                    return True
                for flow_id in e_info.outgoing_flows:
                    if p_state.has_token(flow_id):
                        return True
                return False
            else:
                count_tokens = 0
                for flow_id in e_info.incoming_flows:
                    if p_state.tokens[flow_id] > 0:
                        count_tokens += 1
                if count_tokens == len(e_info.incoming_flows):
                    return True
                if count_tokens > 0 and self.or_join_pred[e_id][1] & p_state.state_mask == 0:
                    return True
                return False
        return False

    def update_process_state(self, e_id, p_state):
        if not self._is_enabled(e_id, p_state):
            return []
        enabled_tasks = list()
        to_execute = [e_id]
        current = 0
        while current < len(to_execute):
            e_info = self.element_info[to_execute[current]]
            for in_flow in e_info.incoming_flows:
                if p_state.tokens[in_flow] > 0:
                    p_state.tokens[in_flow] -= 1
                    p_state.state_mask &= ~self.arcs_bitset[in_flow]
            f_arcs = e_info.outgoing_flows
            if len(f_arcs) > 1:
                if e_info.type is BPMNNodeType.EXCLUSIVE_GATEWAY:
                    f_arcs = [self.element_probability[e_info.id].get_outgoing_flow()]
                else:
                    if e_info.type in [BPMNNodeType.TASK, BPMNNodeType.PARALLEL_GATEWAY, BPMNNodeType.START_EVENT]:
                        f_arcs = copy.deepcopy(e_info.outgoing_flows)
                    elif e_info.type is BPMNNodeType.INCLUSIVE_GATEWAY:
                        f_arcs = self.element_probability[e_info.id].get_multiple_flows()
                random.shuffle(f_arcs)
            for f_arc in f_arcs:
                self._find_next(f_arc, p_state, enabled_tasks, to_execute)
            current += 1
        if len(enabled_tasks) > 1:
            random.shuffle(enabled_tasks)
        return enabled_tasks

    def replay_trace(self, task_sequence: list, f_arcs_frequency: dict) -> (bool, List[bool], ProcessState):
        p_state = ProcessState(self)
        fired_tasks = list()
        fired_or_splits = set()
        for flow_id in self.element_info[self.starting_event].outgoing_flows:
            p_state.add_token(flow_id)
        pending_tasks = dict()
        for current_index in range(len(task_sequence)):
            fired_tasks.append(False)
            self.try_firing(current_index, current_index, task_sequence, fired_tasks, pending_tasks, p_state,
                            f_arcs_frequency, fired_or_splits)
            p_state.add_token(self.element_info[self.from_name[task_sequence[current_index]]].outgoing_flows[0])
            if current_index in pending_tasks:
                for pending_index in pending_tasks[current_index]:
                    self.try_firing(pending_index, current_index, task_sequence, fired_tasks, pending_tasks, p_state,
                                    f_arcs_frequency, fired_or_splits)

        # Firing End Event
        enabled_end, or_fired, path_decisions = self._find_enabled_predecessors(
            self.element_info[self.end_event], p_state)
        self._fire_enabled_predecessors(
            enabled_end, p_state, or_fired, path_decisions, f_arcs_frequency, fired_or_splits)
        end_flow = self.element_info[self.end_event].incoming_flows[0]
        if p_state.has_token(end_flow):
            p_state.tokens[end_flow] = 0

        is_correct = True
        for i in range(0, len(task_sequence)):
            if not fired_tasks[i]:
                is_correct = False
                break

        self.check_unfired_or_splits(fired_or_splits, f_arcs_frequency, p_state)
        return is_correct, fired_tasks, p_state.pending_tokens()

    def try_firing(self, task_index, from_index, task_sequence, fired_tasks, pending_tasks, p_state,
                   f_arcs_frequency, fired_or_splits):
        task_info = self.element_info[self.from_name[task_sequence[task_index]]]
        if not p_state.has_token(task_info.incoming_flows[0]):
            enabled_pred, or_fired, path_decisions = self._find_enabled_predecessors(task_info, p_state)
            firing_index = self.find_firing_index(task_index, from_index, task_sequence, path_decisions, enabled_pred)
            if firing_index == from_index:
                self._fire_enabled_predecessors(enabled_pred, p_state, or_fired, path_decisions, f_arcs_frequency,
                                                fired_or_splits)
            elif firing_index not in pending_tasks:
                pending_tasks[firing_index] = [task_index]
            else:
                pending_tasks[firing_index].append(task_index)
        if p_state.has_token(task_info.incoming_flows[0]):
            p_state.remove_token(task_info.incoming_flows[0])
            fired_tasks[task_index] = True

    def closer_enabled_predecessors(self, e_info, flow_id, enabled_pred, or_firing, path_split, visited, p_state, dist,
                                    min_dist):
        if self._is_enabled(e_info.id, p_state):
            if dist not in enabled_pred:
                enabled_pred[dist] = list()
            enabled_pred[dist].append([e_info, flow_id])
            min_dist[0] = max(min_dist[0], dist)
            return dist, enabled_pred, or_firing, path_split
        elif e_info.type is BPMNNodeType.INCLUSIVE_GATEWAY and e_info.is_join():
            for in_or in e_info.incoming_flows:
                if p_state.has_token(in_or):
                    or_firing[e_info.id] = dist
                    break
        if e_info.type in [BPMNNodeType.INCLUSIVE_GATEWAY, BPMNNodeType.EXCLUSIVE_GATEWAY]:
            path_split[e_info.id] = flow_id
        visited.add(e_info.id)
        if e_info.is_gateway():
            if e_info.type is BPMNNodeType.EXCLUSIVE_GATEWAY and e_info.is_join():
                closer_pred, temp_path, or_f = dict(), dict(), dict()
                c_min = sys.maxsize
                for in_flow in e_info.incoming_flows:
                    pr_info = self._get_predecessor(in_flow)
                    if pr_info.id not in visited:
                        d, e_p, o_f, t_path = self.closer_enabled_predecessors(pr_info, in_flow, dict(), dict(), dict(),
                                                                               visited, p_state, dist + 1, min_dist)
                        if d < c_min:
                            c_min, closer_pred, or_f, temp_path = d, e_p, o_f, t_path
                for e_id in closer_pred:
                    enabled_pred[e_id] = closer_pred[e_id]
                for e_id in temp_path:
                    path_split[e_id] = temp_path[e_id]
                for e_id in or_f:
                    or_firing[e_id] = dist
                return c_min, enabled_pred, or_firing, path_split
            else:
                c_min = dist if e_info.id in or_firing else sys.maxsize
                for in_flow in e_info.incoming_flows:
                    pred_info = self._get_predecessor(in_flow)
                    if pred_info.id not in visited and pred_info.is_gateway():
                        res = self.closer_enabled_predecessors(pred_info, in_flow, enabled_pred, or_firing, path_split,
                                                               visited, p_state, dist + 1, min_dist)
                        c_min = min(res[0], c_min)

                return c_min, enabled_pred, or_firing, path_split
        return sys.maxsize, enabled_pred, or_firing, path_split

    def _find_enabled_predecessors(self, from_task_info, p_state):
        pred_info = self._get_predecessor(from_task_info.incoming_flows[0])
        max_dist = [0]
        closer_pred = self.closer_enabled_predecessors(pred_info, from_task_info.incoming_flows[0], dict(),
                                                       dict(), dict(), set(), p_state, 0,
                                                       max_dist)
        enabled_pred = deque()
        for i in range(0, max_dist[0] + 1):
            if i in closer_pred[1]:
                for pred_id in closer_pred[1][i]:
                    enabled_pred.appendleft(pred_id)
        return enabled_pred, closer_pred[2], closer_pred[3]

    def find_firing_index(self, task_index, from_index, task_sequence, path_decisions, enabled_pred):
        is_conflicting, conflicting_gateways = self.is_conflicting_task(path_decisions, enabled_pred)
        if is_conflicting:
            firing_index = from_index
            for i in range(from_index + 1, len(task_sequence)):
                if task_sequence[i] != task_sequence[task_index]:
                    for or_id in conflicting_gateways:
                        for split_id in conflicting_gateways[or_id]:
                            t_id = self.from_name[task_sequence[i]]
                            if t_id in self.decision_successors[split_id]:
                                return i
            return firing_index
        return from_index

    def is_conflicting_task(self, path_decisions, enabled_pred):
        conflicting_gateways = dict()
        is_conflicting = False
        for or_id in path_decisions:
            if self.element_info[or_id].type is BPMNNodeType.INCLUSIVE_GATEWAY and self.element_info[or_id].is_join():
                conflicting_gateways[or_id] = set()
                for enabled in enabled_pred:
                    e_info = enabled[0]
                    if e_info.id in self.or_join_conflicting_pred[or_id]:
                        conflicting_gateways[or_id].add(e_info.id)
                    if len(conflicting_gateways[or_id]) > 1:
                        is_conflicting = True
        return is_conflicting, conflicting_gateways

    def _fire_enabled_predecessors(self, enabled_pred, p_state, or_firing, path_decisions, f_arcs_frequency,
                                   fired_or_split):
        visited_elements = set()
        if not enabled_pred:
            self.try_firing_or_join(enabled_pred, p_state, or_firing, path_decisions, f_arcs_frequency)
        while enabled_pred:
            [e_info, e_flow] = enabled_pred.popleft()
            if self._is_enabled(e_info.id, p_state):
                visited_elements.add(e_info.id)
                if e_info.type is BPMNNodeType.PARALLEL_GATEWAY:
                    for out_flow in e_info.outgoing_flows:
                        self._update_next(out_flow, enabled_pred, p_state, or_firing, path_decisions, f_arcs_frequency)
                elif e_info.type is BPMNNodeType.EXCLUSIVE_GATEWAY:
                    self._update_next(e_flow, enabled_pred, p_state, or_firing, path_decisions, f_arcs_frequency)
                elif e_info.type is BPMNNodeType.INCLUSIVE_GATEWAY:
                    self._update_next(e_flow, enabled_pred, p_state, or_firing, path_decisions, f_arcs_frequency)
                    if e_info.is_split():
                        fired_or_split.add(e_info.id)
                        for flow_id in e_info.outgoing_flows:
                            if flow_id != e_flow:
                                self._update_next(flow_id, enabled_pred, p_state, or_firing, path_decisions,
                                                  f_arcs_frequency, True)
            for in_flow in e_info.incoming_flows:
                p_state.remove_token(in_flow)
            self.try_firing_or_join(enabled_pred, p_state, or_firing, path_decisions, f_arcs_frequency)

    def try_firing_or_join(self, enabled_pred, p_state, or_firing, path_decisions, f_arcs_frequency):
        fired = set()
        or_firing_list = list()
        for or_join_id in or_firing:
            or_firing_list.append(or_join_id)
        for or_join_id in or_firing_list:
            if self._is_enabled(or_join_id, p_state) or not enabled_pred:
                fired.add(or_join_id)
                e_info = self.element_info[or_join_id]
                self._update_next(e_info.outgoing_flows[0], enabled_pred, p_state, or_firing, path_decisions,
                                  f_arcs_frequency)
                for in_flow in e_info.incoming_flows:
                    p_state.remove_token(in_flow)
                if enabled_pred:
                    break
                if len(or_firing_list) != len(or_firing):
                    for e_id in or_firing:
                        if e_id not in or_firing_list:
                            or_firing_list.append(e_id)
        for or_id in fired:
            del or_firing[or_id]

    def check_unfired_or_splits(self, or_splits, f_arcs_frequency, p_state):
        for or_id in or_splits:
            for flow_id in self.element_info[or_id].outgoing_flows:
                if p_state.tokens[flow_id] > 0:
                    f_arcs_frequency[flow_id] -= p_state.tokens[flow_id]
                    p_state.tokens[flow_id] = 0

    def _update_next(self, flow_id, enabled_pred, p_state, or_firing, path_decisions, f_arcs_frequency, from_or=False):
        if flow_id not in f_arcs_frequency:
            f_arcs_frequency[flow_id] = 1
        else:
            f_arcs_frequency[flow_id] += 1
        p_state.add_token(flow_id)
        if not from_or:
            next_info = self._get_successor(flow_id)
            if next_info.type is BPMNNodeType.PARALLEL_GATEWAY and self._is_enabled(next_info.id, p_state):
                enabled_pred.appendleft([next_info, None])
            elif next_info.id in path_decisions:
                if next_info.type is BPMNNodeType.INCLUSIVE_GATEWAY:
                    if next_info.is_split():
                        enabled_pred.appendleft([next_info, path_decisions[next_info.id]])
                    else:
                        if next_info.id not in or_firing:
                            or_firing[next_info.id] = 1
                else:
                    enabled_pred.appendleft([next_info, path_decisions[next_info.id]])

    def _get_predecessor(self, flow_id):
        return self.element_info[self.flow_arcs[flow_id][0]]

    def _get_successor(self, flow_id):
        return self.element_info[self.flow_arcs[flow_id][1]]

    def compute_branching_probability(self, flow_arcs_frequency):
        gateways_branching = dict()
        for e_id in self.element_info:
            if self.element_info[e_id].type == BPMNNodeType.EXCLUSIVE_GATEWAY and len(
                    self.element_info[e_id].outgoing_flows) > 1:
                total_frequency = 0
                for flow_id in self.element_info[e_id].outgoing_flows:
                    total_frequency += flow_arcs_frequency[flow_id]
                flow_arc_probability = dict()
                for flow_id in self.element_info[e_id].outgoing_flows:
                    flow_arc_probability[flow_id] = flow_arcs_frequency[flow_id] / total_frequency
                gateways_branching[e_id] = flow_arc_probability
        return gateways_branching

    def _find_next(self, f_arc, p_state, enabled_tasks, to_execute):
        p_state.tokens[f_arc] += 1
        p_state.state_mask |= self.arcs_bitset[f_arc]
        next_e = self.flow_arcs[f_arc][1]
        if self._is_enabled(next_e, p_state):
            if self.element_info[next_e].type == BPMNNodeType.TASK:
                enabled_tasks.append(next_e)
            else:
                to_execute.append(next_e)


# Calendar

str_week_days = {"MONDAY": 0, "TUESDAY": 1, "WEDNESDAY": 2, "THURSDAY": 3, "FRIDAY": 4, "SATURDAY": 5, "SUNDAY": 6}
int_week_days = {0: "MONDAY", 1: "TUESDAY", 2: "WEDNESDAY", 3: "THURSDAY", 4: "FRIDAY", 5: "SATURDAY", 6: "SUNDAY"}
convertion_table = {'WEEKS': 604800, 'DAYS': 86400, 'HOURS': 3600, 'MINUTES': 60, 'SECONDS': 1}


class Interval:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.duration = (end - start).total_seconds()

    def merge_interval(self, n_interval):
        self.start = min(n_interval.start, self.start)
        self.end = max(n_interval.end, self.end)
        self.duration = (self.end - self.start).total_seconds()

    def is_before(self, c_date):
        return self.end <= c_date

    def contains(self, c_date):
        return self.start < c_date < self.end

    def is_after(self, c_date):
        return c_date <= self.start


class RCalendar:
    def __init__(self, calendar_id):
        self.calendar_id = calendar_id
        self.default_date = None
        self.new_day = None
        self.work_intervals = dict()
        self.cumulative_work_durations = dict()
        self.work_rest_count = dict()
        self.total_weekly_work = 0
        self.total_weekly_rest = self._to_seconds(1, 'WEEKS')
        for i in range(0, 7):
            self.work_intervals[i] = list()
            self.cumulative_work_durations[i] = list()
            self.work_rest_count[i] = [0, self._to_seconds(1, 'DAYS')]

    @staticmethod
    def _to_seconds(value, from_unit):
        u_from = from_unit.upper()
        return value * convertion_table[u_from] if u_from in convertion_table else value

    @staticmethod
    def _parse_datetime(time, has_date):
        time_formats = ['%H:%M:%S.%f', '%H:%M', '%I:%M%p', '%H:%M:%S', '%I:%M:%S%p'] if not has_date \
            else ['%Y-%m-%dT%H:%M:%S.%f%z', '%b %d %Y %I:%M%p', '%b %d %Y at %I:%M%p', '%B %d, %Y, %H:%M:%S',
                  '%a,%d/%m/%y,%I:%M%p', '%a, %d %B, %Y', '%Y-%m-%dT%H:%M:%SZ']
        for time_format in time_formats:
            try:
                return datetime.datetime.strptime(time, time_format)
            except ValueError:
                pass
        raise ValueError

    def print_calendar_info(self):
        print('Calendar ID: %s' % self.calendar_id)
        print('Total Weekly Work: %.2f Hours' % (self.total_weekly_work / 3600))
        for i in range(0, 7):
            if len(self.work_intervals[i]) > 0:
                print(int_week_days[i])
                for interval in self.work_intervals[i]:
                    print('    from %02d:%02d - to %02d:%02d' % (interval.start.hour, interval.start.minute,
                                                                 interval.end.hour, interval.end.minute))
        print('-----------------------------------------------------------')

    def to_json(self):
        items = []
        for i in range(0, 7):
            if len(self.work_intervals[i]) > 0:
                for interval in self.work_intervals[i]:
                    items.append({
                        'from': int_week_days[i],
                        'to': int_week_days[i],
                        "beginTime": str(interval.start.time()),
                        "endTime": str(interval.end.time())
                    })
        return items

    def add_calendar_item(self, from_day, to_day, begin_time, end_time):
        if from_day.upper() in str_week_days and to_day.upper() in str_week_days:
            try:
                t_interval = Interval(self._parse_datetime(begin_time, False), self._parse_datetime(end_time, False))
                if self.default_date is None:
                    self.default_date = t_interval.start.date()
                    self.new_day = datetime.datetime.combine(self.default_date, datetime.time())
                d_s = str_week_days[from_day]
                d_e = str_week_days[to_day]
                while True:
                    self._add_interval(d_s % 7, t_interval)
                    if d_s % 7 == d_e:
                        break
                    d_s += 1
            except ValueError:
                return

    def compute_cumulative_durations(self):
        for w_day in self.work_intervals:
            cumulative = 0
            for interval in self.work_intervals[w_day]:
                cumulative += interval.duration
                self.cumulative_work_durations[w_day].append(cumulative)

    def _add_interval(self, w_day, interval):
        i = 0
        for to_check in self.work_intervals[w_day]:
            if to_check.end < interval.start:
                i += 1
                continue
            if interval.end < to_check.start:
                break
            prev_dur = to_check.duration
            to_check.merge_interval(interval)
            if to_check.duration > prev_dur:
                self._update_calendar_durations(w_day, to_check.duration - prev_dur)
            return
        self.work_intervals[w_day].insert(i, interval)
        self._update_calendar_durations(w_day, interval.duration)

    def _update_calendar_durations(self, w_day, duration):
        self.work_rest_count[w_day][0] += duration
        self.work_rest_count[w_day][1] -= duration
        self.total_weekly_work += duration
        self.total_weekly_rest -= duration

    def find_idle_time(self, requested_date, duration):
        real_duration = 0
        pending_duration = duration
        if duration > self.total_weekly_work:
            real_duration += self._to_seconds(int(duration / self.total_weekly_work), 'WEEKS')
            pending_duration %= self.total_weekly_work
        # Addressing the first day as an special case
        c_day = requested_date.date().weekday()
        c_date = datetime.datetime.combine(self.default_date, requested_date.time())

        worked_time, total_time = self._find_time_starting(pending_duration, c_day, c_date)
        pending_duration -= worked_time
        real_duration += total_time
        c_date = self.new_day
        while pending_duration > 0:
            c_day += 1
            r_d = c_day % 7
            if pending_duration > self.work_rest_count[r_d][0]:
                pending_duration -= self.work_rest_count[r_d][0]
                real_duration += 86400
            else:
                real_duration += self._find_time_completion(pending_duration, self.work_rest_count[r_d][0], r_d, c_date)
                break
        return real_duration

    def next_available_time(self, requested_date):
        c_day = requested_date.date().weekday()
        c_date = datetime.datetime.combine(self.default_date, requested_date.time())

        for interval in self.work_intervals[c_day]:
            if interval.end == c_day:
                continue
            if interval.is_after(c_date):
                return (interval.start - c_date).total_seconds()
            if interval.contains(c_date):
                return 0
        duration = 86400 - (c_date - self.new_day).total_seconds()
        for i in range(c_day + 1, c_day + 8):
            r_day = i % 7
            if self.work_rest_count[r_day][0] > 0:
                return duration + (self.work_intervals[r_day][0].start - self.new_day).total_seconds()
            duration += 86400
        return duration

    def find_working_time(self, start_date, end_date):
        # print("%s -- %s" % (str(start_date), str(end_date)))
        pending_duration = (end_date - start_date).total_seconds()
        worked_hours = 0

        c_day = start_date.date().weekday()
        c_date = datetime.datetime.combine(self.default_date, start_date.time())

        to_complete_day = 86400 - (c_date - self.new_day).total_seconds()
        available_work = self._calculate_available_duration(c_day, c_date)

        previous_date = c_date
        while pending_duration > to_complete_day:
            pending_duration -= to_complete_day
            worked_hours += available_work
            c_day = (c_day + 1) % 7
            available_work = self.work_rest_count[c_day][0]
            to_complete_day = 86400
            previous_date = self.new_day

        for interval in self.work_intervals[c_day]:
            if interval.is_before(previous_date):
                continue
            interval_duration = interval.duration
            if interval.contains(previous_date):
                interval_duration -= (previous_date - interval.start).total_seconds()
            else:
                pending_duration -= (interval.start - previous_date).total_seconds()
            if pending_duration >= interval_duration:
                worked_hours += interval_duration
            elif pending_duration > 0:
                worked_hours += pending_duration
            pending_duration -= interval_duration
            if pending_duration <= 0:
                break
            previous_date = interval.end
        # print("Worked-hours: %s" % worked_hours)
        # print('-----------------------------------------------------')
        return worked_hours

    def _find_time_starting(self, pending_duration, c_day, from_date):
        available_duration = self._calculate_available_duration(c_day, from_date)
        if available_duration <= pending_duration:
            return available_duration, 86400 - (from_date - self.new_day).total_seconds()
        else:
            return pending_duration, self._find_time_completion(pending_duration, available_duration, c_day, from_date)

    def _calculate_available_duration(self, c_day, from_date):
        i = -1
        passed_duration = 0
        for t_interval in self.work_intervals[c_day]:
            i += 1
            if t_interval.is_before(from_date):
                passed_duration += t_interval.duration
                continue
            if t_interval.is_after(from_date):
                break
            if t_interval.contains(from_date):
                passed_duration += (from_date - self.work_intervals[c_day][i].start).total_seconds()
                break

        return self.work_rest_count[c_day][0] - passed_duration

    def _find_time_completion(self, pending_duration, total_duration, c_day, from_datetime):
        i = len(self.work_intervals[c_day]) - 1
        while total_duration > pending_duration:
            total_duration -= self.work_intervals[c_day][i].duration
            i -= 1
        if total_duration < pending_duration:
            to_datetime = self.work_intervals[c_day][i + 1].start + timedelta(
                seconds=(pending_duration - total_duration))
            return (to_datetime - from_datetime).total_seconds()
        else:
            return (self.work_intervals[c_day][i].end - from_datetime).total_seconds()


def best_fit_distribution(data, bins=200):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    d_min = sys.float_info.max
    d_max = 0
    for d_data in data:
        d_min = min(d_min, d_data)
        d_max = max(d_max, d_data)

    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    distributions = [
        st.alpha, st.anglit, st.arcsine, st.beta, st.betaprime, st.bradford, st.burr, st.cauchy, st.chi, st.chi2,
        st.cosine, st.dgamma, st.dweibull, st.erlang, st.expon, st.exponnorm, st.exponweib, st.exponpow, st.f,
        st.fatiguelife, st.fisk, st.foldcauchy, st.foldnorm, st.genlogistic, st.genpareto,
        st.gennorm, st.genextreme, st.gausshyper, st.gamma, st.gengamma, st.genhalflogistic, st.gilbrat, st.gompertz,
        st.gumbel_r, st.gumbel_l, st.halfcauchy, st.halflogistic, st.halfnorm, st.halfgennorm, st.hypsecant,
        st.invgamma, st.invgauss, st.invweibull, st.johnsonsb, st.johnsonsu, st.ksone, st.kstwobign, st.laplace,
        st.levy, st.levy_l, st.logistic, st.loggamma, st.loglaplace, st.lognorm, st.lomax, st.maxwell, st.mielke,
        st.nakagami, st.ncx2, st.ncf, st.nct, st.norm, st.pareto, st.pearson3, st.powerlaw, st.powerlognorm,
        st.powernorm, st.rdist, st.reciprocal, st.rayleigh, st.rice, st.semicircular, st.t, st.triang, st.truncexpon,
        st.truncnorm, st.tukeylambda, st.uniform, st.vonmises, st.vonmises_line, st.wald, st.weibull_min,
        st.weibull_max, st.wrapcauchy
    ]

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data
    i = 1
    for distribution in distributions:

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            # start = time.time()
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse
            # end = time.time()
            # print("%d- %s: %.3f" % (i, distribution.name, end - start))
            i += 1
        except Exception:
            pass

    return {"distribution_name": best_distribution.name, "distribution_params": best_params}


class StochasticProcessMiner:
    settings: Configuration
    bpmn_graph: BPMNGraph
    gateways_branching: dict

    def __init__(self, settings: Configuration):
        self.settings = settings

    def execute_pipeline(self):
        print_section('Preparing the environment')
        if not os.path.exists(self.settings.output):
            print_step(f'Creating output directories: {self.settings.output}')
            os.makedirs(self.settings.output)
        if self.settings.model_path:
            print_step(f'Copying the model from {self.settings.model_path}')
            shutil.copy(self.settings.model_path, self.settings.output)

        print_section('Parsing the given model')
        self.bpmn_graph = self._parse_simulation_model(self.settings.model_path)

        print_section(f'Model parameters extraction')
        print_step('Calculating sequence flow frequencies')
        arcs_frequencies = self._compute_sequence_flow_frequencies(self.settings.log_path, self.bpmn_graph)
        self.gateways_branching = self.bpmn_graph.compute_branching_probability(arcs_frequencies)
        print_asset(f'Sequence flow frequencies: {self.gateways_branching}')

        print_step('Rewriting the model')
        sequences = []
        for gateway_id in self.gateways_branching:
            for seqflow_id in self.gateways_branching[gateway_id]:
                probability = self.gateways_branching[gateway_id][seqflow_id]
                sequences.append({'elementid': seqflow_id, 'prob': probability})
        parameters = {'sequences': sequences}
        bpmn_path = os.path.join(self.settings.output, self.settings.project_name + '.bpmn')
        bpmn_path_new = os.path.join(self.settings.output, self.settings.project_name + '_updated.bpmn')
        xml_writer.print_parameters(bpmn_path, bpmn_path_new, parameters)

    @staticmethod
    def _compute_sequence_flow_frequencies(log_path: Path, bpmn_graph: BPMNGraph) -> dict:
        flow_arcs_frequency = dict()

        log_traces = xes_importer.apply(log_path.__str__())
        for trace in log_traces:
            task_sequence = list()
            for event in trace:
                task_name = event['concept:name']
                state = event['lifecycle:transition'].lower()
                if state in ["start", "assign"]:
                    task_sequence.append(task_name)

            bpmn_graph.replay_trace(task_sequence, flow_arcs_frequency)

        return flow_arcs_frequency

    @staticmethod
    def _parse_simulation_model(model_path: Path):
        bpmn_element_ns = {'xmlns': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}
        tree = ET.parse(model_path.absolute())
        root = tree.getroot()
        to_extract = {'xmlns:task': BPMNNodeType.TASK,
                      'xmlns:startEvent': BPMNNodeType.START_EVENT,
                      'xmlns:endEvent': BPMNNodeType.END_EVENT,
                      'xmlns:exclusiveGateway': BPMNNodeType.EXCLUSIVE_GATEWAY,
                      # NOTE: no parallel gateways in current Simod models
                      'xmlns:parallelGateway': BPMNNodeType.PARALLEL_GATEWAY,
                      'xmlns:inclusiveGateway': BPMNNodeType.INCLUSIVE_GATEWAY}

        bpmn_graph = BPMNGraph()
        for process in root.findall('xmlns:process', bpmn_element_ns):
            for xmlns_key in to_extract:
                for bpmn_element in process.findall(xmlns_key, bpmn_element_ns):
                    name = bpmn_element.attrib["name"] \
                        if "name" in bpmn_element.attrib and len(bpmn_element.attrib["name"]) > 0 \
                        else bpmn_element.attrib["id"]
                    bpmn_graph.add_bpmn_element(bpmn_element.attrib["id"],
                                                ElementInfo(to_extract[xmlns_key], bpmn_element.attrib["id"], name))
            for flow_arc in process.findall('xmlns:sequenceFlow', bpmn_element_ns):
                bpmn_graph.add_flow_arc(flow_arc.attrib["id"], flow_arc.attrib["sourceRef"],
                                        flow_arc.attrib["targetRef"])
        bpmn_graph.encode_or_join_predecessors()

        return bpmn_graph

    @staticmethod
    def parse_xes_log(project_name: str, log_path: Path, bpmn_graph: BPMNGraph, output_folder: Path):
        def _update_first_last(start_date, end_date, current_date):
            if start_date is None:
                start_date = current_date
                end_date = current_date
            else:
                start_date = min(start_date, current_date)
                end_date = max(end_date, current_date)
            return start_date, end_date

        def _update_calendar_from_log(r_calendar, date_time, is_start, min_eps=15):
            from_date = date_time
            to_date = date_time
            if is_start:
                to_date = date_time + timedelta(minutes=min_eps)
            else:
                from_date = date_time - timedelta(minutes=min_eps)

            from_day = int_week_days[from_date.weekday()]
            to_day = int_week_days[to_date.weekday()]

            if from_day != to_day:
                r_calendar.add_calendar_item(
                    from_day, from_day,
                    "%d:%d:%d" % (from_date.hour, from_date.minute, from_date.second), "23:59:59.999")
                if to_date.hour != 0 or to_date.minute != 0 or to_date.second != 0:
                    r_calendar.add_calendar_item(
                        to_day, to_day, "00:00:00", "%d:%d:%d" % (to_date.hour, to_date.minute, to_date.second))
            else:
                r_calendar.add_calendar_item(
                    from_day, to_day,
                    "%d:%d:%d" % (from_date.hour, from_date.minute, from_date.second),
                    "%d:%d:%d" % (to_date.hour, to_date.minute, to_date.second))

        print('Parsing Event Log %s ...' % project_name)
        process_info = ProcessInfo()
        i = 0
        total_traces = 0
        resource_list = set()

        task_resource = dict()
        task_distribution = dict()
        flow_arcs_frequency = dict()
        correct_traces = 0
        correct_activities = 0
        total_activities = 0
        task_fired_ratio = dict()
        task_missed_tokens = 0
        missed_tokens = dict()

        log_traces = xes_importer.apply(log_path)

        arrival_times = list()
        previous_arrival_date = None

        start_date = end_date = None
        resource_calendars = dict()

        for trace in log_traces:
            if previous_arrival_date is not None:
                arrival_times.append((trace[0]['time:timestamp'] - previous_arrival_date).total_seconds())
            previous_arrival_date = trace[0]['time:timestamp']

            caseid = trace.attributes['concept:name']
            total_traces += 1
            started_events = dict()
            trace_info = Trace(caseid)
            task_sequence = list()
            for event in trace:
                task_name = event['concept:name']
                resource = event['org:resource']
                state = event['lifecycle:transition'].lower()
                timestamp = event['time:timestamp']
                start_date, end_date = _update_first_last(start_date, end_date, timestamp)
                if resource not in resource_list:
                    resource_list.add(resource)
                    resource_calendars[resource] = RCalendar("%s_Schedule" % resource)
                _update_calendar_from_log(resource_calendars[resource], timestamp, state in ["start", "assign"])
                if state in ["start", "assign"]:
                    started_events[task_name] = trace_info.start_event(
                        task_name, task_name, timestamp, resource, timestamp, None)
                    task_sequence.append(task_name)
                elif state == "complete":
                    if task_name in started_events:
                        event_info = trace_info.complete_event(started_events.pop(task_name), timestamp)
                        if task_name not in task_resource:
                            task_resource[task_name] = dict()
                            task_distribution[task_name] = dict()
                        if resource not in task_resource[task_name]:
                            task_resource[task_name][resource] = list()
                        task_resource[task_name][resource].append(event_info)
            is_correct, fired_tasks, pending_tokens = bpmn_graph.replay_trace(task_sequence, flow_arcs_frequency)
            if len(pending_tokens) > 0:
                task_missed_tokens += 1
                for flow_id in pending_tokens:
                    if flow_id not in missed_tokens:
                        missed_tokens[flow_id] = 0
                    missed_tokens[flow_id] += 1
            if is_correct:
                correct_traces += 1
            for i in range(0, len(task_sequence)):
                if task_sequence[i] not in task_fired_ratio:
                    task_fired_ratio[task_sequence[i]] = [0, 0]
                if fired_tasks[i]:
                    correct_activities += 1
                    task_fired_ratio[task_sequence[i]][0] += 1
                task_fired_ratio[task_sequence[i]][1] += 1
            total_activities += len(fired_tasks)
            process_info.traces[caseid] = trace_info
            i += 1

        t_r = 100 * correct_traces / total_traces
        a_r = 100 * correct_activities / total_activities
        print("Correct Traces Ratio %.2f (Pass: %d, Fail: %d, Total: %d)" % (
            t_r, correct_traces, total_traces - correct_traces, total_traces))
        print("Correct Tasks  Ratio %.2f (Fire: %d, Fail: %d, Total%d: d)" % (
            a_r, correct_activities, total_activities - correct_activities, total_activities))
        print("Missed Tokens Ratio  %.2f" % (100 * task_missed_tokens / total_traces))
        print('----------------------------------------------')

        min_dur = sys.float_info.max
        max_dur = 0
        for r_id in resource_calendars:
            min_dur = min(min_dur, resource_calendars[r_id].total_weekly_work)
            max_dur = max(max_dur, resource_calendars[r_id].total_weekly_work)
            # resource_calendars[r_id].print_calendar_info()
        print('Min Resource Weekly Work: %.2f ' % (min_dur / 3600))
        print('Max Resource Weekly Work: %.2f ' % (max_dur / 3600))
        print('Saving Resource Calendars ...')
        json_map = dict()
        for r_id in resource_calendars:
            json_map[r_id] = resource_calendars[r_id].to_json()
        with open(os.path.join(output_folder, f'{project_name}_calendars.json'), 'w') as f:
            json.dump(json_map, f)

        print('Computing Branching Probability ...')
        gateways_branching = bpmn_graph.compute_branching_probability(flow_arcs_frequency)
        with open(os.path.join(output_folder, f'{project_name}_gateways_branching.json'), 'w') as f:
            json.dump(gateways_branching, f)

        print('Computing Arrival Times Distribution ...')
        with open(os.path.join(output_folder, f'{project_name}_arrival_times_distribution.json'), 'w') as f:
            json.dump(best_fit_distribution(arrival_times), f)

        print('Computing Task-Resource Distributions ...')
        for task_id in task_resource:
            for resource_id in task_resource[task_id]:
                real_durations = list()
                for e_info in task_resource[task_id][resource_id]:
                    real_durations.append(resource_calendars[resource_id].find_working_time(
                        e_info.started_at, e_info.completed_at))
                    if real_durations[len(real_durations) - 1] <= 0 and e_info.started_at != e_info.completed_at:
                        x = resource_calendars[resource_id].find_working_time(e_info.started_at, e_info.completed_at)
                        print(real_durations[len(real_durations) - 1])
                task_distribution[task_id][resource_id] = best_fit_distribution(real_durations)
        with open(os.path.join(output_folder, f'{project_name}_task_distribution.json'), 'w') as f:
            json.dump(task_distribution, f)

        return process_info
