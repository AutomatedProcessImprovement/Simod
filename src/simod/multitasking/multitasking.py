import copy
import itertools
import uuid
import xml.etree.ElementTree as et
from functools import reduce
from pathlib import Path
from typing import List, Dict, Tuple

import pendulum

XES_NAMESPACE = 'http://www.xes-standard.org'
XML_EVENT_TAG = f'{{{XES_NAMESPACE}}}event'
XML_TRACE_TAG = f'{{{XES_NAMESPACE}}}trace'
XML_DATE_TAG = f'{{{XES_NAMESPACE}}}date'
XML_STRING_TAG = f'{{{XES_NAMESPACE}}}string'
XPATH_ACTIVITY = "*[@key='concept:name']"
XPATH_RESOURCE = "*[@key='org:resource']"
XPATH_END_TIMESTAMP = "*[@key='time:timestamp']"
XPATH_EVENT_END_TIME = "*[@key='EventEndTime']"
XPATH_EVENT_ID = "*[@key='EventID']"

namespaces = {
    'xes': XES_NAMESPACE,
    'time': f"{XES_NAMESPACE}/time.xesext"
}

et.register_namespace('', XES_NAMESPACE)


def _pre_sweeper(xes_input_path: Path) -> et.ElementTree:
    tree = et.parse(xes_input_path)
    resources = {}
    for event in tree.iter(XML_EVENT_TAG):
        name: str = event.find(XPATH_ACTIVITY).attrib.get('value')
        resource: str = event.find(XPATH_RESOURCE).attrib.get('value')

        if resource not in resources:
            resources[resource] = {}

        if name not in resources[resource]:
            event_id = str(uuid.uuid4())
            resources[resource][name] = {'start_event': event, 'event_id': event_id}
        else:
            event_id = resources[resource][name]['event_id']
            start_event = resources[resource][name]['start_event']
            end_time: str = event.find(XPATH_END_TIMESTAMP).attrib.get('value')
            et.SubElement(start_event, XML_DATE_TAG, {'key': 'EventEndTime', 'value': end_time})
            del resources[resource][name]

        et.SubElement(event, XML_STRING_TAG, {'key': 'EventID', 'value': event_id})

    return tree


def _apply_percentage(tree: et.ElementTree, percent: float = 0.05) -> et.ElementTree:
    global namespaces

    root = tree.getroot()

    resources = _extract_resources(root)

    for resource in resources:
        resources[resource] = list(filter(lambda event: event[0] != event[1], resources[resource]))

    pairs_of_events = _extract_pairs_of_events(resources, percent)

    _update_timestamps_from_pairs(resources, pairs_of_events, root)

    return tree


def _apply_sweeper(tree: et.ElementTree) -> et.ElementTree:
    global namespaces

    root = tree.getroot()

    resources = _extract_resources(root)

    event_end_mapping = {}
    traces = root.findall(XML_TRACE_TAG, namespaces)
    for resource in resources:
        adjusted_ranges = _adjust_ranges(resources[resource], event_end_mapping)
        for trace in traces:
            events = trace.findall(f"./{XML_EVENT_TAG}", namespaces)
            for event in events:
                _update_end_timestamp_node_for_resource(adjusted_ranges, event, resource)

    return tree


def _update_end_timestamp_node_for_resource(adjusted_ranges, event, resource):
    event_resource = event.find(XPATH_RESOURCE)
    if not event_resource:
        return

    if event_resource.attrib.get('value') != resource:
        return

    event_end_time = event.find(XPATH_EVENT_END_TIME)
    if not event_end_time:
        return

    end_timestamp_node = event.find(XPATH_END_TIMESTAMP)
    if not end_timestamp_node:
        return

    event_id = event.find(XPATH_EVENT_ID).attrib.get('value')
    stamp = int(adjusted_ranges[event_id] / 1000.0)
    _update_date_node(end_timestamp_node, stamp)


def _extract_resources(root: et.Element) -> Dict[str, List]:
    global namespaces
    events = root.findall(f"./{XML_TRACE_TAG}/{XML_EVENT_TAG}", namespaces)
    resources = {}
    for event in events:
        resource = event.find(XPATH_RESOURCE).attrib.get('value')
        if resource not in resources:
            resources[resource] = []

        dates = event.findall(f"./{XML_DATE_TAG}", namespaces)
        if len(dates) > 1:
            start = dates[0].attrib.get('value')
            start_date = pendulum.parse(start)
            start_ms = int(round(start_date.timestamp() * 1000))

            end = dates[1].attrib.get('value')
            end_date = pendulum.parse(end)
            end_ms = int(round(end_date.timestamp() * 1000))

            event_id = event.find(XPATH_EVENT_ID).attrib.get('value')
            task_id = event.find(XPATH_ACTIVITY).attrib.get('value')
            resources[resource].append([start_ms, end_ms, task_id, event_id])
    return resources


def _get_events_by_resource(root: et.Element, resource: str) -> List[et.Element]:
    global namespaces
    xpath = f"./{XML_TRACE_TAG}/{XML_EVENT_TAG}/{XPATH_RESOURCE}[@value='{resource}']/.."
    return root.findall(xpath, namespaces)


def _update_date_node(node: et.Element, timestamp_ms: int):
    date = pendulum.from_timestamp(timestamp_ms / 1000.0)
    node.attrib['value'] = date.isoformat()


def _extract_pairs_of_events(resources, percent):
    pairs_of_events = []
    visited_event_ids = []
    for resource, events in resources.items():
        i = 0
        for event in events:
            if event[3] not in visited_event_ids:
                for event2 in events[i:]:
                    visited_event_ids.append(event2[3])
                    visited_event_ids.append(event[3])
                    pairs_of_events.append([event, event2])
                    break
            i += 1
    for pair in pairs_of_events:
        total_time = pair[0][1] - pair[0][0] + pair[1][1] - pair[1][0]
        movement_part = int(total_time * percent)
        pair[0][0] += movement_part
        pair[0][1] += movement_part
    return pairs_of_events


def _update_timestamps_from_pairs(resources: Dict[str, List], pairs_of_events: List, root: et.Element):
    global namespaces

    for resource in resources:
        events = _get_events_by_resource(root, resource)
        for event in events:
            event_id = event.find(XPATH_EVENT_ID).attrib.get('value')
            filtered = list(filter(lambda item: item[0][3] == event_id, pairs_of_events))
            if len(filtered) == 0:
                continue

            stamp = int(filtered[0][0][1] / 1000.0)

            end_timestamp_node = event.find(XPATH_END_TIMESTAMP)
            if end_timestamp_node:
                event_end_time = event.find(XPATH_EVENT_END_TIME)
                if event_end_time and event_end_time.attrib.get('value'):
                    stamp = int(filtered[0][0][0] / 1000.0)
                _update_date_node(end_timestamp_node, stamp)

            end_time_node = event.find(XPATH_EVENT_END_TIME)
            if end_time_node:
                _update_date_node(end_time_node, stamp)


def _adjust_ranges(input: List[Tuple], event_end_mapping: Dict):
    new_ranges = _build_range(input, event_end_mapping)
    new_ranges.sort(key=(lambda v: v[3]))

    adjusted_ranges = {}

    for event, go in itertools.groupby(new_ranges, lambda v: v[3]):
        items = list(go)
        items.sort(key=lambda v: v[1])

        sum_duration = reduce(lambda acc, x: acc + x[1] - x[0], items, 0)
        adjusted_ranges[event] = event_end_mapping[event] - sum_duration

    return adjusted_ranges


def _build_range(input, event_end_mapping: Dict) -> List:
    points = []

    event_task_mapping = {}

    for start, stop, task, eventid in input:
        event_task_mapping[eventid] = task
        event_end_mapping[eventid] = stop
        points.append((start, '+', copy.deepcopy(task), copy.deepcopy(eventid)))
        points.append((stop, '-', copy.deepcopy(task), copy.deepcopy(eventid)))
    points.sort()

    ranges = []
    current_stack = []
    current_stack_eventid = []
    last_start = None
    for offset, pm, task, eventid in points:
        if pm == '+':
            if last_start is not None:
                ranges.append((last_start, offset, copy.deepcopy(
                    current_stack), copy.deepcopy(current_stack_eventid)))
            current_stack.append(copy.deepcopy(task))
            current_stack_eventid.append(copy.deepcopy(eventid))
            last_start = offset
        elif pm == '-':
            ranges.append((last_start, offset, copy.deepcopy(
                current_stack), copy.deepcopy(current_stack_eventid)))
            current_stack.remove(copy.deepcopy(task))
            current_stack_eventid.remove(copy.deepcopy(eventid))
            if (len(current_stack) > 0):
                last_start = offset
            else:
                last_start = None

    new_ranges = []

    for range in ranges:
        pieces = len(range[3])
        if pieces > 1:
            diff = range[1] - range[0]
            piece_size = int(diff / pieces)
            initial_start = range[0]
            max_end = range[1]

            piece_number = 0
            for cur_event_id in range[3]:
                task = event_task_mapping[cur_event_id]
                modified = (initial_start + piece_size * piece_number,
                            min(max_end, initial_start + piece_size * piece_number + piece_size),
                            copy.deepcopy(task),
                            cur_event_id)
                piece_number += 1
                if modified not in new_ranges:
                    new_ranges.append(modified)
        else:
            if pieces == 1:
                cop = copy.deepcopy(range)
                cur_event_id = range[3][0]
                new_range = (cop[0], cop[1], copy.deepcopy(cop[2][0]), cur_event_id)
                if new_range not in new_ranges:
                    new_ranges.append(new_range)

    new_ranges.sort(key=lambda v: v[0])

    return new_ranges


def process_log(xes_input_path: Path, xes_output_path: Path):
    tree = _pre_sweeper(xes_input_path)
    tree = _apply_percentage(tree)
    tree = _apply_sweeper(tree)
    tree.write(xes_output_path)


if __name__ == '__main__':
    import sys

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    process_log(input_path, output_path)
