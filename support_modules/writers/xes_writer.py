# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 14:23:00 2019

@author: manuel.camargo

Read a csv file and convert that in xes file
"""
import itertools as it
from opyenxes.factory.XFactory import XFactory
from opyenxes.data_out.XesXmlSerializer import XesXmlSerializer

def convert_line_in_event(csv_mapping: dict, event: dict, one_timestamp:bool):
    """Read one line and convert in a Xes Event object
    :param type_for_attribute: dictionary with the type of all attribute.
    :param attribute_list: dict with the attribute in string format
    :return: An XEvent with the respective attribute
    """
    # TODO: Add the use of extensions and optimize code
    events = list()
    attribute_map = XFactory.create_attribute_map()
    for attr_type, attr_value in event.items():
        attribute_type = csv_mapping[attr_type]
        if attribute_type in ["Activity", "Resource"]:
            attribute = XFactory.create_attribute_literal(attribute_type, attr_value)
            attribute_map[attribute.get_key()] = attribute
            if attribute_type == "Activity":
                attribute = XFactory.create_attribute_literal('concept:name', attr_value)
                attribute_map[attribute.get_key()] = attribute
            if attribute_type == "Resource":
                attribute = XFactory.create_attribute_literal('org:resource', attr_value)
                attribute_map[attribute.get_key()] = attribute            
        elif attribute_type == 'Complete Timestamp':
            attribute = XFactory.create_attribute_timestamp("time:timestamp", attr_value)
            attribute_map[attribute.get_key()] = attribute
            attribute2 = XFactory.create_attribute_literal('lifecycle:transition', 'complete')
            attribute_map[attribute2.get_key()] = attribute2
        elif attribute_type in ['Case ID', 'Event ID', 'Start Timestamp']:
            next
        else:
            attribute = XFactory.create_attribute_discrete(attribute_type, int(attr_value))
            attribute_map[attribute.get_key()] = attribute
    events.append(XFactory.create_event(attribute_map))
    if not one_timestamp:
        attribute_map = XFactory.create_attribute_map()
        for attr_type, attr_value in event.items():
            attribute_type = csv_mapping[attr_type]
            if attribute_type in ["Activity", "Resource"]:
                attribute = XFactory.create_attribute_literal(attribute_type, attr_value)
                attribute_map[attribute.get_key()] = attribute
                if attribute_type == "Activity":
                    attribute = XFactory.create_attribute_literal('concept:name', attr_value)
                    attribute_map[attribute.get_key()] = attribute
                if attribute_type == "Resource":
                    attribute = XFactory.create_attribute_literal('org:resource', attr_value)
                    attribute_map[attribute.get_key()] = attribute            
            elif attribute_type == 'Start Timestamp':
                attribute = XFactory.create_attribute_timestamp("time:timestamp", attr_value)
                attribute_map[attribute.get_key()] = attribute
                attribute2 = XFactory.create_attribute_literal('lifecycle:transition', 'start')
                attribute_map[attribute2.get_key()] = attribute2
            elif attribute_type in ['Case ID', 'Event ID', 'Complete Timestamp']:
                next
            else:
                attribute = XFactory.create_attribute_discrete(attribute_type, int(attr_value))
                attribute_map[attribute.get_key()] = attribute
        events.insert(0, XFactory.create_event(attribute_map))
        
    return events


def create_xes_file(input_log, output_file, read_options):
    csv_mapping = { v:k for k,v in read_options['column_names'].items()}
    log = XFactory.create_log()
    data = sorted(input_log.data, key=lambda x:x['caseid'])
    for key, group in it.groupby(data, key=lambda x:x['caseid']):
        if read_options['one_timestamp']:
            csv_trace = sorted(list(group), key=lambda x:x['end_timestamp'])
        else:
            csv_trace = sorted(list(group), key=lambda x:x['start_timestamp'])
        events = list()
        for line in csv_trace:
            events.extend(convert_line_in_event(csv_mapping, line, read_options['one_timestamp']))
        trace_attribute=XFactory.create_attribute_literal('concept:name', key)
        trace_attribute_map = XFactory.create_attribute_map()
        trace_attribute_map[trace_attribute.get_key()] = trace_attribute
        trace = XFactory.create_trace(attribute=trace_attribute_map)
        for event in events:
            trace.append(event)
        log.append(trace)

    
    # Save log in xes format
    with open(output_file, "w") as file:
        XesXmlSerializer().serialize(log, file)
