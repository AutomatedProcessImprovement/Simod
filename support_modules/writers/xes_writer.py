# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 14:23:00 2019

@author: manuel.camargo

Read a csv file and convert that in xes file
"""
import itertools as it
import os
from opyenxes.factory.XFactory import XFactory
# from opyenxes.info import XLogInfoFactory
from opyenxes.data_out.XesXmlSerializer import XesXmlSerializer
from opyenxes.extension.std.XLifecycleExtension import XLifecycleExtension as xlc


class XesWriter(object):
    """
    This class writes a process log in .xes format
    """

    def __init__(self, log, settings):
        """constructor"""
        self.log = log
        self.one_timestamp = settings['read_options']['one_timestamp']
        self.column_names = settings['read_options']['column_names']
        self.output_file = os.path.join(settings['output'],
                                        settings['file'].split('.')[0]+'.xes')

        self.create_xes_file()

    def create_xes_file(self):
        csv_mapping = {v: k for k, v in self.column_names.items()}
        log = XFactory.create_log()
        data = sorted(self.log.data, key=lambda x: x['caseid'])
        for key, group in it.groupby(data, key=lambda x: x['caseid']):
            sort_key = ('end_timestamp'
                        if self.one_timestamp else 'start_timestamp')
            csv_trace = sorted(list(group), key=lambda x: x[sort_key])
            events = list()
            for line in csv_trace:
                events.extend(self.convert_line_in_event(csv_mapping, line))
            trace_attribute=XFactory.create_attribute_literal('concept:name', key)
            trace_attribute_map = XFactory.create_attribute_map()
            trace_attribute_map[trace_attribute.get_key()] = trace_attribute
            trace = XFactory.create_trace(attribute=trace_attribute_map)
            for event in events:
                trace.append(event)
            log.append(trace)
            # log.set_info(classifier, info)

        # Save log in xes format
        with open(self.output_file, "w") as file:
            XesXmlSerializer().serialize(log, file)

    def convert_line_in_event(self, csv_mapping, event):
        """
        Parameters
        ----------
        csv_mapping : dictionary with the type of all attribute.
        event : dict with the attribute in string format

        Returns
        -------
        events : An XEvent with the respective attribute

        """
        transitions = [{'column': 'Complete Timestamp',
                        'value': xlc.StandardModel.COMPLETE,
                        'skiped': 'Start Timestamp'}]
        if not self.one_timestamp:
            transitions.insert(0,{'column': 'Start Timestamp',
                                  'value': xlc.StandardModel.START,
                                  'skiped': 'Complete Timestamp'})
        # TODO: Add the use of extensions and optimize code
        events = list()
        for transition in transitions:
            attribute_map = XFactory.create_attribute_map()
            for attr_type, attr_value in event.items():
                attribute_type = csv_mapping[attr_type]
                if attribute_type in ["Activity", "Resource"]:
                    if attribute_type == "Activity":
                        attribute = XFactory.create_attribute_literal(
                            'concept:name', attr_value, extension=None)
                        attribute_map[attribute.get_key()] = attribute
                    if attribute_type == "Resource":
                        attribute = XFactory.create_attribute_literal(
                            'org:resource', attr_value, extension=None)
                        attribute_map[attribute.get_key()] = attribute
                elif attribute_type == transition['column']:
                    attribute = XFactory.create_attribute_timestamp(
                        "time:timestamp", attr_value, extension=None)
                    attribute_map[attribute.get_key()] = attribute
                    attribute2 = XFactory.create_attribute_literal(
                        'lifecycle:transition',
                        transition['value'],
                        extension=xlc)
                    attribute_map[attribute2.get_key()] = attribute2
                elif attribute_type in ['Case ID',
                                        'Event ID', transition['skiped']]:
                    next
                else:
                    attribute = XFactory.create_attribute_discrete(
                        attribute_type, int(attr_value))
                    attribute_map[attribute.get_key()] = attribute
            events.append(XFactory.create_event(attribute_map))
        return events