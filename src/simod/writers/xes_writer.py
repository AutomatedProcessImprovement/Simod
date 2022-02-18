import itertools as it
from pathlib import Path
from typing import Union

import pandas as pd
from opyenxes.data_out.XesXmlSerializer import XesXmlSerializer
from opyenxes.extension.std.XLifecycleExtension import XLifecycleExtension as xlc
from opyenxes.factory.XFactory import XFactory

from ..common_routines import convert_df_to_xes
from ..configuration import ReadOptions
from ..readers.log_reader import LogReader


class XesWriter(object):  # TODO: it makes sense to save data also with LogReader instead of a separate class
    """This class writes a process log in XES-format."""

    def __init__(self, log: Union[LogReader, pd.DataFrame, list], read_options: ReadOptions, output_path: Path):
        if isinstance(log, pd.DataFrame):
            self.log = log.values
        elif isinstance(log, LogReader):
            self.log = log.data
        elif isinstance(log, list):
            self.log = log
        else:
            raise Exception(f'Unimplemented type for {type(log)}')
        self.one_timestamp = read_options.one_timestamp
        self.column_names = read_options.column_names
        self.output_file = output_path
        self.write_xes()

    def write_xes(self):
        log_df = pd.DataFrame(self.log)
        log_df.rename(columns={
            'task': 'concept:name',
            'caseid': 'case:concept:name',
            'event_type': 'lifecycle:transition',
            'user': 'org:resource',
            'end_timestamp': 'time:timestamp'
        }, inplace=True)
        log_df.drop(columns=['@@startevent_concept:name',
                             '@@startevent_org:resource',
                             '@@startevent_Activity',
                             '@@startevent_Resource',
                             '@@duration',
                             'case:variant',
                             'case:variant-index',
                             'case:creator',
                             'Activity',
                             'Resource',
                             'elementId',
                             'processId',
                             'resourceId',
                             'resourceCost',
                             '@@startevent_element',
                             '@@startevent_elementId',
                             '@@startevent_process',
                             '@@startevent_processId',
                             '@@startevent_resourceId',
                             'etype'], inplace=True, errors='ignore')

        log_df.fillna('UNDEFINED', inplace=True)

        convert_df_to_xes(log_df, self.output_file)

    def create_xes_file(self):
        csv_mapping = {v: k for k, v in self.column_names.items()}
        log = XFactory.create_log()
        data = sorted(self.log, key=lambda x: x['caseid'])
        for key, group in it.groupby(data, key=lambda x: x['caseid']):
            sort_key = ('end_timestamp' if self.one_timestamp else 'start_timestamp')
            csv_trace = sorted(list(group), key=lambda x: x[sort_key])  # TODO: why is that "csv_trace" in xes function?
            events = list()
            for line in csv_trace:
                events.extend(self.convert_line_in_event(csv_mapping, line))
            trace_attribute = XFactory.create_attribute_literal('concept:name', key)
            trace_attribute_map = XFactory.create_attribute_map()
            trace_attribute_map[trace_attribute.get_key()] = trace_attribute
            trace = XFactory.create_trace(attribute=trace_attribute_map)
            for event in events:
                trace.append(event)
            log.append(trace)
            # log.set_info(classifier, info)

        # Save log in xes format
        with self.output_file.open('w') as f:
            XesXmlSerializer().serialize(log, f)

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
            transitions.insert(0, {'column': 'Start Timestamp',
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
                        attribute = XFactory.create_attribute_literal('concept:name', attr_value, extension=None)
                        attribute_map[attribute.get_key()] = attribute
                    if attribute_type == "Resource":
                        attribute = XFactory.create_attribute_literal('org:resource', attr_value, extension=None)
                        attribute_map[attribute.get_key()] = attribute
                elif attribute_type == transition['column']:
                    attribute = XFactory.create_attribute_timestamp("time:timestamp", attr_value, extension=None)
                    attribute_map[attribute.get_key()] = attribute
                    attribute2 = XFactory.create_attribute_literal('lifecycle:transition', transition['value'],
                                                                   extension=xlc)
                    attribute_map[attribute2.get_key()] = attribute2
                elif attribute_type in ['Case ID', 'Event ID', transition['skiped']]:
                    continue
                else:
                    attribute = XFactory.create_attribute_discrete(attribute_type, int(attr_value))
                    attribute_map[attribute.get_key()] = attribute
            events.append(XFactory.create_event(attribute_map))
        return events
