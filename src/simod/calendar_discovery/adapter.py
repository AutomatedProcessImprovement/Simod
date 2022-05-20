"""
This module provides adapters for the previous Simod's TimeTablesCreator.
"""
from io import StringIO, BytesIO
from typing import Optional, List

import pendulum
from lxml import etree

import pandas as pd

from simod.configuration import Configuration, CalendarType
from .case_arrival import discover as discover_arrival_calendar, CASE_ID_KEY
from .resource import PoolMapping, UNDIFFERENTIATED_RESOURCE_POOL_KEY, ACTIVITY_KEY, RESOURCE_KEY, END_TIMESTAMP_KEY
from .resource import discover as discover_resource_calendar
from ..cli_formatter import print_notice
from ..event_log import LogReader, read


def _prosimos_calendar_to_time_table(resource_calendar: dict, arrival_calendar: dict) -> etree.ElementTree:
    """
    Converts the Prosimos calendar to the TimeTablesCreator's timetable.
    """
    xml_template = '''<?xml version="1.0" encoding="utf8"?>
<qbp:timetables xmlns:qbp="http://www.qbp-simulator.com/Schema201212">
    <qbp:timetable id="QBP_ARR_DEFAULT_TIMETABLE" default="true" name="24/7">
        <qbp:rules>
            {arrival_rules}
        </qbp:rules>
    </qbp:timetable>
    <qbp:timetable id="Discovered_DEFAULT_CALENDAR" default="false" name="Discovered_DEFAULT_CALENDAR">
        <qbp:rules>
            {resource_rules}
        </qbp:rules>
    </qbp:timetable>
</qbp:timetables>
'''
    resource_xml = __prosimos_calendar_to_qbp_xml_string(resource_calendar)
    arrival_xml = __prosimos_calendar_to_qbp_xml_string(arrival_calendar)
    xml = xml_template.replace('{resource_rules}', resource_xml).replace('{arrival_rules}', arrival_xml)
    xml_io = BytesIO(bytes(xml, encoding='utf8'))
    return etree.parse(xml_io)


def __prosimos_calendar_to_qbp_xml_string(calendar: dict) -> str:
    calendar_items = calendar[UNDIFFERENTIATED_RESOURCE_POOL_KEY]
    xml_items = [__prosimos_calendar_item_to_qbp_rule(item) for item in calendar_items]
    return __et_elements_to_string(xml_items)


def __et_elements_to_string(elements: List[str]) -> str:
    """
    Converts a list of ET.Element to a string.
    """
    return '\n'.join(element for element in elements)


def __format_time(value: str) -> str:
    """
    Parses time from Prosimos and converts it the format readable by downstream dependencies.
    """
    time_format = 'HH:mm:ss.SSSZ'
    return pendulum.parse(value).format(time_format)  # e.g., value is in the format of '06:22:00'


def __prosimos_calendar_item_to_qbp_rule(item: dict) -> str:
    """
    Converts a Prosimos calendar item to a TimeTablesCreator's rule.
    """
    template = '<qbp:rule fromTime="{fromTime}" toTime="{toTime}" fromWeekDay="{fromDay}" toWeekDay="{toDay}"/>'
    return template \
        .replace('{fromTime}', __format_time(item['beginTime'])) \
        .replace('{toTime}', __format_time(item['endTime'])) \
        .replace('{fromDay}', item['from']) \
        .replace('{toDay}', item['to'])


def pool_mapping_to_resource_pool(pool_mapping: PoolMapping):
    """
    Converts PoolMapping to the TimeTablesCreator's resource pool.
    """
    pass


def pool_mapping_to_resource_table(pool_mapping: PoolMapping) -> pd.DataFrame:
    """
    Converts PoolMapping to the TimeTablesCreator's resource table.
    """
    roles = list(pool_mapping.keys())
    resources = list(pool_mapping.values())
    return pd.DataFrame({'role': roles, 'resource': resources})


class TimeTablesCreator:
    """
    This is an adapter class for extractions.schedule_tables.TimeTablesCreator.

    """
    # timetable names
    res_ttable_name: dict
    # arrival and resource timetables in XML
    time_table: etree.ElementTree
    # configuration
    settings: Configuration

    def __init__(self, settings: Configuration):
        print_notice('TimeTablesCreator adapter initialized')
        self.settings = settings

    def create_timetables(self, log: Optional[LogReader] = None):
        # NOTE: The previous version of the TimeTablesCreator supported only default undifferentiated calendars
        # even though it was supposed to support calendars differentiated by resource pools.
        #
        # resource_calendar_method = self.settings.res_cal_met  # NOTE: resource calendar discovery method discarded
        # arrival_calendar_method = self.settings.arr_cal_met   # NOTE: arrival calendar discovery method discarded
        resource_calendar_method = CalendarType.UNDIFFERENTIATED

        if log:
            event_log = pd.DataFrame(log.data)
        else:
            event_log, log_path_csv = read(self.settings.log_path)
            log_path_csv.unlink()

        # handling different column names: standard and Simod names
        columns_mapping = None
        if 'caseid' in event_log.columns:
            columns_mapping = {
                CASE_ID_KEY: 'caseid',
                ACTIVITY_KEY: 'task',
                RESOURCE_KEY: 'user',
                END_TIMESTAMP_KEY: 'end_timestamp',
            }

        resource_calendar = discover_resource_calendar(event_log, resource_calendar_method, columns_mapping=columns_mapping)
        arrival_calendar = discover_arrival_calendar(event_log, columns_mapping=columns_mapping)

        assert resource_calendar is not None, 'Resource calendar discovery failed'
        assert arrival_calendar is not None, 'Arrival calendar discovery failed'

        self.time_table = _prosimos_calendar_to_time_table(resource_calendar, arrival_calendar)
        self.res_ttable_name = {
            'arrival': 'QBP_ARR_DEFAULT_TIMETABLE',
            'resources': 'Discovered_DEFAULT_CALENDAR'
        }
