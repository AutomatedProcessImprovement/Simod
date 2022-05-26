"""
This module provides adapters for the previous Simod's TimeTablesCreator.
"""
import itertools
from io import BytesIO
from pathlib import Path
from typing import Optional, List, Tuple

import pandas as pd
import pendulum
from lxml import etree

from simod.configuration import CalendarType, Configuration
from simod.event_log import LogReader, read
from .case_arrival import discover as discover_arrival_calendar, CASE_ID_KEY
from .resource import UNDIFFERENTIATED_RESOURCE_POOL_KEY, ACTIVITY_KEY, RESOURCE_KEY, END_TIMESTAMP_KEY
from .resource import discover as discover_resource_calendar
from ..resource_pool_discoverer import ResourcePoolDiscoverer
from ... import support_utils
from ...cli_formatter import print_step


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


class TimeTableMiner:
    """
    This is an adapter class for extractions.schedule_tables.TimeTablesCreator.
    """
    timetable_names: dict  # timetable names
    timetable: etree.ElementTree  # arrival and resource timetables in XML
    log: Optional[LogReader]
    log_path: Optional[Path]
    arrival_timetable: dict
    resource_timetable: dict

    def __init__(self, log_path: Optional[Path] = None, log: Optional[LogReader] = None):
        self.log_path = log_path
        self.log = log
        self.__run()

    def __run(self):
        # NOTE: The previous version of the TimeTablesCreator supported only default undifferentiated calendars
        # even though it was supposed to support calendars differentiated by resource pools.
        #
        # resource_calendar_method = self.settings.res_cal_met  # NOTE: resource calendar discovery method discarded
        # arrival_calendar_method = self.settings.arr_cal_met   # NOTE: arrival calendar discovery method discarded
        resource_calendar_method = CalendarType.UNDIFFERENTIATED

        if self.log:
            event_log = pd.DataFrame(self.log.data)
        else:
            event_log, log_path_csv = read(self.log_path)
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

        resource_calendar = discover_resource_calendar(event_log, resource_calendar_method,
                                                       columns_mapping=columns_mapping)
        arrival_calendar = discover_arrival_calendar(event_log, columns_mapping=columns_mapping)

        assert resource_calendar is not None, 'Resource calendar discovery failed'
        assert arrival_calendar is not None, 'Arrival calendar discovery failed'

        self.resource_timetable = resource_calendar
        self.arrival_timetable = arrival_calendar
        self.timetable = _prosimos_calendar_to_time_table(resource_calendar, arrival_calendar)
        self.timetable_names = {
            'arrival': 'QBP_ARR_DEFAULT_TIMETABLE',
            'resources': 'Discovered_DEFAULT_CALENDAR'
        }


def discover_timetables_and_get_default_arrival_resource_pool(log_path: Path) -> Tuple[list, etree.Element]:
    timetables_miner = TimeTableMiner(log_path=log_path)
    arrival_default_resource_pool = [
        {
            'id': 'QBP_DEFAULT_RESOURCE',
            'name': 'SYSTEM',
            'total_amount': '100000',
            'costxhour': '20',
            'timetable_id': timetables_miner.timetable_names['arrival']
        }
    ]
    return arrival_default_resource_pool, timetables_miner.timetable


def discover_timetables_with_resource_pools(log: LogReader, settings: Configuration):
    print_step('Resource Miner')

    def create_resource_pool(resource_table, table_name) -> list:
        """Creates resource pools and associate them the default timetable in BIMP format"""
        resource_pool = [{'id': 'QBP_DEFAULT_RESOURCE', 'name': 'SYSTEM', 'total_amount': '20', 'costxhour': '20',
                          'timetable_id': table_name['arrival']}]
        data = sorted(resource_table, key=lambda x: x['role'])
        for key, group in itertools.groupby(data, key=lambda x: x['role']):
            res_group = [x['resource'] for x in list(group)]
            r_pool_size = str(len(res_group))
            name = (table_name['resources'] if 'resources' in table_name.keys() else table_name[key])
            resource_pool.append(
                {'id': support_utils.gen_id(), 'name': key, 'total_amount': r_pool_size, 'costxhour': '20',
                 'timetable_id': name})
        return resource_pool

    resource_pool_discoverer = ResourcePoolDiscoverer(log, sim_threshold=settings.rp_similarity)
    timetables_miner = TimeTableMiner(log=log)

    args = {
        'res_cal_met': settings.res_cal_met,
        'arr_cal_met': settings.arr_cal_met,
        'resource_table': resource_pool_discoverer.resource_table
    }

    if not isinstance(args['res_cal_met'], CalendarType):
        args['res_cal_met'] = CalendarType.from_str(settings.res_cal_met)
    if not isinstance(args['arr_cal_met'], CalendarType):
        args['arr_cal_met'] = CalendarType.from_str(settings.arr_cal_met)

    resource_pool = create_resource_pool(resource_pool_discoverer.resource_table, timetables_miner.timetable_names)
    resource_table = pd.DataFrame.from_records(resource_pool_discoverer.resource_table)

    return timetables_miner.timetable, resource_pool, resource_table
