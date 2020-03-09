# -*- coding: utf-8 -*-
from support_modules import support as sup
import itertools
from datetime import datetime


class TimeTablesCreator():
    '''
        This class creates the resources timetables and associates them
        to the resource pools
     '''

    def __init__(self, resource_table, dtype=None):
        '''constructor'''
        self.dtype = dtype

        self.resource_table = resource_table
        self.time_table = list()
        self.create_timetables()

        self.resource_pool = list()

        self.analize_schedules()

    def create_timetables(self) -> None:
        """
        Creates predefined timetables for BIMP simulator
        """
        if self.dtype == 'LV917':
            self.time_table.append({'id_t': 'QBP_DEFAULT_TIMETABLE',
                                    'default': 'true',
                                    'name': 'Default',
                                    'from_t': '09:00:00.000+00:00',
                                    'to_t': '17:00:00.000+00:00',
                                    'from_w': 'MONDAY',
                                    'to_w': 'FRIDAY'})
            schedule = {'work_days': [1, 1, 1, 1, 1, 0, 0],
                        'start_hour': datetime(1900, 1, 1, 9, 0, 0),
                        'end_hour': datetime(1900, 1, 1, 17, 0, 0)}
        elif self.dtype == '247':
            self.time_table.append({'id_t': 'QBP_DEFAULT_TIMETABLE',
                                    'default': 'true',
                                    'name': '24/7',
                                    'from_t': '00:00:00.000+00:00',
                                    'to_t': '23:59:59.999+00:00',
                                    'from_w': 'MONDAY',
                                    'to_w': 'SUNDAY'})
            schedule = {'work_days': [1, 1, 1, 1, 1, 1, 1],
                        'start_hour': datetime(1900, 1, 1, 0, 0, 0),
                        'end_hour': datetime(1900, 1, 1, 23, 59, 59)}
        # Add default schedule to resources
        self.resource_table.append({'role': 'SYSTEM', 'resource': 'AUTO'})
        for x in self.resource_table:
            x['schedule'] = schedule

    def analize_schedules(self) -> None:
        """
        Creates resource pools and associate them the default timetable
        in BIMP format
        """
        data = sorted(self.resource_table, key=lambda x: x['role'])
        for key, group in itertools.groupby(data, key=lambda x: x['role']):
            res_group = [x['resource'] for x in list(group)]
            r_pool_size = str(len(res_group)) if key != 'SYSTEM' else '20'
            self.resource_pool.append({'id': sup.gen_id(),
                                       'name': key,
                                       'total_amount': r_pool_size,
                                       'costxhour': '20',
                                       'timetable_id': 'QBP_DEFAULT_TIMETABLE'}
                                      )
        self.resource_pool[0]['id'] = 'QBP_DEFAULT_RESOURCE'
