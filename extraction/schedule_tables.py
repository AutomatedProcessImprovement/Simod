# -*- coding: utf-8 -*-
from support_modules import support as sup
import itertools
from datetime import datetime


def analize_schedules(resource_table, log, default=False, dtype=None):
    resource_table.append({'role': 'SYSTEM', 'resource': 'AUTO'})    
    resource_pool = list()
    if default:
        time_table, resource_table = create_timetables(resource_table, dtype=dtype)
        data = sorted(resource_table, key=lambda x:x['role'])
        for key, group in itertools.groupby(data, key=lambda x:x['role']):
            values = list(group)
            group_resources = [x['resource'] for x in values]
            r_pool_size = str(len(group_resources)) if key != 'SYSTEM' else '20'
            resource_pool.append(
                dict(id=sup.gen_id(), name=key, total_amount=r_pool_size, costxhour="20",
                     timetable_id="QBP_DEFAULT_TIMETABLE"))
        resource_pool[0]['id'] = 'QBP_DEFAULT_RESOURCE'
    else:
        print('test')
        # resource_pool[0]['id'] = 'QBP_DEFAULT_RESOURCE'
        resource_pool.append(dict(id='QBP_DEFAULT_RESOURCE', name = 'Role 0', total_amount = '1', costxhour="0",timetable_id="QBP_DEFAULT_TIMETABLE" ))
    return resource_pool, time_table, resource_table



def analize_log_schedule(resource_table, log):
    # Define and assign schedule tables
    time_table, resource_table = create_timetables(resource_table, dtype='247')
    log_data = log.data
    for resource in resource_table:
        # Calculate worked days
        resource['w_days_intime'], resource['w_days_offtime'] = worked_days(resource, log_data)
        available_time = (resource['schedule']['end_hour']-resource['schedule']['start_hour']).total_seconds()
        # Calculate resource availability
        resource['ava_intime'] = available_time * resource['w_days_intime']
        resource['ava_offtime'] = available_time * resource['w_days_offtime']
    #Print availability per role
    sup.create_csv_file_header(resource_table, 'schedule.csv')
    [print(x) for x in roles_availability(resource_table)]


def create_timetables(resource_table,default=True, dtype='LV917'):
    time_table = list()
    if default:
        if dtype=='LV917':
            time_table.append(dict(id_t="QBP_DEFAULT_TIMETABLE",default="true",name="Default",
            from_t = "09:00:00.000+00:00",to_t="17:00:00.000+00:00",from_w="MONDAY",to_w="FRIDAY"))
            schedule = dict(work_days = [1,1,1,1,1,0,0], start_hour = datetime(1900,1,1,9, 0, 0), end_hour = datetime(1900,1,1,17, 0, 0))
        elif dtype=='247':
            time_table.append(dict(id_t="QBP_DEFAULT_TIMETABLE",default="true",name="24/7",
            from_t = "00:00:00.000+00:00",to_t="23:59:59.999+00:00",from_w="MONDAY",to_w="SUNDAY"))
            schedule = dict(work_days = [1,1,1,1,1,1,1], start_hour = datetime(1900,1,1,0, 0, 0), end_hour = datetime(1900,1,1,23, 59, 59))
        else:
            raise Exception('Default schedule not existent')
        # Add default schedule to resources
        for x in resource_table:
            x['schedule'] = schedule
    return time_table, resource_table

def worked_days(resource_data, log_data):
    resource = resource_data['resource']
    work_days = resource_data['schedule']['work_days']
    events_resource = list(filter(lambda x: x['user']==resource, log_data))
    start_days = set([x['start_timestamp'].date() for x in events_resource])
    end_days = set([x['end_timestamp'].date() for x in events_resource])
    days = list(start_days.union(end_days))
    in_time_table, off_time_table = 0 , 0
    for x in days:
        if work_days[x.weekday()] == 1:
            in_time_table += 1
        else:
            off_time_table += 1
    return in_time_table, off_time_table

def roles_availability(resource_table):
    roles_list = list()
    for key, group in itertools.groupby(resource_table, key=lambda x:x['role']):
        values = list(group)
        group_ava_intime = [x['ava_intime'] for x in values]
        group_ava_offtime = [y['ava_offtime'] for y in values]
        roles_list.append(dict(role=key, ava_intime=sum(group_ava_intime),ava_offtime=sum(group_ava_offtime)))
    return roles_list

def roles_timetables(resource_table):
    roles_list = list()
    for key, group in itertools.groupby(resource_table, key=lambda x:x['role']):
        values = list(group)
        group_ava_intime = [x['ava_intime'] for x in values]
        group_ava_offtime = [y['ava_offtime'] for y in values]
        roles_list.append(dict(role=key, ava_intime=sum(group_ava_intime),ava_offtime=sum(group_ava_offtime)))
    return roles_list
