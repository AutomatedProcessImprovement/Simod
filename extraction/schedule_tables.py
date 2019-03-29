# -*- coding: utf-8 -*-
from support_modules import support as sup
import itertools
from datetime import datetime


# def analize_schedules(roles, log):
#     log_data = log.data
#     # resources = sorted(list(set([x['user'] for x in log_data])))
#     resource_schedule = list()
#     for rol in roles:
#         for resource in rol['members']:
#             events_resource = list(filter(lambda x: x['user']==resource, log_data))
#             start_days = sorted(list(set([x['start_timestamp'].date() for x in events_resource])))
#             for start_day in start_days:
#                 try:
#                     min_timestamp_day = min(list(filter(lambda x: x['start_timestamp'].date()==start_day, events_resource)),
#                     key=itemgetter('start_timestamp'))['start_timestamp']
#                     max_timestamp_day = max(list(filter(lambda x: x['end_timestamp'].date()==start_day, events_resource)),
#                     key=itemgetter('end_timestamp'))['end_timestamp']
#                     min_hour=min_timestamp_day.time()
#                     max_hour=max_timestamp_day.time()
#                     duration=(max_timestamp_day-min_timestamp_day).total_seconds()
#                     if duration > 0.0:
#                         resource_schedule.append(dict(resource=resource,rol=rol['role'],day=start_day, min=min_hour, dur=duration))
#                 except:
#                     pass
#         # print(np.mean([x['dur'] for x in resource_schedule])/3600)
#         sup.create_csv_file_header(resource_schedule, 'schedule.csv')

def analize_schedules(resource_table, log, default=False, dtype=None):
    resource_pool = list()
    if default:
        time_table, resource_table = create_timetables(resource_table, dtype=dtype)
        data = sorted(resource_table, key=lambda x:x['role'])
        for key, group in itertools.groupby(data, key=lambda x:x['role']):
            values = list(group)
            group_resources = [x['resource'] for x in values]
            resource_pool.append(
                dict(id=sup.gen_id(), name=key, total_amount=str(len(group_resources)), costxhour="20",
                     timetable_id="QBP_DEFAULT_TIMETABLE"))
        resource_pool[0]['id'] = 'QBP_DEFAULT_RESOURCE'
        resource_pool.append(dict(id='0', name = 'Role 0', total_amount = '1', costxhour="0",timetable_id="QBP_DEFAULT_TIMETABLE" ))
    else:
        print('test')
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
            time_table.append(dict(id_t="QBP_DEFAULT_TIMETABLE",default="false",name="24/7",
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
