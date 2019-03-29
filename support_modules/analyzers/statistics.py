# -*- coding: utf-8 -*-
import itertools
import numpy as np

def task_metrics_statistics(process_stats):
    case_list = list()
    # Group by source
    data = sorted(process_stats, key=lambda x:x['source'])
    for key, group in itertools.groupby(data, key=lambda x:x['source']):
        # Group by run
        data2 = sorted(list(group), key=lambda x:x['run_num'])
        for key2, group2 in itertools.groupby(data2, key=lambda x:x['run_num']):
            # Group by case
            data3 = sorted(list(group2), key=lambda x:x['caseid'])
            for key3, group3 in itertools.groupby(data3, key=lambda x:x['caseid']):
                # Group by task
                data4 = sorted(list(group3), key=lambda x:x['task'])
                for key4, group4 in itertools.groupby(data4, key=lambda x:x['task']):
                    values = list(group4)
                    group_processing_time = [x['processing_time'] for x in values]
                    group_waiting_time = [x['waiting_time'] for x in values]
                    group_multitasking = [x['multitasking'] for x in values]
                    case_list.append(dict(source=key , run_num=key2, caseid=key3, task=key4,
                        processing_time=np.sum(group_processing_time), waiting_time=np.sum(group_waiting_time),
                        multitasking=np.sum(group_multitasking)))
    run_list = list()
    data = sorted(case_list, key=lambda x:x['source'])
    for key, group in itertools.groupby(data, key=lambda x:x['source']):
        # Group by run
        data2 = sorted(list(group), key=lambda x:x['run_num'])
        for key2, group2 in itertools.groupby(data2, key=lambda x:x['run_num']):
            # Group by task
            data4 = sorted(list(group2), key=lambda x:x['task'])
            for key4, group4 in itertools.groupby(data4, key=lambda x:x['task']):
                values = list(group4)
                group_processing_time = [x['processing_time'] for x in values]
                group_waiting_time = [x['waiting_time'] for x in values]
                group_multitasking = [x['multitasking'] for x in values]
                run_list.append(dict(source=key , run_num=key2, task=key4,
                    processing_time=np.mean(group_processing_time), waiting_time=np.mean(group_waiting_time),
                    multitasking=np.mean(group_multitasking)))

    task_list = list()
    # Group by source
    data = sorted(run_list, key=lambda x:x['source'])
    for key, group in itertools.groupby(data, key=lambda x:x['source']):
        values = list(group)
        data2 = sorted(values, key=lambda x:x['task'])
        # Group by task
        for key2, group2 in itertools.groupby(data2, key=lambda x:x['task']):
            values2 = list(group2)
            group_processing_time = [x['processing_time'] for x in values2]
            group_waiting_time = [x['waiting_time'] for x in values2]
            group_multitasking = [x['multitasking'] for x in values2]
            task_list.append(dict(source=key ,task=key2, processing_time=group_processing_time
                , waiting_time=group_waiting_time, multitasking=group_multitasking))
    return task_list

def process_metrics_statistics(process_stats):
    case_list = list()
    # Group by source
    data = sorted(process_stats, key=lambda x:x['source'])
    for key, group in itertools.groupby(data, key=lambda x:x['source']):
        # Group by run
        data2 = sorted(list(group), key=lambda x:x['run_num'])
        for key2, group2 in itertools.groupby(data2, key=lambda x:x['run_num']):
            # Group by case
            data3 = sorted(list(group2), key=lambda x:x['caseid'])
            for key3, group3 in itertools.groupby(data3, key=lambda x:x['caseid']):
                # Group by task
                values = list(group3)
                group_processing_time = [x['processing_time'] for x in values]
                group_waiting_time = [x['waiting_time'] for x in values]
                group_multitasking = [x['multitasking'] for x in values]
                case_list.append(dict(source=key , run_num=key2, caseid=key3,
                    processing_time=np.sum(group_processing_time), waiting_time=np.sum(group_waiting_time),
                    multitasking=np.sum(group_multitasking)))

    process_run_list = list()
    # Group by source
    data = sorted(case_list, key=lambda x:x['source'])
    for key, group in itertools.groupby(data, key=lambda x:x['source']):
        # Group by run
        data2 = sorted(list(group), key=lambda x:x['run_num'])
        for key2, group2 in itertools.groupby(data2, key=lambda x:x['run_num']):
            # Group by task
            values = list(group2)
            group_processing_time = [x['processing_time'] for x in values]
            group_waiting_time = [x['waiting_time'] for x in values]
            group_multitasking = [x['multitasking'] for x in values]
            process_run_list.append(dict(source=key , run_num=key2, processing_time=np.mean(group_processing_time),
                waiting_time=np.mean(group_waiting_time), multitasking=np.mean(group_multitasking)))

    process_list = list()
    # Group by source
    data = sorted(process_run_list, key=lambda x:x['source'])
    for key, group in itertools.groupby(data, key=lambda x:x['source']):
        # Group by run
        values = list(group)
        group_processing_time = [x['processing_time'] for x in values]
        group_waiting_time = [x['waiting_time'] for x in values]
        group_multitasking = [x['multitasking'] for x in values]
        process_list.append(dict(source=key , processing_time=group_processing_time,
            waiting_time=group_waiting_time, multitasking=group_multitasking))

    return process_list

def role_statistics(process_stats):
    case_list = list()
    # Group by source
    data = sorted(process_stats, key=lambda x:x['source'])
    for key, group in itertools.groupby(data, key=lambda x:x['source']):
        # Group by run
        data2 = sorted(list(group), key=lambda x:x['run_num'])
        for key2, group2 in itertools.groupby(data2, key=lambda x:x['run_num']):
            # Group by case
            data3 = sorted(list(group2), key=lambda x:x['caseid'])
            for key3, group3 in itertools.groupby(data3, key=lambda x:x['caseid']):
                # Group by task
                data4 = sorted(list(group3), key=lambda x:x['role'])
                for key4, group4 in itertools.groupby(data4, key=lambda x:x['role']):
                    values = list(group4)
                    group_processing_time = [x['processing_time'] for x in values]
                    case_list.append(dict(source=key , run_num=key2, caseid=key3, role=key4,
                        processing_time=np.sum(group_processing_time)))
    run_list = list()
    # Group by source
    data = sorted(case_list, key=lambda x:x['source'])
    for key, group in itertools.groupby(data, key=lambda x:x['source']):
        # Group by run
        data2 = sorted(list(group), key=lambda x:x['run_num'])
        for key2, group2 in itertools.groupby(data2, key=lambda x:x['run_num']):
            # Group by task
            data4 = sorted(list(group2), key=lambda x:x['role'])
            for key4, group4 in itertools.groupby(data4, key=lambda x:x['role']):
                values = list(group4)
                group_processing_time = [x['processing_time'] for x in values]
                run_list.append(dict(source=key , run_num=key2, role=key4,
                    processing_time=np.mean(group_processing_time)))

    role_list = list()
    # Group by source
    data = sorted(run_list, key=lambda x:x['source'])
    for key, group in itertools.groupby(data, key=lambda x:x['source']):
        values = list(group)
        data2 = sorted(values, key=lambda x:x['role'])
        # Group by task
        for key2, group2 in itertools.groupby(data2, key=lambda x:x['role']):
            values2 = list(group2)
            group_processing_time = [x['processing_time'] for x in values2]
            role_list.append(dict(source=key ,role=key2, processing_time=group_processing_time))
    return role_list
