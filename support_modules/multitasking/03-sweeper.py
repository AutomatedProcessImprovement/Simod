import sys
import re
import os
import subprocess
import configparser as cp
import getopt
import csv
import os.path
import xml.etree.ElementTree as ET
import queue
import itertools
import copy
from shutil import copyfile
import time
import datetime
from functools import reduce
import uuid

print("---START("+str(datetime.datetime.now())+")---")

# variable for statistics, shows how many sweep line partitions has been made
piecess = 0

model_input = "./PurchasingExample_Processed_Output_ForPercentage.xes" #Con_percentage
model_output = "./PurchasingExample_Processed_Output_ForPercentage_FINAL.xes" #Con_percentage

#model_input = "./PurchasingExample_Processed_Input_SinPercentage.xes" #Sin_Percentage
#model_output = "./PurchasingExample_Processed_Output_SinPercentage.xes" #Sin_Percentage

event_end_mapping = {}

#-------Added_for_index
intersected_ids = []
intersected_resources = []
intersected_activities = []
#----------------------

dict = {
    # 'trace14': {
    #     'res1': [(1, 10, 'a'), (5, 20, 'b'), (3, 15, 'c')]
    # }
}

def sortFirst(val):
    return val[0]
def sortSecond(val):
    return val[1]


def main():
    copy_file_name = './InputCopy.xes'
    copyfile(model_input, copy_file_name)
    ET.register_namespace('', "http://www.xes-standard.org")

    #-------Added_for_index
    totalevents = 0
    partial_log_overlap = []
    total_log_overlap = []
    num_partial_pairs = 0
    num_total_pairs = 0
    
    num_events2 = 0
    #----------------------

    ns1 = {
        'xes': "http://www.xes-standard.org",
        'time': "http://www.xes-standard.org/time.xesext"
    }
    
    tree = ET.parse(copy_file_name)
    root = tree.getroot()

    traces = root.findall("{http://www.xes-standard.org}trace", ns1)
    print("traces number = ", len(traces))

    for trace in traces:
        # if(trace_name.attrib['value'] not in dict):
            # dict[trace_name.attrib['value']] = {}
        events = trace.findall("./{http://www.xes-standard.org}event", ns1)

        #print("ADD--> Events: " + str(len(events)))
        num_events2 += len(events)

        for event in events:
            for d in event:
                if(d.attrib["key"] == "org:resource"):
                    if(d.attrib['value'] not in dict):
                        dict[d.attrib['value']] = []

                    # we assume that event has two dates, first is normal start
                    # second is EventEndTime that we add using function add_event_id_to_log_events
                    dates = event.findall("./{http://www.xes-standard.org}date", ns1)
                    if(len(dates) > 1):
                        start = dates[0].attrib['value']
                        dat_start = datetime.datetime.strptime(start, "%Y-%m-%dT%H:%M:%S.%f")
                        start_ms = int(round(dat_start.timestamp() * 1000))

                        end = dates[1].attrib['value']
                        dat_end = datetime.datetime.strptime(end, "%Y-%m-%dT%H:%M:%S.%f")
                        end_ms = int(round(dat_end.timestamp() * 1000))

                        event_id = next(x.attrib['value'] for x in event.getchildren() if x.attrib['key'] == 'EventID')
                        task_id = next(x.attrib['value'] for x in event.getchildren() if x.attrib['key'] == 'concept:name')

                        # add the event corresponding to the resource 
                        dict[d.attrib["value"]].append((start_ms, end_ms, task_id, event_id))

    for resource in dict:
        print("Current resource = ", resource)
        
        #-------Added_for_index
        overfx_res, num_overpartial_res, numb_overtotal_res = calculate_overlap(dict[resource])
        if overfx_res > 0:
            intersected_resources.append(resource)    
        if num_overpartial_res > 0:
            partial_log_overlap.append( (1/num_overpartial_res) * overfx_res )
            num_partial_pairs += num_overpartial_res     
        if numb_overtotal_res > 0:
            total_log_overlap.append((1/numb_overtotal_res) * overfx_res ) 
            num_total_pairs += numb_overtotal_res
        #----------------------
        
        adjusted_ranges = adjust_ranges(dict[resource])
        # trace_name = trace.find("./{http://www.xes-standard.org}string", ns1)
        # print("Current trace = ", trace_name.attrib['value'])
        
        # debug_counter = 0

        #-------Added_for_index
        totalevents += len(adjusted_ranges)
        #----------------------

        for trace in traces:
            events = trace.findall("./{http://www.xes-standard.org}event", ns1)

            for event in events:
                # debug_counter += 1
                # if(debug_counter % 2000 == 0):
                #     print("counter = ", debug_counter)

                eventid = None
                endmark = False
                event_resource = event.find("./{http://www.xes-standard.org}string[@key='resource']")

                if(event_resource != None and event_resource.attrib["value"] == resource):
                    for d in event:
                        if(d.attrib["key"] == "EventID"):
                            eventid = d.attrib['value']
                        if(d.attrib["key"] == "EventEndTime"):
                            endmark = True

                    if(endmark == True):
                        for d in event:
                            if(d.attrib["key"] == "time:timestamp"):
                                
                                stamp = int(adjusted_ranges[eventid]/1000.0)
                                dtd = datetime.datetime.fromtimestamp(stamp)
                                upd = dtd.strftime("%Y-%m-%dT%H:%M:%S.%f")
                                d.attrib['value'] = upd
                                break

    #-------Added_for_index
    intersected_id_events = list(itertools.chain.from_iterable(intersected_ids)) #Convert to a flat list
    intersected_id_events = list(dict.fromkeys(intersected_id_events)) 
    
    for id_int in intersected_id_events:
        for rec in dict:
            for start, stop, task, eventid in dict[rec]:
                if(eventid == id_int):
                    #if(rec not in intersected_resources):
                        #print("rec:: " + str(rec))
                        #intersected_resources.append(rec)
                    if(task not in intersected_activities):
                        intersected_activities.append(task)

    avg_partial_MTWI = 0
    avg_total_MTWI = 0
    
    if len(partial_log_overlap) > 0:
        avg_partial_MTWI = sum(partial_log_overlap) / len(partial_log_overlap)
    if len(total_log_overlap) > 0:
        avg_total_MTWI = sum(total_log_overlap) / len(total_log_overlap)
    
    print("=======================================")
    print("> AVG(Partial_MTWI): " + str(avg_partial_MTWI))
    print("> AVG(Total_MTWI): " + str(avg_total_MTWI))
    print("> Num_pairs_overlap: " + str(num_partial_pairs))
    print("> Num_pairs_logs: "+ str(num_total_pairs))
    print("> ·····")
    print("> Resources-total[" + str(len(dict))+"]")
    print("> Resources-multitasking["+str(len(intersected_resources))+"]")
    print("> Events-total[" + str(totalevents)+"]")
    print("> Events-multitasking["+str(len(intersected_id_events))+"]")
    print("> Activities_intersected["+str(len(intersected_activities))+"]")
    print("> .....")
    print("> num_events2 (hay que dividir) = " + str(num_events2))
    print("=======================================")
    
    #----------------------
    print("piecess = ", piecess)
    tree.write(open(model_output, 'wb'), "UTF-8", True)
    os.remove(copy_file_name)
    print("---END("+str(datetime.datetime.now())+")---")

#-------Added_for_index
def calculate_overlap(events):
    overlap_fx = 0
    num_total_overlap = 0
    num_partial_overlap = 0

    events_copy = events
    
    if len(events) > 1: #The overlapping is generated between two events. 
        
        for start1, end1, task1, idevent1 in events:
            #print("idevent1:: " + str(idevent1))
            events_copy = delete_event(events_copy, idevent1)
            if len(events_copy) > 0:
                for start2, end2, task2, idevent2 in events_copy:
                    
                    if max((end1-start1),(end2-start2)) > 0: 
                        overlap_temp = (max(min(end1,end2) - max(start1,start2),0)) / max((end1-start1),(end2-start2))
                        if overlap_temp > 0:
                            overlap_fx += overlap_temp
                            num_partial_overlap += 1
                    
                    num_total_overlap += 1 #Equivalent to nCk
                        
    return overlap_fx, num_partial_overlap, num_total_overlap

def delete_event(events, delete_idevent): 
    newinput = []
    for start, end, tasks, idevent in events:
        if idevent != delete_idevent: 
            newinput.append((start, end, tasks, idevent))
    return newinput
#----------------------

# obsolete, use adjust_ranges
def build_range(input):
    # for resource in dict[trace]:
    # print("Resource = ", resource)
    # print("\n")
    # input = dict[trace][resource]
    # input = [(10,100,'a','eve1'),(50,200,'b','eve2'),(30,150,'c','eve3')]
    # print(input)
    # input = [(1, 10, 'a'), (5, 20, 'b'), (3, 15, 'c')]
    points = []  # list of (offset, plus/minus, symbol) tuples
    global piecess
    #-------Added_for_index
    global intersected_ids
    #----------------------
    event_task_mapping = {}
    
    start_time = time.time()

    for start, stop, task, eventid in input:
        event_task_mapping[eventid] = task
        event_end_mapping[eventid] = stop
        points.append(
            (start, '+', copy.deepcopy(task), copy.deepcopy(eventid)))
        points.append(
            (stop, '-', copy.deepcopy(task), copy.deepcopy(eventid)))
    points.sort()
    print("points length = ", len(points))

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
            if(len(current_stack) > 0):
                last_start = offset
            else:
                last_start = None

    new_ranges = []

    # print(ranges)
    counterr = 0
    print("ranges count = ", len(ranges))
    for range in ranges:
        """counterr += 1                        #Counter -- IBET commented this
        if(counterr % 100 == 0):
            print("ranges counter = ", counterr)""" 

        pieces = len(range[3])
        if(pieces > 1):
            
            #-------Added_for_index
            intersected_ids.append(range[3])
            #----------------------
            
            piecess += 1
            diff = range[1] - range[0]
            piece_size = int(diff / pieces)
            # print("piece size = ", piece_size)
            initial_start = range[0]
            max_end = range[1]

            piece_number = 0
            for cur_event_id in range[3]:
                cop = copy.deepcopy(range)
                # filtered = list(filter(lambda x: event_task_mapping[x] == task, cop[3]))[0]
                task = event_task_mapping[cur_event_id]
                modified = (initial_start + piece_size * piece_number,
                            min(max_end, initial_start + piece_size *
                                piece_number + piece_size),
                            copy.deepcopy(task),
                            cur_event_id)
                piece_number += 1
                if(modified not in new_ranges):
                    new_ranges.append(modified)
        else:
            if(pieces == 1):
                cop = copy.deepcopy(range)
                # filtered = list(filter(lambda x: event_task_mapping[x] == cop[2][0], cop[3]))[0]
                cur_event_id = range[3][0]
                task = event_task_mapping[cur_event_id]
                new_range = (cop[0],
                             cop[1],
                             copy.deepcopy(cop[2][0]),
                             cur_event_id)
                if(new_range not in new_ranges):
                    new_ranges.append(new_range)
        
    print("new_ranges size = ", len(new_ranges))
    new_ranges.sort(key=sortFirst)

    end_time = time.time()
    print("time = " + str(end_time - start_time) + " \n")
    return new_ranges

# adds EventID to all events and EventEndTime to start events
def add_event_id_to_log_events(file_to_read="ConsultaDataMining201618.xes", 
                               file_to_write="ConsultaDataMining201618_Processed_Input.xes"):
    ET.register_namespace('', "http://www.xes-standard.org")
    tree = ET.parse(file_to_read)
    
    dicts = {
        # '9348': {
        #     'Traer informacion estudiante - banner': True
        # }
    }

    for el in tree.iter():
        if(el.tag == '{http://www.xes-standard.org}event'):
            endtime = ""
            name = ""
            resource = ""
            for d in el:
                if(d.attrib["key"] == "time:timestamp"):
                    endtime = d.attrib['value']
                if(d.attrib["key"] == "concept:name"):
                    name = d.attrib['value']
                if(d.attrib["key"] == "resource"):
                    resource = d.attrib['value']

            event_id = None
            if(resource not in dicts):
                dicts[resource] = {}
            if(name not in dicts[resource]):
                event_id = str(uuid.uuid4())
                dicts[resource][name] = {'start_event': el, 'event_id': event_id}
            else:
                event_id = dicts[resource][name]['event_id']
                start_ev = dicts[resource][name]['start_event']
                ET.SubElement(start_ev, 'date', {'key': 'EventEndTime', 'value': endtime})
                del dicts[resource][name]
            
            ET.SubElement(el, 'string', {'key': 'EventID', 'value': event_id})

    tree.write(open(file_to_write, 'wb'), "UTF-8", True)

def adjust_ranges(input):
    print("input length = ", len(input))
    # input = [(10, 100, 'a', 'eve1'), (50, 200, 'b', 'eve2'),(30,150,'c','eve3')]
    new_ranges = build_range(input)
    print("built new ranges")
    # print(new_ranges)
    new_ranges.sort(key=(lambda x: x[3]))

    adjusted_ranges = {}

    for event, go in itertools.groupby(new_ranges, lambda x: x[3]):
        items = list(go)
        items.sort(key=sortSecond)

        sum_duration = reduce(lambda acc, x: acc + x[1] - x[0], items, 0)
        adjusted_ranges[event] = event_end_mapping[event] - sum_duration
        
    # print(adjusted_ranges)
    return adjusted_ranges

if __name__ == "__main__":
    # input = [(1, 5, 'a', 'eve1'), (5, 26, 'a', 'eve1'),(26,50,'c','eve2'),(50,94,'c','eve2')]
    # adjust_ranges(input)
    main()
    # add_event_id_to_log_events()
    