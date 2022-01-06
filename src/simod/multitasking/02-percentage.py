import sys
import os
import subprocess
import configparser as cp
import os.path
import xml.etree.ElementTree as ET
import copy
from shutil import copyfile
import time
import datetime
# from sweeper import add_event_id_to_log_events

model_input = "./PurchasingExample_Processed_Input_ForPercentage.xes"
model_output = "./PurchasingExample_Processed_Output_ForPercentage.xes"

percent = 0.05

dict = {
    # 'trace14': {
    #     'res1': [(1, 10, 'a'), (5, 20, 'b'), (3, 15, 'c')]
    # }
}

def main():
    copy_file_name = './InputCopy.xes'
    copyfile(model_input, copy_file_name)
    ET.register_namespace('', "http://www.xes-standard.org")

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
                        dict[d.attrib["value"]].append([start_ms, end_ms, task_id, event_id])

    removed_instant_events = 0
    for res in dict:
        for ev in reversed(dict[res]):
            if(ev[0] == ev[1]):
                dict[res].remove(ev)
                removed_instant_events += 1

    # TODO: sort

    pairs = []
    visited = []
    for res in dict:
        i = 0
        
        for ev in dict[res]:
            if(ev[3] not in visited):
                for ev2 in dict[res][i:]:
                    if(ev2[0] == ev[1]):
                        visited.append(ev2[3])
                        visited.append(ev[3])
                        pairs.append((ev, ev2))
                        break
            i += 1

    for p in pairs:
        total_time = (p[0][1] - p[0][0]) + (p[1][1] - p[1][0])
        movement_part = int(total_time * percent)
        p[0][0] += movement_part
        p[0][1] += movement_part

    print(len(pairs))

    # processed_times = 0

    for resource in dict:
        print("Current resource = ", resource)

        for trace in traces:
            events = trace.findall("./{http://www.xes-standard.org}event", ns1)

            for event in events:
                eventid = None
                endmark = False
                event_resource = event.find("./{http://www.xes-standard.org}string[@key='org:resource']")

                if(event_resource != None and event_resource.attrib["value"] == resource):
                    for d in event:
                        if(d.attrib["key"] == "EventID"):
                            eventid = d.attrib['value']
                        if(d.attrib["key"] == "EventEndTime"):
                            endmark = True

                    filtered = list(filter(lambda x: x[0][3] == eventid, pairs))
                    if(len(filtered) > 0):
                        for d in event:
                            if(d.attrib["key"] == "time:timestamp"):
                                stamp = None
                                # we actually have an endmark on the start event
                                # which refers to the corresponding end event
                                if(endmark == True):
                                    stamp = int(filtered[0][0][0]/1000.0)
                                else:
                                    stamp = int(filtered[0][0][1]/1000.0)
                                dtd = datetime.datetime.fromtimestamp(stamp)
                                upd = dtd.strftime("%Y-%m-%dT%H:%M:%S.%f")
                                d.attrib['value'] = upd
                                # processed_times += 1
                            if(d.attrib["key"] == "EventEndTime"):
                                stamp = int(filtered[0][0][1]/1000.0)
                                dtd = datetime.datetime.fromtimestamp(stamp)
                                upd = dtd.strftime("%Y-%m-%dT%H:%M:%S.%f")
                                d.attrib['value'] = upd
                                break

    tree.write(open(model_output, 'wb'), "UTF-8", True)
    os.remove(copy_file_name)

if __name__ == "__main__":
    main()
    # add_event_id_to_log_events(file_to_read="InsuranceScenarioD_resources.xes", file_to_write="InsuranceScenarioD_resources_Processed_Input.xes")
