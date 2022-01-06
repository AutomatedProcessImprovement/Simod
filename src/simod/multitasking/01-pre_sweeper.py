# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 14:18:20 2019

@author: bedic
"""
import xml.etree.ElementTree as ET
import uuid

file_to_read="PurchasingExample.xes"
file_to_write="PurchasingExample_Processed_Input_ForPercentage.xes" #Con_percentage
#file_to_write="PurchasingExample_Processed_Input_SinPercentage.xes" #Sin_percentage


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