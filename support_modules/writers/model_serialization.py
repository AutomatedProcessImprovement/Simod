# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 22:15:47 2021

@author: Manuel Camargo
"""
import xml.etree.ElementTree as ET
import xmltodict as xtd
import readers.bpmn_reader as br

def serialize_model(filename):
    bpmn = br.BpmnReader(filename)
    tasks = {x['task_id']: x['task_name']for x in bpmn.get_tasks_info()}
    seqs = bpmn.get_edges_info()
    
    ns = {'qbp': "http://www.qbp-simulator.com/Schema201212"}
    tree = ET.parse(filename)
    root = tree.getroot()
    sim_model_xml = ET.tostring(root.find("qbp:processSimulationInfo", 
                                          namespaces=ns))
    sim_model_xml = sim_model_xml.decode()
    sim_model_xml = sim_model_xml.replace(ns['qbp'], 'qbp')
    sim_model_xml = bytes(sim_model_xml, 'utf-8')
    sim_model_xml = xtd.parse(sim_model_xml)
    info = "qbp:processSimulationInfo"
    sim_model_xml[info]['arrival_rate'] = (
        sim_model_xml[info].pop('qbp:arrivalRateDistribution'))
    sim_model_xml[info]['arrival_rate']['dname'] = (
        sim_model_xml[info]['arrival_rate'].pop('@type')) 
    sim_model_xml[info]['arrival_rate']['dparams'] = dict()
    sim_model_xml[info]['arrival_rate']['dparams']['arg1'] = (
        sim_model_xml[info]['arrival_rate'].pop('@arg1'))
    sim_model_xml[info]['arrival_rate']['dparams']['arg2'] = (
        sim_model_xml[info]['arrival_rate'].pop('@arg2')) 
    sim_model_xml[info]['arrival_rate']['dparams']['mean'] = (
        sim_model_xml[info]['arrival_rate'].pop('@mean')) 
    sim_model_xml[info]['arrival_rate'].pop('qbp:timeUnit')
    
    tags = {'element': 'elements_data',
            'resource': 'resource_pool',
            'sequenceFlow': 'sequences',
            'timetable': 'time_table'}
    for k, v in tags.items():
        element = sim_model_xml[info]['qbp:'+k+'s']["qbp:"+k]
        sim_model_xml[info].pop('qbp:'+k+'s')
        sim_model_xml[info][v] = element
    sim_model_xml[info]['instances'] = (
        sim_model_xml[info].pop('@processInstances'))
    sim_model_xml[info]['start_time'] = (
        sim_model_xml[info].pop('@startDateTime'))
    sim_model_xml[info].pop('@currency')
    sim_model_xml[info].pop('@id')
    sim_model_xml[info].pop('@xmlns:qbp')
    element = sim_model_xml[info]
    sim_model_xml.pop(info)
    sim_model_xml = element
    for element in sim_model_xml['elements_data']:
        element['elementid'] = element.pop('@elementId')
        element['id'] = element.pop('@id')
        element['arg1'] = element['qbp:durationDistribution']['@arg1']
        element['arg2'] = element['qbp:durationDistribution']['@arg2']
        element['mean'] = element['qbp:durationDistribution']['@mean']
        element['type'] = element['qbp:durationDistribution']['@type']
        element['resource'] = element['qbp:resourceIds']['qbp:resourceId']
        element['name'] = tasks[element['elementid']]
        element.pop('qbp:durationDistribution')
        element.pop('qbp:resourceIds')
        
    for element in sim_model_xml['sequences']:
        element['elementid'] = element.pop('@elementId')
        element['prob'] = element.pop('@executionProbability')
        seq = list(filter(lambda x: x['sf_id'] ==element['elementid'], seqs))[0]
        element['gatewayid'] = seq['source']
        element['out_path_id'] = seq['target']
    
    return sim_model_xml
