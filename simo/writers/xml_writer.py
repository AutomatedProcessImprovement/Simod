# -*- coding: utf-8 -*-
import uuid
from lxml import etree
from lxml.builder import ElementMaker # lxml only !
import xml.etree.ElementTree as ET

import os
import re

# print(uuid.uuid4())

#--------------- General methods ----------------
def create_file(output_file, element):
	file_exist = os.path.exists(output_file)
	with open(output_file, 'wb') as f:
		f.write(element)
	f.close()

#-------------- kernel --------------
def print_parameters(bpmn_input, output_file, parameters):
	my_doc = xml_template(parameters['arrival_rate'], parameters['time_table'],
		parameters['resource_pool'], parameters['elements_data'], parameters['sequences'], parameters['instances'])
	root = append_parameters(bpmn_input,my_doc)
	create_file(output_file, etree.tostring(root, pretty_print=True))

def xml_template(arrival_rate, time_table, resource_pool, elements_data, sequences, instances):
	E = ElementMaker(namespace="http://www.qbp-simulator.com/Schema201212", nsmap={'qbp' : "http://www.qbp-simulator.com/Schema201212"})
	PROCESSSIMULATIONINFO = E.processSimulationInfo
	ARRIVALRATEDISTRIBUTION = E.arrivalRateDistribution
	TIMEUNIT = E.timeUnit
	TIMETABLES = E.timetables
	TIMETABLE = E.timetable
	RULES = E.rules
	RULE = E.rule
	RESOURCES = E.resources
	RESOURCE = E.resource
	ELEMENTS = E.elements
	ELEMENT = E.element
	DURATION = E.durationDistribution
	RESOURCESIDS = E.resourceIds
	RESOURCESID = E.resourceId
	SEQUENCEFLOWS = E.sequenceFlows
	SEQUENCEFLOW = E.sequenceFlow

	rootid="qbp_"+str(uuid.uuid4())

	my_doc = PROCESSSIMULATIONINFO(
		ARRIVALRATEDISTRIBUTION(
			TIMEUNIT("seconds"),
			type=arrival_rate['dname'],mean=str(arrival_rate['dparams']['mean']),arg1=str(arrival_rate['dparams']['arg1']),arg2=str(arrival_rate['dparams']['arg2'])
		),
		TIMETABLES(
			*[
				TIMETABLE(
					RULES(
						RULE(fromTime=table['from_t'],toTime=table['to_t'],fromWeekDay=table['from_w'],toWeekDay=table['to_w']),
					),
					id=table['id_t'],default=table['default'],name=table['name']
				) for table in time_table
			]
		),
		RESOURCES(
			*[
				RESOURCE(id=res['id'],name=res['name'],totalAmount=res['total_amount'],costPerHour=res['costxhour'],timetableId=res['timetable_id']) for res in resource_pool
			]
		),
		ELEMENTS(
			*[
				ELEMENT(
					DURATION(
						TIMEUNIT("seconds"),
						type=e['type'],mean=e['mean'],arg1=e['arg1'],arg2=e['arg2']
					),
					RESOURCESIDS(
						RESOURCESID(str(e['resource']))
					),
					id=e['id'],elementId=e['elementid']
				) for e in elements_data
			]
		),
		SEQUENCEFLOWS(
			*[
				SEQUENCEFLOW(elementId=seq['elementid'],executionProbability=str(seq['prob'])) for seq in sequences
			]
		),
		id=rootid, processInstances=str(instances), startDateTime="2017-08-14T08:00:00.000Z",currency="EUR"
	)
	return my_doc

def append_parameters(bpmn_input,my_doc):
	node=etree.fromstring(etree.tostring(my_doc, pretty_print=True))
	tree = etree.parse(bpmn_input)
	root = tree.getroot()
	froot= etree.fromstring(etree.tostring(root, pretty_print=True))
	froot.append(node)
	return froot
