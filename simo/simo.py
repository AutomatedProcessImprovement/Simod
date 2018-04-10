# -*- coding: utf-8 -*-
import traces_alignment as tal
import bpmn_reader as br
import log_reader as lr
import process_structure as gph
import log_replayer as rpl
import configparser as cfg
import task_duration_distribution as td
import interarrival_definition as arr
import gateways_probabilities as gt
import role_discovery as rl
import xml_writer as xml
import assets_writer as asst

import uuid
import sys, getopt
import support as sup
import networkx as nx

# --support --
def gen_id():
	return "qbp_"+str(uuid.uuid4())

def find_resource_id(resource_pool,resource_name):
	id = 0
	for resource in resource_pool:
		# print(resource)
		if resource['name'] == resource_name:
			id = resource['id']
	return id

def print_log_stats(conformed_traces, not_conformed_traces, log):
	print('Conformity percentage: ' + str(sup.ffloat((len(conformed_traces)/(len(conformed_traces)+len(not_conformed_traces)))*100, 2))+'%')
	num_events = 0
	for event in log.raw_data:
		if event['task'] != 'Start' and event['task'] != 'End':
			num_events+=1
	effective_events = 0
	for trace in conformed_traces:
		for event in trace:
			if event['user'] != 'AUTO' and event['task'] != 'Start' and event['task'] != 'End':
				effective_events += 1
	print('Total events: '+ str(num_events/2))
	print('Effective events: ' + str(effective_events))

# --setup--
def main(argv):
	"""Main aplication method"""
	log_file_name = ''
	bpmn_file_name = ''
	output_file = ''
	root_dir = ''
	config_file = ''
	#Capture locations from command line
	try:
		opts, args = getopt.getopt(argv,"hi:b:o:r:",["ifile=","bfile=","ofile=","rfolder="])
	except getopt.GetoptError:
		print('test.py -i <logfile> -b <bpmnfile> -o <outputfile> -r <rootdir>')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print('test.py -i <logfile> -b <bpmnfile> -o <outputfile> -r <rootdir>')
			sys.exit()
		elif opt in ("-i", "--ifile"):
			log_file_name = arg
		elif opt in ("-b", "--bfile"):
			bpmn_file_name = arg
		elif opt in ("-o", "--ofile"):
			output_file = arg
		elif opt in ("-r", "--rfolder"):
			root_dir = arg
	log_input = root_dir + chr(92) + log_file_name
	# log_input = root_dir +'\'+log_file_name
	bpmn_input = root_dir + chr(92) + bpmn_file_name
	# bpmn_input = root_dir + '\' + bpmn_file_name
	output = root_dir + chr(92) + output_file
	# output = root_dir + '\' + output_file
	config_file = root_dir + chr(92) + 'simo\config.ini'
	# config_file = root_dir + '\' + 'simo/config.ini'
	assets_output = root_dir + chr(92) + 'statistics\log'

	alignment_input = root_dir + chr(92) + 'inputs'

	kernel(log_input,bpmn_input,output,config_file,assets_output,alignment_input)

# -- kernel --
def kernel(log_input,bpmn_input,output,config_file,assets_output,alignment_input):
	#Reading and parsing of config file
	try:
		config = cfg.ConfigParser(interpolation=None)
		config.read(config_file)
		end_timeformat = config['Test']['endtimeformat']
		start_timeformat = config['Test']['starttimeformat']
		log_columns_numbers = sup.reduce_list(config['Test']['logcolumnsnumbers'])
		alignment = config['Test']['alignment'] in ['true', 'True', '1', 'Yes', 'yes']
		if alignment:
			align_info_file = config['Test']['aligninfofile']
			align_type_file = config['Test']['aligntypefile']
	except Exception as e:
		print('Invalid config file format...' + '\n' + str(e))
	else:
		#Log reading
		try:
			log = lr.LogReader(log_input,log_columns_numbers,start_timeformat,end_timeformat)
		except IOError as e:
		 	print('Input error ' + str(e))
		except Exception as e:
		 	print('Unexpected error...' + '\n' + str(e))
		else:
			#TODO mannage exceptions from bpmn reading...
			bpmn = br.BpmnReader(bpmn_input)
			# Creation of process graph
			process_graph = gph.create_process_structure(bpmn)
			# Aligning the process traces
			if alignment:
				tal.align_traces(alignment_input, log, align_info_file, align_type_file)
			# Process replaying
			conformed_traces, not_conformed_traces, process_stats = rpl.replay(process_graph, log)
			print_log_stats(conformed_traces, not_conformed_traces, log)
			# Determination of first tasks for calculate the arrival rate
			inter_arrival_times = arr.define_interarrival_tasks(process_graph, conformed_traces)
			arrival_rate = (td.get_task_distribution(inter_arrival_times, False, 50))
			# Gateways probabilities 1=Historycal, 2=Random, 3=Equiprobable
			sequences = gt.define_probabilities(process_graph,bpmn,log,1)
			# Analysing resource pool
			roles = rl.read_resource_pool(log,sim_percentage=0.60)
			resource_pool = list()

			for role in roles:
				resource_pool.append(dict(id=gen_id(),name=role['role'],total_amount=str(role['quantity']),costxhour="20",timetable_id="QBP_DEFAULT_TIMETABLE"))
			resource_pool[0]['id'] = 'QBP_DEFAULT_RESOURCE'

			# Tasks id information
			tasks = list(filter(lambda x: process_graph.node[x]['type'] =='task',nx.nodes(process_graph)))
			i = 0
			elements_data=list()
			for task in tasks:
				# Tasks duration_distribution
				values = td.get_task_distribution(process_graph.node[task]['processing_times'])
				# Task resources assignment
				try:
					resource_name = log.read_resource_task(process_graph.node[task]['name'],roles)
				except ValueError:
					resource_name = resource_pool[0]['name']
				# Build data array
				elements_data.append(dict(id=gen_id(),elementid=process_graph.node[task]['id'],type=values['dname'],mean=str(values['dparams']['mean']),
						arg1=str(values['dparams']['arg1']),arg2=str(values['dparams']['arg2']),resource=find_resource_id(resource_pool,resource_name)))
				sup.print_progress(((i / (len(tasks)-1))* 100),'Analysing tasks data ')
				i +=1
			sup.print_done_task()
			# Exporting values to XML BIMP format
			xml.print_parameters(bpmn_input, output, arrival_rate, resource_pool, elements_data, sequences)
			sup.print_progress(((100 /100)* 100),'Printing XML ')
			sup.print_done_task()
			# Exporting assets
			if not alignment and len(not_conformed_traces) > 0:
				asst.export_not_conformed_traces(not_conformed_traces, assets_output)
			asst.export_tasks_statistics(process_graph, assets_output)
			asst.export_process_statistics(process_stats, process_graph, assets_output)


if __name__ == "__main__":
	main(sys.argv[1:])
