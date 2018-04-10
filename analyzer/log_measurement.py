# -*- coding: utf-8 -*-
import sys, getopt
import os
import csv
import support as sup
from operator import itemgetter


# -- kernel --
def create_csv_file_header(index, output_file):
	file_exist = os.path.exists(output_file)
	with open(output_file, 'w', newline='') as f:
		fieldnames = index[0].keys()
		w = csv.DictWriter(f, fieldnames)
		w.writeheader()
		for element in index:
			w.writerow(element)
		f.close()

def create_structure(records, dictionary, main_column):
	keys = records[0].split(',')
	iterecords = iter(records)
	next(iterecords)
	for record in iterecords:
		temp_record = record.split(',')
		temp_dictionary = {}
		for x,y in zip(keys,temp_record):
			if x == main_column:
				temp_dictionary.update({x:y})
			else:
				if y == 'n/a':
					temp_dictionary.update({x:float(0)})
				else:
					temp_dictionary.update({x:float(y)})
		dictionary.append(temp_dictionary)
	return dictionary

def read_file(filename, dictionary, section, main_column):
	records = list()
	is_started = False
	with open(filename) as fp:
		for line in fp:
			if line.rstrip() == section :
				is_started = True
			if line.rstrip() == '' and is_started == True:
				is_started = False
			if is_started == True:
				if line.rstrip() != section :
					if line.rstrip() != '':
						records.append(line.rstrip())
	records = create_structure(records, dictionary, main_column)
	return records

# -------------- kernel --------------


def kernel(root_dir, individual_output, process_output):

	scenario_stats = list()
	individual_task = list()
	for root, dirs, files in os.walk(root_dir + chr(92) + 'outputs'):
		for f in files:
			scenario_stats = read_file(root_dir + chr(92) + 'outputs' + chr(92) + f, scenario_stats, 'Scenario statistics', 'KPI')
			individual_task  = read_file(root_dir + chr(92) + 'outputs' + chr(92) + f, individual_task, 'Individual task statistics', 'Name')


	create_csv_file_header(scenario_stats, process_output)
	create_csv_file_header(individual_task, individual_output)
# --setup--

root_dir = "C:/Users/Asistente/Documents/Repositorio/PaperAnalysisVsSimulacion/Experimento/Implementacion/Simulacion/SiMo_v3"
individual_output = root_dir + chr(92) + 'statistics' + chr(92) + 'simulation' + chr(92) + 'individual_output.csv'
process_output = root_dir + chr(92) + 'statistics' + chr(92) + 'simulation' + chr(92) + 'process_output.csv'

kernel(root_dir, individual_output, process_output)