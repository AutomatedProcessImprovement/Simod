# -*- coding: utf-8 -*-
from support_modules import support as sup

def define_interarrival_tasks(process_graph, conformed_traces):
	# Analysis of start tasks
	tasks = analize_first_tasks(process_graph)
	inter_arrival_times = find_inter_arrival(tasks, conformed_traces)

	# for task in tasks:
	# 	process_graph.node[task]['type']
	sup.print_done_task()
	return inter_arrival_times

def find_inter_arrival(tasks, conformed_traces):
	records = list()
	conformed_traces
	for trace in conformed_traces:
		for row in trace:
			for task in tasks:
				if row['task'] == task:
					records.append(row['start_timestamp'])
	dates = map(lambda x: x.date(), records)
	days = sorted(list(set(dates)))
	# records by day
	list_by_date = list()
	for x in days:
		temp_day_list = list()
		for y in records:
			if x == y.date():
				temp_day_list.append(y)
		list_by_date.append(dict(day=x,times=sorted(temp_day_list)))
	# delta time by day
	list_deltas = list()
	for ld in list_by_date:
		temp_delta = list()
		if len(ld['times']) > 1:
			for t in range(0,len(ld['times'])-1):
				temp_delta.append((ld['times'][t + 1] - ld['times'][t]).total_seconds())
			list_deltas.append(dict(day=ld['day'].isoweekday(),times=temp_delta))
	#concat of all delta times
	inter_arrival_times = list()
	for z in list_deltas:
		#print(z['times'])
		inter_arrival_times = inter_arrival_times + z['times']
		# Kernel methods
	return inter_arrival_times

def extract_target_tasks(process_graph, num):
	tasks_list=list()
	for node in process_graph.neighbors(num):
		if process_graph.node[node]['type']=='task' or process_graph.node[node]['type']=='start' or process_graph.node[node]['type']=='end':
			tasks_list.append([node])
		else:
			tasks_list.append(extract_target_tasks(process_graph, node))
	return 	tasks_list

def find_tasks_predecesors(process_graph,num):
	# Sources
	r = process_graph.reverse(copy=True)
	paths = list(r.neighbors(num))
	task_paths = extract_target_tasks(r, num)
	in_paths = [sup.reduce_list(path) for path in task_paths]
	ins = [dict(in_tasks=y, in_node= x) for x,y in zip(paths, in_paths)]

	return dict(task=num,sources=ins)

def analize_first_tasks(process_graph):
	tasks_list = list()
	for node in process_graph.nodes:
		if process_graph.node[node]['type']=='task':
			tasks_list.append(find_tasks_predecesors(process_graph,node))
	in_tasks = list()
	i=0
	for task in tasks_list:
		sup.print_progress(((i / (len(tasks_list)-1))* 100),'Defining inter-arrival rate ')
		for path in task['sources']:
			for in_task in path['in_tasks']:
				if process_graph.node[in_task]['type']=='start':
					in_tasks.append(process_graph.node[task['task']]['name'])
		i+=1
	return list(set(in_tasks))
