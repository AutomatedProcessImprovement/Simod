# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET

class BpmnReader(object):
	"""
	This class reads and parse the elements of a given bpmn 2.0 model
	"""
	def __init__(self,input):
		"""constructor"""
		self.tree = ET.parse(input)
		self.root = self.tree.getroot()
		self.ns = {'xmlns':'http://www.omg.org/spec/BPMN/20100524/MODEL'}

	def get_tasks_info(self,noTailingEvets=True):
		"""reads and parse all the tasks information"""
		values = list()
		for process in self.root.findall('xmlns:process',self.ns):
			for task in process.findall('xmlns:task',self.ns):
				task_id = task.get('id')
				task_name = task.get('name')
				if not((task_name=='Start'or task_name=='End') and noTailingEvets==True):
					values.append(dict(task_id=task_id,task_name=task_name))
				else:
					values.append(dict(task_id=task_id,task_name=task_name))
		return values

	def get_ex_gates_info(self):
		"""reads and parse all the exclusive gates information"""
		values = list()
		for process in self.root.findall('xmlns:process',self.ns):
			for ex_gateway in process.findall('xmlns:exclusiveGateway',self.ns):
				gate_id = ex_gateway.get('id')
				gate_name = ex_gateway.get('name')
				gate_dir = ex_gateway.get('gatewayDirection')
				values.append(dict(gate_id=gate_id,gate_name=gate_name,gate_dir=gate_dir))
		return values

	def get_inc_gates_info(self):
		"""reads and parse all the inclusive gates information"""
		values = list()
		for process in self.root.findall('xmlns:process',self.ns):
			for inc_gateway in process.findall('xmlns:inclusiveGateway',self.ns):
				gate_id = inc_gateway.get('id')
				gate_name = inc_gateway.get('name')
				gate_dir = inc_gateway.get('gatewayDirection')
				values.append(dict(gate_id=gate_id,gate_name=gate_name,gate_dir=gate_dir))
		return values

	def get_para_gates_info(self):
		"""reads and parse all the parallel gates information"""
		values = list()
		for process in self.root.findall('xmlns:process',self.ns):
			for para_gateway in process.findall('xmlns:parallelGateway',self.ns):
				gate_id = para_gateway.get('id')
				gate_name = para_gateway.get('name')
				gate_dir = para_gateway.get('gatewayDirection')
				values.append(dict(gate_id=gate_id,gate_name=gate_name,gate_dir=gate_dir))
		return values

	def get_start_event_info(self):
		"""reads and parse all the start events information"""
		values = list()
		for process in self.root.findall('xmlns:process',self.ns):
			for start_event in process.findall('xmlns:startEvent',self.ns):
				start_id = start_event.get('id')
				start_name = start_event.get('name')
				values.append(dict(start_id=start_id,start_name=start_name))
		return values

	def get_end_event_info(self):
		"""reads and parse all the start events information"""
		values = list()
		for process in self.root.findall('xmlns:process',self.ns):
			for end_event in process.findall('xmlns:endEvent',self.ns):
				end_id = end_event.get('id')
				end_name = end_event.get('name')
				values.append(dict(end_id=end_id,end_name=end_name))
		return values

	def get_timer_events_info(self):
		"""reads and parse all the start events information"""
		values = list()
		for process in self.root.findall('xmlns:process',self.ns):
			for timer_event in process.findall('xmlns:intermediateCatchEvent',self.ns):
				timer_id = timer_event.get('id')
				timer_name = timer_event.get('name')
				values.append(dict(timer_id=timer_id,timer_name=timer_name))
		return values

	def get_edges_info(self):
		"""reads and parse all the edges information"""
		values = list()
		for process in self.root.findall('xmlns:process',self.ns):
			for sequence in process.findall('xmlns:sequenceFlow',self.ns):
				source = sequence.get('sourceRef')
				target = sequence.get('targetRef')
				values.append(dict(source=source,target=target))
		return values

	def find_sequence_id(self,source,target):
		for process in self.root.findall('xmlns:process',self.ns):
			sequences = process.findall('xmlns:sequenceFlow',self.ns)
		i = 0
		finish=False
		sequence_id = ''
		while(i < len(sequences) and not finish):
			if sequences[i].get('sourceRef') == source and sequences[i].get('targetRef') == target:
				sequence_id = sequences[i].get('id')
				finish= True
			i +=1
		return sequence_id

	def follow_sequence(self, process, flow_id, direction ):
		sequence_flows = process.findall('xmlns:sequenceFlow',self.ns)
		return list(filter(lambda x:x.get('id')==flow_id,sequence_flows))[0].get(direction)

	def getProcessId(self):
		process = self.root.find('xmlns:process', self.ns)
		id = process.get('id')
		return id

	def getStartEventId(self):
		process = self.root.find('xmlns:process', self.ns)
		start = process.find('xmlns:startEvent', self.ns)
		id = start.get('id')
		return id
