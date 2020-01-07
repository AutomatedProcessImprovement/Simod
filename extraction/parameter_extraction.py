# -*- coding: utf-8 -*-
# from support_modules import support as sup
from extraction import log_replayer as rpl
from extraction import pdf_definition as pdf
from extraction import interarrival_definition as arr
from extraction import gateways_probabilities as gt
from extraction import role_discovery as rl
from extraction import schedule_tables as sch
from extraction import tasks_evaluation as te


import pandas as pd

# -- Extract parameters --
def extract_parameters(log, bpmn, process_graph, settings):
    if bpmn != None and log != None:
        bpmnId = bpmn.getProcessId()
        startEventId = bpmn.getStartEventId()
        # Creation of process graph
        #-------------------------------------------------------------------
        # Analysing resource pool LV917 or 247
        roles, resource_table = rl.read_resource_pool(log, drawing=False, sim_percentage=settings['rp_similarity'])
        resource_pool, time_table, resource_table = sch.analize_schedules(resource_table, log, True, '247')
        #-------------------------------------------------------------------
        # Process replaying
        conformed_traces, not_conformed_traces, process_stats = rpl.replay(process_graph, log, settings)
        process_stats = pd.DataFrame.from_records(process_stats)
        # -------------------------------------------------------------------
        # Adding role to process stats
        resource_table = pd.DataFrame.from_records(resource_table)        
        process_stats = process_stats.merge(resource_table, on='resource', how='left')
        #-------------------------------------------------------------------
        # Determination of first tasks for calculate the arrival rate
        inter_arrival_times = arr.define_interarrival_tasks(process_graph, conformed_traces, settings)
        arrival_rate_bimp = (pdf.get_task_distribution(inter_arrival_times, 50))
        arrival_rate_bimp['startEventId'] = startEventId
        #-------------------------------------------------------------------
        # Gateways probabilities 1=Historical, 2=Random, 3=Equiprobable
        sequences = gt.define_probabilities(process_graph, bpmn, log, 1)
        #-------------------------------------------------------------------
        # Tasks id information
        elements_data = te.evaluate_tasks(process_graph, process_stats, resource_pool, settings)
        
        parameters = dict(arrival_rate=arrival_rate_bimp, time_table=time_table, resource_pool=resource_pool,
                              elements_data=elements_data, sequences=sequences, instances=len(log.get_traces(settings['read_options'])),
                                bpmnId=bpmnId)
        return parameters, process_stats