# -*- coding: utf-8 -*-
from extraction import log_replayer as rpl
from extraction import interarrival_definition as arr
# from extraction import gateways_probabilities2 as gt
from extraction import role_discovery as rl
from extraction import schedule_tables as sch
from extraction import tasks_evaluator as te

import pandas as pd
from support_modules import support as sup


def extract_parameters(log, bpmn, process_graph, settings):
    if bpmn and log:
        bpmnId = bpmn.getProcessId()
        startEventId = bpmn.getStartEventId()
        # Creation of process graph
        # -------------------------------------------------------------------
        # Analysing resource pool LV917 or 247
        res_analyzer = rl.ResourcePoolAnalyser(
            log, sim_threshold=settings['rp_similarity'])
        # roles = res_analyzer.roles
        resource_table = res_analyzer.resource_table
        resource_pool, time_table, resource_table = sch.analize_schedules(
            resource_table, log, True, '247')
        # -------------------------------------------------------------------
        # Process replaying
        conformed_traces, not_conformed_traces, process_stats = rpl.replay(
            process_graph, log, settings)
        process_stats = pd.DataFrame.from_records(process_stats)
        # -------------------------------------------------------------------
        # Adding role to process stats
        resource_table = pd.DataFrame.from_records(resource_table)
        process_stats = process_stats.merge(
            resource_table, on='resource', how='left')
        # -------------------------------------------------------------------
        # Determination of first tasks for calculate the arrival rate
        inter_evaluator = arr.InterArrivalEvaluator(process_graph,
                                                    conformed_traces, settings)
        arrival_rate_bimp = inter_evaluator.dist
        arrival_rate_bimp['startEventId'] = startEventId
        # Gateways probabilities 1=Historical, 2=Random, 3=Equiprobable
        # sequences = gt.define_probabilities(process_graph,
        #                                     bpmn,
        #                                     log, settings['gate_management'])
        model_data = pd.DataFrame.from_dict(
            dict(process_graph.nodes.data()), orient='index')[['id']]
        sup.create_json(model_data.to_dict('index'), 'id.csv')
        # sequences = gt.define_probabilities(process_graph,
        #                                     settings['gate_management'])
        # print(sequences)
        # -------------------------------------------------------------------
        # Tasks id information
        elements_data = te.TaskEvaluator(process_graph,
                                         process_stats,
                                         resource_pool,
                                         settings).elements_data

        parameters = dict(arrival_rate=arrival_rate_bimp,
                          time_table=time_table,
                          resource_pool=resource_pool,
                          elements_data=elements_data,
                          sequences=sequences,
                          instances=len(log.get_traces()),
                          bpmnId=bpmnId)
        return parameters, process_stats
