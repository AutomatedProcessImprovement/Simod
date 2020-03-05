# -*- coding: utf-8 -*-
from extraction import log_replayer as rpl
from extraction import interarrival_definition as arr
from extraction import gateways_probabilities as gt
from extraction import role_discovery as rl
from extraction import schedule_tables as sch
from extraction import tasks_evaluator as te

import pandas as pd
from support_modules import support as sup

def extract_parameters(log, bpmn, process_graph, settings):
    if bpmn and log:
        bpmnId = bpmn.getProcessId()
        startEventId = bpmn.getStartEventId()
        # -------------------------------------------------------------------
        # Analysing resource pool LV917 or 247
        res_analyzer = rl.ResourcePoolAnalyser(
            log,
            sim_threshold=settings['rp_similarity'])

        ttcreator = sch.TimeTablesCreator(res_analyzer.resource_table, '247')
        resource_pool = ttcreator.resource_pool
        time_table = ttcreator.time_table
        resource_table = ttcreator.resource_table
        # -------------------------------------------------------------------
        # Process replaying
        model_data = pd.DataFrame.from_dict(
            dict(process_graph.nodes.data()), orient='index')[['id']]
        sup.create_json(model_data.to_dict('index'), 'id.csv')
        sup.save_graph(process_graph, 'process_graph.json')
        sup.create_csv_file_header(log.data, 'log_data.csv')
        sup.create_json(settings, 'settings.json')
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
        gevaluator = gt.GatewaysEvaluator(process_graph,
                                          settings['gate_management'])
        sequences = gevaluator.probabilities
        for x in sequences:
            x['elementid'] = bpmn.find_sequence_id(x['gatewayid'],
                                                   x['out_path_id'])
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
