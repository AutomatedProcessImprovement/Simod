# NOTE: this file is not an actual test though, but could be helpful to debug the new replayer

import os
from pathlib import Path

from simod.readers.log_reader import LogReader
from simod.replayer_datatypes import BPMNGraph, ElementInfo
from simod.configuration import Configuration

bpmn_schema_url = 'http://www.omg.org/spec/BPMN/20100524/MODEL'
bpmn_element_ns = {'xmlns': bpmn_schema_url}

xes_simodbpmn_file_paths = {
    'purchasing_example': ['/input_files/xes_files/PurchasingExample.xes',
                           '/input_files/bpmn_simod_models/PurchasingExample.bpmn'],
    'production': ['/input_files/xes_files/production.xes',
                   '/input_files/bpmn_simod_models/Production.bpmn'],
    'insurance': ['/input_files/xes_files/insurance.xes',
                  '/input_files/bpmn_simod_models/insurance.bpmn'],
    'call_centre': ['/input_files/xes_files/callcentre.xes',
                    '/input_files/bpmn_simod_models/callcentre.bpmn'],
    'bpi_challenge_2012': ['/input_files/xes_files/BPI_Challenge_2012_W_Two_TS.xes',
                           '/input_files/bpmn_simod_models/BPI_Challenge_2012_W_Two_TS.bpmn'],
    'bpi_challenge_2017_filtered': ['/input_files/xes_files/BPI_Challenge_2017_W_Two_TS_filtered.xes',
                                    '/input_files/bpmn_simod_models/BPI_Challenge_2017_W_Two_TS_filtered.bpmn'],
    'bpi_challenge_2017': ['/input_files/xes_files/BPI_Challenge_2017_W_Two_TS.xes',
                           '/input_files/bpmn_simod_models/BPI_Challenge_2017_W_Two_TS.bpmn'],
    'consulta_data_mining': ['/input_files/xes_files/ConsultaDataMining201618.xes',
                             '/input_files/bpmn_simod_models/ConsultaDataMining201618.bpmn']
}

to_execute = {'HC-STRICT': False,
              'HC-FLEX': False,
              'TS-STRICT': False,
              'NSGA-II': False,
              'METRICS': True}

experiment_logs = {0: 'production',
                   1: 'purchasing_example',
                   2: 'consulta_data_mining',
                   3: 'insurance',
                   4: 'call_centre',
                   5: 'bpi_challenge_2012',
                   6: 'bpi_challenge_2017_filtered',
                   7: 'bpi_challenge_2017'}


def reply_event_log(event_log, bpmn_graph, log_path=None):
    traces = event_log.get_traces()

    flow_arcs_prob = []
    flow_arcs_freq = []
    correct_traces = 0
    correct_activities = 0
    total_activities = 0
    task_fired_ratio = dict()
    task_missed_tokens = 0
    missed_tokens = dict()
    total_tokens = 0
    total_traces = 0

    for post_p in [False, True]:
        flow_arcs_frequency = dict()

        for trace in traces:
            sequence = [event['task'] for event in trace]
            is_correct, fired_tasks, pending_tokens = bpmn_graph.replay_trace(sequence, flow_arcs_frequency, post_p)
            if not post_p:
                total_traces += 1
                if len(pending_tokens) > 0:
                    task_missed_tokens += 1
                    for flow_id in pending_tokens:
                        if flow_id not in missed_tokens:
                            missed_tokens[flow_id] = 0
                        missed_tokens[flow_id] += 1
                        total_tokens += 1
                if is_correct:
                    correct_traces += 1
                for i in range(0, len(sequence)):
                    if sequence[i] not in task_fired_ratio:
                        task_fired_ratio[sequence[i]] = [0, 0]
                    if fired_tasks[i]:
                        correct_activities += 1
                        task_fired_ratio[sequence[i]][0] += 1
                    task_fired_ratio[sequence[i]][1] += 1
                total_activities += len(fired_tasks)

        flow_arcs_prob.append(bpmn_graph.compute_branching_probability(flow_arcs_frequency))
        flow_arcs_freq.append(flow_arcs_frequency)
    t_r = 100 * correct_traces / total_traces
    a_r = 100 * correct_activities / total_activities
    print("Correct Traces Ratio %.2f (Pass: %d, Fail: %d, Total: %d)" % (
        t_r, correct_traces, total_traces - correct_traces, total_traces))
    print("Correct Tasks  Ratio %.2f (Fire: %d, Fail: %d, Total: %d)" % (
        a_r, correct_activities, total_activities - correct_activities, total_activities))
    print("Missed Tokens Ratio  %.2f (%d tokens left)" % (100 * task_missed_tokens / total_traces, total_tokens))
    print('----------------------------------------------')
    for task_id in task_fired_ratio:
        print('%s -- %.2f (%d / %d)' % (task_id, task_fired_ratio[task_id][0] / task_fired_ratio[task_id][1],
                                        task_fired_ratio[task_id][0], task_fired_ratio[task_id][1]))
    print('----------------------------------------------')
    print_probabilities(flow_arcs_prob, flow_arcs_freq)
    return flow_arcs_frequency


def print_probabilities(flow_arcs_prob, f_arcs_freq):
    c = 1
    for g_id in flow_arcs_prob[0]:
        print(g_id)
        for i in [0, 1]:
            print('G_%d' % c, end='')
            for flow_id in flow_arcs_prob[i][g_id]:
                print(', %.3f (%d)' % (flow_arcs_prob[i][g_id][flow_id], f_arcs_freq[i][flow_id]), end='')
            print()
        print('........................................')
        c += 1


def main():
    for i in range(7, 8):
        current_dir = str(Path(os.path.dirname(__file__)).parent.parent)
        log_path = current_dir + xes_simodbpmn_file_paths[experiment_logs[i]][0]
        model_path = current_dir + xes_simodbpmn_file_paths[experiment_logs[i]][1]

        settings = Configuration(model_path=Path(model_path), log_path=Path(log_path))
        settings.fill_in_derived_fields()

        event_log = LogReader(log_path, settings.read_options)
        bpmn_graph = BPMNGraph.from_bpmn_path(Path(model_path))
        print('Process: ' + experiment_logs[i])
        f_arcs_freq = reply_event_log(event_log, bpmn_graph, log_path)
        break
    os._exit(0)


if __name__ == '__main__':
    main()
