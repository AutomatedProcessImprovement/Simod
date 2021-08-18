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


def reply_event_log(event_log, bpmn_graph):
    traces = event_log.get_raw_traces()

    def _collect_task_sequence(trace):
        task_sequence = list()
        for event in trace:
            task_name = event['task']  # original: concept:name
            state = event['event_type'].lower()  # original: lifecycle:transition
            if state in ["start", "assign"]:
                task_sequence.append(task_name)
        return task_sequence

    task_sequences = map(lambda trace: _collect_task_sequence(trace), traces)

    flow_arcs_frequency = dict()
    for sequence in task_sequences:
        bpmn_graph.replay_trace(sequence, flow_arcs_frequency)

    return flow_arcs_frequency


def main():
    for i in range(0, 8):
        current_dir = str(Path(os.path.dirname(__file__)).parent.parent)
        log_path = current_dir + xes_simodbpmn_file_paths[experiment_logs[i]][0]
        model_path = current_dir + xes_simodbpmn_file_paths[experiment_logs[i]][1]

        settings = Configuration(model_path=Path(model_path), log_path=Path(log_path))
        settings.fill_in_derived_fields()

        event_log = LogReader(log_path, settings.read_options)
        bpmn_graph = BPMNGraph.from_bpmn_path(Path(model_path))

        x = reply_event_log(event_log, bpmn_graph)

    os._exit(0)


if __name__ == '__main__':
    main()
