import json
import os
import shutil
import sys
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import numpy as np
import scipy.stats as st
from pm4py.objects.log.importer.xes import importer as xes_importer
from simod.parameter_extraction import Operator
from simod.stochastic_miner_datatypes import BPMNNodeType, ElementInfo, ProcessInfo, Trace, BPMNGraph, int_week_days, \
    RCalendar

from .cli_formatter import print_section, print_step
from .configuration import Configuration
from .parameter_extraction import Pipeline
from .writers import xml_writer


# Process Miner

class StochasticProcessMiner:
    settings: Configuration
    bpmn_graph: BPMNGraph
    gateways_branching: dict
    log_traces: list

    def __init__(self, settings: Configuration):
        self.settings = settings

    def execute_pipeline(self):
        self._preprocessing()
        self._extract_parameters()

    def _preprocessing(self):
        print_section('Preparing the environment')

        if not os.path.exists(self.settings.output):
            print_step(f'Creating output directories: {self.settings.output}')
            os.makedirs(self.settings.output)
        if self.settings.model_path:
            print_step(f'Copying the model from {self.settings.model_path}')
            shutil.copy(self.settings.model_path, self.settings.output)

        print_step('Parsing the given model')
        self.bpmn_graph = self._parse_simulation_model(self.settings.model_path)

        print_step('Parsing the event log')
        self.log_traces = xes_importer.apply(self.settings.log_path.__str__())

    def _extract_parameters(self):
        print_section(f'Parameters extraction')

        input = ParameterExtractionInputForStochasticMiner(log_traces=self.log_traces, bpmn=self.bpmn_graph)
        output = ParameterExtractionOutputForStochasticMiner()
        extraction_pipeline = Pipeline(input=input, output=output)
        extraction_pipeline.set_pipeline([
            GatewayProbabilitiesMinerForStochasticMiner
        ])
        extraction_pipeline.execute()

        print_step('Rewriting the model')
        parameters = {'sequences': output.sequences}
        bpmn_path = os.path.join(self.settings.output, self.settings.project_name + '.bpmn')
        xml_writer.print_parameters(bpmn_path, bpmn_path, parameters)

    @staticmethod
    def _parse_simulation_model(model_path: Path):
        bpmn_element_ns = {'xmlns': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}
        tree = ET.parse(model_path.absolute())
        root = tree.getroot()
        to_extract = {'xmlns:task': BPMNNodeType.TASK,
                      'xmlns:startEvent': BPMNNodeType.START_EVENT,
                      'xmlns:endEvent': BPMNNodeType.END_EVENT,
                      'xmlns:exclusiveGateway': BPMNNodeType.EXCLUSIVE_GATEWAY,
                      # NOTE: no parallel gateways in current Simod models
                      'xmlns:parallelGateway': BPMNNodeType.PARALLEL_GATEWAY,
                      'xmlns:inclusiveGateway': BPMNNodeType.INCLUSIVE_GATEWAY}

        bpmn_graph = BPMNGraph()
        for process in root.findall('xmlns:process', bpmn_element_ns):
            for xmlns_key in to_extract:
                for bpmn_element in process.findall(xmlns_key, bpmn_element_ns):
                    name = bpmn_element.attrib["name"] \
                        if "name" in bpmn_element.attrib and len(bpmn_element.attrib["name"]) > 0 \
                        else bpmn_element.attrib["id"]
                    bpmn_graph.add_bpmn_element(bpmn_element.attrib["id"],
                                                ElementInfo(to_extract[xmlns_key], bpmn_element.attrib["id"], name))
            for flow_arc in process.findall('xmlns:sequenceFlow', bpmn_element_ns):
                bpmn_graph.add_flow_arc(flow_arc.attrib["id"], flow_arc.attrib["sourceRef"],
                                        flow_arc.attrib["targetRef"])
        bpmn_graph.encode_or_join_predecessors()

        return bpmn_graph

    @staticmethod
    def parse_xes_log(project_name: str, log_path: Path, bpmn_graph: BPMNGraph, output_folder: Path):
        def _update_first_last(start_date, end_date, current_date):
            if start_date is None:
                start_date = current_date
                end_date = current_date
            else:
                start_date = min(start_date, current_date)
                end_date = max(end_date, current_date)
            return start_date, end_date

        def _update_calendar_from_log(r_calendar, date_time, is_start, min_eps=15):
            from_date = date_time
            to_date = date_time
            if is_start:
                to_date = date_time + timedelta(minutes=min_eps)
            else:
                from_date = date_time - timedelta(minutes=min_eps)

            from_day = int_week_days[from_date.weekday()]
            to_day = int_week_days[to_date.weekday()]

            if from_day != to_day:
                r_calendar.add_calendar_item(
                    from_day, from_day,
                    "%d:%d:%d" % (from_date.hour, from_date.minute, from_date.second), "23:59:59.999")
                if to_date.hour != 0 or to_date.minute != 0 or to_date.second != 0:
                    r_calendar.add_calendar_item(
                        to_day, to_day, "00:00:00", "%d:%d:%d" % (to_date.hour, to_date.minute, to_date.second))
            else:
                r_calendar.add_calendar_item(
                    from_day, to_day,
                    "%d:%d:%d" % (from_date.hour, from_date.minute, from_date.second),
                    "%d:%d:%d" % (to_date.hour, to_date.minute, to_date.second))

        print('Parsing Event Log %s ...' % project_name)
        process_info = ProcessInfo()
        i = 0
        total_traces = 0
        resource_list = set()

        task_resource = dict()
        task_distribution = dict()
        flow_arcs_frequency = dict()
        correct_traces = 0
        correct_activities = 0
        total_activities = 0
        task_fired_ratio = dict()
        task_missed_tokens = 0
        missed_tokens = dict()

        log_traces = xes_importer.apply(log_path)

        arrival_times = list()
        previous_arrival_date = None

        start_date = end_date = None
        resource_calendars = dict()

        for trace in log_traces:
            if previous_arrival_date is not None:
                arrival_times.append((trace[0]['time:timestamp'] - previous_arrival_date).total_seconds())
            previous_arrival_date = trace[0]['time:timestamp']

            caseid = trace.attributes['concept:name']
            total_traces += 1
            started_events = dict()
            trace_info = Trace(caseid)
            task_sequence = list()
            for event in trace:
                task_name = event['concept:name']
                resource = event['org:resource']
                state = event['lifecycle:transition'].lower()
                timestamp = event['time:timestamp']
                start_date, end_date = _update_first_last(start_date, end_date, timestamp)
                if resource not in resource_list:
                    resource_list.add(resource)
                    resource_calendars[resource] = RCalendar("%s_Schedule" % resource)
                _update_calendar_from_log(resource_calendars[resource], timestamp, state in ["start", "assign"])
                if state in ["start", "assign"]:
                    started_events[task_name] = trace_info.start_event(
                        task_name, task_name, timestamp, resource, timestamp, None)
                    task_sequence.append(task_name)
                elif state == "complete":
                    if task_name in started_events:
                        event_info = trace_info.complete_event(started_events.pop(task_name), timestamp)
                        if task_name not in task_resource:
                            task_resource[task_name] = dict()
                            task_distribution[task_name] = dict()
                        if resource not in task_resource[task_name]:
                            task_resource[task_name][resource] = list()
                        task_resource[task_name][resource].append(event_info)
            is_correct, fired_tasks, pending_tokens = bpmn_graph.replay_trace(task_sequence, flow_arcs_frequency)
            if len(pending_tokens) > 0:
                task_missed_tokens += 1
                for flow_id in pending_tokens:
                    if flow_id not in missed_tokens:
                        missed_tokens[flow_id] = 0
                    missed_tokens[flow_id] += 1
            if is_correct:
                correct_traces += 1
            for i in range(0, len(task_sequence)):
                if task_sequence[i] not in task_fired_ratio:
                    task_fired_ratio[task_sequence[i]] = [0, 0]
                if fired_tasks[i]:
                    correct_activities += 1
                    task_fired_ratio[task_sequence[i]][0] += 1
                task_fired_ratio[task_sequence[i]][1] += 1
            total_activities += len(fired_tasks)
            process_info.traces[caseid] = trace_info
            i += 1

        t_r = 100 * correct_traces / total_traces
        a_r = 100 * correct_activities / total_activities
        print("Correct Traces Ratio %.2f (Pass: %d, Fail: %d, Total: %d)" % (
            t_r, correct_traces, total_traces - correct_traces, total_traces))
        print("Correct Tasks  Ratio %.2f (Fire: %d, Fail: %d, Total%d: d)" % (
            a_r, correct_activities, total_activities - correct_activities, total_activities))
        print("Missed Tokens Ratio  %.2f" % (100 * task_missed_tokens / total_traces))
        print('----------------------------------------------')

        min_dur = sys.float_info.max
        max_dur = 0
        for r_id in resource_calendars:
            min_dur = min(min_dur, resource_calendars[r_id].total_weekly_work)
            max_dur = max(max_dur, resource_calendars[r_id].total_weekly_work)
            # resource_calendars[r_id].print_calendar_info()
        print('Min Resource Weekly Work: %.2f ' % (min_dur / 3600))
        print('Max Resource Weekly Work: %.2f ' % (max_dur / 3600))
        print('Saving Resource Calendars ...')
        json_map = dict()
        for r_id in resource_calendars:
            json_map[r_id] = resource_calendars[r_id].to_json()
        with open(os.path.join(output_folder, f'{project_name}_calendars.json'), 'w') as f:
            json.dump(json_map, f)

        print('Computing Branching Probability ...')
        gateways_branching = bpmn_graph.compute_branching_probability(flow_arcs_frequency)
        with open(os.path.join(output_folder, f'{project_name}_gateways_branching.json'), 'w') as f:
            json.dump(gateways_branching, f)

        print('Computing Arrival Times Distribution ...')
        with open(os.path.join(output_folder, f'{project_name}_arrival_times_distribution.json'), 'w') as f:
            json.dump(StochasticProcessMiner.best_fit_distribution(arrival_times), f)

        print('Computing Task-Resource Distributions ...')
        for task_id in task_resource:
            for resource_id in task_resource[task_id]:
                real_durations = list()
                for e_info in task_resource[task_id][resource_id]:
                    real_durations.append(resource_calendars[resource_id].find_working_time(
                        e_info.started_at, e_info.completed_at))
                    if real_durations[len(real_durations) - 1] <= 0 and e_info.started_at != e_info.completed_at:
                        x = resource_calendars[resource_id].find_working_time(e_info.started_at, e_info.completed_at)
                        print(real_durations[len(real_durations) - 1])
                task_distribution[task_id][resource_id] = StochasticProcessMiner.best_fit_distribution(real_durations)
        with open(os.path.join(output_folder, f'{project_name}_task_distribution.json'), 'w') as f:
            json.dump(task_distribution, f)

        return process_info

    @staticmethod
    def best_fit_distribution(data, bins=200):
        """Model data by finding best fit distribution to data"""
        # Get histogram of original data
        d_min = sys.float_info.max
        d_max = 0
        for d_data in data:
            d_min = min(d_min, d_data)
            d_max = max(d_max, d_data)

        y, x = np.histogram(data, bins=bins, density=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0

        # Distributions to check
        distributions = [
            st.alpha, st.anglit, st.arcsine, st.beta, st.betaprime, st.bradford, st.burr, st.cauchy, st.chi, st.chi2,
            st.cosine, st.dgamma, st.dweibull, st.erlang, st.expon, st.exponnorm, st.exponweib, st.exponpow, st.f,
            st.fatiguelife, st.fisk, st.foldcauchy, st.foldnorm, st.genlogistic, st.genpareto,
            st.gennorm, st.genextreme, st.gausshyper, st.gamma, st.gengamma, st.genhalflogistic, st.gilbrat,
            st.gompertz,
            st.gumbel_r, st.gumbel_l, st.halfcauchy, st.halflogistic, st.halfnorm, st.halfgennorm, st.hypsecant,
            st.invgamma, st.invgauss, st.invweibull, st.johnsonsb, st.johnsonsu, st.ksone, st.kstwobign, st.laplace,
            st.levy, st.levy_l, st.logistic, st.loggamma, st.loglaplace, st.lognorm, st.lomax, st.maxwell, st.mielke,
            st.nakagami, st.ncx2, st.ncf, st.nct, st.norm, st.pareto, st.pearson3, st.powerlaw, st.powerlognorm,
            st.powernorm, st.rdist, st.reciprocal, st.rayleigh, st.rice, st.semicircular, st.t, st.triang,
            st.truncexpon,
            st.truncnorm, st.tukeylambda, st.uniform, st.vonmises, st.vonmises_line, st.wald, st.weibull_min,
            st.weibull_max, st.wrapcauchy
        ]

        # Best holders
        best_distribution = st.norm
        best_params = (0.0, 1.0)
        best_sse = np.inf

        # Estimate distribution parameters from data
        i = 1
        for distribution in distributions:

            # Try to fit the distribution
            try:
                # Ignore warnings from data that can't be fit
                # start = time.time()
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')

                    # fit dist to data
                    params = distribution.fit(data)

                    # Separate parts of parameters
                    arg = params[:-2]
                    loc = params[-2]
                    scale = params[-1]

                    # Calculate fitted PDF and error with fit in distribution
                    pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                    sse = np.sum(np.power(y - pdf, 2.0))

                    # identify if this distribution is better
                    if best_sse > sse > 0:
                        best_distribution = distribution
                        best_params = params
                        best_sse = sse
                # end = time.time()
                # print("%d- %s: %.3f" % (i, distribution.name, end - start))
                i += 1
            except Exception:
                pass

        return {"distribution_name": best_distribution.name, "distribution_params": best_params}


# Parameters Extraction Implementations: For Stochastic Process Miner

@dataclass
class ParameterExtractionInputForStochasticMiner:
    log_traces: list = None
    bpmn: BPMNGraph = None


@dataclass
class ParameterExtractionOutputForStochasticMiner:
    sequences: list = field(default_factory=list)


class GatewayProbabilitiesMinerForStochasticMiner(Operator):
    input: ParameterExtractionInputForStochasticMiner
    output: ParameterExtractionOutputForStochasticMiner

    def __init__(self, input: ParameterExtractionInputForStochasticMiner,
                 output: ParameterExtractionOutputForStochasticMiner):
        self.input = input
        self.output = output
        self._execute()

    def _execute(self):
        print_step('Mining gateway probabilities')

        arcs_frequencies = self._compute_sequence_flow_frequencies()
        gateways_branching = self.input.bpmn.compute_branching_probability(arcs_frequencies)

        sequences = []
        for gateway_id in gateways_branching:
            for seqflow_id in gateways_branching[gateway_id]:
                probability = gateways_branching[gateway_id][seqflow_id]
                sequences.append({'elementid': seqflow_id, 'prob': probability})

        self.output.sequences = sequences

    def _compute_sequence_flow_frequencies(self):
        flow_arcs_frequency = dict()
        for trace in self.input.log_traces:
            task_sequence = list()
            for event in trace:
                task_name = event['concept:name']
                state = event['lifecycle:transition'].lower()
                if state in ["start", "assign"]:
                    task_sequence.append(task_name)
            self.input.bpmn.replay_trace(task_sequence, flow_arcs_frequency)
        return flow_arcs_frequency
