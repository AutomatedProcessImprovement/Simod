"""
    The code is taken from external_tools/DiffResBP_Simulator/bpdfr_simulation_engine/simulation_properties_parser.py

    The aim of the module is to extract parameters from a QBP-generated XML and save them as a JSON-file for the new
    simulator to use.
"""
import itertools
import json
import multiprocessing
import time
import xml.etree.ElementTree as ET
from typing import Callable, Tuple

import pandas as pd
from tqdm import tqdm

from simod.common_routines import evaluate_logs, read_stats, execute_shell_cmd, pbar_async
from simod.configuration import Configuration

QBP_NS = {'qbp': 'http://www.qbp-simulator.com/Schema201212'}
BPMN_NS = {'xmlns': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}


def parse_model_and_save_json(qbp_bpmn_path, out_file):
    """Extracts data from BPMN QBP and saves it to JSON for the new simulator."""

    global QBP_NS, BPMN_NS

    tree = ET.parse(qbp_bpmn_path)
    root = tree.getroot()
    simod_root = root.find("qbp:processSimulationInfo", QBP_NS)

    # 1. Extracting gateway branching probabilities
    gateways_branching = dict()
    reverse_map = dict()
    for process in root.findall('xmlns:process', BPMN_NS):
        for xmlns_key in ['xmlns:exclusiveGateway', 'xmlns:inclusiveGateway']:
            for bpmn_element in process.findall(xmlns_key, BPMN_NS):
                if bpmn_element.attrib["gatewayDirection"] == "Diverging":
                    gateways_branching[bpmn_element.attrib["id"]] = dict()
                    for out_flow in bpmn_element.findall("xmlns:outgoing", BPMN_NS):
                        arc_id = out_flow.text.strip()
                        gateways_branching[bpmn_element.attrib["id"]][arc_id] = 0
                        reverse_map[arc_id] = bpmn_element.attrib["id"]
    for flow_prob in simod_root.find("qbp:sequenceFlows", QBP_NS).findall("qbp:sequenceFlow", QBP_NS):
        flow_id = flow_prob.attrib["elementId"]
        gateways_branching[reverse_map[flow_id]][flow_id] = flow_prob.attrib["executionProbability"]

    # 2. Extracting Resource Calendars
    resource_pools = dict()
    calendars_map = dict()
    bpmn_calendars = simod_root.find("qbp:timetables", QBP_NS)
    for calendar_info in bpmn_calendars:
        calendar_id = calendar_info.attrib["id"]
        if calendar_id not in calendars_map:
            calendars_map[calendar_id] = list()

        time_table = calendar_info.find("qbp:rules", QBP_NS).find("qbp:rule", QBP_NS)
        calendars_map[calendar_id].append({"from": time_table.attrib["fromWeekDay"],
                                           "to": time_table.attrib["toWeekDay"],
                                           "beginTime": _format_date(time_table.attrib["fromTime"]),
                                           "endTime": _format_date(time_table.attrib["toTime"])})

    # 3. Extracting Arrival time distribution
    arrival_time_dist = _extract_dist_params(simod_root.find("qbp:arrivalRateDistribution", QBP_NS))

    # 4. Extracting task-resource duration distributions
    bpmn_resources = simod_root.find("qbp:resources", QBP_NS)
    simod_elements = simod_root.find("qbp:elements", QBP_NS)

    resource_calendars = dict()
    for resource in bpmn_resources:
        resource_pools[resource.attrib["id"]] = list()
        calendar_id = resource.attrib["timetableId"]
        for i in range(1, int(resource.attrib["totalAmount"]) + 1):
            nr_id = "%s_%d" % (resource.attrib["id"], i)
            resource_pools[resource.attrib["id"]].append(nr_id)
            resource_calendars[nr_id] = calendars_map[calendar_id]

    task_resource_dist = dict()
    for e_inf in simod_elements:
        task_id = e_inf.attrib["elementId"]
        rpool_id = e_inf.find("qbp:resourceIds", QBP_NS).find("qbp:resourceId", QBP_NS).text
        dist_info = e_inf.find("qbp:durationDistribution", QBP_NS)

        t_dist = _extract_dist_params(dist_info)
        if task_id not in task_resource_dist:
            task_resource_dist[task_id] = dict()
        for rp_id in resource_pools[rpool_id]:
            task_resource_dist[task_id][rp_id] = t_dist

    # 5.Saving all in a single JSON file
    to_save = {
        "arrival_time_distribution": arrival_time_dist,
        "gateway_branching_probabilities": gateways_branching,
        "task_resource_distribution": task_resource_dist,
        "resource_calendars": resource_calendars,
    }
    with open(out_file, 'w') as file_writter:
        json.dump(to_save, file_writter)


def _extract_dist_params(dist_info):
    # time_unit = dist_info.find("qbp:timeUnit", simod_ns).text
    # The time_tables produced by bimp always have the parameters in seconds, although it shouws other time units in
    # the XML file.
    dist_params = {"mean": float(dist_info.attrib["mean"]),
                   "arg1": float(dist_info.attrib["arg1"]),
                   "arg2": float(dist_info.attrib["arg2"])}
    dist_name = dist_info.attrib["type"].upper()
    if dist_name == "EXPONENTIAL":
        return {"distribution_name": "expon", "distribution_params": [dist_params["arg1"], 1]}
    if dist_name == "NORMAL":
        return {"distribution_name": "norm", "distribution_params": [dist_params["mean"], dist_params["arg1"]]}
    if dist_name == "FIXED":
        return {"distribution_name": "fix", "distribution_params": [dist_params["mean"], 0, 1]}
    if dist_name == "LOGNORMAL":
        return {"distribution_name": "lognorm", "distribution_params": [dist_params["mean"], dist_params["arg1"], 0, 1]}
    if dist_name == "UNIFORM":
        return {"distribution_name": "uniform", "distribution_params": [dist_params["arg1"], dist_params["arg2"], 0, 1]}
    if dist_name == "GAMMA":
        return {"distribution_name": "gamma", "distribution_params": [dist_params["mean"], dist_params["arg1"], 0, 1]}
    if dist_name == "TRIANGULAR":
        return {"distribution_name": "triang", "distribution_params": [dist_params["mean"], dist_params["arg1"],
                                                                       dist_params["arg2"], 0, 1]}
    return None


def _format_date(date_str):
    date_splt = date_str.split("+")
    if len(date_splt) == 2 and date_splt[1] == "00:00":
        return date_splt[0]
    return date_str


def simulate(settings: Configuration, process_stats: pd.DataFrame, log_data, evaluate_fn: Callable = None):
    if evaluate_fn is None:
        evaluate_fn = evaluate_logs

    if isinstance(settings, dict):
        settings = Configuration(**settings)

    reps = settings.repetitions
    cpu_count = multiprocessing.cpu_count()
    w_count = reps if reps <= cpu_count else cpu_count
    pool = multiprocessing.Pool(processes=w_count)

    # Simulate
    args = [(settings, rep) for rep in range(reps)]
    p = pool.map_async(execute_simulator, args)
    pbar_async(p, 'simulating:', reps)

    # Read simulated logs
    p = pool.map_async(read_stats, args)
    pbar_async(p, 'reading simulated logs:', reps)

    # Evaluate
    args = [(settings, process_stats, log) for log in p.get()]
    if len(log_data.caseid.unique()) > 1000:
        pool.close()
        results = [evaluate_fn(arg) for arg in tqdm(args, 'evaluating results:')]
        sim_values = list(itertools.chain(*results))
    else:
        p = pool.map_async(evaluate_fn, args)
        pbar_async(p, 'evaluating results:', reps)
        pool.close()
        sim_values = list(itertools.chain(*p.get()))
    return sim_values


def execute_simulator(args: Tuple):
    settings: Configuration
    repetitions: int
    settings, repetitions = args
    args = ['java', '-jar', settings.bimp_path.absolute().__str__(),
            (settings.output / (settings.project_name + '.bpmn')).__str__(),
            '-csv',
            (settings.output / 'sim_data' / (settings.project_name + '_' + str(repetitions + 1) + '.csv')).__str__()]
    # NOTE: the call generates a CSV event log from a model
    # NOTE: might fail silently, because stderr or stdout aren't checked
    execute_shell_cmd(args)
