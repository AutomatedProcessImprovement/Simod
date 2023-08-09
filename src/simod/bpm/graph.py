from pathlib import Path

import networkx as nx
from lxml import etree


def get_activities_ids_by_name_from_bpmn(model_path: Path) -> dict:
    """
    Returns activities' IDs accessed by activity name from the model.

    Sample output: { 'Register Order': '1', 'Verify Order': '2' }
    """
    xml = etree.parse(str(model_path))
    root = xml.getroot()
    namespace = {"xmlns": "http://www.omg.org/spec/BPMN/20100524/MODEL"}
    values = {}
    for process in root.findall("xmlns:process", namespace):
        for task in process.findall("xmlns:task", namespace):
            task_id = task.get("id")
            task_name = task.get("name")
            values[task_name] = task_id
    return values


def get_activities_names_from_bpmn(model_path: Path) -> list[str]:
    """
    Returns activities' names from the model.

    Sample output: ['Register Order', 'Verify Order']
    """
    xml = etree.parse(str(model_path))
    root = xml.getroot()
    namespace = {"xmlns": "http://www.omg.org/spec/BPMN/20100524/MODEL"}
    values = []
    for process in root.findall("xmlns:process", namespace):
        for task in process.findall("xmlns:task", namespace):
            task_name = task.get("name")
            values.append(task_name)
    return values


def from_bpmn_reader(bpmn, verbose=True) -> nx.DiGraph:
    """Creates a process graph from a BPMNReader instance."""
    g = _load_process_structure(bpmn, verbose)
    return g


def _find_node_num(g, id):
    resp = list(filter(lambda x: g.nodes[x]["id"] == id, g.nodes))
    if len(resp) > 0:
        resp = resp[0]
    else:
        resp = -1
    return resp


def _create_nodes(g, total_elements, index, array, node_type, node_name, node_id, verbose):
    i = 0
    while i < len(array):
        # sup.print_progress(((index / (total_elements - 1)) * 100), 'Loading of bpmn structure from file ')
        g.add_node(
            index,
            type=node_type,
            name=array[i][node_name],
            id=array[i][node_id],
            executions=0,
            processing_times=list(),
            waiting_times=list(),
            multi_tasking=list(),
            temp_enable=None,
            temp_start=None,
            temp_end=None,
            tsk_act=False,
            gtact=False,
            xor_gtdir=0,
            gt_num_paths=0,
            gt_visited_paths=0,
        )
        index += 1
        i += 1
    return index


def _load_process_structure(bpmn, verbose) -> nx.DiGraph:
    g = nx.DiGraph()
    # Loading data
    start = bpmn.read_start_events()
    tasks = bpmn.read_activities()
    ex_gates = bpmn.read_exclusive_gateways()
    inc_gates = bpmn.read_inclusive_gateways()
    para_gates = bpmn.read_parallel_gateways()
    end = bpmn.read_end_events()
    timer_events = bpmn.read_intermediate_catch_events()
    # total_elements = (len(start) + len(tasks) + len(ex_gates) + len(inc_gates) + len(para_gates) + len(end) + len(timer_events))
    total_elements = len(tasks) + len(ex_gates) + len(inc_gates) + len(para_gates) + len(timer_events)
    # Adding nodes
    # index = create_nodes(g,total_elements,0,start,'start','start_name','start_id')
    index = _create_nodes(
        g,
        total_elements,
        0,
        list(filter(lambda x: x["task_name"].lower() == "start", tasks)),
        "start",
        "task_name",
        "task_id",
        verbose,
    )
    index = _create_nodes(
        g,
        total_elements,
        index,
        list(filter(lambda x: x["task_name"].lower() not in ["start", "end"], tasks)),
        "task",
        "task_name",
        "task_id",
        verbose,
    )
    index = _create_nodes(
        g,
        total_elements,
        index,
        list(filter(lambda x: x["task_name"].lower() == "end", tasks)),
        "end",
        "task_name",
        "task_id",
        verbose,
    )
    index = _create_nodes(
        g,
        total_elements,
        index,
        list(filter(lambda x: x["gate_dir"] == "Diverging", ex_gates)),
        "gate",
        "gate_name",
        "gate_id",
        verbose,
    )
    index = _create_nodes(
        g,
        total_elements,
        index,
        list(filter(lambda x: x["gate_dir"] == "Converging", ex_gates)),
        "gate2",
        "gate_name",
        "gate_id",
        verbose,
    )
    index = _create_nodes(g, total_elements, index, inc_gates, "gate2", "gate_name", "gate_id", verbose)
    index = _create_nodes(g, total_elements, index, para_gates, "gate3", "gate_name", "gate_id", verbose)
    # index = create_nodes(g, total_elements, index, end,'end','end_name','end_id')
    index = _create_nodes(g, total_elements, index, timer_events, "timer", "timer_name", "timer_id", verbose)
    # Add edges
    for edge in bpmn.read_sequence_flows():
        if edge["source"] != start[0]["start_id"] and edge["target"] != end[0]["end_id"]:
            g.add_edge(_find_node_num(g, edge["source"]), _find_node_num(g, edge["target"]))
    # Define #of in_paths for paralell gateways_probabilities
    para_gates = list(filter(lambda x: g.nodes[x]["type"] == "gate3", nx.nodes(g)))
    for x in para_gates:
        g.nodes[x]["gt_num_paths"] = len(list(g.neighbors(x)))
    return g
