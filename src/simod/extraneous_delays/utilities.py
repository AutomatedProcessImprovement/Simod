import uuid
from pathlib import Path
from typing import List

from extraneous_activity_delays.config import TimerPlacement
from lxml import etree
from lxml.etree import ElementTree, QName

from simod.extraneous_delays.types import ExtraneousDelay


def add_timers_to_bpmn_model(
    process_model: Path,
    delays: List[ExtraneousDelay],
    timer_placement: TimerPlacement = TimerPlacement.BEFORE,
):
    """
    Enhance the BPMN model received by adding a timer previous (or after) to each activity denoted by [timers].

    :param process_model:       Path to the process model (in BPMN format) to enhance.
    :param delays:              Dict with the name of each activity as key, and the timer configuration as value.
    :param timer_placement:     Option to consider the placement of the timers either BEFORE (the extraneous delay
                                is considered to be happening previously to an activity instance) or AFTER (the
                                extraneous delay is considered to be happening afterward an activity instance) each
                                activity.
    """
    if len(delays) > 0:
        # Extract process
        parser = etree.XMLParser(remove_blank_text=True)
        bpmn_model = etree.parse(process_model, parser)
        model, process, namespace = _get_basic_bpmn_elements(bpmn_model)
        # Add a timer for each task
        for task in process.findall("task", namespace):
            task_name = task.attrib["name"]
            delay = next((delay for delay in delays if delay.activity_name == task_name), None)
            if delay is not None:
                # The activity has a prepared timer -> add it!
                _add_timer_to_bpmn_model(task, delay.delay_id, process, namespace, timer_placement=timer_placement)
        # Overwrite enhanced BPMN document
        bpmn_model.write(process_model, pretty_print=True)


def _add_timer_to_bpmn_model(
    task,
    timer_id,
    process,
    namespace,
    timer_placement: TimerPlacement,
):
    # The activity has a prepared timer -> add it!
    task_id = task.attrib["id"]
    # Create a timer to add
    timer = etree.Element(QName(namespace[None], "intermediateCatchEvent"), {"id": timer_id}, namespace)
    # Redirect the edge incoming/outgoing to the task, so it points to the timer
    if timer_placement == TimerPlacement.BEFORE:  # Incoming edge
        edge = process.find("sequenceFlow[@targetRef='{}']".format(task_id), namespace)
        edge.attrib["targetRef"] = timer_id
        # Create edge from the timer to the task
        flow_id = "Flow_{}".format(str(uuid.uuid4()))
        flow = etree.Element(
            QName(namespace[None], "sequenceFlow"),
            {"id": flow_id, "sourceRef": timer_id, "targetRef": task_id},
            namespace,
        )
        # Update incoming flow information inside the task
        task_incoming = task.find("incoming", namespace)
        if task_incoming is not None:
            task_incoming.text = flow_id
        # Add incoming element inside timer
        timer_incoming = etree.Element(QName(namespace[None], "incoming"), {}, namespace)
        timer_incoming.text = edge.attrib["id"]
        # Add outgoing element inside timer
        timer_outgoing = etree.Element(QName(namespace[None], "outgoing"), {}, namespace)
        timer_outgoing.text = flow_id
    else:  # Outgoing edge
        edge = process.find("sequenceFlow[@sourceRef='{}']".format(task_id), namespace)
        edge.attrib["sourceRef"] = timer_id
        # Create edge from the task to the timer
        flow_id = "Flow_{}".format(str(uuid.uuid4()))
        flow = etree.Element(
            QName(namespace[None], "sequenceFlow"),
            {"id": flow_id, "sourceRef": task_id, "targetRef": timer_id},
            namespace,
        )
        # Update outgoing flow information inside the task
        task_outgoing = task.find("outgoing", namespace)
        if task_outgoing is not None:
            task_outgoing.text = flow_id
        # Add outgoing element inside timer
        timer_outgoing = etree.Element(QName(namespace[None], "outgoing"), {}, namespace)
        timer_outgoing.text = edge.attrib["id"]
        # Add incoming element inside timer
        timer_incoming = etree.Element(QName(namespace[None], "incoming"), {}, namespace)
        timer_incoming.text = flow_id
    # Append timer incoming and outgoing info
    timer.append(timer_incoming)
    timer.append(timer_outgoing)
    # Timer definition element
    timer_definition_id = "TimerEventDefinition_{}".format(str(uuid.uuid4()))
    timer_definition = etree.Element(
        QName(namespace[None], "timerEventDefinition"),
        {"id": timer_definition_id},
        namespace,
    )
    timer.append(timer_definition)
    # Add the elements to the process
    process.append(timer)
    process.append(flow)


def _get_basic_bpmn_elements(document: ElementTree) -> tuple:
    model = document.getroot()
    namespace = model.nsmap
    process = model.find("process", namespace)
    # Return elements
    return model, process, namespace
