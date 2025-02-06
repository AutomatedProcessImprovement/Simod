import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from pix_framework.discovery.case_arrival import CaseArrivalModel
from pix_framework.discovery.gateway_probabilities import GatewayProbabilities
from pix_framework.discovery.resource_calendar_and_performance.fuzzy.resource_calendar import FuzzyResourceCalendar
from pix_framework.discovery.resource_model import ResourceModel

from pix_framework.io.bpmn import get_activities_ids_by_name_from_bpmn

from simod.batching.types import BatchingRule
from simod.branch_rules.types import BranchRules
from simod.data_attributes.types import CaseAttribute, GlobalAttribute, EventAttribute
from simod.extraneous_delays.types import ExtraneousDelay
from simod.prioritization.types import PrioritizationRule
from simod.utilities import get_simulation_parameters_path

# Keys for serialization
PROCESS_MODEL_KEY = "process_model"
GATEWAY_PROBABILITIES_KEY = "gateway_branching_probabilities"
CASE_ARRIVAL_DISTRIBUTION_KEY = "arrival_time_distribution"
CASE_ARRIVAL_CALENDAR_KEY = "arrival_time_calendar"
RESOURCE_PROFILES_KEY = "resource_profiles"
RESOURCE_CALENDARS_KEY = "resource_calendars"
RESOURCE_ACTIVITY_PERFORMANCE_KEY = "task_resource_distribution"
EXTRANEOUS_DELAYS_KEY = "event_distribution"
CASE_ATTRIBUTES_KEY = "case_attributes"
GLOBAL_ATTRIBUTES_KEY = "global_attributes"
EVENT_ATTRIBUTES_KEY = "event_attributes"
PRIORITIZATION_RULES_KEY = "prioritisation_rules"
BATCHING_RULES_KEY = "batch_processing"
BRANCH_RULES_KEY = "branch_rules"


@dataclass
class BPSModel:
    """
    Represents a Business Process Simulation (BPS) model containing all necessary components
    to simulate a business process.

    This class manages various elements such as the BPMN process model, resource configurations,
    extraneous delays, case attributes, and prioritization/batching rules. It provides methods
    to convert the model into a format compatible with Prosimos and handle activity ID mappings.

    Attributes
    ----------
    process_model : :class:`pathlib.Path`, optional
        Path to the BPMN process model file.
    gateway_probabilities : List[:class:`GatewayProbabilities`], optional
        Probabilities for gateway-based process routing.
    case_arrival_model : :class:`CaseArrivalModel`, optional
        Model for the arrival of new cases in the simulation.
    resource_model : :class:`ResourceModel`, optional
        Model for the resources involved in the process, their working schedules, etc.
    extraneous_delays : List[:class:`~simod.extraneous_delays.types.ExtraneousDelay`], optional
        A list of delays representing extraneous waiting times before/after activities.
    case_attributes : List[:class:`CaseAttribute`], optional
        Case-level attributes and their update rules.
    global_attributes : List[:class:`GlobalAttribute`], optional
        Global attributes and their update rules.
    event_attributes : List[:class:`EventAttribute`], optional
        Event-level attributes and their update rules.
    prioritization_rules : List[:class:`PrioritizationRule`], optional
        A set of case prioritization rules for process execution.
    batching_rules : List[:class:`BatchingRule`], optional
        Rules defining how activities are batched together.
    branch_rules : List[:class:`BranchRules`], optional
        Branching rules defining conditional flow behavior in decision points.
    calendar_granularity : int, optional
        Granularity of the resource calendar, expressed in minutes.

    Notes
    -----
    - `to_prosimos_format` transforms the model into a dictionary format used by Prosimos.
    - `replace_activity_names_with_ids` modifies activity references to use BPMN IDs instead of names.
    """

    process_model: Optional[Path] = None  # A path to the model for now, in future the loaded BPMN model
    gateway_probabilities: Optional[List[GatewayProbabilities]] = None
    case_arrival_model: Optional[CaseArrivalModel] = None
    resource_model: Optional[ResourceModel] = None
    extraneous_delays: Optional[List[ExtraneousDelay]] = None
    case_attributes: Optional[List[CaseAttribute]] = None
    global_attributes: Optional[List[GlobalAttribute]] = None
    event_attributes: Optional[List[EventAttribute]] = None
    prioritization_rules: Optional[List[PrioritizationRule]] = None
    batching_rules: Optional[List[BatchingRule]] = None
    branch_rules: Optional[List[BranchRules]] = None
    calendar_granularity: Optional[int] = None

    def to_prosimos_format(self) -> dict:
        """
        Converts the BPS model into a dictionary format compatible with the Prosimos simulation engine.

        This method extracts all relevant process simulation attributes, including resource models,
        delays, prioritization rules, and activity mappings, and structures them in a format
        understood by Prosimos.

        Returns
        -------
        dict
            A dictionary representation of the BPS model, ready for simulation in Prosimos.

        Notes
        -----
        - If the resource model contains a fuzzy calendar, the model type is set to "FUZZY";
          otherwise, it defaults to "CRISP".
        - The function ensures activity labels are properly linked to their respective BPMN IDs.
        """

        # Get map activity label -> node ID
        activity_label_to_id = get_activities_ids_by_name_from_bpmn(self.process_model)

        attributes = {}
        if self.process_model is not None:
            attributes[PROCESS_MODEL_KEY] = str(self.process_model)
        if self.gateway_probabilities is not None:
            attributes[GATEWAY_PROBABILITIES_KEY] = [
                gateway_probability.to_dict() for gateway_probability in self.gateway_probabilities
            ]
        if self.case_arrival_model is not None:
            attributes |= self.case_arrival_model.to_dict()
        if self.resource_model is not None:
            attributes |= self.resource_model.to_dict()
        if self.extraneous_delays is not None:
            attributes[EXTRANEOUS_DELAYS_KEY] = [
                extraneous_delay.to_dict() for extraneous_delay in self.extraneous_delays
            ]
        if self.case_attributes is not None:
            attributes[CASE_ATTRIBUTES_KEY] = [case_attribute.to_prosimos() for case_attribute in self.case_attributes]
        if self.global_attributes is not None:
            attributes[GLOBAL_ATTRIBUTES_KEY] = [global_attribute.to_prosimos() for global_attribute in self.global_attributes]
        if self.event_attributes is not None:
            attributes[EVENT_ATTRIBUTES_KEY] = [event_attribute.to_prosimos() for event_attribute in self.event_attributes]
        if self.case_attributes is not None and self.prioritization_rules is not None:
            attributes[PRIORITIZATION_RULES_KEY] = [
                priority_rule.to_prosimos() for priority_rule in self.prioritization_rules
            ]
        if self.batching_rules is not None:
            attributes[BATCHING_RULES_KEY] = [
                batching_rule.to_prosimos(activity_label_to_id) for batching_rule in self.batching_rules
            ]
        if self.branch_rules is not None:
            attributes[BRANCH_RULES_KEY] = [branch_rules.to_dict() for branch_rules in self.branch_rules]
        if isinstance(self.resource_model.resource_calendars[0], FuzzyResourceCalendar):
            attributes["model_type"] = "FUZZY"
        else:
            attributes["model_type"] = "CRISP"
        attributes["granule_size"] = {"value": self.calendar_granularity, "time_unit": "MINUTES"}

        return attributes

    def deep_copy(self) -> "BPSModel":
        """
        Creates a deep copy of the current BPSModel instance.

        This ensures that modifying the copied instance does not affect the original.

        Returns
        -------
        :class:`BPSModel`
            A new, independent copy of the current BPSModel instance.

        Notes
        -----
        This method uses Python's `copy.deepcopy()` to create a full recursive copy of the model.
        """
        return copy.deepcopy(self)

    def replace_activity_names_with_ids(self):
        """
        Replaces activity names with their corresponding IDs from the BPMN process model.

        Prosimos requires activity references to be identified by their BPMN node IDs instead of
        activity labels. This method updates:

        - Resource associations in the resource profiles.
        - Activity-resource distributions.
        - Event attributes referencing activity names.

        Raises
        ------
        KeyError
            If an activity name does not exist in the BPMN model.

        Notes
        -----
        - This method modifies the model in place.
        - It ensures compatibility with Prosimos by aligning activity references with BPMN IDs.
        """
        # Get map activity label -> node ID
        activity_label_to_id = get_activities_ids_by_name_from_bpmn(self.process_model)
        # Update activity labels in resource profiles
        if self.resource_model.resource_profiles is not None:
            for resource_profile in self.resource_model.resource_profiles:
                for resource in resource_profile.resources:
                    resource.assigned_tasks = [
                        activity_label_to_id[activity_label] for activity_label in resource.assigned_tasks
                    ]
        # Update activity labels in activity-resource performance
        if self.resource_model.activity_resource_distributions is not None:
            for activity_resource_distributions in self.resource_model.activity_resource_distributions:
                activity_resource_distributions.activity_id = activity_label_to_id[
                    activity_resource_distributions.activity_id
                ]

        # Update activity label in event attributes
        if self.event_attributes is not None:
            for event_attribute in self.event_attributes:
                event_attribute.event_id = activity_label_to_id[event_attribute.event_id]

    def to_json(self, output_dir: Path, process_name: str) -> Path:
        """
        Saves the BPS model in a Prosimos-compatible JSON format.

        This method generates a structured JSON file containing all necessary simulation parameters,
        ensuring that the model can be directly used by the Prosimos engine.

        Parameters
        ----------
        output_dir : :class:`pathlib.Path`
            The directory where the JSON file should be saved.
        process_name : str
            The name of the process, used for naming the output file.

        Returns
        -------
        :class:`pathlib.Path`
            The full path to the generated JSON file.

        Notes
        -----
        - The JSON file is created in `output_dir` with a filename based on `process_name`.
        - Uses `json.dump()` to serialize the model into a structured format.
        - Ensures all attributes are converted into a valid Prosimos format before writing.
        """
        json_parameters_path = get_simulation_parameters_path(output_dir, process_name)

        with json_parameters_path.open("w") as f:
            json.dump(self.to_prosimos_format(), f)

        return json_parameters_path
