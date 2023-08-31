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
from simod.case_attributes.types import CaseAttribute
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
PRIORITIZATION_RULES_KEY = "prioritisation_rules"
BATCHING_RULES_KEY = "batch_processing"


@dataclass
class BPSModel:
    """
    BPS model class containing all the components to simulate a business process model.
    """

    process_model: Optional[Path] = None  # A path to the model for now, in future the loaded BPMN model
    gateway_probabilities: Optional[List[GatewayProbabilities]] = None
    case_arrival_model: Optional[CaseArrivalModel] = None
    resource_model: Optional[ResourceModel] = None
    extraneous_delays: Optional[List[ExtraneousDelay]] = None
    case_attributes: Optional[List[CaseAttribute]] = None
    prioritization_rules: Optional[List[PrioritizationRule]] = None
    batching_rules: Optional[List[BatchingRule]] = None

    def to_prosimos_format(self, granule_size: int = 15) -> dict:
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
        if self.prioritization_rules is not None:
            attributes[PRIORITIZATION_RULES_KEY] = [
                priority_rule.to_prosimos() for priority_rule in self.prioritization_rules
            ]
        if self.batching_rules is not None:
            attributes[BATCHING_RULES_KEY] = [
                batching_rule.to_prosimos(activity_label_to_id) for batching_rule in self.batching_rules
            ]
        if isinstance(self.resource_model.resource_calendars[0], FuzzyResourceCalendar):
            attributes["model_type"] = "FUZZY"
        else:
            attributes["model_type"] = "CRISP"
        attributes["granule_size"] = {"value": granule_size, "time_unit": "MINUTES"}

        return attributes

    def deep_copy(self) -> "BPSModel":
        return copy.deepcopy(self)

    def replace_activity_names_with_ids(self):
        """
        Updates activity labels with activity IDs from the current (BPMN) process model.

        In BPSModel, the activities are referenced by their name, Prosimos uses IDs instead from the BPMN model.
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

    def to_json(self, output_dir: Path, process_name: str, granule_size: int = 15) -> Path:
        json_parameters_path = get_simulation_parameters_path(output_dir, process_name)

        with json_parameters_path.open("w") as f:
            json.dump(self.to_prosimos_format(granule_size=granule_size), f)

        return json_parameters_path
