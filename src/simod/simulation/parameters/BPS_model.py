import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

from pix_framework.discovery.case_arrival import CaseArrivalModel
from pix_framework.discovery.gateway_probabilities import GatewayProbabilities
from pix_framework.discovery.resource_model import ResourceModel

from simod.batching.types import BatchingRule
from simod.bpm.graph import get_activities_ids_by_name
from simod.bpm.reader_writer import BPMNReaderWriter
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

    def to_dict(self) -> dict:
        attributes = {}

        if self.process_model is not None:
            attributes |= {PROCESS_MODEL_KEY: str(self.process_model)}

        if self.gateway_probabilities is not None:
            attributes |= {
                GATEWAY_PROBABILITIES_KEY: [
                    gateway_probability.to_dict() for gateway_probability in self.gateway_probabilities
                ]
            }

        if self.case_arrival_model is not None:
            attributes |= self.case_arrival_model.to_dict()

        if self.resource_model is not None:
            attributes |= self.resource_model.to_dict()

        if self.extraneous_delays is not None:
            attributes |= {
                EXTRANEOUS_DELAYS_KEY: [extraneous_delay.to_dict() for extraneous_delay in self.extraneous_delays]
            }

        if self.case_attributes is not None:
            attributes |= {
                CASE_ATTRIBUTES_KEY: [case_attribute.to_prosimos() for case_attribute in self.case_attributes]
            }

        if self.prioritization_rules is not None:
            attributes |= {
                PRIORITIZATION_RULES_KEY: list(map(lambda x: x.to_prosimos(), self.prioritization_rules))
            }

        if self.batching_rules is not None:
            attributes |= {BATCHING_RULES_KEY: list(map(lambda x: x.to_prosimos(), self.batching_rules))}

        return attributes

    def deep_copy(self) -> "BPSModel":
        return copy.deepcopy(self)

    @staticmethod
    def from_dict(bps_model: dict) -> "BPSModel":
        process_model_path = Path(bps_model[PROCESS_MODEL_KEY]) if PROCESS_MODEL_KEY in bps_model else None

        gateway_probabilities = (
            [
                GatewayProbabilities.from_dict(gateway_probability)
                for gateway_probability in bps_model[GATEWAY_PROBABILITIES_KEY]
            ]
            if GATEWAY_PROBABILITIES_KEY in bps_model
            else None
        )

        case_arrival_model = (
            CaseArrivalModel.from_dict(bps_model)
            if (CASE_ARRIVAL_DISTRIBUTION_KEY in bps_model and CASE_ARRIVAL_CALENDAR_KEY in bps_model)
            else None
        )

        resource_model = (
            ResourceModel.from_dict(bps_model)
            if (
                    RESOURCE_PROFILES_KEY in bps_model
                    and RESOURCE_CALENDARS_KEY in bps_model
                    and RESOURCE_ACTIVITY_PERFORMANCE_KEY in bps_model
            )
            else None
        )

        extraneous_delays = (
            [
                ExtraneousDelay.from_dict(extraneous_delay)
                for extraneous_delay in bps_model[EXTRANEOUS_DELAYS_KEY]
            ]
            if EXTRANEOUS_DELAYS_KEY in bps_model
            else None
        )

        case_attributes = (
            [
                CaseAttribute.from_dict(case_attribute)
                for case_attribute in bps_model[CASE_ATTRIBUTES_KEY]
            ]
            if CASE_ATTRIBUTES_KEY in bps_model
            else None
        )

        prioritization_rules = (
            [
                PrioritizationRule.from_prosimos(prioritization_rule)
                for prioritization_rule in bps_model[PRIORITIZATION_RULES_KEY]
            ]
            if PRIORITIZATION_RULES_KEY in bps_model
            else None
        )

        bpmn_reader = BPMNReaderWriter(process_model_path)
        activities_names_by_id = bpmn_reader.get_activities_ids_to_names_mapping()
        batching_rules = (
            [
                BatchingRule.from_prosimos(batching_rule, activities_names_by_id)
                for batching_rule in bps_model["batch_processing"]
            ]
            if "batch_processing" in bps_model
            else None
        )

        return BPSModel(
            process_model=process_model_path,
            gateway_probabilities=gateway_probabilities,
            case_arrival_model=case_arrival_model,
            resource_model=resource_model,
            extraneous_delays=extraneous_delays,
            case_attributes=case_attributes,
            prioritization_rules=prioritization_rules,
            batching_rules=batching_rules,
        )

    def replace_activity_names_with_ids(self):
        """
        Updates activity labels with activity IDs from the current process model.

        In BPSModel, the activities are referenced by their name, Prosimos uses IDs instead from the BPMN model.
        """
        activity_label_to_id = get_activities_ids_by_name(BPMNReaderWriter(self.process_model).as_graph())
        for resource_profile in self.resource_model.resource_profiles:
            for resource in resource_profile.resources:
                resource.assigned_tasks = [
                    activity_label_to_id[activity_label] for activity_label in resource.assigned_tasks
                ]
        for activity_resource_distributions in self.resource_model.activity_resource_distributions:
            activity_resource_distributions.activity_id = activity_label_to_id[
                activity_resource_distributions.activity_id
            ]

    def to_json(self, output_dir: Path, process_name: str) -> Path:
        json_parameters_path = get_simulation_parameters_path(output_dir, process_name)

        with json_parameters_path.open("w") as f:
            json.dump(self.to_dict(), f)

        return json_parameters_path
