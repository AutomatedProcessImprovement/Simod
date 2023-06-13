import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

from pix_framework.discovery.case_arrival import CaseArrivalModel
from pix_framework.discovery.gateway_probabilities import GatewayProbabilities
from prosimos.simulation_properties_parser import PRIORITISATION_RULES_SECTION, BATCH_PROCESSING_SECTION

from simod.batching.types import BatchingRule
from simod.bpm.graph import get_activities_ids_by_name
from simod.bpm.reader_writer import BPMNReaderWriter
from simod.prioritization.types import PrioritizationLevel
from simod.simulation.parameters.extraneous_delays import ExtraneousDelay
from simod.simulation.parameters.resource_model import ResourceModel
from simod.utilities import get_simulation_parameters_path


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
    # TODO: do wee need case_attributes in BPS model if they only used once in prioritization discovery?
    # case_attributes: Optional[List[CaseAttribute]] = None
    prioritization_rules: Optional[List[PrioritizationLevel]] = None
    batching_rules: Optional[List[BatchingRule]] = None

    def to_prosimos(self) -> dict:
        attributes = {}

        if self.process_model is not None:
            attributes |= {"process_model": str(self.process_model)}

        if self.gateway_probabilities is not None:
            attributes |= {
                "gateway_branching_probabilities": [
                    gateway_probability.to_dict() for gateway_probability in self.gateway_probabilities
                ]
            }

        if self.case_arrival_model is not None:
            attributes |= self.case_arrival_model.to_dict()

        if self.resource_model is not None:
            attributes |= self.resource_model.to_dict()

        # TODO: extraneous delays?

        # TODO: case attributes?

        if self.prioritization_rules is not None:
            attributes |= {
                PRIORITISATION_RULES_SECTION: list(map(lambda x: x.to_prosimos(), self.prioritization_rules))
            }

        if self.batching_rules is not None:
            attributes |= {BATCH_PROCESSING_SECTION: list(map(lambda x: x.to_prosimos(), self.batching_rules))}

        return attributes

    def deep_copy(self) -> "BPSModel":
        return copy.deepcopy(self)

    @staticmethod
    def from_dict(bps_model: dict) -> "BPSModel":
        # NOTE: this method is not needed if we use copy.deepcopy in self.deep_copy()

        process_model_path = Path(bps_model["process_model"]) if "process_model" in bps_model else None

        gateway_probabilities = (
            [
                GatewayProbabilities.from_dict(gateway_probability)
                for gateway_probability in bps_model["gateway_branching_probabilities"]
            ]
            if "gateway_branching_probabilities" in bps_model
            else None
        )

        case_arrival_model = (
            CaseArrivalModel.from_dict(bps_model)
            if ("arrival_time_distribution" in bps_model and "arrival_time_calendar" in bps_model)
            else None
        )

        resource_model = (
            ResourceModel.from_dict(bps_model)
            if (
                "resource_profiles" in bps_model
                and "resource_calendars" in bps_model
                and "task_resource_distribution" in bps_model
            )
            else None
        )

        # TODO: extraneous delays?

        # TODO: case attributes?

        prioritization_rules = (
            [
                PrioritizationLevel.from_prosimos(prioritization_rule)
                for prioritization_rule in bps_model[PRIORITISATION_RULES_SECTION]
            ]
            if PRIORITISATION_RULES_SECTION in bps_model
            else None
        )

        bpmn_reader = BPMNReaderWriter(process_model_path)
        activities_names_by_id = bpmn_reader.get_activities_ids_to_names_mapping()
        batching_rules = (
            [
                BatchingRule.from_prosimos(batching_rule, activities_names_by_id)
                for batching_rule in bps_model[BATCH_PROCESSING_SECTION]
            ]
            if BATCH_PROCESSING_SECTION in bps_model
            else None
        )

        return BPSModel(
            process_model=process_model_path,
            gateway_probabilities=gateway_probabilities,
            case_arrival_model=case_arrival_model,
            resource_model=resource_model,
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
            json.dump(self.to_prosimos(), f)

        return json_parameters_path


# TODO
#  Implement default method to discover a complete BPS model from scratch.
#  It receives the training+validation log and it discovers a first direct
#  attempt, this would be the Prosimos CRISP method, and then SIMOD iterates
#  overriding each parameter.
