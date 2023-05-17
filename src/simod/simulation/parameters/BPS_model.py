from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

from simod.simulation.parameters.case_arrival_model import CaseArrivalModel
from simod.simulation.parameters.gateway_probabilities import GatewayProbabilities
from simod.simulation.parameters.resource_model import ResourceModel


@dataclass
class BPSModel:
    """
    BPS model class containing all the components to simulate a business process model.
    """

    process_model: Optional[Path] = None  # A path to the model for now, in future the loaded BPMN model
    gateway_probabilities: Optional[List[GatewayProbabilities]] = None
    case_arrival_model: Optional[CaseArrivalModel] = None
    resource_model: Optional[ResourceModel] = None

    # extraneous_delays: Optional[List[ExtraneousDelay]]
    # case_attributes: Optional[List[CaseAttribute]]
    # prioritization_rules: Optional[List[PrioritizationRule]]
    # batching_rules: Optional[List[BatchingRule]]

    def to_dict(self) -> dict:
        dictionary = {}
        # Add model path if present
        if self.process_model is not None:
            dictionary |= {'process_model': str(self.process_model)}
        # Add gateway probabilities if present
        if self.gateway_probabilities is not None:
            dictionary |= {
                'gateway_branching_probabilities': [
                    gateway_probability.to_dict()
                    for gateway_probability in self.gateway_probabilities
                ]
            }
        # Add case arrival model if present
        if self.case_arrival_model is not None:
            dictionary |= self.case_arrival_model.to_dict()
        # Add resource model if present
        if self.resource_model is not None:
            dictionary |= self.resource_model.to_dict()
        # Return dictionary with current parameters
        return dictionary

    def deep_copy(self) -> 'BPSModel':
        return BPSModel.from_dict(self.to_dict())

    @staticmethod
    def from_dict(bps_model: dict) -> 'BPSModel':
        return BPSModel(
            process_model=Path(bps_model['process_model']) if 'process_model' in bps_model else None,
            gateway_probabilities=[
                GatewayProbabilities.from_dict(gateway_probability)
                for gateway_probability in bps_model['gateway_branching_probabilities']
            ] if 'gateway_branching_probabilities' in bps_model else None,
            case_arrival_model=CaseArrivalModel.from_dict(bps_model) if (
                    'arrival_time_distribution' in bps_model and
                    'arrival_time_calendar' in bps_model
            ) else None,
            resource_model=ResourceModel.from_dict(bps_model) if (
                    'resource_profiles' in bps_model and
                    'resource_calendars' in bps_model and
                    'task_resource_distribution' in bps_model
            ) else None
        )

# TODO
#  Implement default method to discover a complete BPS model from scratch.
#  It receives the training+validation log and it discovers a first direct
#  attempt, this would be the Prosimos CRISP method, and then SIMOD iterates
#  overriding each parameter.