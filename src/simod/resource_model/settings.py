from dataclasses import dataclass
from pathlib import Path

from pix_framework.discovery.resource_calendars import CalendarDiscoveryParams, CalendarType

from simod.settings.common_settings import Metric
from simod.utilities import nearest_divisor_for_granularity


@dataclass
class HyperoptIterationParams:
    """Parameters for a single iteration of the Resource Model optimization process."""

    # General settings
    output_dir: Path  # Directory where to output all the files of the current iteration
    model_path: Path  # Path to BPMN model
    project_name: str  # Name of the project for file naming

    optimization_metric: Metric  # Metric to evaluate the candidate of this iteration
    calendar_discovery_params: CalendarDiscoveryParams  # Parameters for the calendar discovery

    def to_dict(self) -> dict:
        """Returns a dictionary with the parameters for this run."""
        # Save common params
        optimization_parameters = {
            "output_dir": str(self.output_dir),
            "model_path": str(self.model_path),
            "project_name": str(self.project_name),
            "optimization_metric": str(self.optimization_metric),
        } | self.calendar_discovery_params.to_dict()
        # Return dict
        return optimization_parameters

    @staticmethod
    def from_hyperopt_dict(
        hyperopt_dict: dict,
        optimization_metric: Metric,
        discovery_type: CalendarType,
        output_dir: Path,
        model_path: Path,
        project_name: str,
    ) -> "HyperoptIterationParams":
        """Create the params for this run from the hyperopt dictionary returned by the fmin function."""
        # Extract model discovery parameters if needed (by default None)
        granularity = None
        confidence = None
        support = None
        participation = None
        # If the discovery type implies a discovery, parse parameters
        if discovery_type in [
            CalendarType.UNDIFFERENTIATED,
            CalendarType.DIFFERENTIATED_BY_RESOURCE,
            CalendarType.DIFFERENTIATED_BY_POOL,
        ]:
            if 1440 % hyperopt_dict["granularity"] != 0:
                granularity = nearest_divisor_for_granularity(hyperopt_dict["granularity"])
            else:
                granularity = hyperopt_dict["granularity"]
            confidence = hyperopt_dict["confidence"]
            support = hyperopt_dict["support"]
            participation = hyperopt_dict["participation"]
        # Return parameters instance
        return HyperoptIterationParams(
            output_dir=output_dir,
            model_path=model_path,
            project_name=project_name,
            optimization_metric=optimization_metric,
            calendar_discovery_params=CalendarDiscoveryParams(
                discovery_type=discovery_type,
                granularity=granularity,
                confidence=confidence,
                support=support,
                participation=participation,
            ),
        )
