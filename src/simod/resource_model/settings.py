from dataclasses import dataclass
from pathlib import Path

from pix_framework.discovery.resource_calendar_and_performance.calendar_discovery_parameters import (
    CalendarDiscoveryParameters,
    CalendarType,
)

from simod.settings.common_settings import Metric
from simod.utilities import nearest_divisor_for_granularity


@dataclass
class HyperoptIterationParams:
    """
    Parameters for a single iteration of the Resource Model optimization process.

    This class defines the necessary parameters for optimizing the resource model of the BPS model.
    It includes the parameter values for the discovery of resource profiles, calendars, etc.

    Attributes
    ----------
    output_dir : :class:`pathlib.Path`
        Directory where all files of the current iteration will be stored.
    process_model_path : :class:`pathlib.Path`
        Path to the BPMN process model used for optimization.
    project_name : str
        Name of the project for file naming purposes.
    optimization_metric : :class:`~simod.settings.common_settings.Metric`
        Metric used to evaluate the quality of the current iteration's candidate.
    calendar_discovery_params : :class:`CalendarDiscoveryParameters`
        Parameters for the resource calendar (i.e., working schedules) discovery.
    discover_prioritization_rules : bool, optional
        Whether to attempt discovering prioritization rules (default: False).
    discover_batching_rules : bool, optional
        Whether to attempt discovering batching rules (default: False).
    """

    # General settings
    output_dir: Path  # Directory where to output all the files of the current iteration
    process_model_path: Path  # Path to BPMN model
    project_name: str  # Name of the project for file naming

    optimization_metric: Metric  # Metric to evaluate the candidate of this iteration
    calendar_discovery_params: CalendarDiscoveryParameters  # Parameters for the calendar discovery
    discover_prioritization_rules: bool = False  # Whether to try to add prioritization or not
    discover_batching_rules: bool = False  # Whether to try to add batching or not

    def to_dict(self) -> dict:
        """
        Converts the parameters of the current iteration into a dictionary format.

        Returns
        -------
        dict
            A dictionary containing the iteration parameters.
        """
        # Save common params
        optimization_parameters = {
            "output_dir": str(self.output_dir),
            "process_model_path": str(self.process_model_path),
            "project_name": str(self.project_name),
            "optimization_metric": str(self.optimization_metric),
            "discover_prioritization_rules": str(self.discover_prioritization_rules),
            "discover_batching_rules": str(self.discover_batching_rules),
        } | self.calendar_discovery_params.to_dict()
        # Return dict
        return optimization_parameters

    @staticmethod
    def from_hyperopt_dict(
        hyperopt_dict: dict,
        optimization_metric: Metric,
        discovery_type: CalendarType,
        output_dir: Path,
        process_model_path: Path,
        project_name: str,
    ) -> "HyperoptIterationParams":
        """Create the params for this run from the hyperopt dictionary returned by the fmin function."""
        # Extract model discovery parameters if needed (by default None)
        granularity = None
        confidence = None
        support = None
        participation = None
        fuzzy_angle = 1.0

        def safe_granularity(granularity: int) -> int:
            if 1440 % granularity != 0:
                return nearest_divisor_for_granularity(granularity)
            return granularity

        if discovery_type in [
            CalendarType.UNDIFFERENTIATED,
            CalendarType.DIFFERENTIATED_BY_RESOURCE,
            CalendarType.DIFFERENTIATED_BY_POOL,
        ]:
            granularity = safe_granularity(hyperopt_dict["granularity"])
            confidence = hyperopt_dict["confidence"]
            support = hyperopt_dict["support"]
            participation = hyperopt_dict["participation"]
        elif discovery_type == CalendarType.DIFFERENTIATED_BY_RESOURCE_FUZZY:
            granularity = safe_granularity(hyperopt_dict["granularity"])
            fuzzy_angle = hyperopt_dict["fuzzy_angle"]

        discover_prioritization_rules = hyperopt_dict.get("discover_prioritization_rules", False)
        discover_batching_rules = hyperopt_dict.get("discover_batching_rules", False)

        return HyperoptIterationParams(
            output_dir=output_dir,
            process_model_path=process_model_path,
            project_name=project_name,
            optimization_metric=optimization_metric,
            calendar_discovery_params=CalendarDiscoveryParameters(
                discovery_type=discovery_type,
                granularity=granularity,
                confidence=confidence,
                support=support,
                participation=participation,
                fuzzy_angle=fuzzy_angle,
            ),
            discover_prioritization_rules=discover_prioritization_rules,
            discover_batching_rules=discover_batching_rules,
        )
