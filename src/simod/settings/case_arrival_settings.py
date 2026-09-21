from enum import Enum
from typing import List, Optional, Tuple, Union

from pix_framework.discovery.gateway_probabilities import GatewayProbabilitiesDiscoveryMethod
from pydantic import BaseModel

from .common_settings import Metric
from ..utilities import parse_single_value_or_interval


class CaseArrivalSettings(BaseModel):
    """
    Case arrival model configuration parameters.

    This class defines the ranges of the configurable parameters for optimizing the case arrival
    model of a discovered BPS model, including the metric to assess the quality of each iteration,
    iteration settings, and the threshold parameter to discard outlier observations. In each
    iteration of the optimization process, the parameters are sampled from these values or ranges.

    Attributes
    ----------
    optimization_metric : :class:`~simod.settings.common_settings.Metric`
        The metric used to evaluate process model quality at each iteration of the optimization process (i.e.,
        loss function).
    num_iterations : int
        The number of optimization iterations to perform.
    num_evaluations_per_iteration : int
        The number of replications for the evaluations of each iteration.
    outlier_threshold : Union[float, Tuple[float, float]], optional
        Fixed value or range for the threshold used to filter outliers.
    use_observed_arrival_distribution : bool
        Boolean indicating whether to use the distribution of observed case arrival times (true), or to discover a
        probability distribution function to model them (false).
    """

    optimization_metric: Metric = Metric.ARRIVAL_EMD
    num_iterations: int = 5
    num_evaluations_per_iteration: int = 3
    outlier_threshold: Optional[Union[float, Tuple[float, float]]] = (5.0, 50.0)
    use_observed_arrival_distribution: bool = False

    @staticmethod
    def one_shot() -> "CaseArrivalSettings":
        """
        Instantiates the case arrival model configuration for the one-shot mode (i.e., no optimization, one single
        iteration).

        Returns
        -------
        :class:`CaseArrivalSettings`
            Instance of the case arrival model configuration for the one-shot mode.
        """
        return CaseArrivalSettings(
            optimization_metric=Metric.THREE_GRAM_DISTANCE,
            num_iterations=1,
            num_evaluations_per_iteration=1,
            outlier_threshold=20.0,
            use_observed_arrival_distribution=False,
        )

    @staticmethod
    def from_dict(config: dict) -> "CaseArrivalSettings":
        """
        Instantiates the case arrival model configuration from a dictionary.

        Parameters
        ----------
        config : dict
            Dictionary with the configuration values for the case arrival model parameters.

        Returns
        -------
        :class:`CaseArrivalSettings`
            Instance of the case arrival model configuration for the specified dictionary values.
        """
        optimization_metric = Metric.from_str(config.get("optimization_metric", "arrival_event_distribution"))
        num_iterations = config.get("num_iterations", 5)
        num_evaluations_per_iteration = config.get("num_evaluations_per_iteration", 3)

        outlier_threshold = parse_single_value_or_interval(config.get("outlier_threshold", (5.0, 50.0)))
        use_observed_arrival_distribution = config.get("use_observed_arrival_distribution", False)

        return CaseArrivalSettings(
            optimization_metric=optimization_metric,
            num_iterations=num_iterations,
            num_evaluations_per_iteration=num_evaluations_per_iteration,
            outlier_threshold=outlier_threshold,
            use_observed_arrival_distribution=use_observed_arrival_distribution,
        )

    def to_dict(self) -> dict:
        """
        Translate the control-flow model configuration stored in this instance into a dictionary.

        Returns
        -------
        dict
            Python dictionary storing this configuration.
        """
        dictionary = {
            "optimization_metric": self.optimization_metric.value,
            "num_iterations": self.num_iterations,
            "num_evaluations_per_iteration": self.num_evaluations_per_iteration,
            "outlier_threshold": self.outlier_threshold,
            "use_observed_arrival_distribution": self.use_observed_arrival_distribution
        }

        return dictionary
