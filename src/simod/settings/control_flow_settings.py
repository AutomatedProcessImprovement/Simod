from enum import Enum
from typing import List, Optional, Tuple, Union

from pix_framework.discovery.gateway_probabilities import GatewayProbabilitiesDiscoveryMethod
from pydantic import BaseModel

from .common_settings import Metric
from ..utilities import parse_single_value_or_interval


class ProcessModelDiscoveryAlgorithm(str, Enum):
    """
    Enumeration of process model discovery algorithms.

    This enum defines the available algorithms for discovering process models from event logs.

    Attributes
    ----------
    SPLIT_MINER_V1 : str
        Represents the first version of the Split Miner algorithm (`"sm1"`).
    SPLIT_MINER_V2 : str
        Represents the second version of the Split Miner algorithm (`"sm2"`).
    """

    SPLIT_MINER_V1 = "sm1"
    SPLIT_MINER_V2 = "sm2"

    @classmethod
    def from_str(cls, value: str) -> "ProcessModelDiscoveryAlgorithm":
        """
        Converts a string representation of a process model discovery algorithm
        into the corresponding :class:`ProcessModelDiscoveryAlgorithm` instance.

        This method allows flexible input formats for each algorithm, supporting
        multiple variations of their names.

        Parameters
        ----------
        value : str
            A string representing a process model discovery algorithm.

        Returns
        -------
        :class:`ProcessModelDiscoveryAlgorithm`
            The corresponding enum instance for the given algorithm name.

        Raises
        ------
        ValueError
            If the provided string does not match any known algorithm.
        """
        if value.lower() in [
            "sm2",
            "splitminer2",
            "split miner 2",
            "split_miner_2",
            "split-miner-2",
            "split_miner_v2",
            "split-miner-v2",
            "splitminer-v2",
            "split miner v2",
        ]:
            return cls.SPLIT_MINER_V2
        elif value.lower() in [
            "sm1",
            "splitminer1",
            "split miner 1",
            "split_miner_1",
            "split-miner-1",
            "split_miner_v1",
            "split-miner-v1",
            "splitminer-v1",
            "split miner v1",
        ]:
            return cls.SPLIT_MINER_V1
        else:
            raise ValueError(f"Unknown process model discovery algorithm: {value}")

    def __str__(self):
        if self == ProcessModelDiscoveryAlgorithm.SPLIT_MINER_V1:
            return "Split Miner v1"
        elif self == ProcessModelDiscoveryAlgorithm.SPLIT_MINER_V2:
            return "Split Miner v2"
        return f"Unknown ProcessModelDiscoveryAlgorithm {str(self)}"


class ControlFlowSettings(BaseModel):
    """
    Control-flow model configuration parameters.

    This class defines the ranges of the configurable parameters for optimizing the control-flow
    structure of a discovered process model, including metric selection, iteration settings,
    and various discovery algorithm parameters. In each iteration of the optimization process, the
    parameters are sampled from these values or ranges.

    Attributes
    ----------
    optimization_metric : :class:`~simod.settings.common_settings.Metric`
        The metric used to evaluate process model quality at each iteration of the optimization process (i.e.,
        loss function).
    num_iterations : int
        The number of optimization iterations to perform.
    num_evaluations_per_iteration : int
        The number of replications for the evaluations of each iteration.
    gateway_probabilities : Union[:class:`GatewayProbabilitiesDiscoveryMethod`, List[:class:`GatewayProbabilitiesDiscoveryMethod`]]
        Fixed method or list of methods to use in each iteration to discover gateway probabilities.
    mining_algorithm : :class:`ProcessModelDiscoveryAlgorithm`, optional
        The process model discovery algorithm to use.
    epsilon : Union[float, Tuple[float, float]], optional
        Fixed number or range for the number of concurrent relations between events to be captured in the discovery
        algorithm (between 0.0 and 1.0).
    eta : Union[float, Tuple[float, float]], optional
        Fixed number or range for the threshold for filtering the incoming and outgoing edges in the discovery
        algorithm (between 0.0 and 1.0).
    replace_or_joins : Union[bool, List[bool]], optional
        Fixed value or list for whether to replace non-trivial OR joins.
    prioritize_parallelism : Union[bool, List[bool]], optional
        Fixed value or list for whether to prioritize parallelism over loops.
    discover_branch_rules : bool, optional
        Whether to discover branch rules for gateways.
    f_score : Union[float, Tuple[float, float]], optional
        Fixed value or range for the minimum f-score value to consider the discovered data-aware branching rules.
    """

    optimization_metric: Metric = Metric.THREE_GRAM_DISTANCE
    num_iterations: int = 10
    num_evaluations_per_iteration: int = 3
    gateway_probabilities: Union[
        GatewayProbabilitiesDiscoveryMethod, List[GatewayProbabilitiesDiscoveryMethod]
    ] = GatewayProbabilitiesDiscoveryMethod.DISCOVERY
    mining_algorithm: Optional[ProcessModelDiscoveryAlgorithm] = ProcessModelDiscoveryAlgorithm.SPLIT_MINER_V1
    epsilon: Optional[Union[float, Tuple[float, float]]] = (0.0, 1.0)  # parallelism threshold (epsilon)
    eta: Optional[Union[float, Tuple[float, float]]] = (0.0, 1.0)  # percentile for frequency threshold (eta)
    discover_branch_rules: Optional[bool] = False
    f_score: Optional[Union[float, Tuple[float, float]]] = 0.7  # quality gateway for branch rules (f_score)
    replace_or_joins: Optional[Union[bool, List[bool]]] = False  # should replace non-trivial OR joins
    prioritize_parallelism: Optional[Union[bool, List[bool]]] = False  # should prioritize parallelism on loops

    @staticmethod
    def one_shot() -> "ControlFlowSettings":
        """
        Instantiates the control-flow model configuration for the one-shot mode (i.e., no optimization, one single
        iteration).

        Returns
        -------
        :class:`ControlFlowSettings`
            Instance of the control-flow model configuration for the one-shot mode.
        """
        return ControlFlowSettings(
            optimization_metric=Metric.THREE_GRAM_DISTANCE,
            num_iterations=1,
            num_evaluations_per_iteration=1,
            gateway_probabilities=GatewayProbabilitiesDiscoveryMethod.DISCOVERY,
            mining_algorithm=ProcessModelDiscoveryAlgorithm.SPLIT_MINER_V1,
            epsilon=0.3,
            eta=0.5,
            replace_or_joins=False,
            prioritize_parallelism=False,
        )

    @staticmethod
    def from_dict(config: dict) -> "ControlFlowSettings":
        """
        Instantiates the control-flow model configuration from a dictionary.

        Parameters
        ----------
        config : dict
            Dictionary with the configuration values for the control-flow model parameters.

        Returns
        -------
        :class:`ControlFlowSettings`
            Instance of the control-flow model configuration for the specified dictionary values.
        """
        optimization_metric = Metric.from_str(config.get("optimization_metric", "n_gram_distance"))
        num_iterations = config.get("num_iterations", 10)
        num_evaluations_per_iteration = config.get("num_evaluations_per_iteration", 3)
        gateway_probabilities = GatewayProbabilitiesDiscoveryMethod.from_str(
            config.get("gateway_probabilities", "discovery")
        )

        mining_algorithm = ProcessModelDiscoveryAlgorithm.from_str(config.get("mining_algorithm", "sm1"))
        epsilon, eta, replace_or_joins, prioritize_parallelism = None, None, None, None
        if mining_algorithm in [ProcessModelDiscoveryAlgorithm.SPLIT_MINER_V1]:
            eta = parse_single_value_or_interval(config.get("eta", (0.0, 1.0)))
            epsilon = parse_single_value_or_interval(config.get("epsilon", (0.0, 1.0)))
            replace_or_joins = config.get("replace_or_joins", False)
            prioritize_parallelism = config.get("prioritize_parallelism", False)
        elif mining_algorithm is ProcessModelDiscoveryAlgorithm.SPLIT_MINER_V2:
            epsilon = parse_single_value_or_interval(config.get("epsilon", (0.0, 1.0)))
        else:
            raise ValueError(f"Unknown process model discovery algorithm: {mining_algorithm}")

        discover_branch_rules = config.get("discover_branch_rules", False)
        f_score = None
        if discover_branch_rules:
            f_score = parse_single_value_or_interval(config.get("f_score", (0.0, 1.0)))

        return ControlFlowSettings(
            optimization_metric=optimization_metric,
            num_iterations=num_iterations,
            num_evaluations_per_iteration=num_evaluations_per_iteration,
            gateway_probabilities=gateway_probabilities,
            mining_algorithm=mining_algorithm,
            epsilon=epsilon,
            eta=eta,
            replace_or_joins=replace_or_joins,
            prioritize_parallelism=prioritize_parallelism,
            discover_branch_rules=discover_branch_rules,
            f_score=f_score
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
        }

        if isinstance(self.gateway_probabilities, GatewayProbabilitiesDiscoveryMethod):
            dictionary["gateway_probabilities"] = self.gateway_probabilities.value
        else:
            dictionary["gateway_probabilities"] = [method.value for method in self.gateway_probabilities]

        if self.mining_algorithm is not None:
            dictionary["mining_algorithm"] = self.mining_algorithm.value
            if self.mining_algorithm == ProcessModelDiscoveryAlgorithm.SPLIT_MINER_V2:
                dictionary["epsilon"] = self.epsilon
            elif self.mining_algorithm == ProcessModelDiscoveryAlgorithm.SPLIT_MINER_V1:
                dictionary["epsilon"] = self.epsilon
                dictionary["eta"] = self.eta
                dictionary["replace_or_joins"] = self.replace_or_joins
                dictionary["prioritize_parallelism"] = self.prioritize_parallelism

        if self.discover_branch_rules and self.f_score is not None:
            dictionary["f_score"] = self.f_score

        return dictionary
