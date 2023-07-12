from dataclasses import dataclass
from enum import Enum
from typing import Union, List, Tuple, Optional

from pix_framework.discovery.gateway_probabilities import GatewayProbabilitiesDiscoveryMethod

from .common_settings import Metric
from ..utilities import parse_single_value_or_interval


class ProcessModelDiscoveryAlgorithm(str, Enum):
    SPLIT_MINER_2 = "sm2"
    SPLIT_MINER_3 = "sm3"

    @classmethod
    def from_str(cls, value: str) -> "ProcessModelDiscoveryAlgorithm":
        if value.lower() in ["sm2", "splitminer2", "split miner 2", "split_miner_2", "split-miner-2"]:
            return cls.SPLIT_MINER_2
        elif value.lower() in ["sm3", "splitminer3", "split miner 3", "split_miner_3", "split-miner-3"]:
            return cls.SPLIT_MINER_3
        else:
            raise ValueError(f"Unknown process model discovery algorithm: {value}")

    def __str__(self):
        if self == ProcessModelDiscoveryAlgorithm.SPLIT_MINER_2:
            return "Split Miner 2"
        elif self == ProcessModelDiscoveryAlgorithm.SPLIT_MINER_3:
            return "Split Miner 3"
        return f"Unknown ProcessModelDiscoveryAlgorithm {str(self)}"


@dataclass
class ControlFlowSettings:
    """
    Control-flow optimization settings.
    """

    optimization_metric: Metric = Metric.THREE_GRAM_DISTANCE
    num_iterations: int = 10
    num_evaluations_per_iteration: int = 3
    gateway_probabilities: Union[
        GatewayProbabilitiesDiscoveryMethod, List[GatewayProbabilitiesDiscoveryMethod]
    ] = GatewayProbabilitiesDiscoveryMethod.DISCOVERY
    mining_algorithm: Optional[ProcessModelDiscoveryAlgorithm] = ProcessModelDiscoveryAlgorithm.SPLIT_MINER_3
    concurrency: Optional[Union[float, Tuple[float, float]]] = None
    epsilon: Optional[Union[float, Tuple[float, float]]] = (0.0, 1.0)  # parallelism threshold (epsilon)
    eta: Optional[Union[float, Tuple[float, float]]] = (0.0, 1.0)  # percentile for frequency threshold (eta)
    replace_or_joins: Optional[Union[bool, List[bool]]] = False  # should replace non-trivial OR joins
    prioritize_parallelism: Optional[Union[bool, List[bool]]] = False  # should prioritize parallelism on loops

    @staticmethod
    def from_dict(config: dict) -> "ControlFlowSettings":
        # Optimization metric
        optimization_metric = Metric.from_str(config.get("optimization_metric", "n_gram_distance"))
        # Number of iterations for the optimization process
        num_iterations = config.get("num_iterations", 10)
        # Num evaluations per iteration
        num_evaluations_per_iteration = config.get("num_evaluations_per_iteration", 3)
        # Gateway probabilities discovery method
        gateway_probabilities = GatewayProbabilitiesDiscoveryMethod.from_str(
            config.get("gateway_probabilities", "discovery")
        )
        # Process model discovery algorithm
        mining_algorithm = ProcessModelDiscoveryAlgorithm.from_str(config.get("mining_algorithm", "sm3"))
        if mining_algorithm is ProcessModelDiscoveryAlgorithm.SPLIT_MINER_2:
            # Split Miner 2, set concurrency threshold
            concurrency = parse_single_value_or_interval(config.get("concurrency", (0.0, 1.0)))
            epsilon, eta, replace_or_joins, prioritize_parallelism = None, None, None, None
        else:
            # Split Miner 3, set epsilon/eta/replace_or_joins/prioritize_parallelism
            concurrency = None
            epsilon = parse_single_value_or_interval(config.get("epsilon", (0.0, 1.0)))
            eta = parse_single_value_or_interval(config.get("eta", (0.0, 1.0)))
            replace_or_joins = config.get("replace_or_joins", False)
            prioritize_parallelism = config.get("prioritize_parallelism", False)
        # Instantiate class
        return ControlFlowSettings(
            optimization_metric=optimization_metric,
            num_iterations=num_iterations,
            num_evaluations_per_iteration=num_evaluations_per_iteration,
            gateway_probabilities=gateway_probabilities,
            mining_algorithm=mining_algorithm,
            concurrency=concurrency,
            epsilon=epsilon,
            eta=eta,
            replace_or_joins=replace_or_joins,
            prioritize_parallelism=prioritize_parallelism,
        )

    def to_dict(self) -> dict:
        # Parse general settings
        dictionary = {
            "optimization_metric": self.optimization_metric.value,
            "num_iterations": self.num_iterations,
            "num_evaluations_per_iteration": self.num_evaluations_per_iteration,
        }
        # Parse gateway probabilities
        if isinstance(self.gateway_probabilities, GatewayProbabilitiesDiscoveryMethod):
            dictionary["gateway_probabilities"] = self.gateway_probabilities.value
        else:
            dictionary["gateway_probabilities"] = [method.value for method in self.gateway_probabilities]
        # Parse discovery algorithm parameters
        if self.mining_algorithm is not None:
            dictionary["mining_algorithm"] = self.mining_algorithm.value
            if self.mining_algorithm is ProcessModelDiscoveryAlgorithm.SPLIT_MINER_2:
                # Split Miner 2, set concurrency threshold
                dictionary["concurrency"] = self.concurrency
            else:
                # Split Miner 3, set epsilon/eta/replace_or_joins/prioritize_parallelism
                dictionary["epsilon"] = self.epsilon
                dictionary["eta"] = self.eta
                dictionary["replace_or_joins"] = self.replace_or_joins
                dictionary["prioritize_parallelism"] = self.prioritize_parallelism
        # Return dictionary
        return dictionary
