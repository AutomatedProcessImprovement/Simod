from enum import Enum
from typing import List, Optional, Tuple, Union

from pix_framework.discovery.gateway_probabilities import GatewayProbabilitiesDiscoveryMethod
from pydantic import BaseModel

from .common_settings import Metric
from ..utilities import parse_single_value_or_interval


class ProcessModelDiscoveryAlgorithm(str, Enum):
    SPLIT_MINER_V1 = "sm1"
    SPLIT_MINER_V2 = "sm2"

    @classmethod
    def from_str(cls, value: str) -> "ProcessModelDiscoveryAlgorithm":
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
    Control-flow optimization settings.
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
    replace_or_joins: Optional[Union[bool, List[bool]]] = False  # should replace non-trivial OR joins
    prioritize_parallelism: Optional[Union[bool, List[bool]]] = False  # should prioritize parallelism on loops

    @staticmethod
    def one_shot() -> "ControlFlowSettings":
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
        )

    def to_dict(self) -> dict:
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

        return dictionary
