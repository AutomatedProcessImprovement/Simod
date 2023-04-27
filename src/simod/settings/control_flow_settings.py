from enum import Enum
from typing import Union, List, Optional

from pydantic import BaseModel

from .common_settings import Metric


class StructureMiningAlgorithm(str, Enum):
    SPLIT_MINER_2 = 'sm2'
    SPLIT_MINER_3 = 'sm3'

    @classmethod
    def from_str(cls, value: str) -> 'StructureMiningAlgorithm':
        if value.lower() in ['sm2', 'splitminer2', 'split miner 2', 'split_miner_2', 'split-miner-2']:
            return cls.SPLIT_MINER_2
        elif value.lower() in ['sm3', 'splitminer3', 'split miner 3', 'split_miner_3', 'split-miner-3']:
            return cls.SPLIT_MINER_3
        else:
            raise ValueError(f'Unknown structure mining algorithm: {value}')

    def __str__(self):
        if self == StructureMiningAlgorithm.SPLIT_MINER_2:
            return 'Split Miner 2'
        elif self == StructureMiningAlgorithm.SPLIT_MINER_3:
            return 'Split Miner 3'
        return f'Unknown StructureMiningAlgorithm {str(self)}'


class GatewayProbabilitiesDiscoveryMethod(str, Enum):
    DISCOVERY = 'discovery'
    EQUIPROBABLE = 'equiprobable'

    @classmethod
    def from_str(cls, value: Union[str, List[str]]) -> 'Union[GatewayProbabilitiesDiscoveryMethod, ' \
                                                       'List[GatewayProbabilitiesDiscoveryMethod]]':
        if isinstance(value, str):
            return GatewayProbabilitiesDiscoveryMethod._from_str(value)
        elif isinstance(value, list):
            return [GatewayProbabilitiesDiscoveryMethod._from_str(v) for v in value]

    @classmethod
    def _from_str(cls, value: str) -> 'GatewayProbabilitiesDiscoveryMethod':
        if value.lower() == 'discovery':
            return cls.DISCOVERY
        elif value.lower() == 'equiprobable':
            return cls.EQUIPROBABLE
        else:
            raise ValueError(f'Unknown value {value}')

    def __str__(self):
        if self == GatewayProbabilitiesDiscoveryMethod.DISCOVERY:
            return 'discovery'
        elif self == GatewayProbabilitiesDiscoveryMethod.EQUIPROBABLE:
            return 'equiprobable'
        return f'Unknown GateManagement {str(self)}'


class StructureSettings(BaseModel):
    """
    Structure discovery settings.
    """

    optimization_metric: Metric = Metric.DL
    max_evaluations: Optional[int] = None
    mining_algorithm: Optional[StructureMiningAlgorithm] = None
    concurrency: Optional[Union[float, List[float]]] = None
    epsilon: Optional[Union[float, List[float]]] = None  # parallelism threshold (epsilon)
    eta: Optional[Union[float, List[float]]] = None  # percentile for frequency threshold (eta)
    gateway_probabilities: Optional[
        Union[GatewayProbabilitiesDiscoveryMethod, List[GatewayProbabilitiesDiscoveryMethod]]] = None
    replace_or_joins: Optional[Union[bool, List[bool]]] = None  # should replace non-trivial OR joins
    prioritize_parallelism: Optional[Union[bool, List[bool]]] = None  # should prioritize parallelism on loops

    @staticmethod
    def default() -> 'StructureSettings':
        return StructureSettings(
            optimization_metric=Metric.DL,
            max_evaluations=1,
            mining_algorithm=StructureMiningAlgorithm.SPLIT_MINER_2,
            concurrency=None,
            epsilon=[0.0, 1.0],
            eta=[0.0, 1.0],
            gateway_probabilities=GatewayProbabilitiesDiscoveryMethod.DISCOVERY,
            replace_or_joins=False,
            prioritize_parallelism=False,
        )

    @staticmethod
    def from_dict(config: dict) -> 'StructureSettings':
        mining_algorithm = StructureMiningAlgorithm.from_str(config['mining_algorithm'])

        gateway_probabilities = [
            GatewayProbabilitiesDiscoveryMethod.from_str(g)
            for g in config['gateway_probabilities']
        ]

        optimization_metric = config.get('optimization_metric')
        if optimization_metric is not None:
            optimization_metric = Metric.from_str(optimization_metric)
        else:
            optimization_metric = Metric.DL

        return StructureSettings(
            optimization_metric=optimization_metric,
            max_evaluations=config['max_evaluations'],
            mining_algorithm=mining_algorithm,
            concurrency=config['concurrency'],
            epsilon=config['epsilon'],
            eta=config['eta'],
            gateway_probabilities=gateway_probabilities,
            replace_or_joins=config['replace_or_joins'],
            prioritize_parallelism=config['prioritize_parallelism'],
        )

    def to_dict(self) -> dict:
        return {
            'optimization_metric': str(self.optimization_metric),
            'max_evaluations': self.max_evaluations,
            'mining_algorithm': str(self.mining_algorithm),
            'concurrency': self.concurrency,
            'epsilon': self.epsilon,
            'eta': self.eta,
            'gateway_probabilities': [str(g) for g in self.gateway_probabilities],
            'replace_or_joins': self.replace_or_joins,
            'prioritize_parallelism': self.prioritize_parallelism,
        }
