from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from simod.settings.common_settings import Metric
from simod.settings.control_flow_settings import GatewayProbabilitiesMethod, ProcessModelDiscoveryAlgorithm


@dataclass
class HyperoptIterationParams:
    """Parameters for a single iteration of the Control-Flow optimization process."""
    # General settings
    output_dir: Path  # each pipeline run creates its own directory
    model_path: Path  # in structure optimizer, this path is assigned after the model is mined
    project_name: str  # this doesn't change and just inherits from the project settings, used for file naming

    optimization_metric: Metric
    gateway_probabilities_method: GatewayProbabilitiesMethod
    mining_algorithm: ProcessModelDiscoveryAlgorithm
    # Split Miner 2
    concurrency: Optional[float]
    # Split Miner 3
    epsilon: Optional[float]  # parallelism threshold (epsilon)
    eta: Optional[float]  # percentile for frequency threshold (eta)
    replace_or_joins: Optional[bool]  # should replace non-trivial OR joins
    prioritize_parallelism: Optional[bool]  # should prioritize parallelism on loops

    def to_dict(self) -> dict:
        """Returns a dictionary with the parameters for this run."""
        # Save common params
        optimization_parameters = {
            'output_dir': self.output_dir,
            'optimization_metric': str(self.optimization_metric),
            'gateway_probabilities': self.gateway_probabilities_method,
            'mining_algorithm': str(self.mining_algorithm),
        }
        # Save params depending on the discovery algorithm
        if self.mining_algorithm is ProcessModelDiscoveryAlgorithm.SPLIT_MINER_2:
            optimization_parameters['concurrency'] = self.concurrency
        elif self.mining_algorithm is ProcessModelDiscoveryAlgorithm.SPLIT_MINER_3:
            optimization_parameters['epsilon'] = self.epsilon
            optimization_parameters['eta'] = self.eta
            optimization_parameters['prioritize_parallelism'] = self.prioritize_parallelism
            optimization_parameters['replace_or_joins'] = self.replace_or_joins
        # Return dict
        return optimization_parameters

    @staticmethod
    def from_hyperopt_dict(
            hyperopt_dict: dict,
            optimization_metric: Metric,
            mining_algorithm: ProcessModelDiscoveryAlgorithm,
            output_dir: Path,
            model_path: Optional[Path],
            project_name: str,
    ) -> 'HyperoptIterationParams':
        """Create the params for this run from the hyperopt dictionary returned by the fmin function."""
        # Extract gateway probabilities method
        gateway_probabilities_method = GatewayProbabilitiesMethod.from_str(hyperopt_dict['gateway_probabilities_method'])
        # Extract model discovery parameters if needed (by default None)
        epsilon = None
        eta = None
        concurrency = None
        prioritize_parallelism = None
        replace_or_joins = None
        # If the model was not provided, extract discovery parameters
        if model_path is None:
            if mining_algorithm is ProcessModelDiscoveryAlgorithm.SPLIT_MINER_2:
                concurrency = hyperopt_dict['concurrency']
            elif mining_algorithm is ProcessModelDiscoveryAlgorithm.SPLIT_MINER_3:
                epsilon = hyperopt_dict['epsilon']
                eta = hyperopt_dict['eta']
                prioritize_parallelism = hyperopt_dict['prioritize_parallelism']
                replace_or_joins = hyperopt_dict.get('replace_or_joins')
        # Return parameters instance
        return HyperoptIterationParams(
            output_dir=output_dir,
            model_path=model_path,
            project_name=project_name,
            optimization_metric=optimization_metric,
            gateway_probabilities_method=gateway_probabilities_method,
            mining_algorithm=mining_algorithm,
            concurrency=concurrency,
            epsilon=epsilon,
            eta=eta,
            prioritize_parallelism=prioritize_parallelism,
            replace_or_joins=replace_or_joins,
        )
