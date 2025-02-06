from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pix_framework.discovery.gateway_probabilities import GatewayProbabilitiesDiscoveryMethod

from simod.settings.common_settings import Metric
from simod.settings.control_flow_settings import ProcessModelDiscoveryAlgorithm


@dataclass
class HyperoptIterationParams:
    """
    Parameters for a single iteration of the Control-Flow optimization process.

    This class defines the configuration settings used during an iteration of the
    optimization process, including process model discovery, optimization metric,
    and gateway probability discovery.

    Attributes
    ----------
    output_dir : :class:`pathlib.Path`
        Directory where all output files for the current iteration will be stored.
    provided_model_path : :class:`pathlib.Path`, optional
        Path to a provided BPMN model, if available (no discovery needed).
    project_name : str
        Name of the project, mainly used for file naming.
    optimization_metric : :class:`Metric`
        Metric used to evaluate the candidate process model in this iteration.
    gateway_probabilities_method : :class:`GatewayProbabilitiesDiscoveryMethod`
        Method for discovering gateway probabilities.
    mining_algorithm : :class:`ProcessModelDiscoveryAlgorithm`
        Algorithm used for process model discovery, if necessary.
    epsilon : float, optional
        Number of concurrent relations between events to be captured in the discovery algorithm (between 0.0 and 1.0).
    eta : float, optional
        Threshold for filtering the incoming and outgoing edges in the discovery algorithm (between 0.0 and 1.0).
    replace_or_joins : bool, optional
        Whether to replace non-trivial OR joins in the discovered model.
    prioritize_parallelism : bool, optional
        Whether to prioritize parallelism or loops for model discovery.
    f_score : float], default=Non, optional
        Minimum f-score value to consider the discovered data-aware branching rules.

    Notes
    -----
    - If `provided_model_path` is specified, process model discovery will be skipped.
    """

    # General settings
    output_dir: Path  # Directory where to output all the files of the current iteration
    provided_model_path: Optional[Path]  # Provided when no need to discover BPMN model
    project_name: str  # Name of the project for file naming

    optimization_metric: Metric  # Metric to evaluate the candidate of this iteration
    gateway_probabilities_method: GatewayProbabilitiesDiscoveryMethod  # Method to discover the gateway probabilities
    mining_algorithm: ProcessModelDiscoveryAlgorithm  # Algorithm to discover the process model
    # Split Miner 2
    # Split Miner 3
    epsilon: Optional[float]  # Parallelism threshold (epsilon)
    eta: Optional[float]  # Percentile for frequency threshold (eta)
    replace_or_joins: Optional[bool]  # Should replace non-trivial OR joins
    prioritize_parallelism: Optional[bool]  # Should prioritize parallelism on loops
    f_score: Optional[float] = None  # quality gateway for branch rules (f_score)

    def to_dict(self) -> dict:
        """
        Converts the instance into a dictionary representation of the optimization parameters.

        The returned dictionary is structured based on whether a process model needs
        to be discovered or if a pre-existing model is provided.

        Returns
        -------
        dict
            A dictionary containing the optimization parameters for this iteration.
        """
        optimization_parameters = {
            "output_dir": str(self.output_dir),
            "project_name": str(self.project_name),
            "optimization_metric": str(self.optimization_metric),
            "gateway_probabilities": self.gateway_probabilities_method.value,
            "mining_algorithm": str(self.mining_algorithm),
        }

        if self.provided_model_path is None:
            if self.mining_algorithm is ProcessModelDiscoveryAlgorithm.SPLIT_MINER_V2:
                optimization_parameters["epsilon"] = self.epsilon
            elif self.mining_algorithm is ProcessModelDiscoveryAlgorithm.SPLIT_MINER_V1:
                optimization_parameters["epsilon"] = self.epsilon
                optimization_parameters["eta"] = self.eta
                optimization_parameters["prioritize_parallelism"] = self.prioritize_parallelism
                optimization_parameters["replace_or_joins"] = self.replace_or_joins
        else:
            optimization_parameters["provided_model_path"] = str(self.provided_model_path)

        if self.f_score:
            optimization_parameters["discover_branch_rules"] = True
            optimization_parameters["f_score"] = self.f_score

        return optimization_parameters

    @staticmethod
    def from_hyperopt_dict(
        hyperopt_dict: dict,
        optimization_metric: Metric,
        mining_algorithm: ProcessModelDiscoveryAlgorithm,
        output_dir: Path,
        provided_model_path: Optional[Path],
        project_name: str,
    ) -> "HyperoptIterationParams":
        """Create the params for this run from the hyperopt dictionary returned by the fmin function."""
        gateway_probabilities_method = GatewayProbabilitiesDiscoveryMethod.from_str(
            hyperopt_dict["gateway_probabilities_method"]
        )

        epsilon, eta, prioritize_parallelism, replace_or_joins = None, None, None, None

        if provided_model_path is None:
            if mining_algorithm == ProcessModelDiscoveryAlgorithm.SPLIT_MINER_V1:
                epsilon = hyperopt_dict["epsilon"]
                eta = hyperopt_dict["eta"]
                prioritize_parallelism = hyperopt_dict["prioritize_parallelism"]
                replace_or_joins = hyperopt_dict.get("replace_or_joins")
            elif mining_algorithm == ProcessModelDiscoveryAlgorithm.SPLIT_MINER_V2:
                epsilon = hyperopt_dict["epsilon"]

        f_score = hyperopt_dict.get("f_score", None)

        return HyperoptIterationParams(
            output_dir=output_dir,
            provided_model_path=provided_model_path,
            project_name=project_name,
            optimization_metric=optimization_metric,
            gateway_probabilities_method=gateway_probabilities_method,
            mining_algorithm=mining_algorithm,
            epsilon=epsilon,
            eta=eta,
            prioritize_parallelism=prioritize_parallelism,
            replace_or_joins=replace_or_joins,
            f_score=f_score
        )
