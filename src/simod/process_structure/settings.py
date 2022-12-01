from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, List, Dict, Any

import yaml

from simod.configuration import StructureMiningAlgorithm, GatewayProbabilitiesDiscoveryMethod, PDFMethod, Configuration, \
    Metric


@dataclass
class StructureOptimizationSettings:
    """Settings for the structure optimizer."""
    project_name: Optional[str]
    base_dir: Optional[Path]
    model_path: Optional[Path]

    optimization_metric: Metric
    gateway_probabilities_method: Optional[Union[GatewayProbabilitiesDiscoveryMethod, List[
        GatewayProbabilitiesDiscoveryMethod]]] = GatewayProbabilitiesDiscoveryMethod.DISCOVERY
    max_evaluations: int = 1
    simulation_repetitions: int = 1
    pdef_method: Optional[PDFMethod] = None  # TODO: rename to distribution_discovery_method

    # Structure Miner Settings can be arrays of values, in that case different values are used for different repetition.
    # Structure Miner accepts only singular values for the following settings:
    #
    # for Split Miner 1 and 3
    epsilon: Optional[Union[float, List[float]]] = None  # percentile for frequency threshold
    eta: Optional[Union[float, List[float]]] = None  # parallelism threshold
    # for Split Miner 2
    concurrency: Optional[Union[float, List[float]]] = 0.0
    #
    # Singular Structure Miner configuration used to compose the search space and split epsilon, eta and concurrency
    # lists into singular values.
    mining_algorithm: Optional[StructureMiningAlgorithm] = None
    #
    # Split Miner 3
    and_prior: List[bool] = field(default_factory=lambda: [False])
    or_rep: List[bool] = field(default_factory=lambda: [False])

    @staticmethod
    def from_stream(
            stream: Union[str, bytes],
            base_dir: Path,
            model_path: Optional[Path] = None) -> 'StructureOptimizationSettings':
        settings = yaml.load(stream, Loader=yaml.FullLoader)

        project_name = settings.get('project_name', None)

        if 'structure' in settings:
            settings = settings['structure']

        gateway_probabilities_method = (
                settings.get('gateway_probabilities', None)
                or settings.get('gate_management')  # legacy key support
        )
        if gateway_probabilities_method is not None:
            if isinstance(gateway_probabilities_method, list):
                gateway_probabilities_method = [
                    GatewayProbabilitiesDiscoveryMethod.from_str(g)
                    for g in gateway_probabilities_method
                ]
            elif isinstance(gateway_probabilities_method, str):
                gateway_probabilities_method = GatewayProbabilitiesDiscoveryMethod.from_str(
                    gateway_probabilities_method)
            elif isinstance(gateway_probabilities_method, GatewayProbabilitiesDiscoveryMethod):
                pass
            else:
                raise ValueError('Gateway probabilities must be a list or a string.')
        else:
            gateway_probabilities_method = GatewayProbabilitiesDiscoveryMethod.DISCOVERY

        max_evaluations = settings.get('max_evaluations', None)
        if max_evaluations is None:
            max_evaluations = settings.get('max_eval_s', 1)  # legacy key support

        simulation_repetitions = settings.get('simulation_repetitions', 1)

        pdef_method = settings.get('pdef_method', None)
        if pdef_method is not None:
            pdef_method = PDFMethod.from_str(pdef_method)
        else:
            pdef_method = PDFMethod.AUTOMATIC

        epsilon = settings.get('epsilon', None)

        eta = settings.get('eta', None)

        concurrency = settings.get('concurrency', 0.0)

        mining_algorithm = settings.get('mining_algorithm', None) or settings.get('mining_alg', None)
        if mining_algorithm is not None:
            mining_algorithm = StructureMiningAlgorithm.from_str(mining_algorithm)
        else:
            mining_algorithm = StructureMiningAlgorithm.SPLIT_MINER_3

        and_prior = settings.get('and_prior', None)
        if and_prior is not None:
            if isinstance(and_prior, str):
                and_prior = [and_prior.lower() == 'true']

        or_rep = settings.get('or_rep', None)
        if or_rep is not None:
            if isinstance(or_rep, str):
                or_rep = [or_rep.lower() == 'true']

        optimization_metric = settings.get('optimization_metric', None)
        if optimization_metric is not None:
            optimization_metric = Metric.from_str(optimization_metric)
        else:
            optimization_metric = Metric.DL

        return StructureOptimizationSettings(
            project_name=project_name,
            base_dir=base_dir,
            model_path=model_path,
            optimization_metric=optimization_metric,
            gateway_probabilities_method=gateway_probabilities_method,
            max_evaluations=max_evaluations,
            simulation_repetitions=simulation_repetitions,
            pdef_method=pdef_method,
            epsilon=epsilon,
            eta=eta,
            concurrency=concurrency,
            mining_algorithm=mining_algorithm,
            and_prior=and_prior,
            or_rep=or_rep
        )

    @staticmethod
    def from_configuration(config: Configuration, base_dir: Path) -> 'StructureOptimizationSettings':
        project_name = config.common.log_path.stem

        return StructureOptimizationSettings(
            project_name=project_name,
            base_dir=base_dir,
            model_path=config.common.model_path,
            optimization_metric=config.structure.optimization_metric,
            gateway_probabilities_method=config.structure.gateway_probabilities,
            max_evaluations=config.structure.max_evaluations,
            simulation_repetitions=config.common.repetitions,
            pdef_method=config.structure.distribution_discovery_type,
            epsilon=config.structure.epsilon,
            eta=config.structure.eta,
            concurrency=config.structure.concurrency,
            mining_algorithm=config.structure.mining_algorithm,
            and_prior=config.structure.and_prior,
            or_rep=config.structure.or_rep
        )


@dataclass
class PipelineSettings:
    """Settings for the structure optimization pipeline."""
    # General settings
    output_dir: Optional[Path]  # each pipeline run creates its own directory
    model_path: Optional[Path]  # in structure optimizer, this path is assigned after the model is mined
    project_name: str  # this doesn't change and just inherits from the project settings, used for file naming

    # Optimization settings
    gateway_probabilities_method: GatewayProbabilitiesDiscoveryMethod
    # for Split Miner 1 and 3
    epsilon: Optional[float] = None
    eta: Optional[float] = None
    # for Split Miner 2
    concurrency: Optional[float] = 0.0
    # for Split Miner 3
    and_prior: Optional[bool] = None
    or_rep: Optional[bool] = None

    def optimization_parameters_as_dict(self, mining_algorithm: StructureMiningAlgorithm) -> Dict[str, Any]:
        """Returns a dictionary of parameters relevant for the optimizer."""
        optimization_parameters = {
            'gateway_probabilities': self.gateway_probabilities_method,
            'output_dir': self.output_dir,
        }

        if mining_algorithm in [StructureMiningAlgorithm.SPLIT_MINER_1,
                                StructureMiningAlgorithm.SPLIT_MINER_3]:
            optimization_parameters['epsilon'] = self.epsilon
            optimization_parameters['eta'] = self.eta
            optimization_parameters['and_prior'] = self.and_prior
            optimization_parameters['or_rep'] = self.or_rep
        elif mining_algorithm is StructureMiningAlgorithm.SPLIT_MINER_2:
            optimization_parameters['concurrency'] = self.concurrency

        return optimization_parameters

    @staticmethod
    def from_hyperopt_dict(
            data: dict,
            initial_settings: StructureOptimizationSettings,
            model_path: Path,
            project_name: str,
    ) -> 'PipelineSettings':
        """
        Create a settings object from a hyperopt's dictionary that returned as a result of the optimization.
        Initial settings are required, because for some settings, hyperopt returns an index of the settings' list.
        """
        gateway_probabilities_index = data.get('gateway_probabilities_method')
        assert gateway_probabilities_index is not None
        gateway_probabilities_method = initial_settings.gateway_probabilities_method[gateway_probabilities_index]

        epsilon = data.get('epsilon')
        eta = data.get('eta')
        concurrency = data.get('concurrency')
        and_prior_index = data.get('and_prior')
        and_prior = None
        or_rep_index = data.get('or_rep')
        or_rep = None

        if initial_settings.mining_algorithm in [StructureMiningAlgorithm.SPLIT_MINER_1,
                                                 StructureMiningAlgorithm.SPLIT_MINER_3]:
            assert epsilon is not None
            assert eta is not None

            if initial_settings.mining_algorithm is StructureMiningAlgorithm.SPLIT_MINER_3:
                assert and_prior_index is not None
                assert or_rep_index is not None

                and_prior = initial_settings.and_prior[and_prior_index]
                or_rep = initial_settings.or_rep[or_rep_index]
        elif initial_settings.mining_algorithm is StructureMiningAlgorithm.SPLIT_MINER_2:
            assert concurrency is not None

        output_dir = model_path.parent

        return PipelineSettings(
            output_dir=output_dir,
            model_path=model_path,
            project_name=project_name,
            gateway_probabilities_method=gateway_probabilities_method,
            epsilon=epsilon,
            eta=eta,
            concurrency=concurrency,
            and_prior=and_prior,
            or_rep=or_rep,
        )

    def to_dict(self) -> dict:
        """Converts the settings to a dictionary."""
        return {
            'output_dir': str(self.output_dir),
            'model_path': str(self.model_path),
            'project_name': self.project_name,
            'gateway_probabilities_method': str(self.gateway_probabilities_method),
            'epsilon': self.epsilon,
            'eta': self.eta,
            'concurrency': self.concurrency,
            'and_prior': str(self.and_prior),
            'or_rep': str(self.or_rep),
        }
