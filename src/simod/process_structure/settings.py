from dataclasses import dataclass, field
from typing import Optional, Union, List

import yaml

from simod.configuration import GateManagement, PDFMethod, StructureMiningAlgorithm, AndPriorORemove


@dataclass
class StructureOptimizationSettings:
    """Settings for the structure optimizer."""
    project_name: Optional[str]

    gateway_probabilities: Optional[Union[GateManagement, List[GateManagement]]] = GateManagement.DISCOVERY
    max_evaluations: int = 1
    simulation_repetitions: int = 1
    pdef_method: Optional[PDFMethod] = None  # TODO: rename to distribution_discovery_method

    # Structure Miner Settings can be arrays of values, in that case different values are used for different repetition.
    # Structure Miner accepts only singular values for the following settings:
    #
    # for Split Miner 1 and 3
    epsilon: Optional[Union[float, List[float]]] = None
    eta: Optional[Union[float, List[float]]] = None
    # for Split Miner 2
    concurrency: Optional[Union[float, List[float]]] = 0.0
    #
    # Singular Structure Miner configuration used to compose the search space and split epsilon, eta and concurrency
    # lists into singular values.
    mining_algorithm: StructureMiningAlgorithm = StructureMiningAlgorithm.SPLIT_MINER_3
    #
    # Split Miner 3
    and_prior: List[AndPriorORemove] = field(default_factory=lambda: [AndPriorORemove.FALSE])
    or_rep: List[AndPriorORemove] = field(default_factory=lambda: [AndPriorORemove.FALSE])

    @staticmethod
    def from_stream(stream: Union[str, bytes]) -> 'StructureOptimizationSettings':
        settings = yaml.load(stream, Loader=yaml.FullLoader)

        project_name = settings.get('project_name', None)

        if 'structure_optimizer' in settings:
            settings = settings['structure_optimizer']

        gateway_probabilities = settings.get('gateway_probabilities', None)
        if gateway_probabilities is None:
            gateway_probabilities = settings.get('gate_management', None)  # legacy key support
        if gateway_probabilities is not None:
            if isinstance(gateway_probabilities, list):
                gateway_probabilities = [GateManagement.from_str(g) for g in gateway_probabilities]
            elif isinstance(gateway_probabilities, str):
                gateway_probabilities = GateManagement.from_str(gateway_probabilities)
            else:
                raise ValueError('Gateway probabilities must be a list or a string.')

        max_evaluations = settings.get('max_evaluations', None)
        if max_evaluations is None:
            max_evaluations = settings.get('max_eval_s', 1)  # legacy key support

        simulation_repetitions = settings.get('simulation_repetitions', 1)

        pdef_method = settings.get('pdef_method', None)
        if pdef_method is not None:
            pdef_method = PDFMethod.from_str(pdef_method)
        else:
            pdef_method = PDFMethod.DEFAULT

        epsilon = settings.get('epsilon', None)

        eta = settings.get('eta', None)

        concurrency = settings.get('concurrency', 0.0)

        mining_algorithm = settings.get('mining_algorithm', None)
        if mining_algorithm is None:
            mining_algorithm = settings.get('mining_alg', None)  # legacy key support
        if mining_algorithm is not None:
            mining_algorithm = StructureMiningAlgorithm.from_str(mining_algorithm)

        and_prior = settings.get('and_prior', None)
        if and_prior is not None:
            if isinstance(and_prior, list):
                and_prior = [AndPriorORemove.from_str(a) for a in and_prior]
            elif isinstance(and_prior, str):
                and_prior = [AndPriorORemove.from_str(and_prior)]
            else:
                raise ValueError('and_prior must be a list or a string.')

        or_rep = settings.get('or_rep', None)
        if or_rep is not None:
            if isinstance(or_rep, list):
                or_rep = [AndPriorORemove.from_str(o) for o in or_rep]
            elif isinstance(or_rep, str):
                or_rep = [AndPriorORemove.from_str(or_rep)]
            else:
                raise ValueError('or_rep must be a list or a string.')

        return StructureOptimizationSettings(
            project_name=project_name,
            gateway_probabilities=gateway_probabilities,
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
