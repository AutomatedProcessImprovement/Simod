import os
import platform as pl
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, List

from simod.configuration import StructureMiningAlgorithm, PROJECT_DIR, AndPriorORemove


@dataclass
class Settings:
    xes_path: Path
    output_model_path: Optional[Path] = None
    mining_algorithm: StructureMiningAlgorithm = StructureMiningAlgorithm.SPLIT_MINER_3

    epsilon: Optional[Union[float, List[float]]] = None
    eta: Optional[Union[float, List[float]]] = None

    # Split Miner 2
    concurrency: Union[float, List[float]] = 0.0

    # Split Miner 3
    and_prior: List[AndPriorORemove] = field(default_factory=lambda: [AndPriorORemove.FALSE])
    or_rep: List[AndPriorORemove] = field(default_factory=lambda: [AndPriorORemove.FALSE])

    # Private
    _sm1_path: Path = PROJECT_DIR / 'external_tools/splitminer/splitminer.jar'
    _sm2_path: Path = PROJECT_DIR / 'external_tools/splitminer2/sm2.jar'
    _sm3_path: Path = PROJECT_DIR / 'external_tools/splitminer3/bpmtk.jar'

    def model_path_without_suffix(self) -> Path:
        if self.output_model_path is not None:
            return self.output_model_path.with_suffix('')
        else:
            raise ValueError('No output model path specified.')


class StructureMiner:
    """Discovers the process structure from a log file."""
    _settings: Settings

    def __init__(self, settings: Settings):
        self._settings = settings
        self._run()

    def _run(self):
        self._mining_structure(self._settings.xes_path)

        assert self._settings.output_model_path.exists(), \
            f"Model file {self._settings.output_model_path} hasn't been mined"

    def _mining_structure(self, xes_path: Path):
        miner = self._get_miner(self._settings.mining_algorithm)
        miner(xes_path, self._settings)

    def _get_miner(self, miner: StructureMiningAlgorithm):
        if miner is StructureMiningAlgorithm.SPLIT_MINER_1:
            return self._sm1_miner
        elif miner is StructureMiningAlgorithm.SPLIT_MINER_2:
            return self._sm2_miner
        elif miner is StructureMiningAlgorithm.SPLIT_MINER_3:
            return self._sm3_miner
        else:
            raise ValueError(miner)

    @staticmethod
    def _sm1_miner(xes_path: Path, settings: Settings):
        output_path = str(settings.model_path_without_suffix())
        args = ['java', '-jar', settings._sm1_path,
                str(settings.epsilon), str(settings.eta),
                str(xes_path),
                output_path]
        subprocess.call(args)

    @staticmethod
    def _sm2_miner(xes_path: Path, settings: Settings):
        output_path = str(settings.model_path_without_suffix())
        sep = ';' if pl.system().lower() == 'windows' else ':'
        args = ['java']
        if not pl.system().lower() == 'windows':
            args.append('-Xmx2G')
        args.extend(['-cp',
                     (settings._sm2_path.__str__() + sep + os.path.join(os.path.dirname(settings._sm2_path), 'lib',
                                                                        '*')),
                     'au.edu.unimelb.services.ServiceProvider',
                     'SM2',
                     str(xes_path),
                     output_path,
                     str(settings.concurrency)])
        subprocess.call(args)

    @staticmethod
    def _sm3_miner(xes_path: Path, settings: Settings):
        output_path = str(settings.model_path_without_suffix())
        sep = ';' if pl.system().lower() == 'windows' else ':'

        args = ['java']

        if not pl.system().lower() == 'windows':
            args.extend(['-Xmx2G', '-Xms1024M'])

        if isinstance(settings.and_prior, list):
            and_prior_setting = str([str(value) for value in settings.and_prior])
        else:
            and_prior_setting = str(settings.and_prior)

        if isinstance(settings.or_rep, list):
            or_rep_setting = str([str(value) for value in settings.or_rep])
        else:
            or_rep_setting = str(settings.or_rep)

        args.extend(['-cp',
                     (settings._sm3_path.__str__() + sep + os.path.join(os.path.dirname(settings._sm3_path), 'lib',
                                                                        '*')),
                     'au.edu.unimelb.services.ServiceProvider',
                     'SMD',
                     str(settings.epsilon), str(settings.eta),
                     and_prior_setting, or_rep_setting, 'false',
                     str(xes_path),
                     output_path])
        subprocess.call(args)
