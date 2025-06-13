from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from simod.settings.common_settings import Metric


@dataclass
class HyperoptIterationParams:
    """
    Parameters for a single iteration of the Case-Arrival optimization process.

    This class defines the configuration settings used during an iteration of the
    optimization process of the case arrival model.

    Attributes
    ----------
    output_dir : :class:`pathlib.Path`
        Directory where all output files for the current iteration will be stored.
    project_name : str
        Name of the project, mainly used for file naming.
    optimization_metric : :class:`Metric`
        Metric used to evaluate the candidate process model in this iteration.
    outlier_threshold : float, optional
        Threshold to use when filtering outliers (positive number).

    Notes
    -----
    - Currently, this process only tries different outlier thresholds. Implemented
    as HyperOpt process for convenience and potential extension with more complex
    case arrival model discovery.
    """

    # General settings
    output_dir: Path  # Directory where to output all the files of the current iteration
    project_name: str  # Name of the project for file naming

    optimization_metric: Metric  # Metric to evaluate the candidate of this iteration
    outlier_threshold: Optional[float]  # Outlier threshold

    def to_dict(self) -> dict:
        """
        Converts the instance into a dictionary representation of the optimization parameters.

        Returns
        -------
        dict
            A dictionary containing the optimization parameters for this iteration.
        """
        optimization_parameters = {
            "output_dir": str(self.output_dir),
            "project_name": str(self.project_name),
            "optimization_metric": str(self.optimization_metric),
            "outlier_threshold": self.outlier_threshold,
        }

        return optimization_parameters

    @staticmethod
    def from_hyperopt_dict(
        hyperopt_dict: dict,
        optimization_metric: Metric,
        output_dir: Path,
        project_name: str,
    ) -> "HyperoptIterationParams":
        """Create the params for this run from the hyperopt dictionary returned by the fmin function."""
        outlier_threshold = hyperopt_dict.get("outlier_threshold", 20.0)

        return HyperoptIterationParams(
            output_dir=output_dir,
            project_name=project_name,
            optimization_metric=optimization_metric,
            outlier_threshold=outlier_threshold
        )
