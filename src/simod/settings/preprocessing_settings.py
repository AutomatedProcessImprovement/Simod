from pix_framework.enhancement.start_time_estimator.config import ConcurrencyThresholds
from pydantic import BaseModel


class PreprocessingSettings(BaseModel):
    """
    Configuration for event log preprocessing.

    This class defines parameters used to preprocess event logs before
    SIMOD main pipeline, including concurrency threshold settings
    and multitasking options.

    Attributes
    ----------
    multitasking : bool
        Whether to preprocess the event log to handle resources working in more than one activity at a time.
    enable_time_concurrency_threshold : float
        Threshold for determining concurrent events (for computing enabled) time based on the ratio of overlapping
        w.r.t. their occurrences. Ranges from 0 to 1 (0.3 means that two activities will be considered concurrent
        when their execution overlaps in 30% or more of the cases).
    concurrency_thresholds : :class:`ConcurrencyThresholds`
        Thresholds for the computation of the start times (if missing) based on the Heuristics miner algorithm,
        including direct-follows (df), length-2-loops (l2l), and length-1-loops (l1l).
    """

    multitasking: bool = False
    enable_time_concurrency_threshold: float = 0.5
    concurrency_thresholds: ConcurrencyThresholds = ConcurrencyThresholds(df=0.75, l2l=0.9, l1l=0.9)

    @staticmethod
    def from_dict(config: dict) -> "PreprocessingSettings":
        """
        Instantiates SIMOD preprocessing configuration from a dictionary.

        Parameters
        ----------
        config : dict
            Dictionary with the configuration values for the preprocessing parameters.

        Returns
        -------
        :class:`PreprocessingSettings`
            Instance of SIMOD preprocessing configuration for the specified dictionary values.
        """
        return PreprocessingSettings(
            multitasking=config.get("multitasking", False),
            enable_time_concurrency_threshold=config.get("enable_time_concurrency_threshold", 0.5),
            concurrency_thresholds=ConcurrencyThresholds(
                df=config.get("concurrency_df", 0.9),
                l2l=config.get("concurrency_l2l", 0.9),
                l1l=config.get("concurrency_l1l", 0.9),
            ),
        )

    def to_dict(self) -> dict:
        """
        Translate the preprocessing configuration stored in this instance into a dictionary.

        Returns
        -------
        dict
            Python dictionary storing this configuration.
        """
        return {
            "multitasking": self.multitasking,
            "enable_time_concurrency_threshold": self.enable_time_concurrency_threshold,
            "concurrency_df": self.concurrency_thresholds.df,
            "concurrency_l2l": self.concurrency_thresholds.l2l,
            "concurrency_l1l": self.concurrency_thresholds.l1l,
        }
