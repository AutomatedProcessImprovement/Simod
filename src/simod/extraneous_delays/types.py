from dataclasses import dataclass

from pix_framework.statistics.distribution import DurationDistribution


@dataclass
class ExtraneousDelay:
    """
    Represents an extraneous delay within a business process activity.

    This class encapsulates the details of an identified extraneous delay,
    including the affected activity, a unique delay identifier, and the
    duration distribution of the delay.

    Attributes
    ----------
    activity_name : str
        The name of the activity where the extraneous delay occurs.
    delay_id : str
        A unique identifier for the delay event.
    duration_distribution : :class:`DurationDistribution`
        The statistical distribution representing the delay duration.
    """

    activity_name: str
    delay_id: str
    duration_distribution: DurationDistribution

    def to_dict(self) -> dict:
        """
        Converts the extraneous delay into a dictionary format.

        The dictionary representation is compatible with the Prosimos simulation
        engine, containing activity details, a unique event identifier, and the
        delay duration distribution.

        Returns
        -------
        dict
            A dictionary representation of the extraneous delay.
        """
        return {
            "activity": self.activity_name,
            "event_id": self.delay_id,
        } | self.duration_distribution.to_prosimos_distribution()

    @staticmethod
    def from_dict(delay: dict) -> "ExtraneousDelay":
        """
        Creates an `ExtraneousDelay` instance from a dictionary.

        This method reconstructs an `ExtraneousDelay` object from a dictionary
        containing activity name, delay identifier, and duration distribution.

        Parameters
        ----------
        delay : dict
            A dictionary representation of an extraneous delay.

        Returns
        -------
        :class:`ExtraneousDelay`
            An instance of `ExtraneousDelay` with the extracted attributes.
        """
        return ExtraneousDelay(
            activity_name=delay["activity"],
            delay_id=delay["event_id"],
            duration_distribution=DurationDistribution.from_dict(delay),
        )
