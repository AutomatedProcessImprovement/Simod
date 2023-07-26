
class Error(Exception):
    """Base class for other exceptions"""
    pass


class InvalidFuzzyIntervalSize(Error):
    """Raised when the number of minutes in each fuzzy interval is invalid, i.e., they should be a factor of 1440"""

    def __str__(self):
        return "Invalid Fuzzy Interval Size"
