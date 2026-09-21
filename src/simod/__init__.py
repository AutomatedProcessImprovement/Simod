import warnings

# Supress HyperOpt TPE warning
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*pkg_resources is deprecated as an API.*"
)

__all__ = ["simod"]
