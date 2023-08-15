import math
import os
import platform
import subprocess
import time
import traceback
from builtins import float
from pathlib import Path
from typing import List, Tuple, Union

from hyperopt import STATUS_FAIL, STATUS_OK


def get_project_dir() -> Path:
    return Path(os.path.dirname(__file__)).parent.parent


def is_windows() -> bool:
    return platform.system().lower() == "windows"


def execute_external_command(args):
    if is_windows():
        subprocess.call(" ".join(args))
    else:
        subprocess.call(args)


def hyperopt_step(status: str, fn, *args) -> Tuple[str, object]:
    """Function executes the provided function with arguments in hyperopt safe way."""
    if status == STATUS_OK:
        try:
            return STATUS_OK, fn(*args)
        except Exception as error:
            print(error)
            traceback.print_exc()
            return STATUS_FAIL, None
    else:
        return status, None


def nearest_divisor_for_granularity(granularity: int) -> int:
    closest = 1440
    closest_diff = abs(granularity - closest)
    for i in range(1, int(math.sqrt(1440)) + 1):
        if 1440 % i == 0:
            divisor1 = i
            divisor2 = 1440 // i
            for divisor in [divisor1, divisor2]:
                if divisor <= granularity:
                    diff = granularity - divisor
                    if diff < closest_diff:
                        closest = divisor
                        closest_diff = diff
    return closest


def parse_single_value_or_interval(value: Union[float, int, List[float]]) -> Union[float, Tuple[float, float]]:
    if isinstance(value, float):
        return value
    elif isinstance(value, int):
        return float(value)
    else:
        return value[0], value[1]


def get_process_name_from_log_path(log_path: Path) -> str:
    # Get name of the file (last component)
    name = log_path.name
    # Remove each of the suffixes, if any
    for suffix in reversed(log_path.suffixes):
        name = name.removesuffix(suffix)
    # Return remaining name
    return name


def get_process_model_path(base_dir: Path, process_name: str) -> Path:
    return base_dir / f"{process_name}.bpmn"


def get_simulation_parameters_path(base_dir: Path, process_name: str) -> Path:
    return base_dir / f"{process_name}.json"


def measure_runtime(output_file: str = "runtime.txt"):
    """
    Decorator for measuring runtime of a function and writing it to a file.
    :param output_file: Path to the output file relative to the project root.
    """

    def decorator(func: callable):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time() - start
            with open(output_file, "a") as f:
                module_name = func.__module__.split(".")[-1]
                func_name = func.__name__
                f.write(f"{module_name}.{func_name}: {end} s\n")
            return result

        return wrapper

    return decorator
