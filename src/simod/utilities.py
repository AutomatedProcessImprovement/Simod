import math
import os
import platform
import subprocess
import traceback
from pathlib import Path
from typing import Tuple

from hyperopt import STATUS_OK, STATUS_FAIL


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
