import math
import os
import subprocess
from pathlib import Path
from sys import stdout

from simod.cli_formatter import print_step


def get_project_dir() -> Path:
    return Path(os.path.dirname(__file__)).parent.parent


def print_progress(percentage, text):
    # printing process functions
    stdout.write("\r%s" % text + str(percentage)[0:5] + chr(37) + "...      ")
    stdout.flush()


def print_done_task():
    stdout.write("[DONE]")
    stdout.flush()
    stdout.write("\n")


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


def run_shell_with_venv(args: list):
    venv_path = os.environ.get('VIRTUAL_ENV', str(Path.cwd() / '../../venv'))
    args[0] = os.path.join(venv_path, 'bin', args[0])
    print_step(f'Executing shell command: {args}')
    os.system(' '.join(args))
