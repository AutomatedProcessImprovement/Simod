import csv
import datetime
import json
import math
import os
import platform as pl
import shutil
import time
import uuid
from pathlib import Path
from sys import stdout
from typing import Optional

import numpy as np

from simod.cli_formatter import print_step


def folder_id(prefix=''):
    return prefix + datetime.datetime.today().strftime('%Y%m%d_%H%M%S_') + str(uuid.uuid4()).upper().replace('-', '_')


def file_id(prefix='', extension='.csv'):
    return prefix + datetime.datetime.today().strftime('%Y%m%d_%H%M%S%f') + extension


def gen_id():
    # generate unique bimp element ids
    return "qbp_" + str(uuid.uuid4())


def print_progress(percentage, text):
    # printing process functions
    stdout.write("\r%s" % text + str(percentage)[0:5] + chr(37) + "...      ")
    stdout.flush()


def print_performed_task(text):
    stdout.write("\r%s" % text + "...      ")
    stdout.flush()


def print_done_task():
    stdout.write("[DONE]")
    stdout.flush()
    stdout.write("\n")


def ffloat(num, dec):
    # printing formated float
    return float("{0:.2f}".format(np.round(num, decimals=dec)))


def reduce_list(input, dtype='int'):
    # reduce list of lists with no repetitions
    text = str(input).replace('[', '').replace(']', '')
    text = [x for x in text.split(',') if x != ' ']
    if text and not text == ['']:
        if dtype == 'int':
            return list(set([int(x) for x in text]))
        elif dtype == 'float':
            return list(set([float(x) for x in text]))
        elif dtype == 'str':
            return list(set([x.strip() for x in text]))
        else:
            raise ValueError(dtype)
    else:
        return list()


def create_csv_file(index, output_file, mode='w'):
    with open(output_file, mode) as f:
        for element in index:
            w = csv.DictWriter(f, element.keys())
            w.writerow(element)
        f.close()


def create_csv_file_header(index, output_file, mode='w'):
    with open(output_file, mode, newline='') as f:
        fieldnames = index[0].keys()
        w = csv.DictWriter(f, fieldnames)
        w.writeheader()
        for element in index:
            w.writerow(element)
        f.close()


def create_json(dictionary, output_file):
    with open(output_file, 'w') as f:
        f.write(json.dumps(dictionary, indent=4, sort_keys=True))
        f.close()


def round_preserve(l, expected_sum):
    """
    Rounding lists values preserving the sum values
    """
    actual_sum = sum(l)
    difference = round(expected_sum - actual_sum, 2)
    if difference > 0.00:
        idx = l.index(min(l))
    else:
        idx = l.index(max(l))
    l[idx] += difference
    return l


def avoid_zero_prob(l):
    if len(l) == 2:
        if l[0] == 0.00:
            l = [0.01, 0.99]
        elif l[1] == 0:
            l = [0.99, 0.01]
    return l


def copy(source, destiny):
    if pl.system().lower() == 'windows':
        os.system('copy "' + source + '" "' + destiny + '"')
    else:
        os.system('cp "' + source + '" "' + destiny + '"')


def get_project_dir() -> Path:
    return Path(os.path.dirname(__file__)).parent.parent


def execute_shell_cmd(args):
    print_step(f'Executing shell command: {args}')
    os.system(' '.join(args))


def file_contains(file_path: Path, substr: str) -> Optional[bool]:
    if not file_path.exists():
        return None

    with file_path.open('r') as f:
        contains = next((line for line in f if substr in line), None)

    return contains is not None


def remove_asset(location: Path):
    if location is None or not location.exists():
        return
    print_step(f'Removing {location}')
    if location.is_dir():
        shutil.rmtree(location)
    elif location.is_file():
        location.unlink()


def timeit(method):
    def timed(*args, **kw):
        start = time.time()
        result = method(*args, **kw)
        end = time.time()
        print('%r  %2.2f sec' % (method.__name__, end - start))
        return result

    return timed


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
