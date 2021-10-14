import csv
import datetime
import json
import os
import platform as pl
import uuid
from pathlib import Path
from sys import stdout

import numpy as np


def folder_id():
    return datetime.datetime.today().strftime('%Y%m%d_') + str(uuid.uuid4()).upper().replace('-', '_')


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
    # print debuging csv file
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
