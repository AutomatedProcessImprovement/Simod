import itertools
import multiprocessing
from pathlib import Path
from typing import Callable, Tuple

import pandas as pd
from tqdm import tqdm

from bpdfr_simulation_engine.simulation_properties_parser import parse_qbp_simulation_process
from simod.cli_formatter import print_notice
from simod.common_routines import evaluate_logs
from simod.common_routines import execute_shell_cmd, pbar_async, read_stats
from simod.configuration import Configuration, SimulatorKind


def diffresbp_simulator(args: Tuple):
    """Custom built simulator."""

    print_notice(f'Custom simulator has been chosen')

    settings, repetitions = args
    bpmn_path = settings.output / (settings.project_name + '.bpmn')
    output_path = settings.output / 'sim_data' / (settings.project_name + '_' + str(repetitions + 1) + '.csv')
    json_path = bpmn_path.with_suffix('.json')
    for path in [output_path, json_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
    total_cases = 1  # TODO: what's the value for total_cases?

    parse_qbp_simulation_process(bpmn_path.__str__(), json_path.__str__())

    args = [
        'diff_res_bpsim', 'start-simulation',
        '--bpmn_path', bpmn_path.__str__(),
        '--json_path', json_path.__str__(),
        '--log_out_path', output_path.__str__(),
        '--total_cases', str(total_cases)
    ]

    execute_shell_cmd(args)


def qbp_simulator(args: Tuple):
    """BIMP simulator."""

    print_notice(f'BIMP simulator has been chosen')

    settings: Configuration
    repetitions: int
    settings, repetitions = args
    args = ['java', '-jar', settings.bimp_path.absolute().__str__(),
            (settings.output / (settings.project_name + '.bpmn')).__str__(),
            '-csv',
            (settings.output / 'sim_data' / (settings.project_name + '_' + str(repetitions + 1) + '.csv')).__str__()]
    # NOTE: the call generates a CSV event log from a model
    # NOTE: might fail silently, because stderr or stdout aren't checked
    execute_shell_cmd(args)


def simulate(settings: Configuration, process_stats: pd.DataFrame, log_data, evaluate_fn: Callable = None):
    """General simulation function that takes in different simulators and evaluators."""

    if evaluate_fn is None:
        evaluate_fn = evaluate_logs

    if isinstance(settings, dict):
        settings = Configuration(**settings)

    # Simulator choice based on configuration
    if settings.simulator is SimulatorKind.BIMP:
        simulate_fn = qbp_simulator
    elif settings.simulator is SimulatorKind.CUSTOM:
        simulate_fn = diffresbp_simulator
    else:
        raise ValueError(f'Unknown simulator {settings.simulator}')

    reps = settings.repetitions
    cpu_count = multiprocessing.cpu_count()
    w_count = reps if reps <= cpu_count else cpu_count
    pool = multiprocessing.Pool(processes=w_count)

    # Simulate
    args = [(settings, rep) for rep in range(reps)]
    p = pool.map_async(simulate_fn, args)
    pbar_async(p, 'simulating:', reps)

    # Read simulated logs
    p = pool.map_async(read_stats, args)
    pbar_async(p, 'reading simulated logs:', reps)

    # Evaluate
    args = [(settings, process_stats, log) for log in p.get()]
    if len(log_data.caseid.unique()) > 1000:
        pool.close()
        results = [evaluate_fn(arg) for arg in tqdm(args, 'evaluating results:')]
        sim_values = list(itertools.chain(*results))
    else:
        p = pool.map_async(evaluate_fn, args)
        pbar_async(p, 'evaluating results:', reps)
        pool.close()
        sim_values = list(itertools.chain(*p.get()))
    return sim_values
