import math
import statistics
import sys

import numpy
import pandas as pd
import scipy
from scipy.stats import wasserstein_distance


def get_best_distribution(data: list) -> dict:
    """
    Returns the best probability distribution for a given data series.

    :param data: List of values.
    :return: Distribution dictionary with its name and parameters.
    """
    data = pd.Series(data)
    distribution_data = _best_fit_distribution_1(data)
    distribution_data['distribution_params'] = [{'value': param} for param in
                                                distribution_data['distribution_params']]

    if distribution_data['distribution_name'] == 'fix':
        value = distribution_data['distribution_params'][0]
        distribution_data['distribution_params'].append(value)
        distribution_data['distribution_params'].append(value)
        distribution_data['distribution_params'].append(value)

    return distribution_data


def _best_fit_distribution_1(data):
    fix_value = _check_fix(data)
    if fix_value is not None:
        return {"distribution_name": "fix", "distribution_params": [_check_fix(data)]}

    mean = statistics.mean(data)
    variance = statistics.variance(data)
    st_dev = statistics.pstdev(data)
    d_min = min(data)
    d_max = max(data)

    dist_candidates = [
        {"distribution_name": "expon", "distribution_params": [0, mean, d_min, d_max]},
        {"distribution_name": "norm", "distribution_params": [mean, st_dev, d_min, d_max]},
        {"distribution_name": "uniform", "distribution_params": [d_min, d_max - d_min, d_min, d_max]},
        {"distribution_name": "default", "distribution_params": [d_min, d_max]}
    ]

    if mean != 0:
        mean_2 = mean ** 2
        phi = math.sqrt(variance + mean_2)
        mu = math.log(mean_2 / phi)
        sigma = math.sqrt(math.log(phi ** 2 / mean_2))

        dist_candidates.append({"distribution_name": "lognorm",
                                "distribution_params": [sigma, 0, math.exp(mu), d_min, d_max]}, )

    if mean != 0 and variance != 0:
        dist_candidates.append({"distribution_name": "gamma",
                                "distribution_params": [pow(mean, 2) / variance, 0, variance / mean, d_min, d_max]}, )

    best_dist = None
    best_emd = sys.float_info.max
    for dist_c in dist_candidates:
        ev_list = list()
        for _ in range(0, len(data)):
            ev_list.append(_evaluate_distribution_function(dist_c["distribution_name"], dist_c["distribution_params"]))

        emd = wasserstein_distance(data, ev_list)
        if emd < best_emd:
            best_emd = emd
            best_dist = dist_c

    return best_dist


def _check_fix(data_list, delta=5):
    for d1 in data_list:
        count = 0
        for d2 in data_list:
            if abs(d1 - d2) < delta:
                count += 1
        if count / len(data_list) > 0.9:
            return d1
    return None


def _evaluate_distribution_function(distribution_name, params):
    if distribution_name == "fix":
        return params[0]
    elif distribution_name == 'default':
        return numpy.random.uniform(params[0], params[1])

    arg = params[:-4]
    loc = params[-4]
    scale = params[-3]
    d_min = params[-2]
    d_max = params[-1]

    dist = getattr(scipy.stats, distribution_name)
    num_param = len(arg)

    f_dist = 0
    while True:
        if num_param == 0:
            f_dist = dist.rvs(loc=loc, scale=scale, size=1)[0]
        elif num_param == 1:
            f_dist = dist.rvs(arg[0], loc=loc, scale=scale, size=1)[0]
        elif num_param == 2:
            f_dist = dist.rvs(arg[0], arg[1], loc=loc, scale=scale, size=1)[0]
        elif num_param == 3:
            f_dist = dist.rvs(arg[0], arg[1], arg[2], loc=loc, scale=scale, size=1)[0]
        elif num_param == 4:
            f_dist = dist.rvs(arg[0], arg[1], arg[2], arg[3], loc=loc, scale=scale, size=1)[0]
        elif num_param == 5:
            f_dist = dist.rvs(arg[0], arg[1], arg[2], arg[3], arg[4], loc=loc, scale=scale, size=1)[0]
        elif num_param == 6:
            f_dist = dist.rvs(arg[0], arg[1], arg[2], arg[3], arg[4], arg[5], loc=loc, scale=scale, size=1)[0]
        elif num_param == 7:
            f_dist = dist.rvs(arg[0], arg[1], arg[2], arg[3], arg[4], arg[5], arg[6], loc=loc, scale=scale, size=1)[0]
        if d_min <= f_dist <= d_max:
            break
    return f_dist
