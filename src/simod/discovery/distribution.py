import pandas as pd
from bpdfr_simulation_engine.probability_distributions import best_fit_distribution_1


def get_best_distribution(data):
    """
    Returns the best probability distribution for a given data series.
    """
    data = pd.Series(data)
    distribution_data = best_fit_distribution_1(data)
    distribution_data['distribution_params'] = [{'value': param} for param in
                                                distribution_data['distribution_params']]

    if distribution_data['distribution_name'] == 'fix':
        value = distribution_data['distribution_params'][0]
        distribution_data['distribution_params'].append(value)
        distribution_data['distribution_params'].append(value)
        distribution_data['distribution_params'].append(value)

    return distribution_data
