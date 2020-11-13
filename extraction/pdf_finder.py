# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 17:28:46 2020

@author: Manuel Camargo
"""
import warnings
import pandas as pd
import scipy.stats as st
import numpy as np

import utils.support as sup
# import support as sup


class DistributionFinder():
    """
        This class evaluates a series of data and adjust a
        Probability Distribution Function
     """

    def __init__(self, data_serie, bins='auto'):
        """constructor"""
        self.data_serie = data_serie
        self.bins = bins

        self.distribution = self.get_task_distribution()

    def get_task_distribution(self):
        """
        Calculate the probability distribution of a series of data
        """
        dist = {'norm': 'NORMAL', 'lognorm': 'LOGNORMAL', 'gamma': 'GAMMA',
                'expon': 'EXPONENTIAL', 'uniform': 'UNIFORM',
                'triang': 'TRIANGULAR', 'fixed': 'FIXED'}
        params = dict()
        if not self.data_serie:
            dname = 'fixed'
            params['mean'] = 0
            params['arg1'] = 0
            params['arg2'] = 0
        else:
            # If all the values of the serie are equal,
            # it is an automatic task with fixed distribution
            if np.min(self.data_serie) == np.max(self.data_serie):
                dname = 'fixed'
                params['mean'] = int(np.min(self.data_serie))
                params['arg1'] = 0
                params['arg2'] = 0
            elif len(self.data_serie) < 100:
                dname = 'expon'
                params['mean'] = 0
                params['arg1'] = sup.ffloat(np.mean(self.data_serie), 1)
                params['arg2'] = 0
            else:
                dname = self.dist_best()
                params = self.dist_params(dname)
        return {'dname': dist[dname], 'dparams': params}

    def dist_best(self):
        """
        Finds the best probability distribution for a given data serie
        """
        # Create a data series from the given list
        data = pd.Series(self.data_serie)
        # Get histogram of original data
        hist, bin_edges = np.histogram(data, bins=self.bins, density=True)
        bin_edges = (bin_edges + np.roll(bin_edges, -1))[:-1] / 2.0
        # Distributions to check
        distributions = [st.norm, st.expon, st.uniform,
                         st.triang, st.lognorm, st.gamma]
        # Best holders
        best_distribution = st.norm
        best_sse = np.inf
        # Estimate distribution parameters from data
        for distribution in distributions:
            # Try to fit the distribution
            try:
                # Ignore warnings from data that can't be fit
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    # fit dist to data
                    params = distribution.fit(data)
                    # Separate parts of parameters
                    arg = params[:-2]
                    loc = params[-2]
                    scale = params[-1]
                    # Calculate fitted PDF and error with fit in distribution
                    pdf = distribution.pdf(bin_edges, loc=loc, scale=scale, *arg)
                    sse = np.sum(np.power(hist - pdf, 2.0))
                    # identify if this distribution is better
                    if best_sse > sse > 0:
                        best_distribution = distribution
                        best_sse = sse
            except:
                pass
        return best_distribution.name

    def dist_params(self, dname):
        """
        Calculate additional parameters once the pdf is found
        """
        params = dict()
        if dname == 'norm':
            # for effects of the XML arg1=std and arg2=0
            params['mean'] = sup.ffloat(np.mean(self.data_serie), 1)
            params['arg1'] = sup.ffloat(np.std(self.data_serie), 1)
            params['arg2'] = 0
        elif dname in ['lognorm', 'gamma']:
            # for effects of the XML arg1=var and arg2=0
            params['mean'] = sup.ffloat(np.mean(self.data_serie), 1)
            params['arg1'] = sup.ffloat(np.var(self.data_serie), 1)
            params['arg2'] = 0
        elif dname == 'expon':
            # for effects of the XML arg1=0 and arg2=0
            params['mean'] = 0
            params['arg1'] = sup.ffloat(np.mean(self.data_serie), 1)
            params['arg2'] = 0
        elif dname == 'uniform':
            # for effects of the XML the mean is always 3600,
            # min = arg1 and max = arg2
            params['mean'] = 3600
            params['arg1'] = sup.ffloat(np.min(self.data_serie), 1)
            params['arg2'] = sup.ffloat(np.max(self.data_serie), 1)
        elif dname == 'triang':
            # for effects of the XML the mode is stored in the mean parameter,
            # min = arg1 and max = arg2
            params['mean'] = sup.ffloat(st.mode(self.data_serie).mode[0], 1)
            params['arg1'] = sup.ffloat(np.min(self.data_serie), 1)
            params['arg2'] = sup.ffloat(np.max(self.data_serie), 1)
        return params
