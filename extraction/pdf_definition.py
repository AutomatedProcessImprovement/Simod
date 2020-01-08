# -*- coding: utf-8 -*-
import warnings
#import seaborn as sns; sns.set()
import pandas as pd
import scipy.stats as st
import numpy as np
from support_modules import support as sup


def get_task_distribution(data_serie, bins=200):
    """Calculate the probability distribution of a series of data
        parameters:
        data_serie: data of time deltas
    """
    dist = {'norm':'NORMAL', 'lognorm':'LOGNORMAL', 'gamma':'GAMMA', 'expon':'EXPONENTIAL',
            'uniform':'UNIFORM', 'triang':'TRIANGULAR', 'fixed':'FIXED'}

    if not data_serie:
        dname = 'fixed'
        dparams = dict(mean=0,arg1=0, arg2=0)
    else:
        # If all the values of the serie are equal is an automatic activity with fixed distribution
        if np.min(data_serie)==np.max(data_serie):
            dname = 'fixed'
            dparams = dict(mean=int(np.min(data_serie)),arg1=0, arg2=0)
        elif len(data_serie) < 100:
            dname = 'expon'
            dparams = dict(mean=0, arg1=sup.ffloat(np.mean(data_serie),1), arg2=0)
        else:
            dname = dist_best(data_serie, bins)
            dparams =  dist_params(dname, data_serie)
    return dict(dname=dist[dname], dparams=dparams)

# -- Find best distribution --
def dist_best(series, bins):
    """Calculate the best probability distribution for a given data serie"""
    # Create a data series from the given list
    data = pd.Series(series)
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0
    # Distributions to check
    distributions = [st.norm,st.expon,st.uniform,st.triang,st.lognorm,st.gamma]
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
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))
                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_sse = sse
        except Exception:
            pass
    return best_distribution.name


def dist_params(dname, data_serie):
    """calculate additional parameters once the probability distribution is found"""
    params = dict()
    if dname=='norm':
        #for effects of the XML arg1=std and arg2=0
        params=dict(mean=sup.ffloat(np.mean(data_serie),1), arg1=sup.ffloat(np.std(data_serie),1), arg2=0)
    elif dname=='lognorm' or dname=='gamma':
        #for effects of the XML arg1=var and arg2=0
        params=dict(mean=sup.ffloat(np.mean(data_serie),1), arg1=sup.ffloat(np.var(data_serie),1), arg2=0)
    elif dname=='expon':
        #for effects of the XML arg1=0 and arg2=0
        params=dict(mean=0, arg1=sup.ffloat(np.mean(data_serie),1), arg2=0)
    elif dname=='uniform':
        #for effects of the XML the mean is always 3600, min = arg1 and max = arg2
        params=dict(mean=3600, arg1=sup.ffloat(np.min(data_serie),1), arg2=sup.ffloat(np.max(data_serie),1))
    elif dname=='triang':
        #for effects of the XML the mode is stored in the mean parameter, min = arg1 and max = arg2
        params=dict(mean=sup.ffloat(st.mode(data_serie).mode[0],1), arg1=sup.ffloat(np.min(data_serie),1), arg2=sup.ffloat(np.max(data_serie),1))
    return params