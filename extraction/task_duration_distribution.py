# -*- coding: utf-8 -*-
import warnings
#import seaborn as sns; sns.set()
import pandas as pd
import scipy.stats as st
import numpy as np
from support_modules import support as sup


def get_task_distribution(task_data, graph=False, bins=200):
    """Calculate the probability distribution of one task of a log
        parameters:
        task_data: data of time delta for one task
        graph: activate the comparision graphs of the process
    """
    dist = {'norm':'NORMAL', 'lognorm':'LOGNORMAL', 'gamma':'GAMMA', 'expon':'EXPONENTIAL',
            'uniform':'UNIFORM', 'triang':'TRIANGULAR', 'fixed':'FIXED'}

    if not task_data:
        dname = 'fixed'
        dparams = dict(mean=0,arg1=0, arg2=0)
    else:
        # if all the values of the serie are equal is an automatic activity with fixed distribution
        if np.min(task_data)==np.max(task_data):
            dname = 'fixed'
            dparams = dict(mean=int(np.min(task_data)),arg1=0, arg2=0)
        else:
            dname = dist_best(task_data, bins)
            dparams =  dist_params(dname, task_data)
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


def dist_params(dname, task_data):
    """calculate additional parameters once the probability distribution is found"""
    params = dict()
    if dname=='norm':
        #for effects of the XML arg1=std and arg2=0
        params=dict(mean=sup.ffloat(np.mean(task_data),1), arg1=sup.ffloat(np.std(task_data),1), arg2=0)
    elif dname=='lognorm' or dname=='gamma':
        #for effects of the XML arg1=var and arg2=0
        params=dict(mean=sup.ffloat(np.mean(task_data),1), arg1=sup.ffloat(np.var(task_data),1), arg2=0)
    elif dname=='expon':
        #for effects of the XML arg1=0 and arg2=0
        params=dict(mean=0, arg1=sup.ffloat(np.mean(task_data),1), arg2=0)
    elif dname=='uniform':
        #for effects of the XML the mean is always 3600, min = arg1 and max = arg2
        params=dict(mean=3600, arg1=sup.ffloat(np.min(task_data),1), arg2=sup.ffloat(np.max(task_data),1))
    elif dname=='triang':
        #for effects of the XML the mode is stored in the mean parameter, min = arg1 and max = arg2
        params=dict(mean=sup.ffloat(st.mode(task_data).mode[0],1), arg1=sup.ffloat(np.min(task_data),1), arg2=sup.ffloat(np.max(task_data),1))
    return params