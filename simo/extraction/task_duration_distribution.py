# -*- coding: utf-8 -*-
# DISTRIBUTIONS = [
	# st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,
	# st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
	# st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,st.genexpon,
	# st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
	# st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
	# st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,st.levy,st.levy_l,st.levy_stable,
	# st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,
	# st.nct,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,
	# st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
	# st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy
# ]

import warnings
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd
import scipy.stats as st
import numpy as np
import matplotlib
import support as sup

# -- Distribution Graphs --
def dis_graphics(data, best_fit_name, best_fir_paramms, dataYLim, ax, bins):
	"""create comparison graphics"""
	matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
	matplotlib.style.use('ggplot')
	best_dist = getattr(st, best_fit_name)
	# Update plots
	ax.set_ylim(dataYLim)
	ax.set_title(u'Data.\n All Fitted Distributions')
	ax.set_xlabel(u'Time (Secs)')
	ax.set_ylabel('Frequency')
	# Make PDF
	pdf = make_pdf(best_dist, best_fir_paramms)
	# Display
	plt.figure(figsize=(12,8))
	ax = pdf.plot(lw=2, label= best_fit_name + ' (Best fit)', legend=True)
	data.plot(kind='hist', bins=bins, normed=True, alpha=0.5, ax=ax)
	param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
	param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_fir_paramms)])
	dist_str = '{}({})'.format(best_fit_name, param_str)
	ax.set_title(u'Data. with best fit distribution \n' + dist_str)
	ax.set_xlabel(u'Time (Secs)')
	ax.set_ylabel('Frequency')
	plt.show()

def make_pdf(dist, params, size=10000):
	"""Generate distributions's Propbability Distribution Function """
	# Separate parts of parameters
	arg = params[:-2]
	loc = params[-2]
	scale = params[-1]
	# Get sane start and end points of distribution
	start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
	end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)
	# Build PDF and turn into pandas Series
	x = np.linspace(start, end, size)
	y = dist.pdf(x, loc=loc, scale=scale, *arg)
	pdf = pd.Series(y, x)
	return pdf

# -- Find best distribution --
def dist_best(series, graph, bins,bimp):
	"""Calculate the best probability distribution for a given data serie"""
	# Create a data series from the given list
	data = pd.Series(series)
	# Plot for comparison (optional)
	if(graph==True):
		print(data)
		plt.figure(figsize=(12,8))
		ax = data.plot(kind='hist', bins=bins, normed=True, alpha=0.5, color='g')
		# Save plot limits
		dataYLim = ax.get_ylim()
		# Find best fit distribution
		best_fit_name, best_fir_paramms = best_fit_distribution(data,bimp, bins, ax)
		# plot the data
		dis_graphics(data, best_fit_name, best_fir_paramms, dataYLim, ax, bins)
	else:
		# Find best fit distribution whitout plot
		best_fit_name, best_fir_paramms = best_fit_distribution(data,bimp, bins)
	return best_fit_name

# Create models from data
def best_fit_distribution(data,bimp, bins=50, ax=None):
	"""Model data by finding best fit distribution to data"""
	# Get histogram of original data
	y, x = np.histogram(data, bins=bins, density=True)
	x = (x + np.roll(x, -1))[:-1] / 2.0
	# Distributions to check
	DISTRIBUTIONS_BIMP = [st.norm,st.expon,st.uniform,st.triang,st.lognorm,st.gamma]
	DISTRIBUTIONS_SCYLLA = [st.uniform, st.norm, st.triang, st.expon]
	if(bimp):
		DISTRIBUTIONS = DISTRIBUTIONS_BIMP
	else:
		DISTRIBUTIONS = DISTRIBUTIONS_SCYLLA
	# Best holders
	best_distribution = st.norm
	best_params = (0.0, 1.0)
	best_sse = np.inf
	# Estimate distribution parameters from data
	for distribution in DISTRIBUTIONS:
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
				# if axis pass in add to plot
				try:
					if ax:
						pd.Series(pdf, x).plot(label=distribution.name, legend=True,ax=ax)
				except Exception:
					pass
				# identify if this distribution is better
				if best_sse > sse > 0:
					best_distribution = distribution
					best_params = params
					best_sse = sse
		except Exception:
			pass
	return (best_distribution.name, best_params)

def dist_params(dname, task_data,bimp):
	"""calculate additional parameters once the probability distribution is found"""
	params = dict()
	if bimp:
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
	else:
		#In order that Scylla simulator works, the param number can't be doubles.
		if dname == 'norm':
			# for effects of the XML arg1=std and arg2=0
			params = dict(mean=(np.mean(task_data)), arg1=(np.std(task_data)), arg2=0)
		elif dname == 'lognorm' or dname == 'gamma':
			# for effects of the XML arg1=var and arg2=0
			params = dict(mean=(np.mean(task_data)), arg1=(np.var(task_data)), arg2=0)
		elif dname == 'expon':
			# for effects of the XML arg1=0 and arg2=0
			params = dict(mean=(np.mean(task_data)), arg1=0, arg2=0)
		elif dname == 'uniform':
			# for effects of the XML the mean is always 3600, min = arg1 and max = arg2
			params = dict(mean=3600, arg1=(np.min(task_data)), arg2=(np.max(task_data)))
		elif dname == 'triang':
			# for effects of the XML the mode is stored in the mean parameter, min = arg1 and max = arg2
			params = dict(mean=(st.mode(task_data).mode[0]), arg1=(np.min(task_data)), arg2=(np.max(task_data)))
		return params

def dist_names(dname,bimp):
	"""transforms the name of the distribution into the XML format"""
	name = ""
	if dname=='norm':
		if bimp:
			name='NORMAL'
		else:
			name = 'normalDistribution'
	elif dname=='lognorm':
		name='LOGNORMAL'
	elif dname=='gamma':
		name='GAMMA'
	elif dname=='expon':
		if bimp:
			name='EXPONENTIAL'
		else:
			name = 'exponentialDistribution'
	elif dname=='uniform':
		if bimp:
			name='UNIFORM'
		else:
			name = 'uniformDistribution'
	elif dname=='triang':
		if bimp:
			name='TRIANGULAR'
		else:
			name = 'triangularDistribution'
	elif dname=='fixed':
		if bimp:
			name='FIXED'
		else:
			name = 'constantDistribution'
	elif dname == 'erlang':
		name = 'erlangDistribution'
	return name

# --kernel--

def get_task_distribution(task_data,bimp, graph=False, bins=200):
	"""
	calculate the probability distribution of one task of a log
		parameters:
		task_data: data of time delta for one task
		graph: activate the comparision graphs of the process
	"""
	# if all the values of the serie are equal is an automatic activity with fixed distribution
	if not task_data:
		dname = 'fixed'
		dparams = dict(mean=0,arg1=0, arg2=0)
	else:
		if np.min(task_data)==np.max(task_data):
			dname = 'fixed'
			dparams = dict(mean=int(np.min(task_data)),arg1=0, arg2=0)
		else:
			dname = dist_best(task_data, graph, bins,bimp)
			dparams =  dist_params(dname, task_data,bimp)
	return dict(dname=dist_names(dname,bimp),dparams=dparams)
