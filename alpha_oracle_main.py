# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 18:03:14 2020

@author: Manuel Camargo
"""

import os
import pandas as pd

from support_modules.readers import log_reader as lr
from support_modules.analyzers import alpha_oracle as ao

def load_event_log(parms):
    # Dataframe creation
    # Filter load local inter-case features or filter them
    log = lr.LogReader(os.path.join('inputs', parms['file_name']), parms['read_options'])
    log_df = pd.DataFrame(log.data)
    ao.discover_concurrency(log_df, parms['read_options'])

# =============================================================================
# Kernel
# This will be removed after integration
# The next parameters need to be defined to perform the query
# =============================================================================

parameters = dict()

# Matching between the csv columns and the event log
column_names = {'Case ID':'caseid', 'Activity':'task',
                'lifecycle:transition':'event_type', 'Resource':'user'}
# Parameters for event-log reading
parameters['read_options'] = {'timeformat': '%Y-%m-%dT%H:%M:%S.%f',
                              'column_names':column_names,
                              'one_timestamp': False,
                              'reorder':False,
                              'ns_include':True,
                              'filter_d_attrib':True}
parameters['file_name'] = 'PurchasingExample.csv'

load_event_log(parameters)