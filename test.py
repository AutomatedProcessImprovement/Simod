# -*- coding: utf-8 -*-
"""
Created on Mon May 13 16:18:33 2019

@author: Manuel Camargo
"""

from support_modules.readers import log_reader_test as lr

input_file='C:\\Users\\Manuel Camargo\\Documents\\Repositorio\\experiments\\sc_simo\\inputs\\PurchasingExample.xes.gz'
log = lr.LogReader(input_file, '%Y-%m-%dT%H:%M:%S.000')
