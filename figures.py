# -*- coding: utf-8 -*-
"""
Created on Wed May  8 10:31:58 2019

@author: Manuel Camargo
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os

file1 = 'Production_20190426_153607417088.csv'
file2 = 'ConsultaDataMining201618_20190427_082803935907.csv'
file3 = 'PurchasingExample_20190425_165334317107.csv'

df1 = pd.read_csv(os.path.join('outputs', file1))
df2 = pd.read_csv(os.path.join('outputs', file2))
df3 = pd.read_csv(os.path.join('outputs', file3))


similarity = lambda x: 1 - x['loss']
df1['similarity'] = df1.apply(similarity, axis=1)
df2['similarity'] = df2.apply(similarity, axis=1)
df3['similarity'] = df3.apply(similarity, axis=1)

fig, [ax1, ax2, ax3] = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), subplot_kw={'projection': '3d'})

#----fig1----
s1 = df1[ df1.alg_manag == 'trace_alignment']
s2 = df1[ df1.alg_manag == 'replacement']
s3 = df1[ df1.alg_manag == 'removal']
ax1.scatter(s1.epsilon, s1.eta, s1.similarity, c='blue', label='trace_alignment')
ax1.scatter(s2.epsilon, s2.eta, s2.similarity, c='red', label='replacement')
ax1.scatter(s3.epsilon, s3.eta, s3.similarity, c='green', label='removal')
ax1.set_xlabel('epsilon')
ax1.set_ylabel('eta')
ax1.set_zlabel('similarity')
ax1.set_title("MP process")

#----fig2----
s1 = df2[ df2.alg_manag == 'trace_alignment']
s2 = df2[ df2.alg_manag == 'replacement']
s3 = df2[ df2.alg_manag == 'removal']
ax2.scatter(s1.epsilon, s1.eta, s1.similarity, c='blue', label='trace_alignment')
ax2.scatter(s2.epsilon, s2.eta, s2.similarity, c='red', label='replacement')
ax2.scatter(s3.epsilon, s3.eta, s3.similarity, c='green', label='removal')
ax2.set_xlabel('epsilon')
ax2.set_ylabel('eta')
ax2.set_zlabel('similarity')
ax2.set_title("AC process")

#----fig3----
s1 = df3[ df3.alg_manag == 'trace_alignment']
s2 = df3[ df3.alg_manag == 'replacement']
s3 = df3[ df3.alg_manag == 'removal']
ax3.scatter(s1.epsilon, s1.eta, s1.similarity, c='blue', label='trace_alignment')
ax3.scatter(s2.epsilon, s2.eta, s2.similarity, c='red', label='replacement')
ax3.scatter(s3.epsilon, s3.eta, s3.similarity, c='green', label='removal')
ax3.set_xlabel('epsilon')
ax3.set_ylabel('eta')
ax3.set_zlabel('similarity')
ax3.set_title("Purchase-to-pay process")

handles, labels = ax2.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper left')
plt.tight_layout()
plt.show()