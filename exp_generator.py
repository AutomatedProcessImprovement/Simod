# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 15:04:49 2021

@author: Manuel Camargo
"""
import os
import time
import utils.support as sup


# =============================================================================
#  Support
# =============================================================================

def create_file_list(path):
    file_list = list()
    for root, dirs, files in os.walk(path):
        for f in files:
            file_list.append(f)
    return file_list

# =============================================================================
# Sbatch files creator
# =============================================================================


def sbatch_creator(log, miner):
    exp_name = (os.path.splitext(log)[0]
                    .lower()
                    .split(' ')[0][:5])
    if imp == 2:
        default = ['#!/bin/bash',
                   '#SBATCH --partition=gpu',
                   '#SBATCH --gres=gpu:tesla:1',
                   '#SBATCH -J ' + exp_name,
                   '#SBATCH -N 1',
                   '#SBATCH --cpus-per-task=20',
                   '#SBATCH --mem=32000',
                   '#SBATCH -t 120:00:00',
                   'export DISPLAY='+ip_num,
                   'module load jdk-1.8.0_25',
                   'module load python/3.6.3/virtenv',
                   'source activate deep_sim3',
                   ]
    else:
        default = ['#!/bin/bash',
                   '#SBATCH --partition=main',
                   '#SBATCH -J '+exp_name,
                   '#SBATCH -N 1',
                   '#SBATCH --cpus-per-task=20',
                   '#SBATCH --mem=32000',
                   '#SBATCH -t 120:00:00',
                   'export DISPLAY='+ip_num,
                   'module load jdk-1.8.0_25',
                   'module load python/3.6.3/virtenv',
                   'source activate deep_sim3',
                   ]

        options = 'python simod_optimizer.py -f ' + log
        options += ' -m '+miner
        
    default.append(options)
    file_name = sup.folder_id()
    sup.create_text_file(default, os.path.join(output_folder, file_name))
    
# =============================================================================
# Sbatch files submission
# =============================================================================

def sbatch_submit(in_batch, bsize=20):
    file_list = create_file_list(output_folder)
    print('Number of experiments:', len(file_list), sep=' ')
    for i, _ in enumerate(file_list):
        if in_batch:
            if (i % bsize) == 0:
                time.sleep(20)
                os.system('sbatch '+os.path.join(output_folder, file_list[i]))
            else:
                os.system('sbatch '+os.path.join(output_folder, file_list[i]))
        else:
            os.system('sbatch '+os.path.join(output_folder, file_list[i]))

# =============================================================================
# Kernel
# =============================================================================


# create output folder
output_folder = 'jobs_files'
# Xserver ip
ip_num = '172.17.37.49:0.0'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
# clean folder
for _, _, files in os.walk(output_folder):
    for file in files:
        os.unlink(os.path.join(output_folder, file))

# parameters definition
imp = 1  # keras lstm implementation 1 cpu, 2 gpu
logs = [('BPI_Challenge_2012_W_Two_TS.xes', 'sm3'),
        ('BPI_Challenge_2017_W_Two_TS.xes', 'sm3'),
        ('PurchasingExample.xes', 'sm3'),
        ('Production.xes', 'sm3'),
        ('ConsultaDataMining201618.xes', 'sm3'),
        ('insurance.xes', 'sm2'),
        ('callcentre.xes', 'sm3'),
        ('poc_processmining.xes', 'sm3')]

for log, miner in logs:
    # sbatch creation
    sbatch_creator(log, miner)
# submission
# sbatch_submit(False)
