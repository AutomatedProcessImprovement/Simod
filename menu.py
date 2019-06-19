# -*- coding: utf-8 -*-
# some handy functions to use along widgets
from IPython.display import clear_output, display
import ipywidgets as widgets
import threading
import time
from ipywidgets import interact, interactive, fixed, interact_manual

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import numpy as np
import pandas as pd
import os
from os import walk

from support_modules import support as sup
import simod as sim

# General components
ON_EXECUTION = False



_, _, files = next(walk('inputs'))

txt_eventlog = widgets.Dropdown(
       options=files,
       value= files[0],
       layout={'width':'95%'})
eventlog = widgets.VBox([widgets.Label('Event Log:', layout={'width':'95%'}),
                         txt_eventlog], layout={'align_items':'stretch'})

dr_exec_mode = widgets.Dropdown(
       options=['single', 'optimizer'],
       value='single',
       layout={'width':'95%'})

exec_mode = widgets.VBox([widgets.Label('Execution mode:', layout={'width':'95%'}),
                         dr_exec_mode], layout={'align_items':'stretch'})

bt_next = widgets.Button(description='Next')

# Simple components
sl_eta = widgets.FloatSlider(
         value=1,
         min=0.0,
         max=1.0,
         step=0.01,
         disabled=True)

eta = widgets.VBox([widgets.Label('Frequency threshold (eta):', layout={'width':'95%'}),
                         sl_eta], layout={'align_items':'stretch'})

sl_epsilon = widgets.FloatSlider(
         value=1,
         min=0.0,
         max=1.0,
         step=0.01,
         disabled=True)

epsilon = widgets.VBox([widgets.Label('Parallelism threshold (epsilon):', layout={'width':'95%'}),
                         sl_epsilon], layout={'align_items':'stretch'})

sl_alg_manag = widgets.Dropdown(
       options=['removal', 'replacement', 'repairment'],
       value='removal',
       layout={'width':'95%'},
       disabled=True)

alg_manag = widgets.VBox([widgets.Label('Non-conformance management:', layout={'width':'95%'}),
                          sl_alg_manag], layout={'align_items':'stretch'})

sl_rep = widgets.IntSlider(
         value=1,
         min=1,
         max=10,
         step=1,
         disabled=True)

rep = widgets.VBox([widgets.Label('Simulation runs:', layout={'width':'95%'}),
                          sl_rep], layout={'align_items':'stretch'})

bt_start_simple = widgets.Button(description='Start', disabled=True)
# Optimizer components

sl_eta_range = widgets.FloatRangeSlider(
    value=[0.5, 0.7],
    min=0,
    max=1.0,
    step=0.01,
    disabled=True)

eta_range = widgets.VBox([widgets.Label('Frequency threshold (eta):', layout={'width':'95%'}),
                          sl_eta_range], layout={'align_items':'stretch'})

sl_epsilon_range = widgets.FloatRangeSlider(
    value=[0.5, 0.7],
    min=0,
    max=1.0,
    step=0.01,
    disabled=True)

epsilon_range = widgets.VBox([widgets.Label('Parallelism threshold (epsilon):', layout={'width':'95%'}),
                          sl_epsilon_range], layout={'align_items':'stretch'})

sl_max_evals = widgets.IntSlider(
         value=1,
         min=1,
         max=100,
         step=1,
         disabled=True)

max_evals = widgets.VBox([widgets.Label('Max evaluations:', layout={'width':'95%'}),
                          sl_max_evals], layout={'align_items':'stretch'})

sl_rep_opt = widgets.IntSlider(
         value=1,
         min=1,
         max=10,
         step=1,
         disabled=True)

rep_opt = widgets.VBox([widgets.Label('Simulation runs:', layout={'width':'95%'}),
                          sl_rep_opt], layout={'align_items':'stretch'})

progress = widgets.FloatProgress(value=0.0, min=0.0, max=1.0, layout={'width':'98%'})

bt_start_opt = widgets.Button(description='Start', disabled=True)

# Output elements
out = widgets.Output(layout={'width':'65%',
                             'height':'350px',
                             'overflow_y':'auto',
                             'border':'1px solid grey'})
# Graph
graph_out=widgets.Output(layout={'width':'65%',
                             'height':'400px'})

# Results
res_out = widgets.HTML(value= '',  layout={'width':'35%',
                                           'height':'400px'})



def work(temp_file):
    global ON_EXECUTION
    file_size = os.path.getsize(os.path.join('outputs', temp_file))    
    while ON_EXECUTION:
#        time.sleep(0.5)
        new_size = os.path.getsize(os.path.join('outputs', temp_file))
        if file_size < new_size:
            file_size = new_size
            df = pd.read_csv(os.path.join('outputs', temp_file))
            similarity = lambda x: 1 - x['loss']
            df['similarity'] = df.apply(similarity, axis=1)
            update_graph(df)
            update_table(df)
            progress.value = float(len(df.index))/sl_max_evals.value
    
def change_enablement(container, state):
    for ele in container.children:
        if hasattr(ele, 'children'):
            for e in ele.children:
                e.disabled = state
        else:
            ele.disabled = state

# Events handling
def on_start_simple_clicked(_):
    # "linking function with output"
    res_out.value = ''
    with out:
        # what happens when we press the button
        clear_output()
        settings = {
                'file': txt_eventlog.value,
                'epsilon': sl_epsilon.value,
                'eta': sl_eta.value,
                'alg_manag': sl_alg_manag.value,
                'repetitions': sl_rep.value,
                'simulation': True
                }
        change_enablement(box_simple, True)
        results = sim.single_exec(settings)
    df = pd.DataFrame.from_records(results)
    df = df[df.status=='ok']
    similarity = lambda x: 1 - x['loss']
    df['similarity'] = df.apply(similarity, axis=1)
    df = df[['alg_manag','epsilon','eta','similarity']].sort_values(by=['similarity'], ascending=False)
    res_out.value = df.to_html(classes="table table-borderless table-sm",
                               float_format='%.3f',
                               border=0,
                               index=False)
    # Reactivate controls
    change_enablement(box_simple, False)


@out.capture(clear_output=True)
def on_start_opt_clicked(_):
    global ON_EXECUTION
    # "linking function with output"
    res_out.value = ''
    graph_out.clear_output()
        # what happens when we press the button
    temp_file = sup.folder_id()
    if not os.path.exists(os.path.join('outputs', temp_file)):
        open(os.path.join('outputs', temp_file), 'w').close()
    settings = {
            'file': txt_eventlog.value,
            'repetitions': sl_rep_opt.value,
            'simulation': True,
            'temp_file': temp_file
            }
    args = {'epsilon': sl_epsilon_range.value,
            'eta': sl_eta_range.value,
            'max_eval': sl_max_evals.value
            }
    ON_EXECUTION = True
    thread = threading.Thread(target=work, args=(temp_file, ))
    thread.start()
    # Deactivate controls
    change_enablement(box_opt, True)
    results, bayes_trials = sim.hyper_execution(settings, args)
    ON_EXECUTION = False
    # Reactivate controls
    change_enablement(box_opt, False)


def on_next_clicked(_):
    if dr_exec_mode.value == 'optimizer':
        tab.selected_index = 2
        change_enablement(box_simple, True)       
        change_enablement(box_opt, False)
    else:
        tab.selected_index = 1
        change_enablement(box_simple, False)       
        change_enablement(box_opt, True)

def update_table(df):
    res_out.value = ''    
    df = df[df.status=='ok']
    df = df[['alg_manag','epsilon','eta','similarity']].sort_values(by=['similarity'], ascending=False)
    res_out.value = df.to_html(classes="table table-borderless table-sm",
                               float_format='%.3f',
                               border=0,
                               index=False)


@graph_out.capture(clear_output=True)
def update_graph(df):
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.clear()
    s1 = df[ df.alg_manag == 'repairment']
    s2 = df[ df.alg_manag == 'replacement']
    s3 = df[ df.alg_manag == 'removal']
    ax.scatter(s1.epsilon, s1.eta, s1.similarity, c='blue', label='repairment')
    ax.scatter(s2.epsilon, s2.eta, s2.similarity, c='red', label='replacement')
    ax.scatter(s3.epsilon, s3.eta, s3.similarity, c='green', label='removal')
    ax.set_xlabel('epsilon')
    ax.set_ylabel('eta')
    ax.set_zlabel('similarity')
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels)
    plt.show(fig)

# Displaying components and output together
box_general = widgets.VBox([eventlog, exec_mode, bt_next])
box_simple = widgets.VBox([eta, epsilon, alg_manag, rep, bt_start_simple])
box_opt = widgets.VBox([eta_range, epsilon_range, max_evals, rep_opt, bt_start_opt])

tab = widgets.Tab([box_general, box_simple, box_opt],
                  layout={'width':'35%', 'height':'350px'})
tab.set_title(0, 'General')
tab.set_title(1, 'Simple')
tab.set_title(2, 'Optimizer')

box = widgets.HBox([tab, out])
down_box = widgets.HBox([graph_out, res_out])
frame = widgets.VBox(children=(box, progress, down_box))

# Events assignment
bt_next.on_click(on_next_clicked)
bt_start_simple.on_click(on_start_simple_clicked)
bt_start_opt.on_click(on_start_opt_clicked)
