# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 15:25:26 2020

@author: Manuel Camargo
"""
import tkinter as tk
import os
from tkinter import Frame, Button, Label, Entry, messagebox, ttk
from tkinter import filedialog, Scale, LabelFrame
from support_modules import support as sup
import simod as sim
import subprocess


class SimodWindow(Frame):
    def __init__(self, master):
        Frame.__init__(self, master=None)
        self.master.title("Simod")
        self.tabControl = ttk.Notebook(root) 
          
        tab1 = ttk.Frame(self.tabControl, width=450, height=450) 
        tab2 = ttk.Frame(self.tabControl, width=450, height=450) 
        tab3 = ttk.Frame(self.tabControl, width=450, height=450) 
        
        self.tabControl.add(tab1, text ='General') 
        self.tabControl.add(tab2, text ='Single', state='disable')
        self.tabControl.add(tab3, text ='Optimizer', state='disable')
        self.tabControl.pack(expand = 1, fill ="both") 

        self.settings = dict()
        self.args = dict()
        self.validated = False
        self.define_general_settings()
        
        self.general_form = list()
        self.single_form = list()
        self.opt_form = list()
        self.create_general_form(tab1)
        self.create_single_form(tab2)
        self.create_optimizer_form(tab3)

       
    def create_general_form(self, tab):
        form = Frame(tab)
        center = LabelFrame(form, padx=90, pady=30, width=450, height=250)
        btm_frame = Frame(form, pady=3, width=450, height=40)      
        
        # layout all of the main containers
        form.pack(fill=tk.BOTH, expand=True)
        center.pack(fill=tk.BOTH, expand=True)
        btm_frame.pack(side=tk.RIGHT, padx=5, pady=5)
        
        center.grid_propagate(False)
        btm_frame.pack_propagate(False)
        center.grid_rowconfigure(0, pad=20)
        center.grid_rowconfigure(1, pad=20)
        center.grid_columnconfigure(1, weight=2, pad=20)
        
        en_file = Entry(center, state='disabled', width=200) 
        en_file.grid(row=0, column=1, columnspan=2, padx=5, sticky='W')
        # Center elements
        def open_file():
            file = filedialog.askopenfilename(
                initialdir = os.path.join(os.getcwd(), 'inputs'),
                title = "Select file",
                filetypes = (("csv files","*.csv"),
                             ("XES files","*.xes"),
                             ("all files","*.*")))
            en_file.config(state='normal')
            en_file.insert(0, os.path.basename(file))
            en_file.config(state='disabled')
        b_select = Button(center, text="Select event-log", command=open_file)
        b_select.grid(row=0, sticky='W')
        self.general_form.append({'name': 'file', 'obj': en_file})


        lb_exec_mode = Label(center, text ="Exec. Mode: ")
        lb_exec_mode.grid(row=1, sticky='W')
        cb_exec_mode = ttk.Combobox(center)
        cb_exec_mode.set('single')
        cb_exec_mode['values'] = ('single', 'optimizer')
        cb_exec_mode.grid(row=1, column=1, padx=5, sticky='W')
        self.general_form.append({'name': 'exec_mode', 'obj': cb_exec_mode})
        
        lb_eval_metric = Label(center, text ="Evaluation metric: ")
        lb_eval_metric.grid(row=2, sticky='W')
        cb_eval_metric = ttk.Combobox(center)
        cb_eval_metric.set('tsd')
        cb_eval_metric['values'] = ('tsd', 'dl_mae', 'tsd_min', 'mae')
        cb_eval_metric.grid(row=2, column=1, padx=5, sticky='W')        
        self.general_form.append({'name': 'sim_metric', 'obj': cb_eval_metric})
        
        lb_rep = Label(center, text ="Repetitions: ")
        lb_rep.grid(row=3, sticky='W')
        var = tk.IntVar()
        sl_rep = Scale(center, variable=var,
                          from_=1, to=30, resolution=1,
                          orient = tk.HORIZONTAL)
        sl_rep.grid(row=3, column=1, padx=5, sticky='W')
        self.general_form.append({'name': 'repetitions', 'obj': sl_rep})
        # Bottom elements
        b_ok = Button(btm_frame, text="Next", command=self.next_tab)
        b_ok.grid(row=0)
        form.pack(side = tk.TOP, fill = tk.X, padx = 5 , pady = 5)
        return form
        
    def create_single_form(self, tab):
        form = Frame(tab)
        center = LabelFrame(form, width=450, height=250, padx=50, pady=20)
        btm_frame = Frame(form, width=450, height=45, pady=3)        
        
        # layout all of the main containers
        form.pack(fill=tk.BOTH, expand=True)
        center.pack(fill=tk.BOTH, expand=True)
        btm_frame.pack(side=tk.RIGHT, padx=5, pady=5)
        
        center.grid_propagate(False)
        btm_frame.pack_propagate(False)
        center.grid_rowconfigure(5, pad=30)

        
        lb_eta = Label(center, text ="Percentile for frequency (Eta): ")
        lb_eta.grid(row=0, sticky='W')
        var = tk.DoubleVar()
        sl_eta = Scale(center, variable=var,
                          from_=0, to=1, resolution=0.01,
                          orient = tk.HORIZONTAL, length=140)
        sl_eta.set(0.4)
        sl_eta.grid(row=0, column=1, columnspan=2, padx=5, sticky='W')
        self.single_form.append({'name': 'eta', 'obj': sl_eta})
        
        lb_epsilon = Label(center, text ="Parallelism (Epsilon): ")
        lb_epsilon.grid(row=1, sticky='W')
        var2 = tk.DoubleVar()
        sl_epsilon = Scale(center, variable=var2,
                          from_=0, to=1, resolution=0.01,
                          orient = tk.HORIZONTAL, length=140)
        sl_epsilon.set(0.1)
        sl_epsilon.grid(row=1, column=1, columnspan=2, padx=5, sticky='W')
        self.single_form.append({'name': 'epsilon', 'obj': sl_epsilon})
        
        lb_gate = Label(center, text ="Non-conformances management: ")
        lb_gate.grid(row=2, sticky='W')
        cb_gate = ttk.Combobox(center)
        cb_gate.set('removal')
        cb_gate['values'] = ('removal', 'replacement', 'repair')
        cb_gate.grid(row=2, column=1, padx=5, sticky='W')
        self.single_form.append({'name': 'alg_manag', 'obj': cb_gate})

        lb_pool = Label(center, text ="Res. Pool Sim: ")
        lb_pool.grid(row=3, sticky='W')
        var3 = tk.DoubleVar()
        sl_pool = Scale(center, variable=var3,
                          from_=0, to=1, resolution=0.01,
                          orient = tk.HORIZONTAL, length=140)
        sl_pool.set(0.85)
        sl_pool.grid(row=3, column=1, columnspan=2, padx=5, sticky='W')
        self.single_form.append({'name': 'rp_similarity', 'obj': sl_pool})

        lb_gate = Label(center, text ="Gateways discovery: ")
        lb_gate.grid(row=4, sticky='W')
        cb_gate = ttk.Combobox(center)
        cb_gate.set('discovery')
        cb_gate['values'] = ('discovery', 'random', 'equiprobable')
        cb_gate.grid(row=4, column=1, padx=5, sticky='W')
        self.single_form.append({'name': 'gate_management', 'obj': cb_gate})

      
        var4 = tk.BooleanVar()
        ck_semi= ttk.Checkbutton(center, variable=var4,
                                 text="Semi-automatic")
        ck_semi.grid(row=5, column=0, padx=5, sticky='W')
        self.single_form.append({'name': 'pdef_method', 'obj': var4})
        
        # Bottom elements
        b_ok = Button(btm_frame, text="Execute", command=self.execute_single)
        b_ok.grid(row=0, sticky='W')
        b_cancel = Button(btm_frame, text="Back", command=self.back)
        b_cancel.grid(row=0, column=1, sticky='W')
        
        form.pack(side = tk.TOP, fill = tk.X, padx = 5 , pady = 5)
        return form


    def create_optimizer_form(self, tab):
        form = Frame(tab)
        center = LabelFrame(form, width=450, height=250, padx=50, pady=10)
        btm_frame = Frame(form, width=450, height=45, pady=3)        
        
        # layout all of the main containers
        form.pack(fill=tk.BOTH, expand=True)
        center.pack(fill=tk.BOTH, expand=True)
        btm_frame.pack(side=tk.RIGHT, padx=5, pady=5)
        
        center.grid_propagate(False)
        btm_frame.pack_propagate(False)
        
        lb_eta = Label(center, text ="Eta: ")
        varmin = tk.DoubleVar()
        varmax = tk.DoubleVar()
        sl_eta_min = tk.Scale(center, variable=varmin,
                          from_=0, to=1, resolution=0.01,
                          orient = tk.HORIZONTAL, label='min')
        sl_eta_max = tk.Scale(center, variable=varmax,
                          from_=0, to=1, resolution=0.01,
                          orient = tk.HORIZONTAL, label='max')
        sl_eta_max.set(1)
        lb_eta.grid(row=0, sticky='W')
        sl_eta_min.grid(row=0, column=1, padx=5, sticky='W')
        sl_eta_max.grid(row=0, column=2, padx=5, sticky='W')
        self.opt_form.append({'name': 'eta_min', 'obj': sl_eta_min})
        self.opt_form.append({'name': 'eta_max', 'obj': sl_eta_max})


        lb_epsilon = Label(center, text ="Epsilon: ")
        varmin = tk.DoubleVar()
        varmax = tk.DoubleVar()
        sl_epsilon_min = tk.Scale(center, variable=varmin,
                          from_=0, to=1, resolution=0.01,
                          orient = tk.HORIZONTAL, label='min')
        sl_epsilon_max = tk.Scale(center, variable=varmax,
                          from_=0, to=1, resolution=0.01,
                          orient = tk.HORIZONTAL, label='max')
        sl_epsilon_max.set(1)
        lb_epsilon.grid(row=1, sticky='W')
        sl_epsilon_min.grid(row=1, column=1, padx=5, sticky='W')
        sl_epsilon_max.grid(row=1, column=2, padx=5, sticky='W')
        self.opt_form.append({'name': 'epsilon_min', 'obj': sl_eta_min})
        self.opt_form.append({'name': 'epsilon_max', 'obj': sl_eta_max})
        
        lb_rpool = Label(center, text ="Res. Pool Sim: ")
        varmin = tk.DoubleVar()
        varmax = tk.DoubleVar()
        sl_rpool_min = tk.Scale(center, variable=varmin,
                          from_=0, to=1, resolution=0.01,
                          orient = tk.HORIZONTAL, label='min')
        sl_rpool_max = tk.Scale(center, variable=varmax,
                          from_=0, to=1, resolution=0.01,
                          orient = tk.HORIZONTAL, label='max')
        sl_rpool_min.set(0.5)
        sl_rpool_max.set(0.9)
        lb_rpool.grid(row=2, sticky='W')
        sl_rpool_min.grid(row=2, column=1, padx=5, sticky='W')
        sl_rpool_max.grid(row=2, column=2, padx=5, sticky='W')
        self.opt_form.append({'name': 'rpool_min', 'obj': sl_rpool_min})
        self.opt_form.append({'name': 'rpool_max', 'obj': sl_rpool_max})

        
        lb_eval = Label(center, text ="Max. Evaluations: ")
        lb_eval.grid(row=3, sticky='W')
        var3 = tk.IntVar()
        sl_eval = tk.Scale(center, variable=var3,
                          from_=1, to=50, resolution=1,
                          orient = tk.HORIZONTAL, length=215)
        sl_eval.grid(row=3, column=1, columnspan=2, padx=5, sticky='W')
        self.opt_form.append({'name': 'max_eval', 'obj': sl_eval})

        
        # Bottom elements
        b_ok = Button(btm_frame, text="Execute", command=self.execute_opt)
        b_ok.grid(row=0, sticky='W')
        b_cancel = Button(btm_frame, text="Back", command=self.back)
        b_cancel.grid(row=0, column=1, sticky='W')
        
        form.pack(side = tk.TOP, fill = tk.X, padx = 5 , pady = 5)
        return form


    def execute_single(self, event=None):
        self.settings['temp_file'] = sup.file_id(prefix='SE_')
        self.settings['simulation'] = True
        for obj in self.general_form:
            self.settings[obj['name']] = obj['obj'].get()
        for obj in self.single_form:
            if obj['name'] == 'pdef_method':
                self.settings[obj['name']] = ('semi-automatic' 
                                              if obj['obj'].get() else 
                                              'automatic')
            else:
                self.settings[obj['name']] = obj['obj'].get()
        self.validated = True
        self.master.destroy()
        
    def execute_opt(self, event=None):
        
        def val_inter(var):
            if self.args[var][0] >= self.args[var][1]:
                tk.messagebox.showerror(
                    title='Validation error',
                    message=('the minimim '+ var +
                             ' must be bigger than the maximum'))
                return False
            else:
                return True
            
        self.args = dict()
        self.args['eta'] = [
           list(filter(lambda x: x['name']=='eta_min', 
                                self.opt_form))[0]['obj'].get(),
           list(filter(lambda x: x['name']=='eta_max', 
                                self.opt_form))[0]['obj'].get()]
        self.args['epsilon'] = [
           list(filter(lambda x: x['name']=='epsilon_min', 
                                self.opt_form))[0]['obj'].get(),
           list(filter(lambda x: x['name']=='epsilon_max', 
                                self.opt_form))[0]['obj'].get()]
        self.args['max_eval'] = list(filter(
            lambda x: x['name']=='max_eval', self.opt_form))[0]['obj'].get()
        self.args['rp_similarity'] = [
           list(filter(lambda x: x['name']=='rpool_min', 
                                self.opt_form))[0]['obj'].get(),
           list(filter(lambda x: x['name']=='rpool_max', 
                                self.opt_form))[0]['obj'].get()]
        self.args['gate_management'] = ['discovery', 'random', 'equiprobable']
        for obj in self.general_form:
            self.settings[obj['name']] = obj['obj'].get()
        
        self.settings['simulation'] = True
        self.settings['temp_file'] = sup.file_id(prefix='OP_')
        self.settings['pdef_method'] = 'automatic'
        if val_inter('eta') and val_inter('epsilon') and val_inter('rp_similarity'):
            self.validated = True
            self.master.destroy()
 
    def back(self, event=None):
        self.tabControl.tab('current', state='disable')
        self.tabControl.tab(0, state='normal')
        self.tabControl.select(0)
        
    def next_tab(self, event=None):
        file = list(filter(lambda x: x['name']=='file', 
                                self.general_form))[0]['obj'].get()
        if file:
            exec_mode = list(filter(lambda x: x['name']=='exec_mode', 
                                    self.general_form))[0]['obj'].get()
            if exec_mode == 'single':
                self.tabControl.tab(1, state='normal')
                self.tabControl.select(1)
            else:
                self.tabControl.tab(2, state='normal')
                self.tabControl.select(2)
            self.tabControl.tab(0, state='disable')
        else:
            tk.messagebox.showerror(
                title='Required field',
                message='You must select an event log to proceed')

        
    def cancel(self, event=None):
        self.master.destroy()

    def define_general_settings(self):
        """ Sets the app general settings"""
        column_names = {'Case ID': 'caseid', 'Activity': 'task',
                        'lifecycle:transition': 'event_type', 'Resource': 'user'}
        # Event-log reading options
        self.settings['read_options'] = {'timeformat': '%Y-%m-%dT%H:%M:%S.%f',
                                    'column_names': column_names,
                                    'one_timestamp': False,
                                    'filter_d_attrib': True,
                                    'ns_include': True}
        # Folders structure
        self.settings['input'] = 'inputs'
        self.settings['output'] = os.path.join('outputs', sup.folder_id())
        # External tools routes
        self.settings['miner_path'] = os.path.join('external_tools',
                                              'splitminer',
                                              'splitminer.jar')
        self.settings['bimp_path'] = os.path.join('external_tools',
                                             'bimp',
                                             'qbp-simulator-engine.jar')
        self.settings['align_path'] = os.path.join('external_tools',
                                              'proconformance',
                                              'ProConformance2.jar')
        self.settings['aligninfo'] = os.path.join(self.settings['output'],
                                             'CaseTypeAlignmentResults.csv')
        self.settings['aligntype'] = os.path.join(self.settings['output'],
                                             'AlignmentStatistics.csv')


if __name__ == "__main__":
    root = tk.Tk()
    window = SimodWindow(root)
    root.mainloop()
    settings = window.settings
    if window.validated:
        if 'exec_mode' in settings.keys() and settings['exec_mode'] == 'single':
            print(settings)
            simod = sim.Simod(settings)
            simod.execute_pipeline(settings['exec_mode'])
        elif 'exec_mode' in settings.keys() and settings['exec_mode'] == 'optimizer':
            args = window.args
            print(args)
            print(settings)
            # Execute optimizer
            if not os.path.exists(os.path.join('outputs',
                                               settings['temp_file'])):
                open(os.path.join('outputs',
                                  settings['temp_file']), 'w').close()
            # start monitor
            var = ['python', 'simod_figs.py', 
                    '-f',settings['temp_file'], '-e', str(args['max_eval'])]
            subprocess.Popen(var)
            # optimizer
            optimizer = sim.DiscoveryOptimizer(settings, args)
            optimizer.execute_trials()
