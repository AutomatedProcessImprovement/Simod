# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 16:11:38 2020

@author: Manuel Camargo
"""
import os
import sys
import getopt
import queue
import time

import pandas as pd
import subprocess


import tkinter
from tkinter import BOTH, BOTTOM
from tkinter import Frame, ttk, messagebox


class SingleResults(Frame):
    def __init__(self, master, settings):
        self.queue = queue
        # Set up the GUI
        Frame.__init__(self, master=None)
        self.master.title("Single execution result")
        self.file = settings['file']
        #
        center = Frame(master, padx=0, pady=0, width=300, height=300)
        btm_frame = Frame(master, pady=0, width=450, height=40)      
        center.pack(fill=BOTH, expand=True)
        btm_frame.pack(side=BOTTOM, padx=0, pady=0)
        
        center.pack_propagate(False)
        btm_frame.pack_propagate(False)
        
        self.tree = self.create_table(center)
        self.tree.pack(fill=BOTH, expand=True)
        
        buttons = Frame(btm_frame, padx=0, pady=0)
        buttons.pack(side=BOTTOM, padx=0, pady=0)
        open_explorer = tkinter.Button(buttons, 
                                       text='Open externally', 
                                       command=self.open_explorer)
        open_explorer.grid(row=0, column=0, padx=0, pady=0)
        go_home = tkinter.Button(buttons,
                                 text='New execution', 
                                 command=self.go_home)
        go_home.grid(row=0, column=1, padx=0, pady=0)

        
    def create_table(self, frame):
        log = pd.read_csv(os.path.join('outputs', self.file))
        log = log[log.status=='ok']
        log = log.groupby('output').mean().reset_index()
        log = log.to_dict('index')[0]
        tree = ttk.Treeview(frame)
        tree['columns']=('value')
        tree.column('value', width=200)
        tree.heading('#0', text="property")
        tree.heading('value', text="value")
        for k, v in log.items():
            if k != 'output':
                tree.insert('', 'end', text=k, values=(v))
        self.file = log['output']
        return tree

    def open_explorer(self):
        try:
            os.startfile(self.file, 'open')
        except:
            messagebox.showerror("Error", "corrupted file")

    def go_home(self):
        try:
            var = ['python', 'simod_ui.py']
            subprocess.Popen(var)
            self.master.destroy()
        except Exception as e:
            messagebox.showerror("Error", e.message())
            

def catch_parameter(opt):
    """Change the captured parameters names"""
    switch = {'-h': 'help', '-f': 'file'}
    try:
        return switch[opt]
    except Exception as e:
        print(e.message)
        raise Exception('Invalid option ' + opt)


if __name__ == "__main__":
    settings = dict()
    # Catch parameters by console
    try:
        opts, _ = getopt.getopt(sys.argv[1:], "hf:", ['file='])
        for opt, arg in opts:
            key = catch_parameter(opt)
            settings[key] = arg
    except getopt.GetoptError:
        print('Invalid option')
        sys.exit(2)

    while os.stat(os.path.join('outputs', settings['file'])).st_size <= 0:
        time.sleep(1)
        
    root = tkinter.Tk()
    window = SingleResults(root, settings)
    root.mainloop()