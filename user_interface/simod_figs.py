# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 16:11:38 2020

@author: Manuel Camargo
"""
import os
import sys
import getopt
import threading
import queue
import time
import subprocess

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

import tkinter
from tkinter import BOTH, BOTTOM, HORIZONTAL, TOP
from tkinter import Frame, Canvas, ttk, messagebox
from PIL import Image, ImageTk


class OptimizerMonitor(Frame):
    def __init__(self, master, queue, endCommand, settings):
        self.queue = queue
        # Set up the GUI
        Frame.__init__(self, master=None)
        self.master.title("Optimizer execution")
        self.file = settings['file']
        self.image_path = os.path.join(
            'outputs', os.path.splitext(self.file)[0]+'.png')
        self.max_eval = settings['max_eval']
        self.metric = settings['sim_metric']
        #
        center = Frame(master, padx=0, pady=0, width=760, height=300)
        btm_frame = Frame(master, pady=0, width=450, height=40)      
        center.pack(fill=BOTH, expand=True)
        btm_frame.pack(side=BOTTOM, padx=0, pady=0)
        
        center.grid_propagate(False)
        btm_frame.pack_propagate(False)
        # create empty figure and draw
        self.canvas = Canvas(center, width = 450, height = 300)
        self.canvas.grid(row=0, column=0, sticky='W')
        # create figure and load
        self.create_figure()
        image = Image.open(self.image_path)
        image = image.resize((450, 300), Image.ANTIALIAS)
        self.img = ImageTk.PhotoImage(image)
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor='nw', image=self.img)
        self.canvas.update()
        
        table_frame = Frame(center, padx=0, pady=0, width=310, height=300)
        table_frame.grid(row=0, column=1, sticky='W')
        
        self.tree = self.create_table(table_frame)
        self.tree.pack(side=TOP, padx=0, pady=0)

        buttons = Frame(table_frame, padx=0, pady=0)
        buttons.pack(side=BOTTOM, padx=0, pady=0)
        open_explorer = tkinter.Button(buttons, 
                                       text='Open externally', 
                                       command=self.open_explorer)
        open_explorer.grid(row=0, column=0, padx=0, pady=0)
        go_home = tkinter.Button(buttons,
                                 text='New execution', 
                                 command=self.go_home)
        go_home.grid(row=0, column=1, padx=0, pady=0)

        self.progress = ttk.Progressbar(btm_frame, orient=HORIZONTAL,
                                    length=760, mode='determinate')
        self.progress.pack(side=BOTTOM, padx=0, pady=0)
        self.update_bar()

    def create_table(self, frame):
        log = pd.read_csv(os.path.join('outputs', self.file))
        log = log[log.status=='ok']
        log = log.groupby('output').mean().reset_index()
        asc = True if self.metric == 'mae' else False 
        log = log.sort_values('similarity', ascending=asc).head(10)
        log = log[['output', 'similarity']].to_dict('records')
        tree = ttk.Treeview(frame)
        tree['columns']=('similarity')
        tree.column('similarity', width=100)
        tree.heading('#0', text="output")
        tree.heading('similarity', text="similarity")
        for data in log:
            tree.insert('', 'end', text=data['output'],
                        values=(data['similarity']))
        return tree

    def create_figure(self) -> Figure:
        log = pd.read_csv(os.path.join('outputs', self.file))
        log = log.rename(columns={'rp_similarity': 'sim_threshold'})
        log = log[log.status=='ok']
        log = log.groupby(
            ['output','gate_management','alg_manag']).mean().reset_index()
        # plot the data
        if not log.empty:
            x = sns.JointGrid(x="eta", y="similarity", data=log, height=5, ratio=9)
            x = x.plot_joint(sns.scatterplot,
                              data=log,
                              palette=sns.color_palette("BrBG", 3),
                              hue='alg_manag',
                              hue_order=['removal', 'repair', 'replacement'],
                              markers=['o', 'D'],
                              size=log['epsilon'],
                              sizes=(10, 300),
                              edgecolor='black',
                              alpha=0.7,
                              # legend=False)
                              legend='brief')
            plt.legend(bbox_to_anchor=(1.2, 1), loc=2, borderaxespad=0.)
            plt.ylabel('accuracy')
            _, xmax, _, ymax = plt.axis()
            x_coord = log.sort_values('similarity', ascending=False).head(1).eta
            y_coord = log.sort_values('similarity', ascending=False).head(1).similarity
            plt.annotate('best',
                          xy=(x_coord, y_coord),
                xytext=(x_coord+((xmax-x_coord)*0.5), y_coord+((ymax-y_coord)*0.5)),
                # Custom arrow
                arrowprops=dict(arrowstyle="fancy",
                                facecolor='red'))
            x = x.plot_marginals(sns.kdeplot, color=".5", shade=True)
            x.savefig(self.image_path)
            plt.close()
        else:
            fig= plt.figure()
            fig.savefig(self.image_path)

    def update_table(self):
        for children in self.tree.get_children():
            self.tree.delete(children)
        log = pd.read_csv(os.path.join('outputs', self.file))
        log = log[log.status=='ok']
        log = log.groupby('output').mean().reset_index()
        asc = True if self.metric == 'mae' else False 
        log = log.sort_values('similarity', ascending=asc).head(10)
        log = log[['output', 'similarity']].to_dict('records')
        for data in log:
            self.tree.insert('', 'end', text=data['output'],
                              values=(data['similarity']))
            
    def redraw_figure(self):
        self.create_figure()
        image = Image.open(self.image_path)
        image = image.resize((450, 300), Image.ANTIALIAS)
        self.img = ImageTk.PhotoImage(image)
        self.canvas.itemconfig(self.image_on_canvas, image=self.img)
        self.canvas.update()
        
    def update_bar(self):
        evals = len(pd.read_csv(
            os.path.join('outputs', self.file)).output.unique())

        self.progress['value'] = int((evals/self.max_eval)*100)
        self.master.update_idletasks()
        
    def open_explorer(self):
        try:
            selected_item = self.tree.selection()[0]
            outputs = self.tree.item(selected_item)['text']
            os.startfile(outputs, 'open')
        except:
            messagebox.showerror("Error", "Please select one model")
            
    def go_home(self):
        try:
            var = ['python', 'simod_ui.py']
            subprocess.Popen(var)
            self.master.destroy()
        except Exception as e:
            messagebox.showerror("Error", e.message())

    def processIncoming(self):
        """
        Handle all the messages currently in the queue (if any).
        """
        while self.queue.qsize():
            try:
                if self.queue.get(0):
                    self.update_table()
                    self.redraw_figure()
                    self.update_bar()
            except queue.Empty:
                pass

class ThreadedClient:
    """
    Launch the main part of the GUI and the worker thread. periodicCall and
    endApplication could reside in the GUI part, but putting them here
    means that you have all the thread controls in a single place.
    """
    def __init__(self, master, settings):
        """
        Start the GUI and the asynchronous threads. We are in the main
        (original) thread of the application, which will later be used by
        the GUI. We spawn a new thread for the worker.
        """
        self.master = master

        # Create the queue
        self.queue = queue.Queue()

        # Set up the GUI part
        self.gui = OptimizerMonitor(master, self.queue, self.endApplication, settings)
        self.file = settings['file']

        # Set up the thread to do asynchronous I/O
        # More can be made if necessary
        self.running = 1
        self.thread1 = threading.Thread(target=self.workerThread1)
        self.thread1.start()

        # Start the periodic call in the GUI to check if the queue contains
        # anything
        self.periodicCall()

    def periodicCall(self):
        """
        Check every 100 ms if there is something new in the queue.
        """
        self.gui.processIncoming()
        if not self.running:
            # This is the brutal stop of the system. You may want to do
            # some cleanup before actually shutting it down.
            import sys
            sys.exit(1)
        self.master.after(100, self.periodicCall)

    def workerThread1(self):
        """
        This is where we handle the asynchronous I/O. For example, it may be
        a 'select()'.
        One important thing to remember is that the thread has to yield
        control.
        """
        filesize = os.stat(os.path.join('outputs', self.file)).st_size
        while self.running:
            new_size = os.stat(os.path.join('outputs', self.file)).st_size
            if filesize < new_size:
                filesize = new_size
                self.queue.put(True)

    def endApplication(self):
        self.running = 0

def catch_parameter(opt):
    """Change the captured parameters names"""
    switch = {'-h': 'help', '-f': 'file', '-e': 'max_eval', '-s': 'sim_metric'}
    try:
        return switch[opt]
    except Exception as e:
        print(e.message)
        raise Exception('Invalid option ' + opt)


if __name__ == "__main__":
    settings = dict()
    # Catch parameters by console
    try:
        opts, _ = getopt.getopt(sys.argv[1:], "hf:e:s:", ['file=', "max_eval=", "sim_metric="])
        for opt, arg in opts:
            key = catch_parameter(opt)
            if key == 'max_eval':
                settings[key] = int(arg)
            else:
                settings[key] = arg
    except getopt.GetoptError:
        print('Invalid option')
        sys.exit(2)

    while os.stat(os.path.join('outputs', settings['file'])).st_size <= 0:
        time.sleep(1)
        
    root = tkinter.Tk()
    window = ThreadedClient(root, settings)
    root.mainloop()