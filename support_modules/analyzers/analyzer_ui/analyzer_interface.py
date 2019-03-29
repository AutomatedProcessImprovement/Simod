# -*- coding: utf-8 -*-
from analyzers.analyzer_ui import menu as mn
from analyzers.analyzer_ui import act_pchange as apc
from analyzers.analyzer_ui import act_duration as adur
from analyzers.analyzer_ui import resource_use as ruse
from analyzers.analyzer_ui import proc_duration as pdur


import tkinter as tk

LARGE_FONT= ("Verdana", 12)

class AnalizerApp(tk.Tk):

    def __init__(self, task_duration, percentual_change, role_use, process_duration, *args, **kwargs):

        tk.Tk.__init__(self, *args, **kwargs)

        tk.Tk.wm_title(self, "SiMo-Discoverer")


        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand = True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        ActPchange = apc.ActPchange(container, self, percentual_change)
        self.frames['ActPchange'] = ActPchange
        ActPchange.grid(row=0, column=0, sticky="nsew")

        ActDuration = adur.ActDuration(container, self, task_duration)
        self.frames['ActDuration'] = ActDuration
        ActDuration.grid(row=0, column=0, sticky="nsew")

        ProcDuration = pdur.ProcDuration(container, self, process_duration)
        self.frames['ProcDuration'] = ProcDuration
        ProcDuration.grid(row=0, column=0, sticky="nsew")

        ResourceUse = ruse.ResourceUse(container, self, role_use)
        self.frames['ResourceUse'] = ResourceUse
        ResourceUse.grid(row=0, column=0, sticky="nsew")


        StartPage = mn.StartPage(container, self)
        self.frames['StartPage'] = StartPage
        StartPage.grid(row=0, column=0, sticky="nsew")


        self.show_frame('StartPage')

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

def analyzer_interface(task_duration, percentual_change, role_use, process_duration):

    app = AnalizerApp(task_duration, percentual_change, role_use, process_duration)
    app.mainloop()
