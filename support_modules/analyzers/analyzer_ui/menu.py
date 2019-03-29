# -*- coding: utf-8 -*-
#import proc_duration as pd

import tkinter as tk
from tkinter import ttk

LARGE_FONT= ("Verdana", 12)


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        label = tk.Label(self, text="Start Page", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        #print(controller.frames)

        button = ttk.Button(self, text="Activities Percentual Change",
                            command=lambda: controller.show_frame('ActPchange')
                            )
        button.pack()

        button2 = ttk.Button(self, text="Activities Duration",
                            command=lambda: controller.show_frame('ActDuration'))
        button2.pack()

        button3 = ttk.Button(self, text="Process Duration",
                            command=lambda: controller.show_frame('ProcDuration'))
        button3.pack()

        button4 = ttk.Button(self, text="Use of Resources",
                            command=lambda: controller.show_frame('ResourceUse'))
        button4.pack()
