# -*- coding: utf-8 -*-
import support as sup
import matplotlib as mpl
mpl.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg

#import matplotlib.pyplot as plt
import numpy as np

import tkinter as tk
from tkinter import ttk

LARGE_FONT= ("Verdana", 12)

class ActDuration(tk.Frame):

    def __init__(self, parent, controller, task_duration):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Activity level", font=LARGE_FONT)
        label.pack(pady=10,padx=10)
        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame('StartPage'))
        button1.pack()

        #----graph----
        #----- Constants -----
        sources =  sorted(list(set([x['source'] for x in task_duration])))
        width = 0.95/len(sources)         # the width of the bars
        bars_distances = sup.create_symetric_list(width, len(sources))
        bars_colors = ['darkgrey','royalblue','darkseagreen','rosybrown','green']
        labels = [y['alias'] for y in list(filter(lambda x: x['source'] == 'log', task_duration))]
        #-----figure 1: Processing time ----
        #----- Data -----
        series, confidence = dict(), dict()
        for source in sources:
            series[source] = [y['processing_time'] for y in list(filter(lambda x: x['source'] == source, task_duration))]
            if source != 'log':
                confidence[source] = [y['pmci'] for y in list(filter(lambda x: x['source'] == source, task_duration))]
        #----- figure ----
        fig = mpl.figure.Figure(figsize=(9, 5), dpi=100)
        ax = fig.add_subplot(121)
        ind = np.arange(len(series['log']))    # the x locations for the groups
        bars_series = dict(log=ax.barh(ind + bars_distances[0], series['log'], width, color='k'))
        series_sources = sorted(list(set([x['source'] for x in task_duration])))
        series_sources.remove('log')
        for i in range(0 , len(series_sources)):
            source = series_sources[i]
            bars_series[source] = ax.barh(ind + bars_distances[i+1], series[source], width, color=bars_colors[i],
                       xerr=confidence[source], ecolor='r', capsize=3)
        ax.set_title('Processing time (seconds)')
        ax.set_yticks(ind)
        ax.set_yticklabels(labels)
        ax.set_xscale('symlog')
        ax.margins(y=0)
        rectangles, series_names = list(), list()
        for source in sources:
            rectangles.append(bars_series[source][0])
            series_names.append(source)
        ax.legend(rectangles, series_names, loc=1, fontsize='xx-small')
        ax.autoscale_view()

        #-----figure 2: Waiting time ----
        #----- Data -----
        series, confidence = dict(), dict()
        for source in sources:
            series[source] = [y['waiting_time'] for y in list(filter(lambda x: x['source'] == source, task_duration))]
            if source != 'log':
                confidence[source] = [y['wmci'] for y in list(filter(lambda x: x['source'] == source, task_duration))]
        #----- figure ----
        ax2 = fig.add_subplot(122)
        ind = np.arange(len(series['log']))    # the x locations for the groups
        bars_series2 = dict(log=ax2.barh(ind + bars_distances[0], series['log'], width, color='k'))
        series_sources = sorted(list(set([x['source'] for x in task_duration])))
        series_sources.remove('log')
        for i in range(0 , len(series_sources)):
            source = series_sources[i]
            bars_series2[source] = ax2.barh(ind + bars_distances[i+1], series[source], width, color=bars_colors[i],
                xerr=confidence[source], ecolor='r', capsize=3)
        ax2.set_title('Waiting time (seconds)')
        ax2.set_yticks(ind)
        ax2.set_yticklabels(labels)
        ax2.set_xscale('symlog')
        ax2.margins(y=0)
        rectangles, series_names = list(), list()
        for source in sources:
            rectangles.append(bars_series2[source][0])
            series_names.append(source)
        ax2.legend(rectangles, series_names, loc=1, fontsize='xx-small')
        ax2.autoscale_view()

        rect_labels = list()
        def add_label(rects, axis):
            
            for rect in rects:
                try:
                    width = int(rect.get_width())
                except:
                    width = 0
                if width != 0:
                    rankStr = str(width)
                else:
                    rankStr = ''
                # The bars aren't wide enough to print the ranking inside
                if (width < 5):
                    xloc = width + 1
                    align = 'left'
                else:
                    xloc = 0.5*width
                    align = 'right'
              
                if rect.get_facecolor() == (0.1, 0.1, 0.1, 1.0):
                    clr = 'white'
                else:
                    clr = 'black'

                # Center the text vertically in the bar
                yloc = rect.get_y() + rect.get_height()/2.0
                
                label = axis.text(xloc, yloc, rankStr, horizontalalignment=align,
                                 verticalalignment='center', color=clr, weight='bold', fontsize='xx-small',
                                 clip_on=True)
                rect_labels.append(label)

        for source in sources:
            add_label(bars_series[source],ax)
            add_label(bars_series2[source],ax2)

        #fig.tight_layout(pad=1)
        #fig.subplots_adjust(hspace=0.5, right=0.9)
        #----drawing----
        canvas = FigureCanvasTkAgg(fig, self)
        canvas.show()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2TkAgg(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
