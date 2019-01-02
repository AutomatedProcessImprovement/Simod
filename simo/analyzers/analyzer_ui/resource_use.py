# -*- coding: utf-8 -*-
import support as sup
import matplotlib as mpl
#mpl.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg

#import matplotlib.pyplot as plt
import numpy as np

import tkinter as tk
from tkinter import ttk

LARGE_FONT= ("Verdana", 12)

class ResourceUse(tk.Frame):

    def __init__(self, parent, controller, resource_use):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Process level", font=LARGE_FONT)
        label.pack(pady=10,padx=10)
        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame('StartPage'))
        button1.pack()

        #----graph----
        #----- Constants -----
        sources =  sorted(list(set([x['source'] for x in resource_use])))
        width = 0.8/len(sources)         # the width of the bars
        bars_distances = sup.create_symetric_list(width, len(sources))
        bars_colors = ['darkgrey','royalblue','darkseagreen','rosybrown','green']
        labels = [y['role'] for y in list(filter(lambda x: x['source'] == 'log', resource_use))]
        #-----figure 1: Processing time ----
        #----- Data -----
        series, confidence = dict(), dict()
        for source in sources:
            series[source] = [y['processing_time'] for y in list(filter(lambda x: x['source'] == source, resource_use))]
            series[source] = sup.zero_to_nan(series[source])
            if source != 'log':
                confidence[source] = [y['pmci'] for y in list(filter(lambda x: x['source'] == source, resource_use))]
        #----- figure ----
        fig = mpl.figure.Figure(figsize=(8, 5), dpi=100)
        ax = fig.add_subplot(111)
        ind = np.arange(len(series['log']))    # the x locations for the groups
        bars_series = dict(log=ax.barh(ind + bars_distances[0], series['log'], width, color='k'))
        series_sources = sorted(list(set([x['source'] for x in resource_use])))
        series_sources.remove('log')
        for i in range(0 , len(series_sources)):
            source = series_sources[i]
            bars_series[source] = ax.barh(ind + bars_distances[i+1], series[source], width, color=bars_colors[i],
                xerr=confidence[source], ecolor='r', capsize=3)
        ax.set_title('Processing time')
        ax.set_yticks(ind)
        ax.set_yticklabels(labels)
        ax.set_xscale('symlog')
        rectangles, series_names = list(), list()
        for source in sources:
            rectangles.append(bars_series[source][0])
            series_names.append(source)
        ax.legend(rectangles, series_names, loc=4, fontsize='small')
        ax.autoscale_view()

        rect_labels = list()
        # Lastly, write in the ranking inside each bar to aid in interpretation
        def add_label(rects):
            for rect in rects:
                # Rectangle widths are already integer-valued but are floating
                # type, so it helps to remove the trailing decimal point and 0 by
                # converting width to int type
                try:
                    width = int(rect.get_width())
                except:
                    width = 0
                rankStr = str(width)
                # The bars aren't wide enough to print the ranking inside
                if (width < 5):
                    # Shift the text to the right side of the right edge
                    xloc = width + 1
                    # Black against white background
                    clr = 'black'
                    align = 'left'
                else:
                    # Shift the text to the left side of the right edge
                    xloc = 0.5*width
                    # White on magenta
                    clr = 'white'
                    align = 'right'

                # Center the text vertically in the bar
                yloc = rect.get_y() + rect.get_height()/2.0
                label = ax.text(xloc, yloc, rankStr, horizontalalignment=align,
                                 verticalalignment='center', color=clr, weight='bold', fontsize='small',
                                 clip_on=True)
                rect_labels.append(label)

        for source in sources:
            add_label(bars_series[source])

        # add_label(p1)
        # if manual: add_label(p2)
        # add_label(p3)
        # ----drawing----
        canvas = FigureCanvasTkAgg(fig, self)
        canvas.show()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2TkAgg(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
