# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 17:25:45 2020

@author: Manuel Camargo
"""
import tkinter as tk
from tkinter import Frame, Button, Toplevel, Label, Entry, messagebox, ttk


class MainWindow(Frame):
    def __init__(self, master, elements_data):
        Frame.__init__(self, master=None)
        self.master.title("Interarrival durations distributions")
        self.new_elements = list()
        self.tree = self.make_form(elements_data)
        self.tree.pack()    
        b1 = Button(self.master, text = 'Modify', command=self.dialogo)
        b1.pack(side = tk.LEFT, padx = 5, pady = 5)
        b2 = Button(self.master, text = 'Continue', command = self.close_window)
        b2.pack(side = tk.LEFT, padx = 5, pady = 5)

    def dialogo(self):
        try:
            selected_item = self.tree.selection()[0]
            values = tuple(self.tree.item(selected_item)['values'])
            d = MyDialog(self.master, values, "PDF edition")
            self.master.wait_window(d.top)
            self.tree.item(selected_item,values=d.values)
        except:
            messagebox.showerror("Error", "Please select one task")
        
    def make_form(self, elements_data):
        tree = ttk.Treeview(self.master)
        tree['columns']=('pdf','mean','arg1','arg2')
        tree.column('pdf', width=100 )
        tree.column('mean', width=100)
        tree.column('arg1', width=100 )
        tree.column('arg2', width=100)    
        tree.heading('pdf', text="PDF")
        tree.heading('mean', text="Mean")
        tree.heading('arg1', text="Arg1")
        tree.heading('arg2', text="Arg2")
        tree.insert('', 'end', text='Inter-arrival distribution',
                    values=(elements_data['dname'],elements_data['dparams']['mean'],
                            elements_data['dparams']['arg1'],elements_data['dparams']['arg2']))
        return tree
    
    def close_window(self):
        for item in self.tree.get_children():
            values = self.tree.item(item)['values']
            self.new_elements.append(
                {'id':item,'type':values[0],'mean':str(values[1]),
                 'arg1':str(values[2]),'arg2':str(values[3])})
        self.master.destroy()



class MyDialog:
    def __init__(self, parent, values, title):
        self.values = values
        self.top = Toplevel(parent)
        self.top.transient(parent)
        self.top.grab_set()
        
        if len(title) > 0: self.top.title(title)
        
        dist = ('NORMAL', 'LOGNORMAL','GAMMA','EXPONENTIAL','UNIFORM','TRIANGULAR','FIXED')
        fields = [{'name': 'cb1','label':'Probability distribution', 'type': 'Combobox', 'values': dist, 'current':values[0]},
                  {'name': 'in1','label':'Mean (Seconds)', 'type': 'Input', 'values': values[1]},
                  {'name': 'in2','label':'Min', 'type': 'Input', 'values': values[2]},
                  {'name': 'in3','label':'Max', 'type': 'Input', 'values': values[3]}]
        
        self.entries = self.make_edit_form(fields)
        row = Frame(self.top)
        b = Button(row, text="OK", command=self.ok)
        b.pack(side = tk.LEFT, padx = 5, pady=5)
        b = Button(row, text="CANCEL", command=self.cancel)
        b.pack(side = tk.LEFT, padx = 5, pady=5)
        row.pack(side = tk.TOP, fill = tk.X, padx = 5 , pady = 5)
 
    def make_edit_form(self, fields):
        entries = {}
        for field in fields:
            row = Frame(self.top)
            lab = Label(row, width=22, text=field['label']+": ", anchor='w')
            if field['type'] == 'Input':
                ent = Entry(row)
                ent.insert(0, field['values'])
            else:
                ent = ttk.Combobox(row)
                ent.set(field['current'])
                ent['values'] = field['values']
            row.pack(side = tk.TOP, fill = tk.X, padx = 5 , pady = 5)
            lab.pack(side = tk.LEFT)
            ent.pack(side = tk.RIGHT, expand = tk.YES, fill = tk.X)
            entries[field['name']] = ent
        return entries
    
    def ok(self, event=None):
        self.values = (self.entries['cb1'].get(),
                        self.entries['in1'].get(),
                        self.entries['in2'].get(),
                        self.entries['in3'].get())
        self.top.destroy()
 
    def cancel(self, event=None):
        self.top.destroy()
