# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 17:25:45 2020

@author: Manuel Camargo
"""
from tkinter import *
from tkinter import ttk


class MainWindow(Frame):
    def __init__(self, master, elements_data):
        Frame.__init__(self, master=None)
        self.master.title("Probando Dialogos - Manejando datos")
        #self.master.geometry("300x50")
        # Button(root, text="cambiar valor", command=self.dialogo).pack()
        self.valor = StringVar()
        self.valor.set("Hola Manejando datos")
        # Label(self.master, textvariable=self.valor).pack()
        self.tree = self.make_form(elements_data)
        self.tree.pack()    
        # root.bind('<Return>', (lambda event, e = ents: fetch(e)))
        
        b1 = Button(self.master, text = 'Modify', command=self.dialogo)
        b1.pack(side = LEFT, padx = 5, pady = 5)
        # b2 = Button(root, text='Accept',
        # command=(lambda e = ents: monthly_payment(e)))
        # b2.pack(side = LEFT, padx = 5, pady = 5)
        b3 = Button(self.master, text = 'Quit', command = master.quit)
        b3.pack(side = LEFT, padx = 5, pady = 5)

 
    def dialogo(self):
        selected_item = self.tree.selection()[0]
        values = tuple(self.tree.item(selected_item)['values'])

        d = MyDialog(self.master, values, "Probando Dialogo")
        self.master.wait_window(d.top)
        #self.valor.set(d.ejemplo)
        
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
        for element in elements_data:
            tree.insert('', 'end', text=element['name'], 
                        values=(element['type'],element['mean'],
                                element['arg1'],element['arg2']))
        return tree


class MyDialog:
    def __init__(self, parent, values, title):
        self.values = values

        self.top = Toplevel(parent)
        self.top.transient(parent)
        self.top.grab_set()
        
        if len(title) > 0: self.top.title(title)
        
        dist = ('NORMAL', 'LOGNORMAL','GAMMA','EXPONENTIAL','UNIFORM','TRIANGULAR','FIXED')
        fields = [{'name': 'Probability distribution', 'type': 'Combobox', 'values': dist},
                  {'name': 'Mean (Seconds)', 'type': 'Input', 'values': values[1]},
                  {'name': 'Min', 'type': 'Input', 'values': values[2]},
                  {'name': 'Max', 'type': 'Input', 'values': values[3]}]
        
        ent = self.make_edit_form(fields)
        row = Frame(self.top)
        b = Button(row, text="OK", command=self.ok)
        b.pack(side = LEFT, padx = 5, pady=5)
        b = Button(row, text="CANCEL", command=self.cancel)
        b.pack(side = LEFT, padx = 5, pady=5)
        row.pack(side = TOP, fill = X, padx = 5 , pady = 5)
 
    def make_edit_form(self, fields):
        entries = {}
        for field in fields:
            row = Frame(self.top)
            lab = Label(row, width=22, text=field['name']+": ", anchor='w')
            if field['type'] == 'Input':
                ent = Entry(row)
                ent.insert(0, field['values'])
            else:
                ent = ttk.Combobox(row)
                ent['values'] = field['values']
            row.pack(side = TOP, fill = X, padx = 5 , pady = 5)
            lab.pack(side = LEFT)
            ent.pack(side = RIGHT, expand = YES, fill = X)
            entries[field['name']] = ent
        return entries
    
    def ok(self, event=None):
        print("Has escrito ...", self.e.get())
        self.valor.set(self.e.get())
        self.top.destroy()
 
    def cancel(self, event=None):
        self.top.destroy()
