# -*- coding: utf-8 -*-
import scipy
from scipy.stats import pearsonr
import networkx as nx
import matplotlib.pyplot as plt
from support_modules import support as sup
from operator import itemgetter
import random
import pandas as pd

# == support
def random_color(size):
    number_of_colors = size
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]
    return color

def find_index(dictionary, value):
    finish = False
    i = 0
    resp = -1
    while i < len(dictionary) and not finish:
        if dictionary[i]['data']==value:
            resp = dictionary[i]['index']
            finish = True
        i+=1
    return resp

def det_freq_matrix(unique, dictionary):
    freq_matrix = list()
    for u in unique:
        freq = 0
        for d in dictionary:
            if u == d:
                freq += 1
        freq_matrix.append(dict(task=u[0],user=u[1],freq=freq))
    return freq_matrix

def build_profile(users,freq_matrix,prof_size):
    profiles=list()
    for user in users:
        exec_tasks = list(filter(lambda x: x['user']==user['index'],freq_matrix))
        profile = [0,] * prof_size
        for exec_task in exec_tasks:
            profile[exec_task['task']]=exec_task['freq']
        profiles.append(dict(user=user['index'],profile=profile))
    return profiles

def det_correlation_matrix(profiles):
    correlation_matrix = list()
    for profile_x in profiles:
        for profile_y in profiles:
            x = scipy.array(profile_x['profile'])
            y = scipy.array(profile_y['profile'])
            r_row, p_value = pearsonr(x, y)
            correlation_matrix.append(dict(x=profile_x['user'],y=profile_y['user'],distance=r_row))
    return correlation_matrix


def graph_network(g, sub_graphs):
    #IDEA se debe calcular el centroide de los clusters....pos es un diccionario de posiciones y el centroide es el promedio de los puntos x y y
    #despues se debe determinar el punto mas lejano del centroide y ese sera el radio y con esos datos pintar un circulo con patches
    pos = nx.spring_layout(g, k=0.5,scale=10)
    color = random_color(len(sub_graphs))
    for i in range(0,len(sub_graphs)):
        subgraph = sub_graphs[i]
        nx.draw_networkx_nodes(g,pos, nodelist=list(subgraph), node_color=color[i], node_size=200, alpha=0.8)
        nx.draw_networkx_edges(g,pos,width=1.0,alpha=0.5)
        nx.draw_networkx_edges(g,pos, edgelist=subgraph.edges, width=8,alpha=0.5,edge_color=color[i])
    plt.draw()
    plt.show() # display


def role_definition(sub_graphs,users):
    records= list()
    for i in range(0,len(sub_graphs)):
        users_names = list()
        for user in sub_graphs[i]:
            users_names.append(list(filter(lambda x: x['index']==user,users))[0]['data'])
        records.append(dict(role='Role '+ str(i + 1),quantity =len(sub_graphs[i]),members=users_names))
    #Sort roles by number of resources
    records = sorted(records, key=itemgetter('quantity'), reverse=True)
    for i in range(0,len(records)):
        records[i]['role']='Role '+ str(i + 1)
    resource_table = list()
    for record in records:
        for member in record['members']:
            resource_table.append(dict(role=record['role'], resource=member))
    return records, resource_table
# --kernel--


def role_discovery(data, drawing, sim_percentage):
    tasks = data.task.unique()
    users = data.user.unique()
    users = [dict(index=i, data=users[i]) for i in range(0, len(users))]
    data = data.to_dict('records')
    data_transform = list(map(lambda x: [find_index(tasks, x['task']), find_index(users, x['user'])], data))
    unique = list(set(tuple(i) for i in data_transform))
    unique = [list(i) for i in unique]
    # [print(uni) for uni in users]
    # building of a task-size profile of task execution per resource
    profiles = build_profile(users,det_freq_matrix(unique,data_transform),len(tasks))
    sup.print_progress(((20 / 100)* 100),'Analysing resource pool ')
    # building of a correlation matrix between resouces profiles
    correlation_matrix = det_correlation_matrix(profiles)
    sup.print_progress(((40 / 100)* 100),'Analysing resource pool ')
    # creation of a relation network between resouces
    g = nx.Graph()
    for user in users:
        g.add_node(user['index'])
    for relation in correlation_matrix:
        # creation of edges between nodes excluding the same element correlation
        # and those below the 0.7 threshold of similarity
        if relation['distance'] > sim_percentage and relation['x']!=relation['y'] :
            g.add_edge(relation['x'],relation['y'],weight=relation['distance'])
    sup.print_progress(((60 / 100)* 100),'Analysing resource pool ')
    # extraction of fully conected subgraphs as roles
    sub_graphs = list(nx.connected_component_subgraphs(g))
    sup.print_progress(((80 / 100)* 100),'Analysing resource pool ')
    # role definition from graph
    roles = role_definition(sub_graphs,users)
    # plot creation (optional)
    if drawing == True:
        graph_network(g, sub_graphs)
    sup.print_progress(((100 / 100)* 100),'Analysing resource pool ')
    sup.print_done_task()
    return roles

def read_resource_pool(log, drawing=False, sim_percentage=0.7):
    filtered_list = pd.DataFrame(log.data)[['task', 'user']]
    filtered_list = filtered_list[~filtered_list.task.isin(['Start', 'End'])]
    filtered_list = filtered_list[filtered_list.user != 'AUTO']
    return role_discovery(filtered_list, drawing, sim_percentage)
