#!/usr/bin/env python3

"""Script to run the peft collection for multiple profiles."""

import argparse
import sys
import errno
import os
import datetime
import subprocess
import time
import networkx as nx
import functools
import random
import re
import csv
import tempfile
import pandas as pd
import collections
from multiprocessing import Pool
from subprocess import check_output
import matplotlib.font_manager as font_manager

import numpy as np
import matplotlib as mpl

DEFAULT_FIGURE_SIZE = 0.9

# Function to calculate figure size in LaTeX document
def figsize(scale=DEFAULT_FIGURE_SIZE, extra_width=0.0, extra_height=0.0):
    """Determine a good size for the figure given a scale."""
    fig_width_pt = 469.755  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    # Aesthetic ratio (you could change this)
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0
    if scale < 0.7:
        golden_mean *= 1.2
    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    fig_height = fig_width * golden_mean  # height in inches
    fig_size = [fig_width + extra_width, fig_height + extra_height]
    return fig_size

figcount = 1
def figsave(file, include_count=True, data=None):
    global figcount
    if include_count:
        file = f'figures/{figcount:02}_{file}'
        figcount += 1
    else:
        file = f'figures/{file}'
    plt.savefig(file + '.svg')
    plt.savefig(file + '.pdf')
    if data is not None:
        data.to_csv(file + '.csv')

# pgf settings for use in LaTeX
latex = {  # setup matplotlib to use latex for output
    "font.family": "serif",
    # "axes.labelsize":  10,
    # "font.size":       10,
    # "legend.fontsize": 10,
    # "xtick.labelsize": 8,
    # "ytick.labelsize": 8,
}
mpl.rcParams.update(latex)
print(mpl.rcParams.find_all)
import matplotlib.pyplot as plt
import math
import re
import sys
import seaborn as sns
import os

import logging

# import json_tricks as json
from tabulate import tabulate
# pd.options.display.max_columns = None
# pd.options.display.max_rows = None

title_size = 16

# %matplotlib inline
# %matplotlib notebook

sys.path.insert(1, '..')
import peft.ranger_v2 as peft

PEFT_DIR = '..'
SCRIPT_DIR = os.getcwd()

MODELS = [
            'incv3', 
            # 'resnet', 
            # 'vgg',
            # 'unet',
         ]

ARCHS = [
            'ranger', 
            'streaming_flat', 
            'vanilla',
         ]

ACCELS = [
    {
        'name': 'conv1024',
        'type': 'conv',
        'size': 1024,
        'count': 1,
    },
    # {
    #     'name': 'conv512',
    #     'type': 'conv',
    #     'size': 512,
    #     'count': 1,
    # },
    # {
    #     'name': 'conv256',
    #     'type': 'conv',
    #     'size': 256,
    #     'count': 1,
    # },
    # {
    #     'name': 'conv128',
    #     'type': 'conv',
    #     'size': 128,
    #     'count': 1,
    # },
    # {
    #     'name': 'conv64',
    #     'type': 'conv',
    #     'size': 64,
    #     'count': 1,
    # },
    {
        'name': 'bn1024',
        'type': 'bn',
        'size': 1024,
        'count': 1,
    },
    # {
    #     'name': 'bn512',
    #     'type': 'bn',
    #     'size': 512,
    #     'count': 1,
    # },
    # {
    #     'name': 'bn256',
    #     'type': 'bn',
    #     'size': 256,
    #     'count': 1,
    # },
    # {
    #     'name': 'bn128',
    #     'type': 'bn',
    #     'size': 128,
    #     'count': 1,
    # },
    # {
    #     'name': 'bn64',
    #     'type': 'bn',
    #     'size': 64,
    #     'count': 1,
    # },
    # {
    #     'name': 'bn32',
    #     'type': 'bn',
    #     'size': 32,
    #     'count': 1,
    # },
    # {
    #     'name': 'bn16',
    #     'type': 'bn',
    #     'size': 16,
    #     'count': 1,
    # },
    # {
    #     'name': 'bn8',
    #     'type': 'bn',
    #     'size': 8,
    #     'count': 1,
    # },
    {
        'name': 'dense1024',
        'type': 'dense',
        'size': 1024,
        'count': 1,
    },
    # {
    #     'name': 'dense512',
    #     'type': 'dense',
    #     'size': 512,
    #     'count': 1,
    # },
    # {
    #     'name': 'dense256',
    #     'type': 'dense',
    #     'size': 256,
    #     'count': 1,
    # },
    # {
    #     'name': 'dense128',
    #     'type': 'dense',
    #     'size': 128,
    #     'count': 1,
    # },
    # {
    #     'name': 'dense64',
    #     'type': 'dense',
    #     'size': 64,
    #     'count': 1,
    # },
    # {
    #     'name': 'dense32',
    #     'type': 'dense',
    #     'size': 32,
    #     'count': 1,
    # },
    # {
    #     'name': 'dense16',
    #     'type': 'dense',
    #     'size': 16,
    #     'count': 1,
    # },
    # {
    #     'name': 'dense8',
    #     'type': 'dense',
    #     'size': 8,
    #     'count': 1,
    # },
]


def run(command):
    """Print command then run command"""
    print(command)
    print(check_output(command, shell=True))
          
def convert_1_to_inf(file):
    """Convert -1s to inf in file using sed."""
    command = f"sed -i 's/-1/inf/g' {file}"
    run(command)

def read_dep_file(filename):
    """Read dependency file and return NetworkX DAG."""

    with open(filename, 'r') as fd:
        lines = fd.readlines()

    dag = nx.DiGraph()

    for l in lines:
        l = l.strip()
        fields = re.split(r'\s*:\s*', l)
        task = fields[0]
        fields[1] = re.sub(r'\s*', '', fields[1])                      
        dep_tasks = re.split(r'\s*,\s*', fields[1])                   
        dep_tasks = [ d for d in dep_tasks if d != '' ]              

        # Add task
        dag.add_node(task)

        # Add dependencies
        for dtask in dep_tasks:
            dag.add_edge(dtask, task)

    # Add start
    root = peft._get_root_node(dag)
    dag.add_edge('T_s', root)

    # Add end
    end = peft._get_end_node(dag)
    dag.add_edge(end, 'T_e')

    return dag


def expand_accelerators():
    accels = {}
    accel_names = []

    for accel in ACCELS:
        for i in range(accel['count']):
            name = f"{accel['name']}_{i}"
            accels[name] = {
                'name': name,
                'type': accel['type'],
                'size': accel['size'],
            }
            accel_names.append(name)
    
    return accel_names, accels


def update_dag_ranger(dag, model):
    # Communication Weights
    for edge in dag.edges():
        dag.edges[edge]['weight'] = 0

    # Computation Weights
    accel_names, accel_details = expand_accelerators()
    processor_num = len(accel_names)

    type_lookup = {'bn': "BatchNormalization", 'conv': 'Conv2D', 'dense': 'Dense'}
    
    for i, accel in enumerate(accel_names):
        filename = f"stream/{model}/{accel_details[accel]['type']}/taskwise_core1_{accel_details[accel]['size']}{accel_details[accel]['type']}.csv"
        header = ['task', 'type', 'start', 'stop', 'duration']

        with open(filename, newline='') as csvfile:
            reader = csv.DictReader(csvfile, fieldnames=header)
            for row in reader:
                if (row['task'] == 'Start' or row['task'] == 'End'):
                    continue
                
                if (type_lookup[accel_details[accel]['type']] not in row['type']):
                    continue

                # Add exe_time
                if 'exe_time' not in dag.nodes[row['task']]:
                    dag.nodes[row['task']]['exe_time'] = [float('inf'),] * processor_num

                dag.nodes[row['task']]['exe_time'][i] = float(row['duration'])
                
    dag.graph['number_of_processors'] = processor_num
    dag.graph['processor_names'] = accel_names

    return dag


def update_dag_streaming_flat(dag, model):
    ndag = nx.DiGraph()

    accel_names, accel_details = expand_accelerators()
    processor_num = len(accel_names)
    type_lookup = {'BatchNormalization': 'bn', 'Conv2D': 'conv', 'Dense': 'dense'}

    # Get task -> type mapping
    filename = f"stream/{model}/{accel_details[accel_names[1]]['type']}/comp_only_core1_{accel_details[accel_names[1]]['size']}{accel_details[accel_names[1]]['type']}.csv"
    header = ['task', 'type', 'start', 'stop', 'duration']
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=header, skipinitialspace=True)
        for row in reader:
            if (row['task'] == 'Start' or row['task'] == 'End'):
                continue
            dag.nodes[row['task']]['type'] = type_lookup[row['type']]

    nnode_lookup = {}
    
    # For each accelerator build the nodes
    for accel_idx, accel in enumerate(accel_names):
        filename = f"stream/{model}/{accel_details[accel]['type']}/instance_core1_{accel_details[accel]['size']}{accel_details[accel]['type']}.csv"
        
        with open(filename, newline='') as csvfile:
            reader = csv.DictReader(csvfile, skipinitialspace=True)
            for row in reader:                    
                node_name = f"{row['task_name']}_{row['instance']}"
                if dag.nodes[row['task_name']]['type'] != accel_details[accel]['type']:
                    continue
                
                # Add node if new
                if node_name not in ndag.nodes():
                    ndag.add_node(node_name, **{
                        'original_node': row['task_name'],
                        'index': row['instance'],
                        'exe_time': [float('inf'),] * processor_num,
                        'dma_in_time': ['inf',] * processor_num,
                        'dma_out_time': ['inf',] * processor_num,
                    })
                    
                    # Add task to reverse lookup.
                    if row['task_name'] not in nnode_lookup:
                        nnode_lookup[row['task_name']] = []
                    if node_name not in nnode_lookup[row['task_name']]:
                        nnode_lookup[row['task_name']].append(node_name)

                # Update times
                ndag.nodes[node_name]['exe_time'][accel_idx] = float(row['acc'])
                ndag.nodes[node_name]['dma_in_time'][accel_idx] = float(row['dma_in'])
                ndag.nodes[node_name]['dma_out_time'][accel_idx] = float(row['dma_out'])

    # Add start and end
    ndag.add_node('T_s', **{
        'original_node': 'T_s',
        'index': 0,
        'exe_time': [0,] * processor_num,
        'dma_in_time': [0,] * processor_num,
        'dma_out_time': [0,] * processor_num,
    })
    nnode_lookup['T_s'] = ['T_s']

    ndag.add_node('T_e', **{
        'original_node': 'T_e',
        'index': 0,
        'exe_time': [0,] * processor_num,
        'dma_in_time': [0,] * processor_num,
        'dma_out_time': [0,] * processor_num,
    })
    nnode_lookup['T_e'] = ['T_e']

    # For each node
    for c, n in enumerate(nx.bfs_tree(dag, peft._get_root_node(dag))):
        # For each edge
        for edge in dag.in_edges(n):
            # For each new node u
            for u in nnode_lookup[edge[0]]:
                # For each new v
                for v in nnode_lookup[edge[1]]:
                    exec_time = ndag.nodes[u]['exe_time']
                    exec_time2 = ndag.nodes[v]['exe_time']
                    dma_out = [i for i in ndag.nodes[u]['dma_out_time'] if i != 'inf']
                    dma_in = [i for i in ndag.nodes[v]['dma_in_time'] if i != 'inf']
                    weight = np.mean(dma_out) + np.mean(dma_in)
                    # weight = np.mean(ndag.nodes[u]['dma_out_time']) + np.mean(ndag.nodes[v]['dma_in_time'])
                    ndag.add_edge(u, v, **{
                        'weight': weight,
                    })

    ndag.graph['number_of_processors'] = processor_num
    ndag.graph['processor_names'] = accel_names

    return ndag


def update_dag_vanilla(dag, model):
    # Communication Weights
    with open(f'nostream/{model}/conv_nostream/communication_core1_1024conv_nostream.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for item, value in row.items():
                if item != 'Task' and value != '0':
                    task = row['Task']
                    dag[task][item]['weight'] = float(value)

    # Computation Weights
    accel_names, accel_details = expand_accelerators()
    processor_num = len(accel_names)

    type_lookup = {'bn': "BatchNormalizationNS", 'conv': 'Conv2DNS', 'dense': 'DenseNS'}
    
    with open(f'nostream/{model}/no_stream_comp_only.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Add exe_time
            if 'exe_time' not in dag.nodes[row['task']]:
                dag.nodes[row['task']]['exe_time'] = [float('inf'),] * processor_num
            
            for i, accel in enumerate(accel_names):
                lookup_name = f"{type_lookup[accel_details[accel]['type']]}{accel_details[accel]['size']}"
                exe_time = float(row[lookup_name])
                dag.nodes[row['task']]['exe_time'][i] = exe_time if exe_time >= 0 else float('inf')
                
    dag.graph['number_of_processors'] = processor_num
    dag.graph['processor_names'] = accel_names

    return dag

def verify_dag(dag):
    # Check Start
    assert('T_s' in dag.nodes())

    # Check End
    assert('T_e' in dag.nodes())

    # Check processor names
    assert('processor_names' in dag.graph)

    # Check number of processors
    assert('number_of_processors' in dag.graph)
    assert(len(dag.graph['processor_names']) == dag.graph['number_of_processors'])

    # Check weight
    for edge in dag.edges():
        if 'weight' not in dag.edges[edge]:
            print(f'Warning: {edge} is missing a weight. Setting to zero.')
            dag.edges[edge]['weight'] = 0

    # Check exe_time
    for node in dag.nodes():
        if 'exe_time' not in dag.nodes[node]:
            print(f'Warning: {node} is missing a exe_time. Setting to zero.')
            dag.nodes[node]['exe_time'] = [0,] * dag.graph['number_of_processors']

    # Add idle proc

    return dag
    
def process_model(args, model, arch):
    """ Process each model to run peft."""
    
    print(f'Processing: {model} {arch}')

    # Read in the base dependencies
    dag = read_dep_file(f'model_dep_{model}.dep')

    print(f'Tasks (before update): {dag.number_of_nodes()}')
    
    # Read in values for graph
    if arch == 'ranger':
        dag = update_dag_ranger(dag, model)
    elif arch == 'streaming_flat':
        dag = update_dag_streaming_flat(dag, model)
    elif arch == 'vanilla':
        dag = update_dag_vanilla(dag, model)

    dag = verify_dag(dag)

    print(f'Tasks (after update): {dag.number_of_nodes()}')

    # Show the DAG
    if args.showDAG:
        nx.draw(dag, pos=nx.nx_pydot.graphviz_layout(dag, prog='dot'), with_labels=True)
        plt.show()
    
    # Save the DAG
    nx.draw(dag, pos=nx.nx_pydot.graphviz_layout(dag, prog='dot'), with_labels=True)
    plt.savefig(f'{args.output}/{model}_{arch}_dag.png')
    
    # Run Peft
    processor_schedules, task_schedules, dict_output = peft.schedule_dag(
        dag,
        self_penalty = True if model == 'ranger' else False,
        include_idle = True,
    )

    # # Output Result
    # print(f"taskname,start,end,duration,acclname")
    # proc_names = dag.graph['processor_names'] + ['Idle']
    # for i, task in task_schedules.items():
    #     print(f"{task.task},{task.start},{task.end},{task.end-task.start},{proc_names[task.proc]}")

    # Save result to CSV
    with open(f'{args.output}/{model}_{arch}_schedule.csv', 'w') as fd:
        fd.write(f"taskname,start,end,duration,acclname\n")
        for i, task in task_schedules.items():
            fd.write(f"{task.task},{task.start},{task.end},{task.end-task.start},{proc_names[task.proc]}\n")

    # Display Gantt
    if args.showGantt:
        peft.showGanttChart(processor_schedules)

    # Save Gantt
    lookup = {-1: "Idle"}
    lookup.update({i:n for i, n in enumerate(dag.graph['processor_names'])})
    peft.saveGanttChart(processor_schedules, f'{args.output}/{model}_{arch}_gantt', lookup)


def generate_argparser():
    parser = argparse.ArgumentParser(description="A tool for finding PEFT schedules for given DAG task graphs")
    parser.add_argument("-l", "--loglevel",
                        help="The log level to be used in this module. Default: INFO",
                        type=str, default="INFO", dest="loglevel", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--showDAG",
                        help="Switch used to enable display of the incoming task DAG",
                        dest="showDAG", action="store_true")
    parser.add_argument("--showGantt",
                        help="Switch used to enable display of the final scheduled Gantt chart",
                        dest="showGantt", action="store_true")
    parser.add_argument("--output",
                        help="Folder to store output.",
                        type=str, default='output')
    return parser


def main(argv):
    argparser = generate_argparser()
    args = argparser.parse_args()

    logger = logging.getLogger('peft')
    logger.setLevel(logging.getLevelName(args.loglevel))
    consolehandler = logging.StreamHandler()
    consolehandler.setLevel(logging.getLevelName(args.loglevel))
    consolehandler.setFormatter(logging.Formatter("%(levelname)8s : %(name)16s : %(message)s"))
    logger.addHandler(consolehandler)

    # Make the output directory
    try:
        os.makedirs(args.output)
    except FileExistsError as e:
        if e.errno != errno.EEXIST:
            raise

    # Process each model
    for model in MODELS:
        for arch in ARCHS:
            process_model(args, model, arch)

    return 0

if __name__ == '__main__':
    main(sys.argv)
