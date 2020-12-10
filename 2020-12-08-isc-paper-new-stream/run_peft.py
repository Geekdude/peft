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
import json
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
# print(mpl.rcParams.find_all)
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

from update_dag_streaming_flat_parallel_edge import update_dag_streaming_flat_parallel_edge
from update_dag_streaming_flat_parallel_node import update_dag_streaming_flat_parallel_node
from update_dag_streaming_flat_serial_edge import update_dag_streaming_flat_serial_edge
from update_dag_streaming_flat_serial_node import update_dag_streaming_flat_serial_node

PEFT_DIR = '..'
SCRIPT_DIR = os.getcwd()

MODELS = [
            'incv3',
            'resnet50',
            'vgg16', # Note this model has no BN
            'unet', # Note this model has no Dense
         ]

ARCHS = [
            'ranger',
            'streaming_flat_serial_edge',
            'streaming_flat_serial_node',
            'streaming_flat_parallel_edge',
            'streaming_flat_parallel_node',
            'vanilla',
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


def expand_accelerators(args, model):
    with open(args.accelerators) as fd:
        ACCELS = json.load(fd)

    accels = {}
    accel_names = []

    for accel in ACCELS:
        # Limit accel type for models without the accelerator.
        if model == 'vgg16' and accel['type'] == 'bn':
            continue
        if model == 'unet' and accel['type'] == 'dense':
            continue

        for i in range(accel['count']):
            name = f"{accel['name']}_{i}"
            accels[name] = {
                'name': name,
                'type': accel['type'],
                'size': accel['size'],
            }
            accel_names.append(name)

    return accel_names, accels


def update_dag_ranger(args, dag, model):
    # Communication Weights
    for edge in dag.edges():
        dag.edges[edge]['weight'] = args.l_overhead

    # Computation Weights
    accel_names, accel_details = expand_accelerators(args, model)
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


def update_dag_vanilla(args, dag, model):

    accel_names, accel_details = expand_accelerators(args, model)
    processor_num = len(accel_names)

    # Get task -> type mapping
    type_lookup = {'BatchNormalization': 'bn', 'Conv2D': 'conv', 'Dense': 'dense'}
    filename = f"stream/{model}/{accel_details[accel_names[1]]['type']}/comp_only_core1_{accel_details[accel_names[1]]['size']}{accel_details[accel_names[1]]['type']}.csv"
    header = ['task', 'type', 'start', 'stop', 'duration']
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=header, skipinitialspace=True)
        for row in reader:
            if (row['task'] == 'Start' or row['task'] == 'End'):
                continue
            dag.nodes[row['task']]['type'] = type_lookup[row['type']]

    # Communication Weights
    with open(f'nostream/{model}/conv_nostream/communication_core1_1024conv_nostream.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for item, value in row.items():
                if item != 'Task' and value != '0':
                    task = row['Task']
                    dag[task][item]['weight'] = float(value)

    type_lookup = {'bn': "BatchNormalizationNS", 'conv': 'Conv2DNS', 'dense': 'DenseNS'}

    # Computation Weights
    # Read computation times
    with open(f'nostream/{model}/no_stream_comp_only.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Add exe_time
            if 'exe_time' not in dag.nodes[row['task']]:
                dag.nodes[row['task']]['exe_time'] = [float('inf'),] * processor_num

            for i, accel in enumerate(accel_names):
                lookup_name = f"{type_lookup[accel_details[accel]['type']]}{accel_details[accel]['size']}"
                exe_time = float(row[lookup_name])
                dag.nodes[row['task']]['exe_time'][i] = exe_time + args.l_overhead if exe_time >= 0 else float('inf')

    # Read init_comm_overhead
    for i, accel in enumerate(accel_names):
        filename = f"nostream/{model}/{accel_details[accel]['type']}_nostream/init_comm_core1_{accel_details[accel]['size']}{accel_details[accel]['type']}_nostream.csv"

        with open(filename, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if dag.nodes[row['Task']]['type'] != accel_details[accel]['type']:
                    continue
                dag.nodes[row['Task']]['exe_time'][i] += float(row['InitTransferTime'])

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

    print(f'\n***Processing: {model} {arch}')

    # Read in the base dependencies
    dag = read_dep_file(f'model_dep_{model}.dep')

    print(f'Tasks (before update): {dag.number_of_nodes()}')

    # Read in values for graph
    if arch == 'ranger':
        dag = update_dag_ranger(args, dag, model)
    elif arch == 'streaming_flat_parallel_edge':
        dag = update_dag_streaming_flat_parallel_edge(args, dag, model)
    elif arch == 'streaming_flat_parallel_node':
        dag = update_dag_streaming_flat_parallel_node(args, dag, model)
    elif arch == 'streaming_flat_serial_edge':
        dag = update_dag_streaming_flat_serial_edge(args, dag, model)
    elif arch == 'streaming_flat_serial_node':
        dag = update_dag_streaming_flat_serial_node(args, dag, model)
    elif arch == 'vanilla':
        dag = update_dag_vanilla(args, dag, model)

    dag = verify_dag(dag)

    print(f'Tasks (after update): {dag.number_of_nodes()}')

    # Show the DAG
    if args.showDAG:
        fig = plt.figure(figsize=figsize())
        nx.draw(dag, pos=nx.nx_pydot.graphviz_layout(dag, prog='dot'), with_labels=True)
        plt.show()

    # # Save the DAG
    # fig = plt.figure(figsize=figsize())
    # nx.draw(dag, pos=nx.nx_pydot.graphviz_layout(dag, prog='dot'), with_labels=True)
    # plt.savefig(f'{args.output}/{model}_{arch}_dag.png')
    # plt.savefig(f'{args.output}/{model}_{arch}_dag.svg')

    # Save Dot file
    nx.nx_agraph.write_dot(dag, f'{args.output}/{model}_{arch}_dag.dot')

    # Run Peft
    processor_schedules, task_schedules, dict_output = peft.schedule_dag(
        dag,
        self_penalty = False if model == 'vanilla' else True,
        include_idle = True,
    )

    proc_names = dag.graph['processor_names'] + ['Idle']

    # # Output Result
    # print(f"taskname,start,end,duration,acclname")
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

    # # Save Gantt
    # lookup = {-1: "Idle"}
    # lookup.update({i:n for i, n in enumerate(dag.graph['processor_names'])})
    # peft.saveGanttChart(processor_schedules, f'{args.output}/{model}_{arch}_gantt', lookup)


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
    parser.add_argument('-a', '--accelerators',
                        help='The excelerators to use in the experiment',
                        default='all_accelerators.json', type=str)
    parser.add_argument('--l_overhead',
                        help='l overhead value.',
                        default=150000, type=float)
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

    runs = []

    # Process each model
    for model in MODELS:
        for arch in ARCHS:
            if arch == 'vgg16' and 'parallel' in model:
                continue
            runs.append((args, model, arch))
            # process_model(args, model, arch)

    print(f'Launching with {len(runs)} tasks')

    with Pool(len(runs)) as p:
        p.starmap(process_model, runs)

    return 0

if __name__ == '__main__':
    main(sys.argv)
