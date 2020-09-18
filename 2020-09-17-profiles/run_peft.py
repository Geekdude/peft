"""Script to run the peft collection for multiple profiles."""

import argparse
import sys
import os
import datetime
import subprocess
import time
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
# import json_tricks as json
from tabulate import tabulate
# pd.options.display.max_columns = None
# pd.options.display.max_rows = None

title_size = 16

# %matplotlib inline
# %matplotlib notebook

PEFT_DIR = '..'
SCRIPT_DIR = os.getcwd()
MODELS = [
            'incv3', 
            'resnet', 
            'vgg',
            'unet',
         ]

def run(command):
    """Print command then run command"""
    print(command)
    print(check_output(command, shell=True))
          
def convert_1_to_inf(file):
    """Convert -1s to inf in file using sed."""
    command = f"sed -i 's/-1/inf/g' {file}"
    run(command)
    
def reorder_cols(file, outfile):
    """Reorder the accelerators in computation table"""
    output = []
    fieldnames = None

    with open(file, newline='') as fd:
        out_reader = csv.DictReader(fd)
        fieldnames = out_reader.fieldnames
        for row in out_reader:
            output.append(row)
 
    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        '''
        alist.sort(key=natural_keys) sorts in human order
        http://nedbatchelder.com/blog/200712/human_sorting.html
        (See Toothy's implementation in the comments)
        '''
        return [atoi(c) for c in re.split(r'(\d+)', text)]
 
    fieldnames = fieldnames[0:1] + sorted(fieldnames[1:], key=natural_keys) 
    newfieldnames = [f.replace('NS','') for f in fieldnames]

    outputnew = []
    for o in output:
        row = {}
        for k,v in o.items():
            row[k.replace('NS','')] = v
        outputnew.append(row)
 
    with open(outfile, 'w', newline='') as fd:
        writer = csv.DictWriter(fd, fieldnames=newfieldnames)
        writer.writeheader()
        for o in outputnew:
            writer.writerow(o)

def create_order(file, prefix):
    """Extract an order file from an output file."""
    output = []
    with open(file, newline='') as fd:
        out_reader = csv.DictReader(fd)
        for row in out_reader:
            output.append(row)
 
    order =  sorted(output, key=lambda x: float(x['start']))
  
    with open(f'{prefix}order.csv', 'w', newline='') as fd:
        fieldnames = ['taskname', 'acclname']
        writer = csv.DictWriter(fd, fieldnames=fieldnames)
        writer.writeheader()
        for o in order:
            if o['taskname'] != 'Idle':
                taskname = o['taskname']
                acclname = o['acclname'].replace('NS','')
                writer.writerow({'taskname': taskname, 'acclname': acclname})

def reduce_accl(infile, outfile, accl):
    """Reduce the number of accelerators"""
    output = []
    fieldnames = None

    with open(infile, newline='') as fd:
        out_reader = csv.DictReader(fd)
        fieldnames = out_reader.fieldnames
        for row in out_reader:
            output.append(row)
    
    fieldnames = ['task',] + [f for f in fieldnames if f in accl]
 
    with open(outfile, 'w', newline='') as fd:
        writer = csv.DictWriter(fd, fieldnames=fieldnames)
        writer.writeheader()
        for o in output:
            row = {k: v for k, v in o.items() if k in fieldnames}
            writer.writerow(row)

def run_model(model, prefix, comm, no_stream_comp, stream_comp):
    """Run the model though peft"""
    # Run Non-Streaming
    print(f'{prefix} {model} Non-Streaming')
    run(f'PYTHONPATH={PEFT_DIR} python3 -m peft.peft -d {comm} -t {no_stream_comp} --save {model}/{prefix}no_stream_gantt -o task --idle > {model}/{prefix}no_stream_output.csv')

    # Run Streaming
    print(f'{prefix} {model} Streaming')
    run(f'PYTHONPATH={PEFT_DIR} python3 -m peft.ranger_toy -d {comm} -t {stream_comp} --save {model}/{prefix}stream_gantt -o task --idle > {model}/{prefix}stream_output.csv')

    # Run Streaming Same Order
    create_order(file=f'{model}/{prefix}no_stream_output.csv', prefix=f'{model}/{prefix}no_stream_')
    print(f'{prefix} {model} Streaming Same Order')
    run(f'PYTHONPATH={PEFT_DIR} python3 -m peft.ranger_toy -d {comm} -t {stream_comp} -m {model}/{prefix}no_stream_order.csv --save {model}/{prefix}stream_same_order_gantt -o task --idle > {model}/{prefix}stream_same_order_output.csv')
    
def process_model(model):
    """ Process each model to run peft."""
    
    # Convert -1s to inf
    convert_1_to_inf(os.path.join(model, 'no_stream_comp_only.csv'))
    convert_1_to_inf(os.path.join(model, 'stream_taskwise.csv'))
    
    # Reorder accelerators 
    reorder_cols(f'{model}/no_stream_comp_only.csv', f'{model}/no_stream_comp_only_accl.csv')
    reorder_cols(f'{model}/stream_taskwise.csv', f'{model}/stream_taskwise_accl.csv')

    # Reorder execution tasks
    run(f'PYTHONPATH={PEFT_DIR} python3 -m peft.reorder_exe_time -d {model}/communication_core1_512conv_nostream.csv -t {model}/no_stream_comp_only_accl.csv > {model}/no_stream_comp_only_sorted.csv') 
    run(f'PYTHONPATH={PEFT_DIR} python3 -m peft.reorder_exe_time -d {model}/communication_core1_512conv_nostream.csv -t {model}/stream_taskwise_accl.csv > {model}/stream_taskwise_sorted.csv')
   
    # Merge time together
    run(f'PYTHONPATH={PEFT_DIR} python3 -m peft.init_time_merge -t {model}/no_stream_comp_only_sorted.csv -i {model}/init_comm_core1_512conv.csv > {model}/no_stream_taskwise_sorted.csv')
    
    # Run the model
    run_model(model=model, prefix='all_', comm=f'{model}/communication_core1_512conv_nostream.csv', no_stream_comp=f'{model}/no_stream_taskwise_sorted.csv', stream_comp=f'{model}/stream_taskwise_sorted.csv')

    # Run the model with less accelerators
    accl = [
        'BatchNormalization1024',
        'BatchNormalization128',
        'BatchNormalization16',
        'BatchNormalization256',
        'BatchNormalization512',
        'BatchNormalization8',
        'Conv2D1024',
        'Conv2D128',
        'Conv2D256',
        'Conv2D512',
        'Dense1024',
        'idle',
    ]
    reduce_accl(f'{model}/no_stream_taskwise_sorted.csv', f'{model}/reduce_no_stream_taskwise_sorted.csv', accl)
    reduce_accl(f'{model}/stream_taskwise_sorted.csv', f'{model}/reduce_stream_taskwise_sorted.csv', accl)
    run_model(model=model, prefix='reduce_', comm=f'{model}/communication_core1_512conv_nostream.csv', no_stream_comp=f'{model}/reduce_no_stream_taskwise_sorted.csv', stream_comp=f'{model}/reduce_stream_taskwise_sorted.csv')
            
def collect_results(model):
    """Collect the results from runing the model."""
    summation = collections.OrderedDict()
    makespan = collections.OrderedDict()
    parallel_time = collections.OrderedDict()

    # Non-Streaming
    data = pd.read_csv(f'{model}/all_no_stream_output.csv')
    data_summary = data.groupby(by='acclname')['duration'].sum()
    data_summary.to_csv(f'{model}/all_no_stream_output_summary.csv')
    makespan['no_streaming'] = data.sort_values('end').iloc[-1]['end']
    summation['no_streaming'] = data_summary.sum()

    # Streaming
    data = pd.read_csv(f'{model}/all_stream_output.csv')
    data_summary = data.groupby(by='acclname')['duration'].sum()
    data_summary.to_csv(f'{model}/all_stream_output_summary.csv')
    makespan['streaming'] = data.sort_values('end').iloc[-1]['end']
    summation['streaming'] = data_summary.sum()

    # Streaming Same Order
    data = pd.read_csv(f'{model}/all_stream_same_order_output.csv')
    data_summary = data.groupby(by='acclname')['duration'].sum()
    data_summary.to_csv(f'{model}/all_stream_same_order_output_summary.csv')
    makespan['streaming_same_order'] = data.sort_values('end').iloc[-1]['end']
    summation['streaming_same_order'] = data_summary.sum()

    for k in makespan.keys():
        parallel_time[k] = summation[k] - makespan[k]
     
    return makespan, summation, parallel_time

def main(argv):
    # Process each model
    for model in MODELS:
        process_model(model)
         
    # Collect Results
    results = collections.OrderedDict()
    for model in MODELS:
        results[model] = collections.OrderedDict()
        results[model]['makespan'], results[model]['summation'], results[model]['parallel_time'] = collect_results(model)
    
    for model in MODELS:
        print(model)
        for m in results[model]:
            print(f'\t{m}')
            for v in results[model][m]:
                vv = v+':'
                print(f'\t\t{vv:21} {results[model][m][v]:>15}')

    def plot_makespan(results):            
        model_rename = {
            "incv3": "Invc3",
            "vgg": "VGG",
            "resnet": "ResNet",
            "unet": "Unet",
        }
        case_rename = {
            "streaming": "Streaming",
            "no_streaming": "Non-Streaming",
            "streaming_same_order": "Streaming Same Sched",
            "Unet": "Unet",
        }
        data = []
        
        for model in MODELS:
            for k, v in results[model]['makespan'].items():
                ms = (v / results[model]['makespan']['no_streaming']) * 100
                data.append({"Makespan": ms, "Model": model_rename[model], "Case": case_rename[k]})

        sns.set_style(style="whitegrid")
        dataf = pd.DataFrame(data)
        fig = plt.figure(figsize=figsize())
        ax = sns.barplot(x="Model", y="Makespan", hue="Case", data=dataf, palette="Blues_d",)
        ax.legend(loc="lower right")
        # plt.ylim(0,140)
        # for p in ax.patches:
        #     ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
        #          ha='center', va='center', fontsize=8, color='gray', xytext=(0, 20),
        #          textcoords='offset points')
        figsave("grouped_makespan", data=dataf)
    
    plot_makespan(results)

if __name__ == '__main__':
    main(sys.argv)



