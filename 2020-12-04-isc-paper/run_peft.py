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

import ranger_v2

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
            # 'streaming_flat', 
            # 'vanilla',
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

def process_model(model, arch, args):
    """ Process each model to run peft."""
    
    if arch == 'vanilla':
        # Computation Matrix
        computation_matrix = readCsvToNumpyMatrix(args.task_execution_file)

        # Communication Matrix
        communication_matrix = np.ones((computation_matrix.shape[1]+1, computation_matrix.shape[1]+1)) # Add one for the idle processor.


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
    return parser


def main(argv):
    argparser = generate_argparser()
    args = argparser.parse_args()

    logger.setLevel(logging.getLevelName(args.loglevel))
    consolehandler = logging.StreamHandler()
    consolehandler.setLevel(logging.getLevelName(args.loglevel))
    consolehandler.setFormatter(logging.Formatter("%(levelname)8s : %(name)16s : %(message)s"))
    logger.addHandler(consolehandler)

    # Process each model
    for model in MODELS:
        for arch in ARCHS:
            process_model(model, arch, args)

    return 0

if __name__ == '__main__':
    main(sys.argv)
