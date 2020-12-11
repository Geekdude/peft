#!/usr/bin/env python3

"""Script to run the peft collection for accelerator configurations."""

import argparse
import sys
import math
import errno
import os
import glob
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

sys.path.insert(1, '..')
import peft.ranger_v2 as peft
SCRIPT_DIR = os.getcwd()


def run(command):
    """Print command then run command"""
    print(command)
    print(check_output(command, shell=True))


def generate_argparser():
    parser = argparse.ArgumentParser(description="A tool for running peft scalability tests.")
    parser.add_argument("--output",
                        help="Folder to store output.",
                        type=str, default='test')
    parser.add_argument('--l_overhead',
                        help='l overhead value.',
                        default=150000, type=float)
    parser.add_argument('-d', '--duplicate',
                        help='Number of times in include the DAG',
                        default=1, type=int)
    return parser


def main(argv):
    argparser = generate_argparser()
    args = argparser.parse_args()

    # Make the output directory
    try:
        os.makedirs(args.output)
    except FileExistsError as e:
        if e.errno != errno.EEXIST:
            raise

    runs = []

    files = glob.glob('scalability/*.json')
    for accel in files:
        output_folder = f'scalability_output/{args.output}/{os.path.splitext(os.path.basename(accel))[0]}_overhead{args.l_overhead:.0f}_dup{args.duplicate}'
        runs.append((f'python3 run_peft.py --output {output_folder} --accelerators {accel} --l_overhead {args.l_overhead} --duplicate {args.duplicate}',))
    
    for r in runs:
        print(r)
    
    with Pool(math.ceil(len(runs)/2)) as p:
        p.starmap(run, runs)

    return 0

if __name__ == '__main__':
    main(sys.argv)
