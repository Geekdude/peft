#!/usr/bin/env python3

"""Description"""

import argparse
import sys
import glob
import os
import datetime
import subprocess
import time
import functools
import csv
import random
import re
from tabulate import tabulate
from multiprocessing import Pool
from subprocess import check_output

MODELS = [
            'incv3',
            'resnet50',
            'unet', # Note this model has no Dense
            'vgg16', # Note this model has no BN
         ]

ARCHS = [
            'ranger',
            'streaming_flat_serial_node',
            'streaming_flat_serial_edge',
            'streaming_flat_parallel_node',
            'streaming_flat_parallel_edge',
            'vanilla',
         ]

def run(command):
    """Print command then run command"""
    print(command)
    print(check_output(command, shell=True))


def main(argv):
    # Parse the arguments
    parser = argparse.ArgumentParser(description="""Description""")
    parser.add_argument('-i', '--input', help='Input directory', required=True)
    parser.add_argument('-o', '--output', help='Output file', default='collect.csv')
    args = parser.parse_args(argv[1:])

    results = {}

    # Collect results
    for model in MODELS:
        results[model] = {}
        for arch in ARCHS:
            filename = f'{args.input}/{model}_{arch}_schedule.csv'
            try:
                with open(filename, newline='') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        if row['taskname'] == 'T_e':
                            results[model][arch] = row['end']
            except Exception as e:
                pass

    # Build table
    table = []
    headers = ['Experiment'] + [i for i in ARCHS]
    for model in MODELS:
        table.append([])
        table[-1].append(model)
        for arch in ARCHS:
            try:
                table[-1].append(f'{float(results[model][arch]):.0f}')
            except:
                table[-1].append('N/A')
    
    with open(args.output, 'w') as csvfile:
        csvfile.write(','.join(headers))
        csvfile.write('\n')
        for row in table:
            csvfile.write(','.join(row))
            csvfile.write('\n')

    print(table) 
    print(headers)
    print(tabulate(table, headers=headers))



if __name__ == '__main__':
    main(sys.argv)

