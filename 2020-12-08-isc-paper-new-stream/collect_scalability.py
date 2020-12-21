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
            # 'streaming_flat_serial_edge',
            # 'streaming_flat_parallel_node',
            # 'streaming_flat_parallel_edge',
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
    parser.add_argument('-o', '--output', help='Output file', default='collect_scalability.csv')
    args = parser.parse_args(argv[1:])

    results = []

    input_folder = f'scalability_output/{args.input}'

    for subdir, dirs, files in os.walk(input_folder):
        for d in dirs:
            print(d)
            match = re.match(r'all_accelerators_conv(\d+)_bn(\d+)_dense(\d+)_overhead(\d+)_dup(\d+)', d)
            if match:
                conv, bn, dense, overhead, dup = match.group(1, 2, 3, 4, 5)
                print(conv, bn, dense, overhead, dup)

                # Collect results
                for model in MODELS:
                    for arch in ARCHS:
                        try:
                            filename = f'{subdir}/{d}/{model}_{arch}_schedule.csv'
                            print(filename)
                            with open(filename, newline='') as csvfile:
                                reader = csv.DictReader(csvfile)
                                tasks = 0
                                for row in reader:
                                    if row['taskname'] != 'T_s' and row['taskname'] != 'T_e' and 'idle' not in row['taskname']:
                                        tasks += 1
                                    if row['taskname'] == 'T_e':
                                        results.append({
                                            'conv': conv,
                                            'bn': bn,
                                            'dense': dense,
                                            'overhead': overhead,
                                            'dup': dup,
                                            'model': model,
                                            'arch': arch,
                                            'makespan': row['end'],
                                            'tasks': None,
                                            'time': None
                                        })
                                results[-1]['tasks'] = tasks

                            filename = f'{subdir}/{d}/{model}_{arch}_time.txt'
                            with open(filename) as fd:
                                line = fd.readline()
                                match = re.match(r'(.*) seconds', line)
                                if match:
                                    results[-1]['time'] = match.group(1)

                        except Exception as e:
                            pass
    

    with open(args.output, 'w', newline='') as csvfile:
        fieldnames = ['conv', 'bn', 'dense', 'overhead', 'dup', 'model', 'arch', 'makespan', 'tasks', 'time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

if __name__ == '__main__':
    main(sys.argv)

