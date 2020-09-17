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
from multiprocessing import Pool
from subprocess import check_output

PEFT_DIR = '..'
SCRIPT_DIR = os.getcwd()
MODELS = [
            'incv3', 
            'resnet', 
            'vgg',
            'unet',
         ]

def run(command):
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
 
    tokenize = re.compile(r'(\d+)|(\D+)').findall
    def natural_sortkey(string):          
        return tuple(int(num) if num else alpha for num, alpha in tokenize(string))
 
    fieldnames = fieldnames[0:1] + sorted(fieldnames[1:], key=natural_sortkey) 
 
    with open(outfile, 'w', newline='') as fd:
        writer = csv.DictWriter(fd, fieldnames=fieldnames)
        writer.writeheader()
        for o in output:
            writer.writerow(o)
    
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
    # TODO


def run_model(model, prefix, comm, no_stream_comp, stream_comp):
    # Run Non-Streaming
    print(f'{prefix} {model} Non-Streaming')
    run(f'PYTHONPATH={PEFT_DIR} python3 -m peft.peft -d {comm} -t {no_stream_comp} --showDAG --showGantt -o task --idle > {model}/{prefix}no_stream_output.csv')

    # Run Streaming
    print(f'{prefix} {model} Streaming')
    run(f'PYTHONPATH={PEFT_DIR} python3 -m peft.ranger_toy -d {comm} -t {stream_comp} --showGantt -o task --idle > {model}/{prefix}stream_output.csv')

    # Run Streaming Same Order
    create_order(file=f'{model}/{prefix}no_stream_output.csv', prefix=f'{model}/{prefix}')
    print(f'{prefix} {model} Streaming Same Order')
    run(f'PYTHONPATH={PEFT_DIR} python3 -m peft.ranger_toy -d {comm} -t {stream_comp} -m {model}/{prefix}order.csv --showGantt -o task --idle > {model}/{prefix}stream_same_order_output.csv')

def create_order(file, prefix):
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
            if o['taskname'] != 'idle':
                taskname = o['taskname']
                acclname = o['acclname'].replace('NS','')
                writer.writerow({'taskname': taskname, 'acclname': acclname})

def main(argv):
    # Process each model
    for model in MODELS:
        process_model(model)

if __name__ == '__main__':
    main(sys.argv)



