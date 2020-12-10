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
import random
import re
from multiprocessing import Pool
from subprocess import check_output


def run(command):
    """Print command then run command"""
    print(command)
    print(check_output(command, shell=True))


def main(argv):
    # Parse the arguments
    parser = argparse.ArgumentParser(description="""Description""")
    parser.add_argument('-i', '--input', help='Input directory', required=True)
    parser.add_argument('-o', '--output', help='Output directory', required=True)
    args = parser.parse_args(argv[1:])

    for file in glob.glob(f"{args.input}/*.dot"):
        base = os.path.splitext(file)[0]
        run(f'dot -Tpng {file} > {base}_dot.png')


if __name__ == '__main__':
    main(sys.argv)

