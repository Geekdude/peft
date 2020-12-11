import re
import os
import sys
import subprocess
import pdb
import itertools
import shlex
import tempfile
import argparse
import time
import json
import commentjson

accelerator_sizes = [
    "1024", "512", "256", "128", "64"
]

acc_input = [ 
    [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9],
               [2, 1, 1], [3, 1, 1], [4, 1, 1], [5, 1, 1], [6, 1, 1], [7, 1, 1], [8, 1, 1], [9, 1, 1], [10, 1, 1],
                          [3, 2, 2], [4, 2, 2], [5, 2, 2], [6, 2, 2], [7, 2, 2], [8, 2, 2], [9, 2, 2], [10, 2, 2],
            ]

def getBufferSize(MAC, atype):
    if atype != 'conv':
        return 1
    if MAC == 64:
        return 128
    return 256

def getAcceleratorArea(config):
    type = config["type"]
    MAC = config["size"]
    count = config["count"]
    Buffer = getBufferSize(MAC, type)
    ctrl = 0.30086957 
    if type != "conv":
        ctrl = 0.0111
    Area = ctrl + 0.0010394*MAC + 0.00173913*Buffer
    #print("type:{} MAC:{} Buffer:{} count:{} Area:{}".format(type, MAC, Buffer, count, Area))
    return Area * int(count)


def generateConfig(tconv, tbn, tdense):
    def getAccConfig(acount, acc_size, atype):
        d = { "name" : atype+acc_size, 
              "type" : atype,
              "size" : int(acc_size),
              "count" : int(acount)
            }
        return d
    all_config = []
    for j in accelerator_sizes:
        all_config.append(getAccConfig(tconv, j, "conv"))
    for j in accelerator_sizes:
        all_config.append(getAccConfig(tbn, j, "conv"))
    for j in accelerator_sizes:
        all_config.append(getAccConfig(tdense, j, "conv"))
    fname = "all_accelerators_conv{}_bn{}_dense{}.json".format(tconv, tbn, tdense)
    with open(fname, "w") as fh:
        json.dump(all_config, fh, indent=4)
        fh.close()
    return (all_config, fname)

def getTotalArea(config):
    total_area = 0
    for acc in config:
        total_area = total_area + getAcceleratorArea(acc)
    return total_area

def main():
    name = "all_accelerators.json"
    with open(name) as infile:
        data = commentjson.load(infile)
        #total_area = getTotalArea(data)
        #print("Total area: "+str(total_area)+" mm2")
    print("{}, {}".format("File", "Area"))
    for each_config in acc_input:
        (all_config, fname) = generateConfig(each_config[0], each_config[1], each_config[2])   
        area = getTotalArea(all_config)
        print("{}, {}".format(fname, area))

if __name__ == "__main__":
    main()

