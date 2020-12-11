import sys
import networkx as nx
import csv
import numpy as np

sys.path.insert(1, '..')
import peft.ranger_v2 as peft


def update_dag_streaming_flat_serial_edge(args, dag, model):
    ndag = nx.DiGraph()

    from run_peft import expand_accelerators
    accel_names, accel_details = expand_accelerators(args, model)
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
            try:
                dag.nodes[row['task']]['type'] = type_lookup[row['type']]
            except Exception as e:
                raise

    nnode_lookup = {}

    # For each accelerator build the nodes
    for accel_idx, accel in enumerate(accel_names):
        filename = f"stream/{model}/{accel_details[accel]['type']}/instance_core1_{accel_details[accel]['size']}{accel_details[accel]['type']}.csv"

        with open(filename, newline='') as csvfile:
            reader = csv.DictReader(csvfile, skipinitialspace=True)
            for row in reader:
                try:
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
                except Exception as e:
                    raise

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

    # Connect interier nodes
    for node, nnodes in nnode_lookup.items():
        for idx in range(len(nnodes)-1):
            u = nnodes[idx]
            v = nnodes[idx+1]
            dma_out = [i for i in ndag.nodes[u]['dma_out_time'] if i != 'inf']
            dma_in = [i for i in ndag.nodes[v]['dma_in_time'] if i != 'inf']
            weight = np.mean(dma_out) + np.mean(dma_in)
            ndag.add_edge(u, v, **{
                'weight': weight + args.l_overhead,
            })

    # Connect exterior nodes
    # For each node
    for c, n in enumerate(nx.bfs_tree(dag, peft._get_root_node(dag))):
        # For each edge
        for edge in dag.in_edges(n):
            u = nnode_lookup[edge[0]][-1]
            v = nnode_lookup[edge[1]][0]
            dma_out = [i for i in ndag.nodes[u]['dma_out_time'] if i != 'inf']
            dma_in = [i for i in ndag.nodes[v]['dma_in_time'] if i != 'inf']
            weight = np.mean(dma_out) + np.mean(dma_in)
            ndag.add_edge(u, v, **{
                'weight': weight + args.l_overhead,
            })

    ndag.graph['number_of_processors'] = processor_num
    ndag.graph['processor_names'] = accel_names

    return ndag