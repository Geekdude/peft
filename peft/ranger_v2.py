"""Core code to be used for scheduling a task DAG with HEFT"""

from collections import deque, namedtuple
from math import inf
from peft.gantt import showGanttChart
from peft.gantt import saveGanttChart
from types import SimpleNamespace

import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import math

logger = logging.getLogger('peft')

ScheduleEvent = namedtuple('ScheduleEvent', 'task start end proc')

"""
Default computation matrix - taken from Arabnejad 2014 PEFT paper
computation matrix: v x q matrix with v tasks and q PEs
"""
W0 = np.array([
    [22, 21, 36],
    [22, 18, 18],
    [32, 27, 19],
    [7, 10, 17],
    [29, 27, 10],
    [26, 17, 9],
    [14, 25, 11],
    [29, 23, 14],
    [15, 21, 20],
    [13, 16, 16]
])

"""
Default communication matrix - not listed in Arabnejad 2014 PEFT paper
communication matrix: q x q matrix with q PEs

Note that a communication cost of 0 is used for a given processor to itself
"""
C0 = np.array([
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
])

def schedule_dag(dag,
    self_penalty,
    time_offset=0,
    relabel_nodes=False,
    manual_order=[],
    include_idle=False,
):
    """
    Given an application DAG and a set of matrices specifying PE bandwidth and (task, pe) execution times, computes the HEFT schedule
    of that DAG onto that set of PEs
    """
    proc_schedules = {}

    _self = {
        'self_penalty': self_penalty,
        'task_schedules': {},
        'proc_schedules': proc_schedules,
        'time_offset': time_offset,
        'root_node': None,
        'optimistic_cost_table': None,
        'manual_order': manual_order,
        'ranger_overhead': False,
    }
    _self = SimpleNamespace(**_self)
    
    _self.root_node = _get_root_node(dag)
    _self.end_node = _get_end_node(dag)

    _self.number_of_tasks = dag.number_of_nodes()
    _self.number_of_processors = dag.graph['number_of_processors']

    _self.numExistingJobs = 0

    # Setup arrays to hold the task and proc schedules.
    for i in dag.nodes():
        _self.task_schedules[i] = None
    for i in range(_self.number_of_processors):
        if i not in _self.proc_schedules:
            _self.proc_schedules[i] = []
    _self.proc_schedules[-1] = []

    # Reapply existing schedules
    for proc in proc_schedules:
        for schedule_event in proc_schedules[proc]:
            raise(NotImplemented("Existing schedules is broken."))
            _self.task_schedules[schedule_event.task] = schedule_event

    logger.debug(""); logger.debug("====================== Performing Optimistic Cost Table Computation ======================\n"); logger.debug("")
    _self.optimistic_cost_table = _compute_optimistic_cost_table(_self, dag)

    # Run the schedular.
    if len(_self.manual_order) == 0: 
        logger.debug(""); logger.debug("====================== Computing EFT for each (task, processor) pair and scheduling in order of decreasing Rank-U ======================"); logger.debug("")
        sorted_nodes = sorted(dag.nodes(), key=lambda node: dag.nodes()[node]['rank'], reverse=True)
        if sorted_nodes[0] != _self.root_node:
            logger.debug("Root node was not the first node in the sorted list. Must be a zero-cost and zero-weight placeholder node. Rearranging it so it is scheduled first\n")
            idx = sorted_nodes.index(_self.root_node)
            sorted_nodes.insert(0, sorted_nodes.pop(idx))
        logger.debug(f"Scheduling tasks in this order: {sorted_nodes}")
        for node in sorted_nodes:
            if _self.task_schedules[node] is not None:
                continue
            minTaskSchedule = ScheduleEvent(node, inf, inf, -1)
            minOptimisticCost = inf
            if 'T_s' in node or 'T_e' in node:
                minOptimisticCost = 0
                minTaskSchedule = _compute_eft(_self, dag, node, -1, 0)
            else:
                for proc in range(_self.number_of_processors):
                    taskschedule = _compute_eft(_self, dag, node, proc)
                    if (taskschedule.end + _self.optimistic_cost_table[node][proc] < minTaskSchedule.end + minOptimisticCost):
                        minTaskSchedule = taskschedule
                        minOptimisticCost = _self.optimistic_cost_table[node][proc]
            _self.task_schedules[node] = minTaskSchedule
            _self.proc_schedules[minTaskSchedule.proc].append(minTaskSchedule)
            _self.proc_schedules[minTaskSchedule.proc] = sorted(_self.proc_schedules[minTaskSchedule.proc], key=lambda schedule_event: schedule_event.end)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('\n')
                for proc, jobs in _self.proc_schedules.items():
                    logger.debug(f"Processor {proc} has the following jobs:")
                    logger.debug(f"\t{jobs}")
                logger.debug('\n')
            for proc in range(_self.number_of_processors):
                for job in range(len(_self.proc_schedules[proc])-1):
                    first_job = _self.proc_schedules[proc][job]
                    second_job = _self.proc_schedules[proc][job+1]
                    assert first_job.end <= second_job.start, \
                    f"Jobs on a particular processor must finish before the next can begin, but job {first_job.task} on processor {first_job.proc} ends at {first_job.end} and its successor {second_job.task} starts at {second_job.start}"

    # Use the manual schedule
    else:
        logger.debug(""); logger.debug("====================== Using the manual schedule ======================"); logger.debug("")
        if _self.manual_order[0][0] != _self.root_node:
            logger.error("Root node was not the first node in the manual order. Must be a zero-cost and zero-weight placeholder node. Rearranging it so it is scheduled first\n")
        logger.debug(f"Scheduling tasks in this order: {[x[0] for x in _self.manual_order]}")
        for node, proc in _self.manual_order:
            taskschedule = _compute_eft(_self, dag, node, proc)
            _self.task_schedules[node] = taskschedule
            _self.proc_schedules[taskschedule.proc].append(taskschedule)
            _self.proc_schedules[taskschedule.proc] = sorted(_self.proc_schedules[taskschedule.proc], key=lambda schedule_event: schedule_event.end)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('\n')
                for proc, jobs in _self.proc_schedules.items():
                    logger.debug(f"Processor {proc} has the following jobs:")
                    logger.debug(f"\t{jobs}")
                logger.debug('\n')
            for proc in range(_self.number_of_processors):
                for job in range(len(_self.proc_schedules[proc])-1):
                    first_job = _self.proc_schedules[proc][job]
                    second_job = _self.proc_schedules[proc][job+1]
                    assert first_job.end <= second_job.start, \
                    f"Jobs on a particular processor must finish before the next can begin, but job {first_job.task} on processor {first_job.proc} ends at {first_job.end} and its successor {second_job.task} starts at {second_job.start}"

    # Add the idle time
    if include_idle:
        sorted_events = []
        for task in _self.task_schedules.values():
            sorted_events.append((task.start, "start"))
            sorted_events.append((task.end, "end"))

        sorted_events = sorted(sorted_events, key=lambda x: x[0])

        start = 0
        count = 0
        idle_count = 0
        for t, s in sorted_events:
            if s == "end":
                count -= 1
                start = t
            elif s == "start":
                if count == 0:
                    end = t
                    delta = end - start

                    if delta > np.float(0): 
                        idle_task_name = f'idle_{idle_count}'
                        idle_count += 1
                        event = ScheduleEvent(idle_task_name, start, end, -1)
                        _self.task_schedules[idle_task_name] = event
                        _self.proc_schedules[-1].append(event)
                        _self.proc_schedules[-1] = sorted(_self.proc_schedules[-1], key=lambda schedule_event: schedule_event.end)
                count += 1

    dict_output = {}
    for proc_num, proc_tasks in _self.proc_schedules.items():
        for idx, task in enumerate(proc_tasks):
            if idx > 0 and (proc_tasks[idx-1].end - proc_tasks[idx-1].start > 0):
                dict_output[task.task] = (proc_num, idx, [proc_tasks[idx-1].task])
            else:
                dict_output[task.task] = (proc_num, idx, [])

    return _self.proc_schedules, _self.task_schedules, dict_output


def _get_root_node(dag):
    # Nodes with no successors cause the any expression to be empty
    root_node = [node for node in dag.nodes() if not any(True for _ in dag.predecessors(node))]
    assert len(root_node) == 1, f"Expected a single root node, found {len(root_node)}"
    root_node = root_node[0]
    return root_node


def _get_end_node(dag):
    # Nodes with no successors cause the any expression to be empty
    end_node = [node for node in dag.nodes() if not any(True for _ in dag.successors(node))]
    assert len(end_node) == 1, f"Expected a single end node, found {len(end_node)}"
    end_node = end_node[0]
    return end_node


def _compute_optimistic_cost_table(_self, dag):
    """
    Uses a basic BFS approach to traverse upwards through the graph building the optimistic cost table along the way
    """

    optimistic_cost_table = {}

    terminal_node = [node for node in dag.nodes() if not any(True for _ in dag.successors(node))]
    assert len(terminal_node) == 1, f"Expected a single terminal node, found {len(terminal_node)}"
    terminal_node = terminal_node[0]

    optimistic_cost_table[terminal_node] = _self.number_of_processors * [0]
    dag.nodes[terminal_node]['rank'] = 0
    visit_queue = deque(dag.predecessors(terminal_node))

    node_can_be_processed = lambda node: all(successor in optimistic_cost_table for successor in dag.successors(node))
    while visit_queue:
        node = visit_queue.pop()

        while node_can_be_processed(node) is not True:
            try:
                node2 = visit_queue.pop()
            except IndexError:
                raise RuntimeError(f"Node {node} cannot be processed, and there are no other nodes in the queue to process instead!")
            visit_queue.appendleft(node)
            node = node2

        optimistic_cost_table[node] = _self.number_of_processors * [0]

        logger.debug(f"Computing optimistic cost table entries for node: {node}")

        # Perform OCT kernel
        # Need to build the OCT entries for every task on each processor
        for curr_proc in range(_self.number_of_processors):
            # Need to maximize over all the successor nodes
            max_successor_oct = -inf
            for succnode in dag.successors(node):
                logger.debug(f"\tLooking at successor node: {succnode}")
                # Need to minimize over the costs across each processor
                min_proc_oct = inf
                for succ_proc in range(_self.number_of_processors):
                    successor_oct = optimistic_cost_table[succnode][succ_proc]
                    successor_comp_cost = dag.nodes[succnode]['exe_time'][succ_proc]

                    # In RANGER there is comm cost for the same proc.
                    if _self.self_penalty == False and curr_proc == succ_proc:
                        successor_comm_cost = 0 
                    else:
                        successor_comm_cost = dag[node][succnode]['weight'] 

                    cost = successor_oct + successor_comp_cost + successor_comm_cost
                    logger.debug(f"If node {node} is on {curr_proc} and successor {succnode} is on {succ_proc}, the optimistic cost entry is {cost}")
                    if cost < min_proc_oct:
                        min_proc_oct = cost
                if min_proc_oct > max_successor_oct:
                    max_successor_oct = min_proc_oct
            assert max_successor_oct != -inf, f"No node should have a maximum successor OCT of {-inf} but {node} does when looking at processor {curr_proc}"
            optimistic_cost_table[node][curr_proc] = max_successor_oct
        # End OCT kernel
        dag.nodes[node]['rank'] = np.mean(optimistic_cost_table[node])
        visit_queue.extendleft([prednode for prednode in dag.predecessors(node) if prednode not in visit_queue])

    return optimistic_cost_table

def _compute_eft(_self, dag, node, proc, exe_time_override=None):
    """
    Computes the EFT of a particular node if it were scheduled on a particular processor
    It does this by first looking at all predecessor tasks of a particular node and determining the earliest time a task would be ready for execution (ready_time)
    It then looks at the list of tasks scheduled on this particular processor and determines the earliest time (after ready_time) a given node can be inserted into this processor's queue
    """
    ready_time = _self.time_offset
    ranger_communication_overhead = 0
    logger.debug(f"Computing EFT for node {node} on processor {proc}")
    for prednode in list(dag.predecessors(node)):
        predjob = _self.task_schedules[prednode]
        assert predjob != None, f"Predecessor nodes must be scheduled before their children, but node {node} has an unscheduled predecessor of {prednode}"
        logger.debug(f"\tLooking at predecessor node {prednode} with job {predjob} to determine ready time")
        if _self.self_penalty == False and predjob.proc == proc:
            ready_time_t = predjob.end
            ranger_communication_overhead_t = 0
        else:
            ready_time_t = predjob.end + dag[predjob.task][node]['weight']
            ranger_communication_overhead_t = dag[predjob.task][node]['weight']
        logger.debug(f"\tNode {prednode} can have its data routed to processor {proc} by time {ready_time_t}")
        if ready_time_t > ready_time:
            ready_time = ready_time_t
            ranger_communication_overhead = ranger_communication_overhead_t
    logger.debug(f"\tReady time determined to be {ready_time}")
	
	# If not ranger then discard ranger overhead
    if not _self.ranger_overhead:
        ranger_communication_overhead = 0
    
    if exe_time_override is not None:
        computation_time = exe_time_override
    else:   
        computation_time = dag.nodes[node]['exe_time'][proc]
    job_list = _self.proc_schedules[proc]
    for idx in range(len(job_list)):
        prev_job = job_list[idx]
        if idx == 0:
            if (prev_job.start - computation_time) - ready_time - (2 * ranger_communication_overhead) > 0:
                logger.debug(f"Found an insertion slot before the first job {prev_job} on processor {proc}")
                job_start = ready_time + ranger_communication_overhead
                min_schedule = ScheduleEvent(node, job_start, job_start+computation_time, proc)
                break
        if idx == len(job_list)-1:
            job_start = max(ready_time, prev_job.end + ranger_communication_overhead)  # Need 3 cycle communication gap between tasks running on a processor
            min_schedule = ScheduleEvent(node, job_start, job_start + computation_time, proc)
            break
        next_job = job_list[idx+1]
        #Start of next job - computation time == latest we can start in this window
        #Max(ready_time, previous job's end) == earliest we can start in this window
        #If there's space in there, schedule in it
        logger.debug(f"\tLooking to fit a job of length {computation_time} into a slot of size {next_job.start - max(ready_time, prev_job.end)}")
        if (next_job.start - computation_time) - max(ready_time, prev_job.end) >= 0:
            job_start = max(ready_time, prev_job.end)
            logger.debug(f"\tInsertion is feasible. Inserting job with start time {job_start} and end time {job_start + computation_time} into the time slot [{prev_job.end}, {next_job.start}]")
            min_schedule = ScheduleEvent(node, job_start, job_start + computation_time, proc)
            break
    else:
        #For-else loop: the else executes if the for loop exits without break-ing, which in this case means the number of jobs on this processor are 0
        min_schedule = ScheduleEvent(node, ready_time, ready_time + computation_time, proc)
    logger.debug(f"\tFor node {node} on processor {proc}, the EFT is {min_schedule}")
    return min_schedule

def readCsvToNumpyMatrix(csv_file):
    """
    Given an input file consisting of a comma separated list of numeric values with a single header row and header column,
    this function reads that data into a numpy matrix and strips the top row and leftmost column
    """
    with open(csv_file) as fd:
        logger.debug(f"Reading the contents of {csv_file} into a matrix")
        contents = fd.read()
        contentsList = contents.split('\n')
        contentsList = list(map(lambda line: line.split(','), contentsList))
        contentsList = contentsList[0:len(contentsList)-1] if contentsList[len(contentsList)-1] == [''] else contentsList

        matrix = np.array(contentsList)
        matrix = np.delete(matrix, 0, 0) # delete the first row (entry 0 along axis 0)
        matrix = np.delete(matrix, 0, 1) # delete the first column (entry 0 along axis 1)
        matrix = matrix.astype(float)
        logger.debug(f"After deleting the first row and column of input data, we are left with this matrix:\n{matrix}")
        return matrix

def getTaskAndAcclNames(csv_file):
    """
    Given an input file consisting of a comma separated list of numeric values with a single header row and header column,
    this function returns the names of the tasks and the accelerators.
    """
    with open(csv_file) as fd:
        lines = fd.readlines()

    accls = {-1: "Idle"}
    line = lines.pop(0).split(',')
    for i, a in enumerate(line):
        if i == 0:
            continue
        accls[i-1] = a.strip()

    tasks = {-1: "Idle"}
    for i, line in enumerate(lines):
        tasks[i] = line.split(',')[0].strip()

    accls_r = {}
    for i, v in accls.items():
        accls_r[v] = i

    tasks_r = {}
    for i, v in tasks.items():
        tasks_r[v] = i

    return tasks, accls, tasks_r, accls_r

def readCsvToDict(csv_file):
    """
    Given an input file consisting of a comma separated list of numeric values with a single header row and header column,
    this function reads that data into a dictionary with keys that are node numbers and values that are the CSV lists
    """
    with open(csv_file) as fd:
        matrix = readCsvToNumpyMatrix(csv_file)

        outputDict = {}
        for row_num, row in enumerate(matrix):
            outputDict[row_num] = row
        return outputDict

def readDagMatrix(dag_file):
    """
    Given an input file consisting of a connectivity matrix, reads and parses it into a networkx Directional Graph (DiGraph)
    """
    matrix = readCsvToNumpyMatrix(dag_file)

    dag = nx.DiGraph(matrix)
    dag.remove_edges_from(
        # Remove all edges with weight of 0 since we have no placeholder for "this edge doesn't exist" in the input file
        [edge for edge in dag.edges() if dag.get_edge_data(*edge)['weight'] == '0.0']
    )

    # Duplicate weight attribute to data size.
    for edge in dag.edges():
        dag.edges[edge]['data'] = dag.edges[edge]['weight']

    return dag

def readManualOrder(order_csv, tasks_r, accls_r):
    """
    Given an input file consitity of (task, device map), read and parse into a scheduling order.
    """
    order = []

    if order_csv:
        with open(order_csv) as fd:
            order_v = fd.readlines()

        order_v.pop(0)

        for o in order_v:
            task, device = o.split(',')
            order.append((tasks_r[task.strip()], accls_r[device.strip()]))

    return order

def update_dag(dag, model, computation_matrix, spm, transfer, showDAG):
    """Update the DAG for the different models."""
    # Ranger
    if model == 'ranger':
        for edge in dag.edges():
            dag.edges[edge]['weight'] = 0
        for n in dag.nodes():
            for p in range(computation_matrix.shape[1]):
                if computation_matrix[dag.nodes[n]['index']][p] != 0: # Do not add to empty start/stop nodes.
                    computation_matrix[dag.nodes[n]['index']][p] += (spm / transfer) * 2
            dag.nodes[n]['exe_time'] = computation_matrix[dag.nodes[n]['index']]

    # Streaming Flat
    if model == 'streaming_flat':
        ndag = nx.DiGraph()

        # Expand each node
        for c, n in enumerate(nx.bfs_tree(dag, _get_root_node(dag))):
            end_point = dag.out_degree(n) == 0 

            edge_data = [0] + [dag.edges[i]['data'] for i in dag.in_edges(n)] 
            incoming_data = max(edge_data)

            if end_point: # Do not split the endpoint
                n_tasks = 1
            else:
                n_tasks = max(math.ceil(incoming_data/spm), 1)

            assert(c != 0 or n_tasks == 1)

            # Generate Sub Tasks
            # For each node
            for i in range(n_tasks): 
                current = f'{n}.{i}'
                ndag.add_node(current, **{
                    'original_node': n,
                    'index': dag.nodes[n]['index'],
                    'id': f'{n}.{i}',
                    'exe_time': computation_matrix[dag.nodes[n]['index']] / n_tasks
                })

                # For each edge
                for edge in dag.in_edges(n):
                    for nn in ndag.nodes:
                        if edge[0] == ndag.nodes[nn]['original_node']:
                            ndag.add_edge(nn, current, **{
                                'weight' : spm / transfer,
                                'data': dag.edges[edge]['data'],
                            })
                
        # Create new computation matrix
        dag = ndag

    # Vanilla
    if model == 'vanilla':
        for edge in dag.edges():
            dag.edges[edge]['weight'] = dag.edges[edge]['data'] / transfer
        for n in dag.nodes():
            dag.nodes[n]['exe_time'] = computation_matrix[dag.nodes[n]['index']]
            
    dag.graph['number_of_processors'] = computation_matrix.shape[1]

    if showDAG:
        nx.draw(dag, pos=nx.nx_pydot.graphviz_layout(dag, prog='dot'), with_labels=True)
        plt.show()

    return dag

def generate_argparser():
    parser = argparse.ArgumentParser(description="A tool for finding PEFT schedules for given DAG task graphs")
    parser.add_argument("-d", "--dag_file",
                        help="File containing input DAG to be scheduled. Uses default 10 node dag from Arabnejad 2014 if none given.",
                        type=str, default="test/peftgraph_task_connectivity.csv")
    parser.add_argument("-t", "--task_execution_file",
                        help="File containing execution times of each task on each particular PE. Uses a default 10x3 matrix from Arabnejad 2014 if none given.",
                        type=str, default="test/peftgraph_task_exe_time.csv")
    parser.add_argument("-l", "--loglevel",
                        help="The log level to be used in this module. Default: INFO",
                        type=str, default="INFO", dest="loglevel", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--showDAG",
                        help="Switch used to enable display of the incoming task DAG",
                        dest="showDAG", action="store_true")
    parser.add_argument("--showGantt",
                        help="Switch used to enable display of the final scheduled Gantt chart",
                        dest="showGantt", action="store_true")
    parser.add_argument("-o", "--output",
                        help="Output format to use for results",
                        choices=['default', 'task', 'csv'], default='default')
    parser.add_argument("--save",
                        help="Save the Gantt chart picture.",
                        type=str, default='')
    parser.add_argument("-m", "--manual", 
                        help="Specify a csv file containing a manual order to follow. File contains (task, device map).",
                        type=str, default="")
    parser.add_argument("--idle",
                        help="Include idle time in output.",
                        action="store_true")
    parser.add_argument("--model",
                        help="Specify the model to use for the evaluation",
                        type=str, default="ranger",
                        choices=['ranger', 'streaming_flat', 'vanilla'])
    parser.add_argument("--transfer",
                        help="Specify the transfer rate of data between nodes.",
                        type=float)
    parser.add_argument("--spm", help='Scratch pad memory size', type=float)
    return parser

if __name__ == "__main__":
    argparser = generate_argparser()
    args = argparser.parse_args()

    logger.setLevel(logging.getLevelName(args.loglevel))
    consolehandler = logging.StreamHandler()
    consolehandler.setLevel(logging.getLevelName(args.loglevel))
    consolehandler.setFormatter(logging.Formatter("%(levelname)8s : %(name)16s : %(message)s"))
    logger.addHandler(consolehandler)

    computation_matrix = readCsvToNumpyMatrix(args.task_execution_file)

    if args.model == 'ranger':
        self_penalty = True
    else:
        self_penalty = False

    dag = readDagMatrix(args.dag_file)

    tasks, accls, tasks_r, accls_r = getTaskAndAcclNames(args.task_execution_file)
    order = readManualOrder(args.manual, tasks_r, accls_r)

    # Relabel nodes
    for i in dag.nodes:
        dag.nodes[i]['index'] = i
    dag = nx.relabel_nodes(dag, tasks)

    dag = update_dag(
        dag=dag,
        model=args.model,
        computation_matrix=computation_matrix,
        spm=args.spm,
        transfer=args.transfer,
        showDAG=args.showDAG,
    )

    processor_schedules, task_schedules, dict_output = schedule_dag(
        dag, 
        self_penalty=self_penalty,
        manual_order=order, 
        include_idle=args.idle, 
    )

    if args.output == 'default':
        for proc, jobs in processor_schedules.items():
            logger.info(f"Processor {proc} has the following jobs:")
            logger.info(f"\t{jobs}")
    elif args.output == 'task':
        print(f"taskname,start,end,duration,acclname")
        for i, task in task_schedules.items():
            print(f"{task.task},{task.start},{task.end},{task.end-task.start},{accls[task.proc]}")
    else: # CSV
        with open('output.csv', 'w') as fd:
            fd.write(f"taskname,start,end,duration,acclname\n")
            for i, task in task_schedules.items():
                fd.write(f"{task.task},{task.start},{task.end},{task.end-task.start},{accls[task.proc]}\n")


    if args.showGantt:
        showGanttChart(processor_schedules)

    if args.save != '':
        saveGanttChart(processor_schedules, args.save, accls)

