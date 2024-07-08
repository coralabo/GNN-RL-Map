import numpy as np
import pulp
from graph_routing import Graph


# Load data
def loadData(path):
    data = []
    with open(path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(",")
            temp = []
            for i in line:
                temp.append(int(i))
            print(temp)
            data.append(temp)
    return np.array(data)


def export_result(outdict, data):
    # Initialize the output matrix, the first column is node number, the second column is time step, 3-6 column is child node, and the last column is routing information
    out = np.zeros([len(data), 7], dtype=int)
    # Enter the node number in the first column, and the child node numbers in columns 1, 3, 5, and 7
    input_col = [0, 1, 3, 5, 7]
    # copy the required input information into the output matrix
    output_col = [0, 2, 3, 4, 5]
    out[:, output_col] = data[:, input_col]
    for nodes in out:
        cur_node = nodes[0]
        positions = np.argwhere(out == cur_node)
        for position in positions:
            if position[1] == 0:
                out[position[0], 1] = outdict[cur_node]
    return out


def export_result_edge(outdict, data):
    # Initialize the output matrix, the first column is node number, the second column is time step, the middle 8 columns are 4 child node numbers and corresponding edge types, and the last column is routing information
    out = np.zeros([len(data), 11], dtype=int)
    # Enter the node number in the first column, and the child node number and corresponding edge type in columns 1 to 8
    input_col = list(range(9))
    # copy the required input information into the output matrix
    output_col = [0, 2, 3, 4, 5, 6, 7, 8, 9]
    out[:, output_col] = data[:, input_col]
    for nodes in out:
        cur_node = nodes[0]
        positions = np.argwhere(out == cur_node)
        for position in positions:
            if position[1] == 0:
                out[position[0], 1] = outdict[cur_node]
    return out


def countMinPE(data, II):
    # The third-to-last number is the latest end time step
    lastTime = max(data[:, -3])
    # List of available times, list of available nodes
    times = [i for i in range(0, lastTime + 1)]
    nodes = [i for i in range(1, len(data) + 1)]

    # Take the minimum task
    prob = pulp.LpProblem("MinPE", sense=pulp.LpMinimize)
    # The total number of pe is between 0 and 15
    peSum = pulp.LpVariable("peSum", lowBound=0, upBound=16, cat=pulp.LpInteger)
    # Scheduling scheme
    plan = pulp.LpVariable.dicts("Plan", (times, nodes), lowBound = 0 , upBound = 1, cat = pulp.LpInteger)
    # Add target minimizes peSum
    prob += peSum
    # Resource constraint
    for i in range(0, II):
        prob += pulp.lpSum([plan[time][node] for time in range(i, lastTime, II) for node in nodes]) <= peSum
    # Unique start time
    for i in nodes:
        prob += pulp.lpSum([plan[time][i] for time in times]) == 1
    # Dependency constraints & Long dependency constraints
    for node in nodes:
        if data[node - 1][9] == data[node - 1][10]:
            prob += plan[data[node - 1][9]][node] == 1
        # TODO:
        #   Note that subIndex here only sets up 4 children at most
        for subIndex in [1, 3, 5, 7]:
            subNode = data[node - 1][subIndex]
            subOutLoop = data[node - 1][subIndex+1]
            # No child node is skipped
            if subNode == 0:
                continue
            # Dependency constraint
            rootTimes = [i for i in range(data[node - 1][9], data[node - 1][10] + 1)]
            subTimes = [i for i in range(data[subNode - 1][9], data[subNode - 1][10] + 1)]
            # Intra-loop dependency
            if not subOutLoop:
                prob += pulp.lpSum([(time + 1) * plan[time][subNode] for time in subTimes]) - \
                        pulp.lpSum([(time + 1) * plan[time][node] for time in rootTimes]) >= 1
            # Interloop dependency
            else:
                prob += pulp.lpSum([(time + 1 + II) * plan[time][subNode] for time in subTimes]) - \
                        pulp.lpSum([(time + 1) * plan[time][node] for time in rootTimes]) >= 1


            # Long dependent constraint
            prob += pulp.lpSum([(time % II + 1) * plan[time][subNode] for time in subTimes]) != \
                    pulp.lpSum([(time % II + 1) * plan[time][node] for time in rootTimes])
    prob.writeLP("solution.lp")
    prob.solve()
    # print("Status: ", pulp.LpStatus[prob.status])
    out_dict = {}
    # # Output scheduling result
    for time in times:
        print(time, end="| ")
        for node in nodes:
            if plan[time][node].varValue == 1:
                out_dict[node] = time
                print(node, end=' | ')
        if time % II == II - 1:
            print("\n---------------------------------------------", end="")
        print("\n")
    print("\n\nMinPE = ", peSum.varValue)
    embedding = export_result_edge(out_dict, data)

    return embedding
