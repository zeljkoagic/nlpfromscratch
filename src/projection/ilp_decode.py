from __future__ import print_function, division

from collections import namedtuple, defaultdict
from random import random

import numpy as np
from gurobipy import Model, GRB, quicksum
import networkx as nx

np.random.seed(42)


class Arc(namedtuple('Arc', 'u v u_pos v_pos weight')):
    def __str__(self):
        return "{}_{}_{}_{}".format(self.u, self.v, self.u_pos, self.v_pos)


def make_dense_arcs(num_nodes, num_pos):
    # edge_weights = np.random.rand(num_tokens, num_tokens)
    arcs = []
    for u in range(num_nodes):
        for v in range(1, num_nodes):
            if u == v:
                continue
            for u_pos in range(num_pos):
                for v_pos in range(num_pos):
                    arcs.append(Arc(u, v, u_pos, v_pos, random()))
    return arcs


def build_joint_model(arc_list, num_nodes):
    outgoing_arcs = defaultdict(list)
    incoming_arcs = defaultdict(list)

    for arc in arc_list:
        outgoing_arcs[arc.u].append(arc)
        incoming_arcs[arc.v].append(arc)

    model = Model("single_commodity")

    # Variables: a cont. flow variable and an binary variable for each edge
    flow_vars = {}
    edge_vars = {}
    for arc in arc_list:
        flow_vars[arc] = model.addVar(name="flow_" + str(arc))
        edge_vars[arc] = model.addVar(vtype=GRB.BINARY, name="edge_" + str(arc))

    # Variable: a POS variable for each possible token value
    pos_vars = {}
    for n in range(1, num_nodes):
        possible_pos = {arc.u_pos for arc in outgoing_arcs[n]} | {arc.v_pos for arc in incoming_arcs[n]}
        pos_vars[n] = [model.addVar(vtype=GRB.BINARY, name='pos_{}_{}'.format(n, pos))
                       for pos in possible_pos]
        assert len(pos_vars[n])

    model.update()

    # Node constraints
    for n in range(1, num_nodes):
        # Constraint: each node has exactly one parent
        incoming_vars = [edge_vars[arc] for arc in incoming_arcs[n]]
        model.addConstr(quicksum(incoming_vars) == 1)

        # Constraint: each pos has exactly one value
        model.addConstr(quicksum(pos_vars[n]) == 1)

        # Constraint: Each node consumes one unit of flow
        in_flow = [flow_vars[arc] for arc in incoming_arcs[n]]
        out_flow = [flow_vars[arc] for arc in outgoing_arcs[n]]
        model.addConstr(quicksum(in_flow) - quicksum(out_flow) == 1)

    # Connectivity constraint. Root sends flow to each node
    root_out_flow = [flow_vars[arc] for arc in outgoing_arcs[0]]
    model.addConstr(quicksum(root_out_flow) == (num_nodes - 1))

    # Inactive arcs have no flow
    LARGE_NUMBER = 1000
    for arc in arc_list:
        model.addConstr(flow_vars[arc] <= (edge_vars[arc] * LARGE_NUMBER))

    # Setup objective
    terms = [edge_vars[arc] * arc.weight for arc in arc_list]
    model.setObjective(quicksum(terms))

    # Missing: Constraint POS with respect to edges
    for arc in arc_list:
        if arc.u != 0:
            model.addConstr(pos_vars[arc.u][arc.u_pos] >= edge_vars[arc])
        model.addConstr(pos_vars[arc.v][arc.v_pos] >= edge_vars[arc])

    model.update()
    return model


arc_list = make_dense_arcs(4, 3)
model = build_joint_model(arc_list, 4)
print(model)
model.optimize()


G = nx.DiGraph()
for var in model.getVars():
    # print(var.varName, var.x)
    if var.varName.startswith("edge") and var.x == 1.0:
        parts = var.varName.split("_")
        u, v = map(int, parts[1:3])
        G.add_edge(u, v, u_pos=parts[3], v_pos=parts[4])

    elif var.varName.startswith("pos") and var.x == 1.0:
        parts = var.varName.split("_")
        u = int(parts[1])
        G.add_node(u)
        G.node[u]['pos'] = parts[2]

print(G.edges(data=True))
print("Is tree", nx.is_tree(G))
print(G.nodes(data=True))
