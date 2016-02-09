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
    # print(var.x)

    # if edge_var.x == 1:
    #     G.add_edge(*edge)




#
# num_tokens = 4
#
#
#
#
# model = Model("single_commodity")
#
# # Add a variable for each possible edge
# flow_vars = {}
# edge_vars = {}
# for u in range(num_tokens):
#     for v in range(1, num_tokens):
#         if u == v:
#             continue
#
#         flow_vars[u, v] = model.addVar(name="flow_{}_{}".format(u, v))
#         edge_vars[u, v] = model.addVar(vtype=GRB.BINARY, name="edge_{}_{}".format(u, v))
#
# model.update()
#
# # Each node has exactly one parent
# for i in range(1, num_tokens):
#     in_edges = [edge_vars[k, i] for k in range(num_tokens) if k != i]
#     model.addConstr(quicksum(in_edges) == 1)
#
#
#
# # Flow constraints
# for i in range(1, num_tokens):
#     in_flow = [flow_vars[k, i]
#                for k in range(num_tokens)
#                if k != i]
#     out_flow = [flow_vars[i, k]
#                 for k in range(1, num_tokens)
#                 if i != k]
#
#     model.addConstr(quicksum(in_flow) - quicksum(out_flow) == 1)
#
# root_out_flow = [flow_vars[0, k] for k in range(1, num_tokens)]
# model.addConstr(quicksum(root_out_flow) == (num_tokens - 1))
#
#
# for edge in edge_vars.keys():
#     # Disable flow on inactive edges
#     model.addConstr(flow_vars[edge] <= (edge_vars[edge] * 1000))
#
# # Setup objective
# terms = []
# for edge, edge_var in edge_vars.items():
#     terms.append(edge_vars[edge] * edge_weights[edge])
#
#
# model.setObjective(quicksum(terms), GRB.MAXIMIZE)
# model.optimize()
#
# G = nx.DiGraph()
# for edge, edge_var in edge_vars.items():
#     if edge_var.x == 1:
#         G.add_edge(*edge)
#
# print(nx.info(G))
# print("Is tree", nx.is_tree(G))
#
# for edge, flow_var in flow_vars.items():
#     print(edge, flow_var.x, edge_vars[edge].x)
