from __future__ import print_function, division

from collections import defaultdict
from random import random

import networkx as nx
import numpy as np
from gurobipy import Model, GRB, quicksum

np.random.seed(45)


class Arc:
    __slots__ = ['u', 'v', 'u_pos', 'v_pos', 'weight', 'flow_var', 'edge_var']

    def __init__(self, u, v, u_pos, v_pos, weight):
        self.u = u
        self.v = v
        self.u_pos = u_pos
        self.v_pos = v_pos
        self.weight = weight

    def __repr__(self):
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
    for arc in arc_list:
        arc.flow_var = model.addVar(name="flow_" + str(arc))
        arc.edge_var = model.addVar(vtype=GRB.BINARY, name="edge_" + str(arc))

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
        incoming_vars = [arc.edge_var for arc in incoming_arcs[n]]
        model.addConstr(quicksum(incoming_vars) == 1)

        # Constraint: each pos has exactly one value
        model.addConstr(quicksum(pos_vars[n]) == 1)

        # Constraint: Each node consumes one unit of flow
        in_flow = [arc.flow_var for arc in incoming_arcs[n]]
        out_flow = [arc.flow_var for arc in outgoing_arcs[n]]
        model.addConstr(quicksum(in_flow) - quicksum(out_flow) == 1)

    # Connectivity constraint. Root sends flow to each node
    root_out_flow = [arc.flow_var for arc in outgoing_arcs[0]]
    model.addConstr(quicksum(root_out_flow) == (num_nodes - 1))

    # Inactive arcs have no flow
    LARGE_NUMBER = 1000
    for arc in arc_list:
        model.addConstr(arc.flow_var <= (arc.edge_var * LARGE_NUMBER))

    # Setup objective
    terms = [arc.edge_var * arc.weight for arc in arc_list]
    model.setObjective(quicksum(terms))

    # Missing: Constraint POS with respect to edges
    for arc in arc_list:
        if arc.u != 0:
            model.addConstr(pos_vars[arc.u][arc.u_pos] >= arc.edge_var)
        model.addConstr(pos_vars[arc.v][arc.v_pos] >= arc.edge_var)

    model.update()
    return model


def extract_solution(arc_list, num_nodes):
    heads = [-1] * num_nodes
    pos = [-1] * num_nodes

    for arc in arc_list:
        if arc.edge_var.x == 1.0:
            heads[arc.v] = arc.u

            if arc.u != 0:
                assert pos[arc.u] == -1 or pos[arc.u] == arc.u_pos, [str(arc), pos]
                pos[arc.u] = arc.u_pos

            assert pos[arc.v] == -1 or pos[arc.v] == arc.v_pos, [str(arc), pos]
            pos[arc.v] = arc.v_pos

    return heads, pos


def check_is_tree(arc_list, num_nodes):
    G = nx.DiGraph()
    for arc in arc_list:
        if arc.edge_var.x == 1.0:
            G.add_edge(arc.u, arc.v, u_pos=arc.u_pos, v_pos=arc.v_pos)

    return nx.is_tree(G) and G.number_of_nodes() == num_nodes


if __name__ == '__main__':
    num_nodes = 50
    arc_list = make_dense_arcs(num_nodes, 3)
    model = build_joint_model(arc_list, num_nodes)
    print(model)
    model.optimize()

    print("Is tree", check_is_tree(arc_list, num_nodes))
    print(extract_solution(arc_list, num_nodes))
