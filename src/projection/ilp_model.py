from typing import List

from gurobipy import Model, GRB, quicksum
import networkx as nx

from collections import defaultdict

from arc import Arc


def build_joint_model(arc_list: List[Arc], num_nodes: int):
    outgoing_arcs = defaultdict(list)
    incoming_arcs = defaultdict(list)

    for arc in arc_list:
        outgoing_arcs[arc.u].append(arc)
        incoming_arcs[arc.v].append(arc)

    model = Model("single_commodity")

    # Variables: a cont. flow variable and an binary variable for each edge
    for arc in arc_list:
        arc.flow_var = model.addVar(name="flow_{}_{}_{}_{}".format(arc.u, arc.v, arc.u_pos, arc.v_pos))
        arc.edge_var = model.addVar(vtype=GRB.BINARY,
                                    name="edge_{}_{}_{}_{}".format(arc.u, arc.v, arc.u_pos, arc.v_pos))

    # Variable: a POS variable for each possible token value
    pos_vars = {}
    for n in range(1, num_nodes):
        possible_pos = {arc.u_pos for arc in outgoing_arcs[n]} | {arc.v_pos for arc in incoming_arcs[n]}
        pos_vars[n] = {pos: model.addVar(vtype=GRB.BINARY, name='pos_{}_{}'.format(n, pos))
                       for pos in possible_pos}
        assert len(pos_vars[n])

    model.update()

    # Node constraints
    for n in range(1, num_nodes):
        # Constraint: each node has exactly one parent
        incoming_vars = [arc.edge_var for arc in incoming_arcs[n]]
        model.addConstr(quicksum(incoming_vars) == 1,
                        name='one_parent_' + str(n))

        # Constraint: each pos has exactly one value
        model.addConstr(quicksum(pos_vars[n].values()) == 1,
                        name='one_pos_' + str(n))

        # Constraint: Each node consumes one unit of flow
        in_flow = [arc.flow_var for arc in incoming_arcs[n]]
        out_flow = [arc.flow_var for arc in outgoing_arcs[n]]
        model.addConstr(quicksum(in_flow) - quicksum(out_flow) == 1,
                        name='consume_one_flow_unit_' + str(n))

    # Connectivity constraint. Root sends flow to each node
    root_out_flow = [arc.flow_var for arc in outgoing_arcs[0]]
    model.addConstr(quicksum(root_out_flow) == (num_nodes - 1),
                    name='root_flow')

    # Inactive arcs have no flow
    LARGE_NUMBER = 1000
    for arc in arc_list:
        model.addConstr(arc.flow_var <= (arc.edge_var * LARGE_NUMBER),
                        name='inactive_no_flow_{}_{}_{}_{}'.format(arc.u, arc.v, arc.u_pos, arc.v_pos))

    # Setup objective
    terms = [arc.edge_var * arc.weight for arc in arc_list]
    model.setObjective(quicksum(terms), GRB.MAXIMIZE)

    # Missing: Constraint POS with respect to edges
    for arc in arc_list:
        if arc.u != 0:
            model.addConstr(pos_vars[arc.u][arc.u_pos] >= arc.edge_var,
                            name='arc_{}_{}_{}_{}_pos_must_match_u'.format(arc.u, arc.v, arc.u_pos, arc.v_pos))
        model.addConstr(pos_vars[arc.v][arc.v_pos] >= arc.edge_var,
                        name='arc_{}_{}_{}_{}_pos_must_match_v'.format(arc.u, arc.v, arc.u_pos, arc.v_pos))

    model.update()
    return model

def solution_exists(model):
    return model.Status == GRB.OPTIMAL

def extract_solution(arc_list: List[Arc], num_nodes: int, unk_pos: int=-1) -> (List[int], List[str]):
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

            # Re-attach tokens with unknown POS to -1
            if arc.u_pos == unk_pos:
                heads[arc.v] = -1

    return heads, pos


def check_is_tree(arc_list, num_nodes):
    G = nx.DiGraph()
    for arc in arc_list:
        if arc.edge_var.x == 1.0:
            G.add_edge(arc.u, arc.v, u_pos=arc.u_pos, v_pos=arc.v_pos)

    return nx.is_tree(G) and G.number_of_nodes() == num_nodes
