import argparse

import pickle
from collections import defaultdict
from itertools import groupby, product, count

from arc import Arc
from ilp_model import build_joint_model, extract_solution, solution_exists
from parallel_sentence import ParallelSentence

import numpy as np
from dependency_decoding import chu_liu_edmonds

parser = argparse.ArgumentParser(description="Projects dependency trees from source to target via word alignments.")

parser.add_argument("parallel_corpus", help="Pickle file with parallel sentences")
parser.add_argument("--partial", help="Project partial structures", action='store_true')
args = parser.parse_args()

parallel_sentences = pickle.load(open(args.parallel_corpus, "rb"))

pos_counter = count()
pos_vocab = defaultdict(pos_counter.__next__)
# pos_vocab = defaultdict(lambda: 0)
UNK_POS = pos_vocab['UNK']



def index_by_source(alignments):
    # FIXME here we are aligning roots and roots
    alignments_by_source = {0: [(0, 0, 1)]}

    for s_i, group_of_alignments in groupby(alignments, key=lambda t: t[0]):
        alignments_by_source[s_i] = list(group_of_alignments)

    return alignments_by_source


def build_arc_for_source(parallel_sent: ParallelSentence):
    arc_list = []

    for source_sent in parallel_sent.sources:
        alignments_by_source = index_by_source(source_sent.alignments)

        # Note that we are transposing the matrix here!
        # After transposing the cell (i, j) corresponds to the arc (i, j).
        source_row = source_sent.weights.col
        source_col = source_sent.weights.row
        source_data = source_sent.weights.data

        for i in range(source_row.shape[0]):
            s_i = source_row[i]
            s_j = source_col[i]
            assert s_j > 0
            source_edge_score = source_data[i]

            for ta_i, ta_j in product(alignments_by_source.get(s_i, []), alignments_by_source.get(s_j, [])):
                _, t_i, w_ii = ta_i
                _, t_j, w_jj = ta_j

                arc_list.append(Arc(u=t_i, v=t_j,
                                    u_pos=pos_vocab[source_sent.pos[s_i]],
                                    v_pos=pos_vocab[source_sent.pos[s_j]],
                                    weight=w_ii * w_jj * source_edge_score
                                    ))

    return arc_list


for sent_i, parallel_sent in enumerate(parallel_sentences):
    num_target_nodes = len(parallel_sent.target)
    all_arcs = build_arc_for_source(parallel_sent)

    # Take the max over all arcs that share (u, v, u_pos, v_pos)
    maxed_arcs = []
    all_arcs = sorted(all_arcs, key=lambda arc: (arc.u, arc.v, arc.u_pos, arc.v_pos, -arc.weight))
    for _, arc_group in groupby(all_arcs, key=lambda arc: (arc.u, arc.v, arc.u_pos, arc.v_pos)):
        max_arc = next(arc_group)
        maxed_arcs.append(max_arc)
        assert max_arc.u <= len(parallel_sent.target)
        assert max_arc.v <= len(parallel_sent.target)


    if args.partial:
        # Allow partial structures
        min_arc_weight = min(arc.weight for arc in maxed_arcs)
        for n in range(1, num_target_nodes):
            maxed_arcs.append(Arc(0, n, 0, UNK_POS, min_arc_weight - 0.01))
    else:
        # Do we have a possible head for each of the tokens?
        has_head = {arc.v for arc in maxed_arcs}
        if set(range(1, num_target_nodes)) != has_head:
            continue


    # Building model
    model = build_joint_model(maxed_arcs, num_nodes=num_target_nodes)
    model.write('model_{}.lp'.format(sent_i))
    model.setParam('LogToConsole', 0)
    model.optimize()

    if solution_exists(model):
        heads, pos = extract_solution(maxed_arcs, num_nodes=num_target_nodes)
        # print(parallel_sent.gold)
        print('SOLVED: ', sent_i)
        #print('WE GOT A SOLUTION')
        rev_pos_vocab = {i: pos_ for pos_, i in pos_vocab.items()}
        print(heads)
        print(pos)
        print([rev_pos_vocab[tag] for tag in pos[1:]])
    else:
        print("-", end=' ')