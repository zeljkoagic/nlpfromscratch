import argparse

import pickle
from collections import defaultdict
from itertools import groupby, product, count

from arc import Arc
from ilp_model import build_joint_model, extract_solution, solution_exists
from parallel_sentence import ParallelSentence

parser = argparse.ArgumentParser(description="Projects dependency trees from source to target via word alignments.")

parser.add_argument("parallel_corpus", help="Pickle file with parallel sentences")
args = parser.parse_args()

parallel_sentences = pickle.load(open(args.parallel_corpus, "rb"))

pos_counter = count()
pos_vocab = defaultdict(pos_counter.__next__)

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

                # SUBTLE code
                arc_list.append(Arc(u=t_i, v=t_j,
                                    u_pos=pos_vocab[source_sent.pos[s_i]],
                                    v_pos=pos_vocab[source_sent.pos[s_j]],
                                    weight=w_ii * w_jj * source_edge_score
                                    ))

    return arc_list


for parallel_sent in parallel_sentences:
    all_arcs = build_arc_for_source(parallel_sent)

    # Take the max over all arcs that share (u, v, u_pos, v_pos)
    maxed_arcs = []
    all_arcs = sorted(all_arcs, key=lambda arc: (arc.u, arc.v, arc.u_pos, arc.v_pos, -arc.weight))
    for _, arc_group in groupby(all_arcs, key=lambda arc: (arc.u, arc.v, arc.u_pos, arc.v_pos)):
        maxed_arcs.append(next(arc_group))

    # Building model
    model = build_joint_model(maxed_arcs, num_nodes=len(parallel_sent.target))
    model.optimize()

    if solution_exists(model):
        heads, pos = extract_solution(maxed_arcs, num_nodes=len(parallel_sent.target))
        print(heads)
        print(pos)
    else:
        print('X', end=' ')