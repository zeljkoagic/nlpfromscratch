import argparse
import pickle
from collections import defaultdict
from itertools import groupby, product, count, repeat
from math import exp
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score

from arc import Arc
from ilp_model import build_joint_model, extract_solution, solution_exists
from parallel_sentence import ParallelSentence
from utils.conll import write_conll06_sentence
from utils.vocab import Vocab

parser = argparse.ArgumentParser(description="Projects dependency trees from source to target via word alignments.")

parser.add_argument("parallel_corpus", help="Pickle file with parallel sentences")
parser.add_argument("--partial", help="Project partial structures", action='store_true')
parser.add_argument("--softmax", help="Apply softmax normalization after projection", action='store_true')
parser.add_argument("--coarse", help="Coarse POS for constraints", action='store_true')
parser.add_argument("--solution-file", help="Pickle solutions here. "
                                            "Solutions are aligned with the input parallel sentences",
                    type=Path)
parser.add_argument("--conll-file", help="Projected sentences written here", type=Path)

args = parser.parse_args()

parallel_sentences = pickle.load(open(args.parallel_corpus, "rb"))

if args.coarse:
    pos_vocab = Vocab(next_fn=repeat(1))
    pos_vocab.set('ROOT', -1)
    pos_vocab.set('UNK', 0)
    UNK_POS = 0
else:
    pos_vocab = Vocab()
    pos_vocab.set('ROOT', -1)
    UNK_POS = pos_vocab.map('UNK')


def index_by_source(alignments):
    # FIXME here we are aligning roots and roots
    alignments_by_source = {0: [(0, 0, 1)]}

    for s_i, group_of_alignments in groupby(alignments, key=lambda t: t[0]):
        alignments_by_source[s_i] = list(group_of_alignments)

    return alignments_by_source


def print_eval_solutions(parallel_sentences, solutions):
    # Evaluate on gold trees, if they exist
    predictions = defaultdict(list)
    sol_id = 0
    for parallel_sent, solution in zip(parallel_sentences, solutions):
        if solution is None:
            continue
        heads, pos_ids = solution

        # Extract predictions for evaluation
        if parallel_sent.gold:
            predictions['sol_id'].extend([sol_id] * (len(heads) - 1))
            predictions['pos'].extend(pos_ids[1:])
            predictions['head'].extend(heads[1:])

            tokens, gold_pos, gold_heads = parallel_sent.gold
            predictions['gold_head'].extend(gold_heads[1:])
            predictions['gold_pos'].extend(gold_pos[1:])

        sol_id += 1

    predictions = pd.DataFrame(predictions)

    print("{} gold trees projected".format(len(predictions.sol_id.unique())))
    print("{} partial tokens skipped".format((predictions['head'] == -1).sum()))
    nonpartial_mask = predictions['head'] >= 0
    predictions = predictions[nonpartial_mask]
    print("UAS: ", accuracy_score(predictions['gold_head'],
                                  predictions['head']))
    print("POS accuracy", accuracy_score(predictions['gold_pos'],
                                         predictions['pos']))


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


solutions = []
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

    # Normalize edge scores for each head decision by a soft-max
    if args.softmax:
        maxed_arcs = sorted(maxed_arcs, key=lambda arc: arc.v)
        for _, arc_group in groupby(maxed_arcs, key=lambda arc: arc.v):
            arc_list = list(arc_group)

            # Get normalization term
            softmax_norm = 0
            for arc in arc_list:
                arc.weight = exp(arc.weight)
                softmax_norm += arc.weight

            # Apply normalization
            for arc in arc_list:
                arc.weight /= softmax_norm

    if args.partial:
        # Allow partial structures
        # min_arc_weight = min(arc.weight for arc in maxed_arcs)
        # mean_arc_weight = sum(arc.weight for arc in maxed_arcs) / len(maxed_arcs)
        for n in range(1, num_target_nodes):
            maxed_arcs.append(Arc(0, n, 0, UNK_POS, 0))
    else:
        # Do we have a possible head for each of the tokens?
        has_head = {arc.v for arc in maxed_arcs}
        if set(range(1, num_target_nodes)) != has_head:
            solutions.append(None)
            continue

    # Building model
    model = build_joint_model(maxed_arcs, num_nodes=num_target_nodes)
    # model.write('model_{}.lp'.format(sent_i))
    model.setParam('LogToConsole', 0)
    model.optimize()

    if solution_exists(model):
        heads, pos_ids = extract_solution(maxed_arcs, num_nodes=num_target_nodes, unk_pos=UNK_POS)
        pos = list(map(pos_vocab.rev_map, pos_ids))
        solutions.append((heads, pos))
    else:
        solutions.append(None)
        print("-", end=' ')

if args.solution_file:
    with args.solution_file.open('wb') as out_file:
        pickle.dump(solutions, out_file, protocol=-1)

print_eval_solutions(parallel_sentences, solutions)

# Output CONLL file
if args.conll_file:
    with args.conll_file.open('w') as out_file:
        for parallel_sent, solution in zip(parallel_sentences, solutions):
            if solution is None:
                continue

            pred_heads, pred_pos = solution
            write_conll06_sentence(forms=parallel_sent.target,
                                   heads=pred_heads,
                                   pos_tags=pred_pos,
                                   file=out_file)