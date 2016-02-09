import argparse
import numpy as np
from pathlib import Path
import utils.conll as conll
from dependency_decoding import chu_liu_edmonds

parser = argparse.ArgumentParser(description="Sagae & Lavie (2006) decoding on delexicalized parses of the target.")
parser.add_argument("--parses", required=True, help="path to individual CoNLL files", type=Path, nargs="+")

args = parser.parse_args()
vote_handles = [projection_file.open() for projection_file in args.parses]

while True:

    # get sentences and trees for all source parses
    sentences_and_trees = [conll.get_next_sentence_and_tree_old(handle) for handle in vote_handles]

    # save the sentence tokens for later printout
    the_sentence = sentences_and_trees[0][0]

    # we hit EOF
    if not the_sentence:
        break

    # instantiate the votes graph
    votes_graph = np.zeros(sentences_and_trees[0][1].shape)

    # collect the votes
    for sentence, tree, _ in sentences_and_trees:  # the last item is the list of POS tags
        votes_graph += tree

    # print(votes_graph)
    # run MST to get heads
    decoded_heads, score = chu_liu_edmonds(votes_graph)

    it = 1
    for token in the_sentence:
        token.head = decoded_heads[it]  # assign the newly-decoded head
        it += 1
        print(token)
    print()
