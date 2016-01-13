import argparse
from collections import Counter
import utils.alignments as align
import utils.conll as conll
import utils.normalize as norm
from functools import partial
import sys
import time
from pathlib import Path
from scipy import sparse
import numpy as np
import pyximport; pyximport.install()
import utils.project_deps as project
from utils.coo_matrix_nocheck import CooMatrix

start_time = time.time()  # timing the script

# choose weight matrix normalizers
normalizers = {"softmax": None,
               "rank": partial(norm.rank, use_integers=False),
               "intrank": partial(norm.rank, use_integers=True),
               "stdev": norm.stdev_norm,
               "identity": lambda x: x,
               "standardize": norm.standardize}

parser = argparse.ArgumentParser(description="Projects dependency trees from source to target via word alignments.")

parser.add_argument("--source", required=True, help="source CoNLL file", type=Path)
parser.add_argument("--target", required=True, help="target CoNLL file")
parser.add_argument("--word_alignment", required=True, help="word alignments file")
parser.add_argument("--sentence_alignment", required=True, help="sentence alignments file")
parser.add_argument("--norm_before", required=True, choices=normalizers.keys(), help="normalization before projection")
parser.add_argument("--norm_after", required=True, choices=normalizers.keys(), help="normalization after projection")
parser.add_argument('--with_pp', required=True, choices=[0, 1], help="project POS with alignment probabilities instead unit votes", type=int)
parser.add_argument('--trees', required=True, choices=[0, 1], help="project dependency trees instead of weight matrices", type=int)
parser.add_argument('--binary', required=True, choices=[0, 1], help="use binary alignments instead of alignment probabilities", type=int)
parser.add_argument('--use_similarity', action='store_true', help="use word alignment-derived language similarity proxy")
parser.add_argument("--stop_after", required=False, help="stop after n sentences")
parser.add_argument("--temperature", required=False, help="softmax temperature", type=float, default=1.0)

args = parser.parse_args()

normalizers['softmax'] = partial(norm.softmax, temperature=args.temperature)

source_language_name = args.source.stem.split(".", 1)[0]

normalize_before_projection = normalizers[args.norm_before]
normalize_after_projection = normalizers[args.norm_after]

if args.trees:
    normalize_before_projection = normalizers["identity"]

# source sentence getter is determined by args.trees
source_data_getters = {0: conll.get_next_sentence_and_graph,
                       1: conll.get_next_sentence_and_tree}

get_source_data = source_data_getters[args.trees]

# get the sentence alignments
sentence_alignments = align.read_sentence_alignments(args.sentence_alignment)

# get the word alignments and source-target similarity
word_alignments, similarity = align.read_word_alignments(args.word_alignment)

# get the source and target conll file handlers
source_file_handle = args.source.open()
target_file_handle = open(args.target)

# sentence counters, word alignment counter for matching to sentence alignments
source_sid_counter = 0
target_sid_counter = 0
walign_counter = 0

for target_sentence in conll.sentences(target_file_handle, sentence_getter=conll.get_next_sentence):

    if args.stop_after and int(args.stop_after) == target_sid_counter:
        break

    # target sentence found in sentence alignment, get source sentence id and confidence
    if target_sid_counter in sentence_alignments:
        source_sid, sal_confidence = sentence_alignments[target_sid_counter]
        # get word alignments for that sentence pair
        walign_pairs, walign_probs = word_alignments[walign_counter]
        walign_counter += 1

    if (target_sid_counter not in sentence_alignments) or (len(walign_probs) == 0):
        for _ in target_sentence:  # if not found, just print out dummy to maintain the number of lines/sentences
            print("_")
        print()
        target_sid_counter += 1
        continue

    # skip source and target sentences that are not in the sentence alignments
    while source_sid_counter != source_sid:
        _ = conll.get_next_sentence_and_graph(source_file_handle)
        source_sid_counter += 1

    # check if sentence ids match
    assert source_sid_counter == source_sid

    # now that the sentence ids match and word alignments are in place,
    # get the sentence, POS, and graph from the source
    source_sentence, S_sparse, source_pos_tags = get_source_data(source_file_handle)

    # source and target sentences are retrieved, increment counters
    source_sid_counter += 1
    target_sid_counter += 1

    # source matrix normalization
    S = np.full(S_sparse.shape, fill_value=np.nan)
    S[S_sparse.row, S_sparse.col] = S_sparse.data
    S = normalize_before_projection(S)
    non_nan_mask = np.argwhere(~np.isnan(S))
    rows = non_nan_mask[:, 0]
    cols = non_nan_mask[:, 1]
    S_sparse = CooMatrix(rows, cols, S[rows, cols], S.shape)

    # project parts of speech and dependency labels
    P = align.project_token_labels(source_pos_tags, walign_pairs, walign_probs, args.with_pp)
    # L = align.project_token_labels(source_dep_labels, walign_pairs, walign_probs, args.with_pp)

    # project the dependencies from source to target
    m = len(source_sentence)
    n = len(target_sentence)

    A_sparse = align.get_alignment_matrix((m + 1, n + 1), walign_pairs, walign_probs, args.binary)
    T = project.project_dependencies_faster(S_sparse, A_sparse)  # We now use sparse matrices

    # normalize the target matrix
    T = normalize_after_projection(T)

    # TODO Do we need to verify whether this is a good proxy for language similarity?
    # TODO: Should we think about when we apply the language pair similarity?
    # apply the pair similarity factor
    if args.use_similarity:
        T *= similarity

    # print the results
    for token in target_sentence:
        # get the POS projections for the current target token
        projected_tags = P.get(token.idx) if token.idx in P else Counter({"_": 0})
        # projected_labels = L.get(token.idx) if token.idx in L else Counter({"_": 0})

        print("%s\t%s\t%s" % (source_language_name,
                              " ".join(["{}:{}".format(t[0], t[1]) for t in projected_tags.most_common()]),
                              # " ".join(["{}:{}".format(l[0], l[1]) for l in projected_labels.most_common()]),
                              " ".join(map(str, T[token.idx]))))
    print()

print("Execution time: %s sec" % (time.time() - start_time), file=sys.stderr)
