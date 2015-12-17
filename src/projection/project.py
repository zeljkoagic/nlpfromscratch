import argparse
from collections import defaultdict, Counter
import utils.alignments as align
import utils.conll as conll
import utils.normalize as norm
from functools import partial
import sys
import time

start_time = time.time()  # timing the script

parser = argparse.ArgumentParser(description="Projects dependency trees from source to target via word alignments.")

parser.add_argument("--source", required=True, help="source CoNLL file")
parser.add_argument("--target", required=True, help="target CoNLL file")
parser.add_argument("--word_alignment", required=True, help="word alignments file")
parser.add_argument("--sentence_alignment", required=True, help="sentence alignments file")
parser.add_argument("--norm_before", required=True, choices=["softmax", "rank", "intrank", "stdev", "identity"], help="normalization before projection")
parser.add_argument("--norm_after", required=True, choices=["softmax", "rank", "intrank", "stdev", "identity"], help="normalization after projection")
parser.add_argument('--with_pp', action='store_true', help="project POS with alignment probabilities instead unit votes")
parser.add_argument('--trees', action='store_true', help="project dependency trees instead of weight matrices")
parser.add_argument('--binary', action='store_true', help="use binary alignments instead of alignment probabilities")
parser.add_argument('--use_similarity', action='store_true', help="use word alignment-derived language similarity proxy")

args = parser.parse_args()

# choose weight matrix normalizers
normalizers = {"softmax": norm.softmax,
               "rank": partial(norm.rank, use_integers=False),
               "intrank": partial(norm.rank, use_integers=True),
               "stdev": norm.stdev_norm,
               "identity": lambda x: x}

normalize_before_projection = normalizers[args.norm_before]
normalize_after_projection = normalizers[args.norm_after]

# source sentence getter is determined by args.trees
source_data_getters = {True: conll.get_next_sentence_and_tree,
                       False: conll.get_next_sentence_and_graph}

get_source_data = source_data_getters[args.trees]

# get the sentence alignments
sentence_alignments = align.read_sentence_alignments(args.sentence_alignment)

# get the word alignments and source-target similarity
word_alignments, similarity = align.read_word_alignments(args.word_alignment)

# get the source and target conll file handlers
source_file_handle = open(args.source)
target_file_handle = open(args.target)

# sentence counters, word alignment counter for matching to sentence alignments
source_sid_counter = 0
target_sid_counter = 0
walign_counter = 0

for target_sentence in conll.sentences(target_file_handle, sentence_getter=conll.get_next_sentence):
    # target sentence found in sentence alignment
    if target_sid_counter in sentence_alignments:
        source_sid, sal_confidence = sentence_alignments[target_sid_counter]
    else:
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

    # now that the sentence ids match, get the sentence, POS, and graph from the source
    source_sentence, S, source_pos_tags, source_dep_labels = get_source_data(source_file_handle)

    # source and target sentences are retrieved, increment counters
    source_sid_counter += 1
    target_sid_counter += 1

    # source matrix normalization
    S = normalize_before_projection(S)

    # get word alignments for that sentence pair
    walign_pairs, walign_probs = word_alignments[walign_counter]
    walign_counter += 1

    # project parts of speech and dependency labels
    P = align.project_token_labels(source_pos_tags, walign_pairs, walign_probs, args.with_pp)
    L = align.project_token_labels(source_dep_labels, walign_pairs, walign_probs, args.with_pp)

    # project the dependencies from source to target
    m = len(source_sentence)
    n = len(target_sentence)

    A = align.get_alignment_matrix((m + 1, n + 1), walign_pairs, walign_probs, args.binary)
    T = align.project_dependencies_to_target(S, A)

    # normalize the target matrix
    T = normalize_after_projection(T)

    # apply the pair similarity factor TODO: Should we think about when we apply the similarity?
    if args.use_similarity:
        T *= similarity  # TODO Do we need to verify whether this is a good proxy for language similarity?

    # print the results
    for token in target_sentence:
        # get the POS projections for the current target token
        projected_tags = P.get(token.idx)# if token.idx in P else Counter({"_": 0})
        projected_labels = L.get(token.idx)# if token.idx in L else Counter({"_": 0})

        print("%s\t%s\t%s" % (" ".join(["{}:{}".format(t[0], t[1]) for t in projected_tags.most_common()]),
                              " ".join(["{}:{}".format(l[0], l[1]) for l in projected_labels.most_common()]),
                              " ".join(map(str, T[token.idx]))))
    print()

print("Execution time: %s sec" % (time.time() - start_time), file=sys.stderr)
