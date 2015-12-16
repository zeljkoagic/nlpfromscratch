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
normalize = {"softmax": norm.softmax,
             "rank": partial(norm.rank, use_integers=False),
             "intrank": partial(norm.rank, use_integers=True),
             "stdev": norm.stdev_norm,
             "identity": lambda x: x}

normalize_before_projection = normalize[args.norm_before]
normalize_after_projection = normalize[args.norm_after]

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

# iterate through sentence alignments
for source_sid, target_data in sentence_alignments.items():
    target_sid, sal_confidence = target_data

    # skip source and target sentences that are not in the sentence alignments
    while source_sid_counter != source_sid:
        _ = conll.get_next_sentence_and_graph(source_file_handle)
        source_sid_counter += 1

    while target_sid_counter != target_sid:
        target_sentence = conll.get_next_sentence(target_file_handle)
        # print to ensure the same number of lines for each projection to target
        for _ in target_sentence:
            print("_")
        print()
        target_sid_counter += 1

    # check if sentence ids match
    assert source_sid_counter == source_sid
    assert target_sid_counter == target_sid

    # now that the sentence ids match, get the sentence, POS, and graph from the source
    if args.trees:
        source_sentence, S, source_pos_tags, source_dep_labels = conll.get_next_sentence_and_tree(source_file_handle)
    else:
        source_sentence, S, source_pos_tags, source_dep_labels = conll.get_next_sentence_and_graph(source_file_handle)

    # source matrix normalization
    S = normalize_before_projection(S)

    # for the target, we just need the sentence part
    target_sentence = conll.get_next_sentence(target_file_handle)

    source_sid_counter += 1
    target_sid_counter += 1

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
        projected_tags = P.get(token.idx)
        projected_labels = L.get(token.idx)

        if projected_tags is None:
            projected_tags = Counter({"_": 0})  # fix the target tokens with no projected tags
        if projected_labels is None:
            projected_labels = Counter({"_": 0})  # fix the target tokens with no projected tags

        print("%s\t%s\t%s" % (" ".join(["{}:{}".format(t[0], t[1]) for t in projected_tags.most_common()]),
                              " ".join(["{}:{}".format(l[0], l[1]) for l in projected_labels.most_common()]),
                              " ".join(map(str, T[token.idx]))))
    print()

print("Execution time: %s sec" % (time.time() - start_time), file=sys.stderr)
