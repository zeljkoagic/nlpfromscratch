import argparse
import numpy as np
from collections import Counter
import utils.conll as conll
import mst.cle as cle
import string
import sys
import time
import copy

start_time = time.time()  # timing the script

parser = argparse.ArgumentParser(description="Votes.")

parser.add_argument("--target", required=True, help="target CoNLL file")
parser.add_argument("--votes", required=True, help="file with all votes merged")
parser.add_argument('--inner_vote_pos', action='store_true', help="intra-language POS tag voting")
parser.add_argument('--inner_vote_labels', action='store_true', help="intra-language dependency label voting")
parser.add_argument('--pretagged', action='store_true', help="use preassigned target POS tags instead of voted tags")

args = parser.parse_args()
target_file_handle = open(args.target)


def update_scores(gold_token, system_token):
    update = np.array([0.0, 0.0, 0.0, 0.0])
    if system_token.cpos == gold_token.cpos:
        update[0] = 1.0
    if system_token.head == gold_token.head and system_token.deprel == gold_token.deprel:
        update[1] = 1.0
    if system_token.head == gold_token.head:
        update[2] = 1.0
    if system_token.deprel == gold_token.deprel:
        update[3] = 1.0
    return update

count = 0
scores = np.array([0.0, 0.0, 0.0, 0.0])  # POS, LAS, UAS, LA

current_sentence_matrix = []
current_pos_tags = []
current_dep_labels = []
skip_sentence = False  # skip sentences with empty sources

for line in open(args.votes):
    line = line.strip()

    if line and not skip_sentence:
        # voting for tags, dependency labels and heads
        overall_pos_votes = Counter()
        overall_label_votes = Counter()
        overall_head_votes = None

        # a line has n source languages delimited by "#"
        sources = line.split("#")

        # iterate through blocks provided by singled-out sources
        for source in sources:
            labels_and_heads = source.split("\t")
            if len(labels_and_heads) != 3:
                continue  # skip empty source

            # get POS and dependencies parts
            source_pos_votes, source_label_votes, source_head_votes = labels_and_heads

            # turn POS part & label part into Counters
            source_pos_votes = source_pos_votes.split()
            source_pos_counter = Counter()
            source_label_votes = source_label_votes.split()
            source_label_counter = Counter()

            for vote in source_pos_votes:
                pos, num = vote.split(":")
                source_pos_counter.update({pos: int(num)})
                if args.inner_vote_pos:
                    break

            for vote in source_label_votes:
                label, num = vote.split(":")
                source_label_counter.update({label: int(num)})
                if args.inner_vote_labels:  # TODO Do we want inner voting to apply for dependency labels too?
                    break

            # add single source counts to the overall pool
            overall_pos_votes.update(source_pos_counter)
            overall_label_votes.update(source_label_counter)

            # collect heads for current source
            source_head_votes = np.array(list(map(float, source_head_votes.strip().split())))

            # do weight summing
            if overall_head_votes is None:
                overall_head_votes = source_head_votes
            else:
                overall_head_votes += source_head_votes

        if overall_head_votes is None:
            skip_sentence = True  # all sources were empty, skip sentence
            continue

        current_pos_tags.append(overall_pos_votes.most_common(1)[0][0])
        current_dep_labels.append(overall_label_votes.most_common(1)[0][0])
        current_sentence_matrix.append(overall_head_votes)

        # skip sentences with at least one placeholder "_" POS tag or dependency label
        if current_pos_tags[-1] == "_":  # or current_dep_labels[-1] == "_": TODO everything gets skipped if dep=="_"!
            skip_sentence = True

    elif not line:

        current_sentence = conll.get_next_sentence(target_file_handle)  # has to be run even if skip_sentence is True!

        if not skip_sentence:

            current_sentence_matrix = np.array(current_sentence_matrix)  # TODO Should be np.array to begin with!
            decoded_heads = cle.mdst(np.array(current_sentence_matrix))  # do the MST magic

            jt = 0
            for token in current_sentence:
                old_pos = token.cpos
                old_head = token.head
                old_label = token.deprel

                old_token = copy.copy(token)

                # get the decoded head
                token.head = decoded_heads[jt]
                # token.head = np.argmax(current_sentence_matrix[jt, ])  # placeholder, per-token voting

                # get the voted POS tag and dependency label
                if not args.pretagged:
                    token.cpos = "PUNCT" if token.form in string.punctuation else current_pos_tags[jt]

                token.deprel = current_dep_labels[jt]

                # evaluation TODO Makes sense only if the target language is one of the source languages
                count += 1
                scores += update_scores(old_token, token)

                # print(token, " ".join(map(str, current_sentence_matrix[jt, ])))
                print(token)  # we don't need the weights anymore
                jt += 1
            print()

        skip_sentence = False
        current_sentence_matrix = []
        current_pos_tags = []
        current_dep_labels = []

scores = scores / count
print(" ".join(map(str, scores)), time.time() - start_time, file=sys.stderr)
