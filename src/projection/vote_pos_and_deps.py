import argparse
import numpy as np
from collections import Counter
import utils.conll as conll
# import mst.cle as cle
import string
import sys
import time
import copy
from pathlib import Path
import utils.score as score
from dependency_decoding import chu_liu_edmonds
import warnings
import math
import utils.normalize as norm


def add_root_row(tensor):
    first_row = np.ones([1, tensor.shape[1], tensor.shape[2]]) * np.nan
    return np.vstack([first_row, tensor])


def eliminate_all_nan_rows(M_proj):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        by_row = np.nanmax(M_proj, axis=1)
        all_nan_rows = np.isnan(by_row)
        M_proj[all_nan_rows] = np.nanmin(M_proj)


def vote_weight_matrix(sentence_tensor):
    """
    TODO
    :param sentence_tensor:
    :return:
    """
    return np.nansum(sentence_tensor, axis=2)

start_time = time.time()  # timing the script

parser = argparse.ArgumentParser(description="Voting and CLE decoding on projected labels and weight matrices.")

parser.add_argument("--target", required=True, help="target CoNLL file", type=Path)
parser.add_argument("--projections", required=True, help="path to vote files", type=Path, nargs="+")
parser.add_argument("--stop_after", required=False, help="stop after n sentences")
parser.add_argument('--unit_vote_pos', required=True, choices=[0, 1], help="use unit votes for POS tag voting", type=int)
parser.add_argument('--pretagged', action='store_true', help="use preassigned target POS tags instead of voted tags")
parser.add_argument('--dump_npz', action='store_true', help="dump NPZ debug files")
parser.add_argument('--skip_untagged', action='store_true', help="skip sentences with untagged tokens")
parser.add_argument('--decode', action='store_true', help="perform CLE decoding")

args = parser.parse_args()

# TODO
pos_vote_casts = {1: math.ceil,
                  0: lambda x: x}
pos_vote_caster = pos_vote_casts[args.unit_vote_pos]

target_file_handle = args.target.open()

token_count = 0
sentence_count = 0
scorer = score.TokenScorer()  # for scoring

# current sentence information: tags, dependency labels, tensor, and source languages list
current_pos_tags = []
# current_dep_labels = []
current_sentence_tensor = []
current_sentence_source_languages = []

skip_sentence = False  # skip sentences with empty sources

vote_handles = [projection_file.open() for projection_file in args.projections]


for lines in zip(*vote_handles):

    if lines[0] != "\n" and not skip_sentence:
        current_sentence_source_languages.clear()  # TODO: This is redundant! Should be ordered set.

        # voting for tags, dependency labels and heads
        overall_pos_votes = Counter()
        # overall_label_votes = Counter()
        overall_head_votes = None

        projection_weights_per_token = []
        current_sentence_tensor.append(projection_weights_per_token)

        # iterate through blocks provided by singled-out sources
        for source in lines:
            labels_and_heads = source.split("\t")
            # assert len(labels_and_heads) == 3
            if len(labels_and_heads) != 3:
                continue  # skip empty source

            # get POS and dependencies parts
            source_language_name, source_pos_votes, source_head_votes = labels_and_heads
            current_sentence_source_languages.append(source_language_name)

            # turn POS part & label part into Counters
            source_pos_votes = source_pos_votes.split()
            source_pos_counter = Counter()
            # source_label_votes = source_label_votes.split()
            source_label_counter = Counter()

            for vote in source_pos_votes:
                pos, num = vote.split(":")
                source_pos_counter.update({pos: pos_vote_caster(float(num))})

            #for vote in source_label_votes:
            #    label, num = vote.split(":")
            #    source_label_counter.update({label: int(num)})

            # add single source counts to the overall pool
            overall_pos_votes.update(source_pos_counter)
            # overall_label_votes.update(source_label_counter)

            # collect heads for current source
            head_scores = list(map(float, source_head_votes.strip().split()))
            for i, val in enumerate(head_scores):
                if i >= len(projection_weights_per_token):
                    projection_weights_per_token.append([])
                projection_weights_per_token[i].append(val)

        if not len(projection_weights_per_token):  # if none of the source languages contributed any projections
            skip_sentence = True
            continue

        current_pos_tags.append(overall_pos_votes.most_common(1)[0][0])
        # current_dep_labels.append(overall_label_votes.most_common(1)[0][0])

        # skip sentences with at least one placeholder "_" POS tag (or dependency label?)
        if args.skip_untagged and current_pos_tags[-1] == "_":  # or current_dep_labels[-1] == "_": TODO everything gets skipped if dep=="_"!
            skip_sentence = True

    elif lines[0] == "\n":

        current_sentence = conll.get_next_sentence(target_file_handle)  # has to be run even if skip_sentence == True!

        if not skip_sentence:

            sentence_count += 1

            # construct a 3-dim tensor where each slice along the third dimension corresponds
            # to a weight matrix for a given source language
            current_sentence_tensor = add_root_row(np.array(current_sentence_tensor))

            # FIXME Dummy to get stats for number of contributing languages
            print(current_sentence_tensor.shape[2])
            current_sentence_tensor = []
            continue

            # unify the source language matrices into a single a matrix
            # first we sum, then per-row normalize using softmax
            current_sentence_matrix = norm.softmax(vote_weight_matrix(current_sentence_tensor))
            eliminate_all_nan_rows(current_sentence_matrix)

            # dump the raw projections into a file, for debug purposes
            if args.dump_npz:
                raw_projections_filename = "{}.{}".format(args.target.name.split('.', 1)[0], sentence_count)
                np.savez(raw_projections_filename,
                         projection_tensor=current_sentence_tensor,
                         source_languages=current_sentence_source_languages,
                         heads=[token.head for token in current_sentence],
                         tokens=[token.form for token in current_sentence])

            if args.decode:
                # decoded_heads = cle.mdst(current_sentence_matrix)  # do the MST magic TODO Change to new CLE!!!
                decoded_heads, tree_score = chu_liu_edmonds(current_sentence_matrix)
                decoded_heads = decoded_heads[1:]
            else:
                decoded_heads = [0 for _ in current_sentence]

            jt = 0
            for token in current_sentence:
                old_token = copy.copy(token)  # for evaluation

                # get the decoded head
                token.head = decoded_heads[jt]
                # token.head = np.argmax(current_sentence_matrix[jt, ])  # placeholder, per-token voting

                # get the voted POS tag and dependency label
                # token.deprel = current_dep_labels[jt]
                if not args.pretagged:
                    token.cpos = "PUNCT" if token.form in string.punctuation else current_pos_tags[jt]

                # evaluation TODO Makes sense only if the target language is one of the source languages
                scorer.update(old_token, token)

                # print(token, " ".join(map(str, current_sentence_matrix[jt, ])))
                print(token)  # we don't need the weights anymore
                jt += 1
            print()

        skip_sentence = False  # we don't yet know whether to skip the next one or not, so reset the flag
        current_sentence_tensor = []
        current_pos_tags = []
        current_dep_labels = []

        if args.stop_after and int(args.stop_after) == sentence_count:
            break

# assert all(h.read() == "" for h in vote_handles), "Projections differ in size"

print("Scores:", " ".join(map(str, scorer.get_score_list())), file=sys.stderr)
print("Execution time: %s sec" % (time.time() - start_time), file=sys.stderr)
