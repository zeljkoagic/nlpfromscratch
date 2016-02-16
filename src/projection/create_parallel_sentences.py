import argparse
from collections import defaultdict
from pathlib import Path

import pickle

from utils import conll

from parallel_sentence import SourceSentence, ParallelSentence
from utils.alignments import read_word_alignments
from utils.conll import get_next_sentence_and_graph

import pandas as pd

def read_sents_from_conll(conll_file):
    pass


def read_parses(conll_file):
    parses = []
    with src_conll_file.open() as conll_fh:
        for tokens_and_weights_and_pos in conll.sentences(conll_fh, sentence_getter=get_next_sentence_and_graph):
            tokens, weights, pos = tokens_and_weights_and_pos
            forms = ['ROOT'] + [token.form for token in tokens]
            pos = ['ROOT'] + pos

            assert len(forms) == len(pos)
            assert len(forms) == weights.shape[0]
            assert len(forms) == weights.shape[1]

            parses.append((forms, pos, weights))

    return parses

parser = argparse.ArgumentParser(description="Projects dependency trees from source to target via word alignments.")
parser.add_argument("--pairs", required=True, help="Pairs", nargs="+")
parser.add_argument("--corpus", required=True, help="Name of corpus")
parser.add_argument("--base-dir", required=True, help="Root of preprocessed dir", type=Path)
parser.add_argument('--out', required=True, help="Output file", type=Path)

args = parser.parse_args()

# Reading source sentences
source_sents_by_target = defaultdict(list)
for pair in args.pairs:
    src_lang, _ = pair.split("-")

    sent_align_file = (args.base_dir / 'salign' / "{}.{}.sal".format(pair, args.corpus))
    sent_align = pd.read_csv(str(sent_align_file), sep="\t", names=['src_sent_id', 'target_sent_id', 'prob'])

    word_align_file = (args.base_dir / 'walign' / "{}.{}.ibm1.reverse.wal".format(pair, args.corpus))
    word_align = read_word_alignments(word_align_file)

    # CONLL file
    src_conll_file = (args.base_dir / 'conll' / '{}.{}.conll'.format(src_lang, args.corpus))
    src_parses = read_parses(src_conll_file)

    for i, src_parse in enumerate(src_parses):
        forms, pos, weights = src_parse

        target_ids = sent_align.query('src_sent_id = @i').target_sent_id
        if len(target_ids):
            target_id = target_ids.iloc[0]
            pair_id = target_ids.index[0]

            source_sent = SourceSentence(weights, pos, forms, language=src_lang,
                                         alignments=word_align[pair_id])
            source_sents_by_target[target_id].append(source_sent)


# Read in target
target_langs = {pair.split("-")[0] for pair in args.pairs}
assert len(target_langs) == 1
target_lang = list(target_langs)[0]

target_sent_file = (args.base_dir / 'conll' / '{}.{}.conll'.format(target_lang, args.corpus))
target_sents = read_sents_from_conll(target_sent_file)

target_gold_file = (args.base_dir / 'gold' / '{}.{}.conll'.format(target_lang, args.corpus))
if target_gold_file.is_file():
    target_gold_parses = read_parses(target_gold_file)

# Assemble parallel sentences
parallel_sents = []
for target_sent_id, target_sent in enumerate(target_sents):
    parallel_sent = ParallelSentence(target=target_sent,
                                     sources=source_sents_by_target[target_sent_id])
    if len(parallel_sent.sources) >= 1:
        parallel_sents.append(parallel_sent)

with args.out.open('wb') as out_fh:
    pickle.dump(parallel_sents, out_fh, protocol=pickle.HIGHEST_PROTOCOL)