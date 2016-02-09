import argparse
from pathlib import Path

import conll
from itertools import islice

parser = argparse.ArgumentParser(description="Convert a CONLL file into a file ready for the `jdageem` software.")

parser.add_argument("conll_file", type=Path)
parser.add_argument("--heads", help="Output heads for evaluation", action='store_true')
parser.add_argument("--stop_after", help="Only output this many", type=int)
parser.add_argument("--length_cutoff", help="Supress outputting sentences longer than this cutoff", type=int)
args = parser.parse_args()

sent_id = 0
with args.conll_file.open() as conll_file:
    for tokens in islice(conll.sentences(conll_file, conll.get_next_sentence), args.stop_after):
        tags = [token.cpos for token in tokens]
        heads = [str(token.head) for token in tokens]
        assert len(tags) == len(heads)

        if args.length_cutoff and len(tags) > args.length_cutoff:
            continue

        if sent_id > 0 and args.heads:
            print()

        print(" ".join(tags))
        if args.heads:
            print(" ".join(heads))

        sent_id += 1
