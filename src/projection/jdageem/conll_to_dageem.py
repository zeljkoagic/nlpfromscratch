import argparse
from pathlib import Path

import conll

parser = argparse.ArgumentParser(description="Convert a CONLL file into a file ready for the `jdageem` software.")

parser.add_argument("conll_file", type=Path)
args = parser.parse_args()

sent_id = 0
with args.conll_file.open() as conll_file:
    for tokens in conll.sentences(conll_file, conll.get_next_sentence):
        forms = [token.form for token in tokens]
        heads = [str(token.head) for token in tokens]
        assert len(forms) == len(heads)

        if sent_id > 0:
            print()

        print(" ".join(forms))
        print(" ".join(heads))

        sent_id += 1
