import argparse
import numpy as np
from collections import Counter
import utils.conll as conll
import mst.cle as cle
import string
import sys
import time
import copy
from pathlib import Path

start_time = time.time()  # timing the script

parser = argparse.ArgumentParser(description="")
parser.add_argument("projection_files", help='Files with projection tensors and gold parses', type=Path, nargs='+')
args = parser.parse_args()

def sum_normalize(T_proj):
    return T_proj.sum(axis=2)

for projection_file in args.projection_files:
    data = np.load(str(projection_file))
    T_projection = data['projection_tensor']
    source_languages = list(data['source_languages'])
    heads = list(data['heads'])

    # Normalization
    M_projection = sum_normalize(T_projection)
    decoded_heads = cle.mdst(M_projection)

    assert len(heads) == len(decoded_heads)
    num_correct = sum(pred_head == gold_head for gold_head, pred_head in zip(heads, decoded_heads))
    uas = num_correct / len(heads)
    print('{:.2f}'.format(uas))