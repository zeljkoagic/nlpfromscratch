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
from multiprocessing import Pool
import pandas as pd

start_time = time.time()  # timing the script

parser = argparse.ArgumentParser(description="")
parser.add_argument("projection_files", help='Files with projection tensors and gold parses', type=Path, nargs='+')
args = parser.parse_args()

def sum_normalize(T_proj):
    return T_proj.sum(axis=2)

def threshold_normalize(T_proj, threshold=0.1, binary=False):
    mask = T_proj < threshold
    T_proj[mask] = 0.0
    if binary:
        T_proj[~mask] = 1.0

    return T_proj

normalize_fn = threshold_normalize

def get_prediction_score(filename):
    data = np.load(str(filename))
    T_projection = data['projection_tensor']
    source_languages = list(data['source_languages'])
    heads = list(data['heads'])

    # Normalization
    M_projection = normalize_fn(T_projection)
    decoded_heads = cle.mdst(M_projection)

    assert len(heads) == len(decoded_heads)
    num_correct = sum(pred_head == gold_head for gold_head, pred_head in zip(heads, decoded_heads))
    uas = num_correct / len(heads)
    return uas


pool = Pool(processes=20)
scores = pool.map(get_prediction_score, args.projection_files)
print(pd.Series(scores).describe())
