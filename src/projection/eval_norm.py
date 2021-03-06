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
from functools import partial

start_time = time.time()  # timing the script

parser = argparse.ArgumentParser(description="")
parser.add_argument("projection_files", help='Files with projection tensors and gold parses', type=Path, nargs='+')
args = parser.parse_args()


def softmax(matrix, temperature=1):
    m_exp = np.exp(matrix/temperature)
    return (m_exp.T / m_exp.sum(axis=1)).T

def sum_normalize(T_proj):
    return T_proj.sum(axis=2)

def threshold_normalize(T_proj, threshold=0.1, binary=False):
    mask = T_proj < threshold
    T_proj[mask] = 0.0
    if binary:
        T_proj[~mask] = 1.0

    return sum_normalize(T_proj)

def row_normalize(M):
    return (M.T / M.sum(axis=1)).T

def secret_normalize(T_proj):
    #for i in range(T_proj.shape[2]):
    #    T_proj[:, :, i] = (T_proj[:, :, i].T * T_proj[:, :, i].sum(axis=1)).T

    T_proj -= T_proj.mean()
    T_proj /= T_proj.std()


    M_norm = sum_normalize(T_proj)

    return M_norm

    #return softmax(M_norm, 1)

def get_prediction_score(filename, normalize_fn):
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
# normalize_fn = partial(threshold_normalize, threshold=(threshold / 1000)*5)
normalize_fn = secret_normalize
score_fn = partial(get_prediction_score, normalize_fn=normalize_fn)
scores = pool.map(score_fn, args.projection_files)
print(pd.Series(scores).describe())

