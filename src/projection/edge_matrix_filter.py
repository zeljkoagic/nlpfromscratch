import argparse
import numpy as np
from collections import Counter
import utils.conll as conll
import string
import sys
import time
import copy
from pathlib import Path
import pandas as pd
import mst.cle as cle


NEGINF = float("-inf")
ONLYLEAF = set("ADP AUX CONJ DET PUNCT PRT SCONJ".split())
CONTENT = set("ADJ NOUN VERB PROPN".split())

def ud_filter(P,M_in):
    M = copy.copy(M_in)


    #TODO: Flag each step w a binary parameter
    #1.- enforce leafness of certain POS
    should_be_leaves = [i+1 for i,pos in enumerate(P) if pos in ONLYLEAF] #the +1 is to adjust the column indices
    for row in M:
        row = [NEGINF if i in should_be_leaves else x for i,x in enumerate(row)]

    #2.- If there is no verb, promote first content word to head
    # Note that a copula verb should be AUX for this to work, this convention is f.i. not followed in the English UD
    if "VERB" not in P and CONTENT.intersection(set(P)):
        first_content_word = 0
        for i,p in enumerate(P):
             if p in CONTENT:
                first_content_word = i
                likeliest_root = i
                break
        for i in range(M.shape[0]):
            if i != first_content_word:
                M[i,0] = NEGINF

    # 3.- enforce single root
    max_root_score = max(M[:, 0])
    root_counter = 0  # leaves only one possible root, in the binarized case it keeps the very first one
    for i in range(M.shape[0]):
        if root_counter > 0:
            M[i, 0] = NEGINF
        elif M[i, 0] < max_root_score:
            M[i, 0] = NEGINF
        else:
            likeliest_root = i
            root_counter += 1

    # 4.- Attach last punctuation to likeliest root node
    if P[-1] == "PUNCT":
        for x in range(M.shape[1]):
            if x != likeliest_root:
                M[-1,x]=NEGINF
            else:
                M[-1, x] = M_in[-1,likeliest_root]

    print(cle.mdst(M))

    return M


def printmatrix(M):
    for r in M:
        print(" ".join([str(x) for x in r]))

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--infile", help='Files with projection tensors and gold parses',
                        default='/Users/hmartine/proj/nlpfromscratch/data/conll/en.bible100.conll')
    args = parser.parse_args()

    P = [] # list to store POS values
    L = [] #dictionary list to later build a matrix proper

    for line in open(args.infile):
        line = line.strip()
        if line:
            parts = line.split("\t")
            #retrieve column row
            kv_pairs = [kv_pair.split(":") for kv_pair in parts[-1].split()]
            score_dict = {int(k): float(v) for k, v in kv_pairs}
            L.append(score_dict)
            P.append(parts[3])
        else:
            M = []
            dimensions = len(L) + 1 #the 0-th position encodes root attachment edges
            for rindex,rowdict in enumerate(L):
                R = [NEGINF] * dimensions
                for k in rowdict.keys():
                    R[k]=rowdict[k]
                M.append(R)
            M = np.array(M)
            Mfiltered = ud_filter(P,M)
            printmatrix(Mfiltered)
            print()
            P = []  # list to store POS values
            L = []  # dictionary list to later build a matrix proper


if __name__ == "__main__":
    main()