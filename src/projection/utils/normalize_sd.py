# SD normalization by AS

from __future__ import division
import sys
import numpy as np


def clip(x):
    if x < 0:
        return 0.0
    elif x > 1:
        return 1.0
    else:
        return x


def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out


def rank(X):
    y=[str((i+1)/len(X)) for i in range(len(X))]
    return dict(zip(sorted(X),y))


def intrank(X):
    y=[str((i+1)) for i in range(len(X))]
    return dict(zip(sorted(X),y))


def quantilize(X):
    D={}
    SD=np.std(X)
    M=np.mean(X)
    Z=[clip(0.5 + float(x) - M / (2*SD)+0.0000001) for x in X]
    D_prime=dict(zip(X,Z))
    y=chunkIt(Z,3)
    for x in X:
        for chunk in y:
            if D_prime[x] in chunk:
                D[x]=str(np.array(chunk).mean())
    return dict(D)


def normalize(X):
    SD=np.std(X)
    M=np.mean(X)
    D=dict(zip(
        X,[str(clip(0.5 + float(x) - M / (2*SD)+0.0000001)) for x in X]
    ))
    return dict(D)

conll = [l.strip().split() for l in open(sys.argv[1]).readlines()]

sent = []

for c in conll:
    if len(c) < 2:
        for w in sent:
            numbers=[(int(x.split(":")[0]),float(x.split(":")[1])) for x in w[8:]]
            numbers=dict(numbers)
            numbers_list=[(float(x.split(":")[1])) for x in w[8:]]

            if sys.argv[2] == "normalize":
                D=normalize(numbers_list)
            elif sys.argv[2] == "rank":
                D=rank(numbers_list)
            elif sys.argv[2] == "intrank":
                D=intrank(numbers_list)
            elif sys.argv[2] == "quantilize":
                D=quantilize(numbers_list)
            else:
                raise Exception("sys.argv[2] must be one of: normalize, rank, intrank, quantilize")

            string="\t".join(w[:8])+'\t'
            weights=[]
            for i in range(len(sent)+1):
                if i in numbers:
                    weights.append(str(i) + ":" + D[numbers[i]])
                    #weights.append(D[numbers[i]])
                else:
                    weights.append(str(i) + ":0.0")
                    #weights.append("0.0")
            string+="\t".join(weights)
            print(string)
        sent = []
        print()
    else:
        sent.append(c)
