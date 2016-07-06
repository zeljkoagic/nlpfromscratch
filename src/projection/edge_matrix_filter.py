import argparse
import numpy as np
from collections import Counter
import utils.conll as conll
import string
import sys

import copy

from utils.is_projective import is_projective
import networkx as nx
from dependency_decoding import chu_liu_edmonds
NEGINF = float("-inf")
ONLYLEAF = set("ADP AUX CONJ DET PUNCT PRT SCONJ".split())
CONTENT = set("ADJ NOUN VERB PROPN".split())






class DependencyTree(nx.DiGraph):
    """
    A DependencyTree as networkx graph:
    nodes store information about tokens
    edges store edge related info, e.g. dependency relations
    """

    def __init__(self):
        nx.DiGraph.__init__(self)
        self.LEAFNODES = "ADP AUX DET PUNCT CONJ SCONJ".split()

    def pathtoroot(self, child):
        path = []
        newhead = self.head_of(self, child)
        while newhead:
            path.append(newhead)
            newhead = self.head_of(self, newhead)
        return path

    def head_of(self, n):
        for u, v in self.edges():
            if v == n:
                return u
        return None

    def get_sentence_as_string(self,printid=False):
        out = []
        for token_i in range(1, max(self.nodes()) + 1):
            if printid:
                out.append(str(token_i)+":"+self.node[token_i]['form'])
            else:
                out.append(self.node[token_i]['form'])
        return u" ".join(out)

    def subsumes(self, head, child):
        if head in self.pathtoroot(self, child):
            return True

    def get_highest_index_of_span(self, span):  # retrieves the node index that is closest to root
        #TODO: CANDIDATE FOR DEPRECATION
        distancestoroot = [len(self.pathtoroot(self, x)) for x in span]
        shortestdistancetoroot = min(distancestoroot)
        spanhead = span[distancestoroot.index(shortestdistancetoroot)]
        return spanhead

    def get_deepest_index_of_span(self, span):  # retrieves the node index that is farthest from root
        #TODO: CANDIDATE FOR DEPRECATION
        distancestoroot = [len(self.pathtoroot(self, x)) for x in span]
        longestdistancetoroot = max(distancestoroot)
        lownode = span[distancestoroot.index(longestdistancetoroot)]
        return lownode

    def span_makes_subtree(self, initidx, endidx):
        G = nx.DiGraph()
        span_nodes = list(range(initidx,endidx+1))
        span_words = [self.node[x]["form"] for x in span_nodes]
        G.add_nodes_from(span_nodes)
        for h,d in self.edges():
            if h in span_nodes and d in span_nodes:
                G.add_edge(h,d)
        return nx.is_tree(G)

    def _remove_node_properties(self,fields):
        for n in sorted(self.nodes()):
            for fieldname in self.node[n].keys():
                if fieldname in fields:
                    self.node[n][fieldname]="_"

    def is_fully_projective(self):
        countProjectiveRelation=0
        countNonProjectiveRelation=0
        punctNonProj=0

        for i,j in self.edges():
            #i = instance[edge]
            #j = edge

            if j < i:
                i,j=j,i
            for k in range(i+1,j):
                headk = self.head_of(k)
                if i <= headk <= j or j <= headk <= i:
                    projEdge = True
                    countProjectiveRelation+=1
                else:
                    #print("non-projective")
                    #print("{} <= {} <= {} ? ".format(i,headk,j))
                    isProjective=False
                    countNonProjectiveRelation+=1
        return countNonProjectiveRelation == 0

    def punct_proj_violations(self):

        punctuation_indixes = []
        countProjectiveRelation=0
        countNonProjectiveRelation=0
        punctNonProj=0

        for i,j in self.edges():
            #i = instance[edge]
            #j = edge

            if j < i:
                i,j=j,i
            for k in range(i+1,j):
                headk = self.head_of(k)
                if i <= headk <= j or j <= headk <= i:
                    projEdge = True
                    countProjectiveRelation+=1
                else:
                    #print("non-projective")
                    #print("{} <= {} <= {} ? ".format(i,headk,j))
                    isProjective=False
                    countNonProjectiveRelation+=1
                    if self.node[k]["cpostag"] == "PUNCT":
                        punctNonProj+=1
                        punctuation_indixes.append(k) #CONLL -1, positions in the POS array
        return set(punctuation_indixes)

    def leaf_violations(self):
        viol = 0
        for head, dep in self.edges():
            if self.node[head]["cpostag"] in self.LEAFNODES:
                viol+=1
        return viol,len(self.nodes())




def ud_filter(P,M_in):
    M = copy.copy(M_in)
    P = ["ROOT"] + P # We assume the pos list has no padding


    #TODO: Flag each step w a binary parameter
    #1.- enforce leafness of certain POS
    should_be_leaves = [i for i,pos in enumerate(P) if pos in ONLYLEAF]
    #print(should_be_leaves)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if j in should_be_leaves:
                pass
                #M[i,j] = NEGINF #[NEGINF if i in should_be_leaves else x for i,x in enumerate(row)]

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
    #
    # 4.- Attach last punctuation to likeliest root node
    if P[-1] == "PUNCT":
         for x in range(M.shape[1]):
             if x != likeliest_root:
                 M[-1,x]=NEGINF
             else:
                 M[-1, x] = M_in[-1,likeliest_root]
    #
    heads=chu_liu_edmonds(M)
    heads=heads[0]

    sent = DependencyTree()
    for n,p in enumerate(P):
        sent.add_node(n,{'cpostag':p})
    for n in sent.nodes()[1:]:
        sent.add_edge(heads[n],n)

    #print(sent.punct_proj_violations(), is_projective(heads), heads, P)
    for i in sent.punct_proj_violations(): #block the existing head
        M[i,heads[i]]=NEGINF
    return M


def printmatrix(M):
    for r in M:
        print(" ".join([str(x) for x in r]))

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--infile", help='Files with projection tensors and gold parses',
                        default='/Users/hector/proj/nlpfromscratch/data/conll/en.biblenonproj.conll')
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

            dimensions = len(L) + 1 #the 0-th position encodes root attachment edges
            M= [[NEGINF]*dimensions]
            for rindex,rowdict in enumerate(L):
                R = [NEGINF] * dimensions
                for k in rowdict.keys():
                    R[k]=rowdict[k]
                M.append(R)
            M = np.array(M)
            Mfiltered = ud_filter(P,M)
            #printmatrix(Mfiltered)
            #print()
            P = []  # list to store POS values
            L = []  # dictionary list to later build a matrix proper


if __name__ == "__main__":
    main()