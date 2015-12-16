# This is an implemenation of Tarjan's Implementation of CLE for MST, based on Uri Zwick's lecture notes
import numpy as np
import heapq, sys
from string import ascii_lowercase


class Vertex:
    def __init__(self, w_index, ranking):
        self.ranking = ranking
        self.w_index = w_index  #index of word in sentence
        self.in_edge = None
        self.const = 0
        self.prev = None
        self.next = None
        self.parent = None
        self.children = []
        self.target = []
        self.P = []  #if ranking, this will be the edge list ordered by rank

    def __str__(self):
        return str(self.w_index)

    def getPLength(self):
        return len(self.P)

    def getPNext(self):
        #		if self.ranking:
        #			return self.P.pop(0)
        #		else:
        if True:
            return heapq.heappop(self.P)

    def notEmpty(self):
        return not (len(self.P) == 0)


def initialise(M, ranking):
    #head is row, col is dependent
    shape = M.shape
    vertices = []
    for v in range(shape[0]):  #rows=heads, columns=heads
        vertices.append(Vertex(v, ranking))
        for u in range(shape[0]):
            #			if ranking: #ranking starts at 1
            #				vertices[v].P.append((M[u][v],(u,v), M[u][v]))
            #			else:
            if True:
                heapq.heappush(vertices[v].P, (M[u][v], (u, v), M[u][v]))
            #		if ranking:
            #			sorted(vertices[v].P)
            #		else: 
        if True:
            heapq.heapify(vertices[v].P)
    return vertices


def meldQueues(c, a, vertices, ranking):
    index = 0
    while a.notEmpty():
        temp_next = a.getPNext()
        if temp_next[1][0] == temp_next[1][1]:  # get rid of simple loops
            continue
        next = (weight_naive(temp_next, vertices), temp_next[1], temp_next[2])  #adjust weights
        #		if ranking:
        #			while next[0]<c.P[index][0]: #while c's edge is heavier
        #				index+=1
        #			c.P.insert(index,next)
        #			index+=1
        #		else:
        if True:
            heapq.heappush(c.P, next)
    heapq.heapify(c.P)


def find_naive(u):
    while u.parent is not None:
        u = u.parent
    return u


def weight_naive(edge, vertices):  #edge is a pair (weight, (source vertex, target vertex))
    w = edge[2]
    #print 'weight', edge
    v = vertices[edge[1][1]]
    while v.parent is not None:
        w = w + v.const
        v = v.parent
    return w


def expand(vertices):
    R = []
    root = vertices[0]
    dismantle(root, R)
    count = 0
    while len(R) > 0:
        c = R.pop(0)
        uv = c.in_edge
        v = vertices[uv[1][1]]
        v.in_edge = uv
        dismantle(v, R)
        count += 1
    #	return [(vertices[x].in_edge[2], (vertices[x].in_edge[1][0],x)) for x in range(len(vertices))]
    return [vertices[x].in_edge[1][0] for x in range(1, len(vertices))]


def dismantle(u, R):  #u is a Vertex
    while u.parent is not None:
        u = u.parent
        for v in u.children:
            v.parent = None
            if len(v.children) > 0:
                R.append(v)
        del u.children[:]


def contract(M, ranking=False):
    dim = M.shape[0]
    vertices = initialise(M, ranking)
    a = vertices[0]
    count = 0
    letter = M.shape[0] + 1

    while a.notEmpty():
        uv = a.getPNext()
        u = vertices[uv[1][0]]
        b = find_naive(u)
        if a is b:  #self loop
            continue
        else:
            a.in_edge = uv
            a.prev = b
            if u.in_edge is None:
                b.next = a
                a = b  #b is the new first vertex on the path
            else:
                count += 1
                c = Vertex(letter, ranking)  #-1 word index for super vertices
                letter += 1
                while a.parent is None:
                    a.parent = c
                    c.target.append(a.in_edge[1][1])
                    a.const = -weight_naive(a.in_edge, vertices)
                    c.children.append(a)
                    index = 0
                    #merging a's priority queue into c's
                    meldQueues(c, a, vertices, ranking)
                    a = a.prev
                for child in c.children:
                    if child.next is not None:
                        if child.next not in c.children:
                            c.next = child.next
                            child.next.prev = c
                            break
                a = c
    return vertices


def mdst(M, column_heads=True, ranking=False, maximum=True, greedy=False):
    #M is numpy weight matrix with heads on columns, unless column_heads=False
    M = np.vstack([[0.0 for p in range(M.shape[1])], M])
    if maximum and not ranking:
        M = -1 * M
    elif ranking:
        maximum = False
    if greedy:
        min = float('inf')
        index = -1
        for i in range(M.shape[0]):
            #	print M[i][0]
            if M[i][0] < min:
                min = M[i][0]
                index = i
        #print 'min', min 
        for i in range(M.shape[0]):
            if i == index:
                M[i][0] = -100000.0
            else:
                M[i][0] = 0.0
    if column_heads:
        vertices = contract(M.transpose(), ranking)
    else:
        vertices = contract(M, ranking)
    return expand(vertices)


if __name__ == "__main__":
    print
    'other testing script...'
