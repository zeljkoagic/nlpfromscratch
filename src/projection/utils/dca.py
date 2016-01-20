import sys, os, glob, argparse
from utils.coo_matrix_nocheck import CooMatrix
import numpy as np


def hua_et_al_2005(align_ts, align_st, depths, stree, ttree):
    unaligned_source(align_ts, align_st, ttree)
    one2one(align_ts, align_st, stree, ttree)
    one2many(align_ts, align_st, ttree)
    many2one(align_ts, align_st, ttree, stree, depths)


#	#many2many() #equiv to one2many(); many2one()
#	#unaligned_target() #we don't need to do this

def hua_et_al_2005_nodummies(align_ts, align_st, depths, stree, ttree):
    orig_target_length = len(ttree)
    hua_et_al_2005(align_ts, align_st, depths, stree, ttree)
    remove_dummies(ttree, orig_target_length)


def remove_dummies(ttree, orig_target_length):
    intersection = [i for i in range(orig_target_length + 1, len(ttree) + 1) if i in ttree[:orig_target_length]]
    while len(intersection) > 0:
        for x in range(orig_target_length, len(ttree)):  # for all dummies
            for y in range(orig_target_length):  # for all non-dummies
                if ttree[y] == (x + 1):
                    ttree[y] = ttree[x]
        intersection = [i for i in range(orig_target_length + 1, len(ttree) + 1) if i in ttree[:orig_target_length]]


def setroot(stree, ttree, align_st):
    # sets the root of the target tree
    sroot = stree.index(-1)
    for t in align_st[sroot]:
        ttree[t] = -1


def one2one(align_ts, align_st, stree, ttree):
    # maps aligned nodes so long as single source for both source head and dependent
    for x in range(1, len(align_ts.keys()) + 1):
        t = align_ts[x]
        if len(t) == 1 and len(align_st[t[0]]) == 1 and (stree[t[0] - 1] == 0 or len(align_st[stree[t[0] - 1]]) == 1):
            if stree[t[0] - 1] > 0:
                ttree[x - 1] = align_st[stree[t[0] - 1]][0]
            else:
                ttree[x - 1] = 0


def unaligned_source(align_ts, align_st, ttree):
    # introduce new dummy node for each unaligned source node
    # NOTE: can do without this for evaluation, but necessary for tree-building
    # edges get mapped over by one2one
    for i in range(1, len(align_st) + 1):
        s = align_st[i]
        if len(s) == 0:
            align_ts[len(align_ts) + 1] = [i]  # dummy node
            align_st[i].append(len(align_ts.keys()))
            ttree.append(-10)


def one2many(align_ts, align_st, ttree):
    # introduce dummy node for source node mapped to more than 1 target
    # introduce edges from dummy to relevant target nodes
    # aligns source with dummy node
    for i in range(1, len(align_st) + 1):
        s = align_st[i]
        if len(s) > 1:
            ttree.append(-10)  # create dummy node
            for node in s:
                ttree[node - 1] = len(ttree)  # head is the dummy node
            align_st[i] = [len(ttree)]  # align source with dummy node
            align_ts[len(align_ts) + 1] = [i]  # align dummy node with source


def many2one(align_ts, align_st, ttree, stree, depths):
    # remove all alignment except shallowest source in the source tree
    # map result to target tree
    for i in range(1, len(align_ts) + 1):
        t = align_ts[i][:]
        if len(t) > 1:
            align_ts[i] = [t[0]]
            # find shallowest depth in list
            for x in range(1, len(t)):
                if depths[t[x] - 1] < depths[align_ts[i][0] - 1]:
                    align_ts[i] = [t[x]]
            if stree[align_ts[i][0] - 1] == 0:
                ttree[i - 1] = 0
            else:
                ttree[i - 1] = align_st[stree[align_ts[i][0] - 1]][0]


def getAlignments(align_data, slength, tlength):
    # returns adj list of alignments
    # 1) target node --> [source nodes]
    # 2) source node --> [target nodes]
    # the indices of nodes in adj lists are same as the id from the conll files!
    align_ts = {(x + 1): [] for x in range(tlength)}
    align_st = {(x + 1): [] for x in range(slength)}
    for a in align_data:
        st_pair = a.split('-')
        st_pair = (int(st_pair[0]) + 1, int(st_pair[1]) + 1)
        align_ts[st_pair[1]].append(st_pair[0])
        align_st[st_pair[0]].append(st_pair[1])
    return align_ts, align_st


def getAlignmentsSparse(A_sparse):
    # returns adj list of alignments
    # 1) target node --> [source nodes]
    # 2) source node --> [target nodes]
    # the indices of nodes in adj lists are same as the id from the conll files!
    align_ts = {(x + 1): [] for x in range(A_sparse.shape[1] - 1)}  # remove pseudo-root
    align_st = {(x + 1): [] for x in range(A_sparse.shape[0] - 1)}
    for a in range(len(A_sparse.row)):
        if A_sparse.row[a] > 0:
            align_st[A_sparse.row[a]].append(A_sparse.col[a])
            align_ts[A_sparse.col[a]].append(A_sparse.row[a])
    return align_ts, align_st


def makeSourceTreeSparse(S_sparse):
    stree = [-20] * len(S_sparse.row)
    for x in range(len(S_sparse.row)):
        stree[S_sparse.row[x] - 1] = S_sparse.col[x]
    depths = getDepths(stree)
    return stree, depths


def getDepths(stree):
    depths = [-10] * len(stree)  # depth of each node in the tree
    q = []
    rootindex = stree.index(0)
    depths[rootindex] = 0
    for i in range(len(stree)):  # putting children of root onto queue q
        if stree[i] == (rootindex + 1):
            q.append(i)
    while len(q) > 0:
        curr = q.pop(0)
        depths[curr] = depths[stree[curr] - 1] + 1
        for i in range(len(stree)):
            if stree[i] == (curr + 1):
                q.append(i)
    return depths


def makeSourceTree(s_sent):
    # create tree and calculate list of depths corr. to node index
    stree = getTreeFromConllData(s_sent)
    # getting depths of nodes
    depths = getDepths(stree)
    return stree, depths


def getTreeFromConllData(sent):
    # get head list=tree representation for split conll data
    tree = [int(line[6]) for line in sent]  # head list w.r.t. index
    return tree


def getMatchNumber(ttree, gold_ttree):
    matches = [1.0 if ttree[x] == gold_ttree[x] else 0 for x in range(len(gold_ttree))]
    return sum(matches)


def readAllSentsToMemory(sourcefile, targetfile):
    sfile = open(sourcefile, 'r');
    slines = sfile.readlines();
    sfile.close()
    tfile = open(targetfile, 'r');
    tlines = tfile.readlines();
    tfile.close()

    s_sents = [];
    t_sents = []
    templines = []
    for line in slines:
        line = line.strip()
        if len(line) == 0:
            if len(templines) > 0:
                s_sents.append(templines[:])
            del templines[:]
        else:
            templines.append(line.split('\t'))
    del templines[:]
    for line in tlines:
        line = line.strip()
        if len(line) == 0:
            if len(templines) > 0:
                t_sents.append(templines[:])
            del templines[:]
        else:
            templines.append(line.split('\t'))
    return s_sents, t_sents


def projectArrays(t_sent, s_sent, align_data, salign_data):
    ttree = [-10] * len(t_sent)  # tree data structure, recording only heads of nodes
    align_ts, align_st = getAlignments(align_data, len(s_sent), len(t_sent))
    stree, depths = makeSourceTree(s_sent)
    hua_et_al_2005_nodummies(align_ts, align_st, depths, stree, ttree)
    #	hua_et_al_2005(align_ts, align_st, depths, stree, ttree)
    return stree, ttree


def readFiles(sourcefile, targetfile, alignmentfile, salignmentfile):
    s_sents, t_sents = readAllSentsToMemory(sourcefile, targetfile)
    afile = open(alignmentfile, 'r');
    alines = afile.readlines();
    afile.close()
    safile = open(salignmentfile, 'r');
    salines = safile.readlines();
    safile.close()

    a_index = 0
    sum_tp = 0;
    sum_total = 0;
    sum_score = 0
    while a_index < len(alines):
        # print '\r',a_index,
        salign_data = salines[a_index].strip().split('\t')
        align_data = alines[a_index].strip().split()[::2];
        a_index += 1

        s_sent = s_sents[int(salign_data[0])]
        t_sent = t_sents[int(salign_data[1])]

        stree, ttree = projectArrays(t_sent, s_sent, align_data, salign_data)
        gold_ttree = getTreeFromConllData(t_sent)
        tp = getMatchNumber(ttree, gold_ttree)
        total = len(ttree)
        sum_total += total
        sum_tp += tp
        sum_score += tp / total
    return (sum_score / a_index, sum_tp / sum_total)


def testSparseMatrices(sourcefile, targetfile, alignmentfile, salignmentfile):
    # just for testing the project method
    s_sents, t_sents = readAllSentsToMemory(sourcefile, targetfile)
    afile = open(alignmentfile, 'r');
    alines = afile.readlines();
    afile.close()
    safile = open(salignmentfile, 'r');
    salines = safile.readlines();
    safile.close()

    a_index = 0
    sum_tp = 0;
    sum_total = 0;
    sum_score = 0
    while a_index < len(alines):
        # print '\r',a_index,
        salign_data = salines[a_index].strip().split('\t')
        align_data = alines[a_index].strip().split()[::2];
        a_index += 1
        s_sent = s_sents[int(salign_data[0])]
        t_sent = t_sents[int(salign_data[1])]

        # make A_sparse matrix
        row = [0];
        col = [0];
        data = [1]
        for item in align_data:
            pair = item.split('-')
            row.append(int(pair[0]) + 1)
            col.append(int(pair[1]) + 1)
            data.append(1)
        A_sparse = CooMatrix(row, col, data, [len(s_sent) + 1, len(t_sent) + 1])

        stree1, depths1 = makeSourceTree(s_sent)
        row1 = [];
        col1 = [];
        data1 = []
        for x in range(len(stree1)):
            row1.append(x + 1)
            col1.append(stree1[x])
            data1.append(1)
        S_sparse = CooMatrix(row1, col1, data1, [len(row1) + 1, len(col) + 1])

        T_matrix = project(S_sparse, A_sparse)
        ttree = [-10] * (len(t_sent))
        for x in range(1, len(T_matrix)):
            for y in range(len(T_matrix[x])):
                if T_matrix[x][y] == 1:
                    ttree[x - 1] = y
        stree_orig, ttree_orig = projectArrays(t_sent, s_sent, align_data, salign_data)
        for x in range(len(ttree)):
            if ttree[x] != ttree_orig[x]:
                return False


def printScores(outfilebasename, scores):
    latexfile = open(outfilebasename + '.tex', 'w')
    csvfile = open(outfilebasename + '.csv', 'w')

    latexfile.write('\\begin{table}\\begin{tabular}{l  l  l| c  c}\n')
    latexfile.write('source & target & align alg & uas (macro) & uas (micro)\\\\\\hline\n')
    csvfile.write('source,target,align_alg, uas(macro),uas(micro)\n')
    for score in sorted(scores):
        latexfile.write(score[0] + '&' + score[1] + '&' + score[2] + '&' + str(round(score[3], 3)) + '&' + str(
            round(score[4], 3)) + '\\\\\n')
        csvfile.write(score[0] + ',' + score[1] + ',' + score[2] + ',' + str(score[3]) + ',' + str(score[4]) + '\n')

    latexfile.write('\\end{tabular}\\end{table}')
    latexfile.close()
    csvfile.close()


def getLanguageSubset(filename):
    infile = open(filename, 'r')
    langs = [lang.strip() for lang in infile.readlines()]
    return langs


def iterateThroughDir_Bible():
    # first implementation to test for all source languages
    if align_dir.args is None:  # then they all are
        align_dir = '/home/bplank/preprocess-holy-data/data/walign/'
        conll_dir = '/home/bplank/parse-holy-data/data/2project/bible/'
        salign_dir = '/home/bplank/preprocess-holy-data/data/salign/'
        suffix = '.2proj.conll'  # suffix of the files to project
    if args.sourcefile is None:
        sources = ['ar', 'bg', 'cs', 'da', 'de', 'en', 'es', 'eu', 'fa', 'fi', 'fr', 'he', 'hi', 'hr', 'id', 'it', 'no',
                   'pl', 'pt', 'sl', 'sv']
    # sources=['de','en','es','fr']
    else:
        source = getLanguageSubset(sourcefile)
    scores = []
    count = 0
    for afile in glob.glob(align_dir + '*bible*reverse.wal'):
        adata = afile.strip().split('/')[-1].split('.')
        safile = salign_dir + adata[0] + '.bible.sal'
        langs = adata[0].split('-')
        if (langs[0] not in sources) or (langs[1] not in sources):
            continue
        align_model = adata[2]
        source = conll_dir + langs[0] + suffix
        target = conll_dir + langs[1] + suffix
        if os.path.isfile(source) and os.path.isfile(target):
            score = readFiles(source, target, afile, safile)
            scores.append((langs[0], langs[1], align_model, score[0], score[1]))
    printScores('bible_scores_all', scores)


def project(S_sparse, A_sparse_1):
    # takes two CooMatrix objects (coo_matrix_nocheck.py), returns projection
    # S_sparse must be just a tree (a single one per column)
    # return a numpy matrix (n+1)x(n+1) first row is all zeroes

    A_sparse = A_sparse_1.tocoo(copy=True)

    align_ts, align_st = getAlignmentsSparse(A_sparse)
    stree, depths = makeSourceTreeSparse(S_sparse)
    ttree = [-10] * (A_sparse.shape[1] - 1)  # don't include pseudo-root
    tlength = A_sparse.shape[1] - 1
    hua_et_al_2005_nodummies(align_ts, align_st, depths, stree, ttree)
    T_matrix = [0] * pow(tlength + 1, 2)
    for x in range(tlength):
        if ttree[x] > -1:
            T_matrix[((x + 1) * (tlength + 1)) + ttree[x]] = 1
    return np.ndarray(shape=(tlength + 1, tlength + 1), buffer=np.array(T_matrix), dtype=int)


def get_arguments(parser=None):
    # take in parameters
    parser = argparse.ArgumentParser(description="""Project (DCA)""")
    parser.add_argument('--align_dir', help='Directory for .wal files')
    parser.add_argument('--salign_dir', help='Directory for .sal files')
    parser.add_argument('--conll_dir', help='Directory for .conll files')
    parser.add_argument('--sourcefile', help='Source language subset file')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    #	args=get_arguments()
    # Nothing to do

    target = '/home/bplank/parse-holy-data/data/2project/bible/en.2proj.conll'
    source = '/home/bplank/parse-holy-data/data/2project/bible/de.2proj.conll'
    alignments = '/home/bplank/preprocess-holy-data/data/walign/de-en.bible.ibm2.reverse.wal'
    salignments = '/home/bplank/preprocess-holy-data/data/salign/de-en.bible.sal'
    testSparseMatrices(source, target, alignments, salignments)
