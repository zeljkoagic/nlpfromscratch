from __future__ import division
import numpy as np
from collections import defaultdict, Counter


def read_sentence_alignments(filename):
    """Reads hunalign-style sentence alignments, and stores them in a dictionary. Sentence alignments are 1:1.

    :param filename the sentence alignments file
    :return: a source sentence id-indexed dictionary of two-item lists, src_id -> [trg_id, confidence]
    """
    saligns = defaultdict(list)

    for line in open(filename):

        line = line.strip()
        line = line.split()

        if len(line) == 3:
            src_id = int(line[0])
            trg_id = int(line[1])
            confidence = float(line[2])
            saligns[trg_id] = [src_id, confidence]

    return saligns


def read_word_alignments(filename):
    """Reads fast-align word alignments.

    :param filename: fast_align-formatted word alignments file
    :return: a list of <word pairs, probabilities> pairs for all aligned sentences, and the language similarity estimate
    """
    waligns = []
    similarity = 0
    count = 0

    for line in open(filename):

        alignment_items = line.strip().split()

        if alignment_items:
            # account for the alignment file format
            pairs = alignment_items[::2]
            probabilities = [float(p) for p in alignment_items[1::2]]
            waligns.append((pairs, probabilities))
            count += len(probabilities)
            similarity += sum(probabilities)

            assert len(probabilities) == len(pairs), "Prob-pair mismatch"

    return waligns, similarity / count

def get_alignment_matrix(shape, pairs, probabilities, binary=False):
    """Creates (m+1 x n+1) source-target alignment probability matrix.

    :param shape: matrix shape pair, m = source dimension (row indices), n = target dimension (column indices)
    :param pairs: list of source-target index pairs
    :param probabilities: list of probabilities associated with the pairs
    :param binary: use prob = 1 for all confirmed alignments, instead of their actual probabilities
    :return: the alignment matrix
    """

    if len(pairs) != len(probabilities):
        raise Exception("Mismatch in sizes of pairs (%s) and probabilities (%s)" % (len(pairs), len(probabilities)))

    matrix = np.ones(shape) * np.nan  # change here for non-zero default

    if binary:
        probabilities = np.ones_like(probabilities)

    for it in range(len(pairs)):
        source_id, target_id = pairs[it].split("-")
        probability = probabilities[it]
        matrix[int(source_id)+1, int(target_id)+1] = float(probability)

    matrix[0, 0] = 1.0  # source root always aligns to target root
    # return np.where(matrix == 0, [0.5], matrix)
    return matrix


def project_dependencies_to_target(S, A):  # TODO Matrix S must be normalized, i.e., negative values are not allowed!
    """Projects source graph to target graph via source-target word alignment.

    :param S: source graph (m+1 x m+1 matrix)
    :param A: word alignment matrix (m+1 x n+1)
    :return: target graph
    """
    m_plus_one, n_plus_one = A.shape
    
    T = np.ones(shape=(n_plus_one, n_plus_one)) * np.nan  # target graph
    T_edge = np.ones_like(T) * np.nan

    # for each dependent d in source graph (dependents are rows!)
    for d in range(m_plus_one):
            # and for each head of dependent d (heads are columns!)
            for h in range(m_plus_one):
                np.dot(A[d].reshape(-1, 1), A[h].reshape(1, -1), out=T_edge)
                T_edge *= S[d, h]  # multiply by confidence of edge h->d from the source parse
                # T += T_edge
                np.fmax(T, T_edge, out=T)
    return T


def project_token_labels(source_labels, wa_pairs, wa_probs, weigh_votes=True):
    """Projects the token labels from source to target tokens in a sentence.

    :param source_labels: list of source labels ordered by source token id
    :param wa_pairs: word alignment pairs in "sid:tid" format
    :param wa_probs: probabilities corresponding to the alignment pairs
    :param weigh_votes: whether to use weighted voting or not
    :return: a dictionary of vote counters, indexed by target token CoNLL id
    """
    label_votes = defaultdict(Counter)  # dictionary of counters

    it = 0
    for pair in wa_pairs:
        sid, tid = pair.split("-")
        if weigh_votes:
            label_votes[int(tid) + 1].update({source_labels[int(sid)]: wa_probs[it]})
        else:
            label_votes[int(tid) + 1].update({source_labels[int(sid)]: 1})
        it += 1

    return label_votes
