from collections import defaultdict, Counter
import numpy as np
from scipy import sparse


def read_alignments(filename_sa, filename_wa):
    """Reads sentence alignments and word alignments for a given source-target language pair.
    The alignments are paired through sentence ids.

    :param filename_sa: sentence alignment filename, hunalign format
    :param filename_wa: word alignment filename, efmaral/fast_align format
    :return: dictionaries of sentence and word alignments, and the language similarity estimate
    """
    saligns = defaultdict(list)
    waligns = defaultdict(list)
    similarity = 0
    count = 0

    for line_sa, line_wa in zip(open(filename_sa), open(filename_wa)):

        sa_items = line_sa.strip().split()
        wa_items = line_wa.strip().split()

        src_id = int(sa_items[0])
        trg_id = int(sa_items[1])
        confidence = float(sa_items[2])
        saligns[trg_id] = [src_id, confidence]

        if wa_items:
            pairs = [(int(sid), int(tid)) for sid, tid in [pair.split("-") for pair in wa_items[::2]]]
            probabilities = [float(p) for p in wa_items[1::2]]
            waligns[(trg_id, src_id)] = (pairs, probabilities)
            count += len(probabilities)
            similarity += sum(probabilities)
        else:
            waligns[(trg_id, src_id)] = None  # for efmaral's empty alignment lines

    return saligns, waligns, similarity / count


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

    src_indices = []
    trg_indices = []

    for it in range(len(pairs)):
        source_id, target_id = pairs[it]
        src_indices.append(source_id + 1)  # word alignments are 0-indexed, here we move to CoNLL indexing (+1)
        trg_indices.append(target_id + 1)

    # root always aligns to root, accommodate for that
    src_indices.append(0)
    trg_indices.append(0)
    probabilities.append(1.0)

    if binary:
        probabilities = np.ones_like(probabilities)

    matrix = sparse.coo_matrix((probabilities, (src_indices, trg_indices)), shape=shape)

    return matrix.tocsr()


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
                np.fmax(T, T_edge, out=T)
    return T


def project_token_labels(source_labels, wa_pairs, wa_probs):
    """Projects the token labels from source to target tokens in a sentence.

    :param source_labels: list of source labels ordered by source token id
    :param wa_pairs: word alignment pairs in "sid:tid" format
    :param wa_probs: probabilities corresponding to the alignment pairs
    :param weigh_votes: whether to use weighted voting or not
    :return: a dictionary of vote counters, indexed by target token CoNLL id
    """
    label_votes = defaultdict(Counter)  # dictionary of counters

    it = 0
    for sid, tid in wa_pairs:
        # word alignments are 0-indexed, here we move to CoNLL indexing (+1) for target
        label_votes[tid + 1].update({source_labels[sid]: wa_probs[it]})
        it += 1

    return label_votes

def read_word_alignments(word_alignment_file):
    alignment_pairs = []
    with word_alignment_file.open() as in_file:
        for line in in_file:
            alignments_per_sent = []

            parts = line.strip().split(" ")
            if len(parts) > 1:
                assert len(parts) % 2 == 0
                for token_pair, prob in zip(parts[::2], parts[1::2]):
                    src_token_id, target_token_id = map(int, token_pair.split("-"))
                    alignments_per_sent.append((src_token_id + 1, target_token_id + 1, float(prob)))

            alignment_pairs.append(alignments_per_sent)

    return alignment_pairs









