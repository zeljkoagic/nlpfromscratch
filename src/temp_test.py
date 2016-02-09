import timeit

import numpy as np
from scipy import sparse


def project_dependencies_to_target(S, A):
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

def standardize(sentence_matrix):
    normalized = sentence_matrix.copy()
    normalized -= np.nanmean(normalized)
    normalized /= np.nanstd(normalized)

    return normalized

def softmax(sentence_matrix, temperature=1.0):
    """Softmax normalization.

    :param sentence_matrix: (n+1 x n+1) weight matrix from the parser
    :param temperature: softmax temperature
    :return: softmaxed weight matrix
    """
    m_exp = np.exp(sentence_matrix/temperature)
    return (m_exp.T / np.nansum(m_exp, axis=1)).T


np.random.seed(40)
S = np.random.random([40, 40])
S = np.where(S > 0.5, S, 0)

# A = np.random.random([40, 45])
# A = np.where(A > 0.9, A, 0)
A = np.diag(np.ones(45))[:40,:]

proj_normed_softmax = softmax(project_dependencies_to_target(standardize(S), A))
proj_softmax = softmax(project_dependencies_to_target(S, A))