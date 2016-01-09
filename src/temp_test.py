import timeit

import numpy as np
from scipy import sparse


def project_dependencies_to_target_new(S, A):  # TODO Matrix S must be normalized, i.e., negative values are not allowed!
    """Projects source graph to target graph via source-target word alignment.

    :param S: source graph (m+1 x m+1 matrix)
    :param A: word alignment matrix (m+1 x n+1)
    :return: target graph
    """
    m_plus_one, n_plus_one = A.shape

    T = np.ones(shape=(n_plus_one, n_plus_one)) * np.nan  # target graph
    T_edge = np.ones_like(T) * np.nan

    # for each dependent d in source graph (dependents are rows!)
    rows = np.expand_dims(A, axis=-1)
    targets_for_row = np.zeros([A.shape[0], T.shape[0], T.shape[0]])

    for h in range(m_plus_one):
        np.dot(rows, A[h].reshape(1, -1), out=targets_for_row)
        targets_for_row *= S[:, h].reshape(-1, 1, 1)


        np.nanmax(targets_for_row, axis=0, out=T_edge)
        np.fmax(T, T_edge, out=T)

    return T


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




np.random.seed(40)
S = np.random.random([40, 40])
S = np.where(S > 0.5, S, 0)

S_coo = sparse.coo_matrix(S)

# A = np.random.random([40, 45])
# A = np.where(A > 0.9, A, 0)
A = np.diag(np.ones(45))[:40,:]

A_coo = sparse.coo_matrix(A)
A_csr = sparse.csr_matrix(A)


import pyximport; pyximport.install()
from project_deps import project_dependencies_fast, project_dependencies_faster

variations = ["project_dependencies_fast(S, A_coo)",
              "project_dependencies_faster(S_coo, A_csr)",
              "project_dependencies_to_target(S, A)"
              ]

for variation in variations:
    runs = 1000
    elapsed = timeit.timeit(variation, number=runs, globals=globals())
    secs_per_run = elapsed / runs

    print("Running '{}'. Took {} microseconds per round".format(variation, secs_per_run * 1E6))




