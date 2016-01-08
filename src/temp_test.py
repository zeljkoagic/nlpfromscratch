import numpy as np


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
        targets_for_row *= S[:,h].reshape(-1, 1, 1)

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


S = np.random.random([40, 40])
A = np.random.random([40, 45])


for i in range(100):
    project_dependencies_to_target(S, A)