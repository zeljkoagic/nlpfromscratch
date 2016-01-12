import cython
import numpy as np

@cython.boundscheck(False)
def project_dependencies_faster(S, A):
    # S must be a coo_matrix

    cdef:
        double[:,:] T = np.full(shape=(A.shape[1], A.shape[1]), fill_value=np.nan)
        int i, k, l
        int s_i, s_j, t_i, t_j

        # A is a sparse matrix in the compressed column format
        int[:] align_indices = A.indices
        int[:] align_indptr = A.indptr
        double[:] align_data = A.data

        double align_prob_i_to_i, align_prob_j_to_j

        # S is a sparse matrix the in coordinate format
        int[:] source_row = S.row
        int[:] source_col = S.col
        double[:] source_data = S.data
        double source_edge_score


    # The projection finds pairs of source tokens (s_i, s_j) that have a non-zero
    # weight in the source graph and couples them with all pairs of target tokens (t_i, t_j)
    # that are connected with the source tokens through word alignments. Specifically,
    # s_i should be aligned with t_i, and s_j should align with t_j.
    for i in range(source_row.shape[0]):
        s_i = source_row[i]
        s_j = source_col[i]
        source_edge_score = source_data[i]

        # Alignments for s_i
        for k in range(align_indptr[s_i], align_indptr[s_i+1]):
            t_i = align_indices[k]
            align_prob_i_to_i = align_data[k]

            # Alignments for s_j
            for l in range(align_indptr[s_j], align_indptr[s_j+1]):
                t_j = align_indices[l]

                if t_i == t_j:
                    continue

                align_prob_j_to_j = align_data[l]

                T[t_i, t_j] = max(source_edge_score * align_prob_i_to_i  * align_prob_j_to_j, T[t_i, t_j])

    return np.array(T, copy=False)

@cython.boundscheck(False)
def project_dependencies_fast(double[:,:] S, A):
    """Projects source graph to target graph via source-target word alignment.

    :param S: source graph (m+1 x m+1 matrix)
    :param A: word alignment matrix (m+1 x n+1) in SPARSE format
    :return: target graph
    """

    cdef:
        double[:,:] T = np.ones(shape=(A.shape[1], A.shape[1])) * np.nan
        int[:] source_ids = A.row
        int[:] target_ids = A.col
        double[:] alignment_p = A.data
        int i, j, s_i, s_j, t_i, t_j
        double val


    # Find all pairs of source tokens (s_i, s_j) for which alignments exist.
    # Transfer this edge with the weight of the edge (s_i, s_j) in the source graph.
    #
    # (s_i, s_j) is a possible source edge that can be transferred using the alignments,
    # and (t_i, t_j) is a possible target edge.
    for i in range(len(source_ids)):
        s_i = source_ids[i]
        t_i = target_ids[i]
        for j in range(len(source_ids)):
            if i == j:
                continue
            s_j = source_ids[j]
            t_j = target_ids[j]
            # print(i, j)
            val = alignment_p[i] * alignment_p[j] * S[s_i, s_j]
            # TODO how do nans compare here
            T[t_i, t_j] = max(val, T[t_i, t_j])

    return np.array(T, copy=False)