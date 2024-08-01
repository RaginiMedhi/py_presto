# pylint: disable=too-many-arguments,invalid-name

"""
File for Python conversions of the R-C++ functions in Presto.
"""
# inbuilt

from typing import Tuple, List

# pip
import numpy as np


def cpp_in_place_rank_mean(
    v_temp: np.ndarray, idx_begin: int, idx_end: int
) -> Tuple[np.ndarray, List]:
    """
    Arguments:
    x -- numpy array from utils function 'get_sparse_matrix_vectors'
    p_start -- start index
    p_end -- end index

    Returns:
    ties -- numpy array
    """

    ties = []

    if idx_begin > idx_end:
        return ties

    v_sort = [(v_temp[i], i - idx_begin) for i in range(idx_begin, idx_end + 1)]
    v_sort.sort()

    rank_sum = 0
    n = 1
    i = 1
    while i < len(v_sort):
        if v_sort[i][0] != v_sort[i - 1][0]:
            for j in range(n):
                v_temp[v_sort[i - 1 - j][1] + idx_begin] = (rank_sum / n) + 1
            rank_sum = i
            if n > 1:
                ties.append(n)
            n = 1
        else:
            rank_sum += i
            n += 1
        i += 1

    for j in range(n):
        v_temp[v_sort[i - 1 - j][1] + idx_begin] = (rank_sum / n) + 1

    return ties


def cpp_sum_groups_sparse(
    x: np.ndarray, p: np.ndarray, i: np.ndarray, ncol: int, y: List, ngroups: int
) -> np.ndarray:
    """
    Arguments:
    x, p and i -- numpy array from utils function 'get_sparse_matrix_vectors'
    ncol -- number of columns of the original sparse matrix
    ngroups -- number of groups

    Returns:
    res -- numpy array
    """
    res = np.zeros((ngroups, ncol))

    for c in range(ncol):
        for j in range(p[c], p[c + 1]):
            res[y[i[j]], c] += x[j]

    return res


def cpp_sum_groups_sparse_t(
    x: np.ndarray,
    p: np.ndarray,
    i: np.ndarray,
    ncol: int,
    nrow: int,
    y: List,
    ngroups: int,
) -> np.ndarray:
    """
    Arguments:
    x, p and i -- numpy array from utils function 'get_sparse_matrix_vectors'
    ncol -- number of columns of the original sparse matrix
    nrow -- number of columns of the original sparse matrix
    y -- list of encoded group types
    ngroups -- number of groups

    Returns:
    res -- numpy array
    """
    res = np.zeros((ngroups, nrow))

    for c in range(ncol):
        for j in range(p[c], p[c + 1]):
            res[y[c], i[j]] += x[j]

    return res


def cpp_sum_groups_dense(data: np.ndarray, y: List, ngroups: int) -> np.ndarray:
    """
    Arguments:
    data -- numpy array (not from a dgCMatrix)
    y -- list of encoded group types
    ngroups -- number of groups

    Returns:
    res -- numpy array
    """

    res = np.zeros((ngroups, data.shape[1]))

    for r in range(data.shape[0]):
        res[y[r], :] += data[r, :]

    return res


def cpp_sum_groups_dense_t(data: np.ndarray, y: List, ngroups: int) -> np.ndarray:
    """
    Arguments:
    data -- numpy array (not from a dgCMatrix)
    y -- list of encoded group types
    ngroups -- number of groups

    Returns:
    res -- numpy array
    """
    res = np.zeros((ngroups, data.shape[0]))

    for c in range(data.shape[1]):
        res[y[c], :] += data[:, c].T

    return res


def cpp_nnzero_groups_sparse(
    p: np.ndarray, i: np.ndarray, ncol: int, y: List, ngroups: int
) -> np.ndarray:
    """
    Arguments:
    p and i -- numpy array from utils function 'get_sparse_matrix_vectors'
    ncol -- number of columns of the original sparse matrix
    y -- list of encoded group types
    ngroups -- number of groups

    Returns:
    res -- numpy array
    """
    res = np.zeros((ngroups, ncol))
    for c in range(ncol):
        for j in range(p[c], p[c + 1]):
            res[y[i[j]], c] += 1

    return res


def cpp_nnzero_groups_sparse_t(
    p: np.ndarray, i: np.ndarray, ncol: int, nrow: int, y: List, ngroups: int
) -> np.ndarray:
    """
    Arguments:
    p and i -- numpy array from utils function 'get_sparse_matrix_vectors'
    ncol -- number of columns of the original sparse matrix
    nrow -- number of columns of the original sparse matrix
    y -- list of encoded group types
    ngroups -- number of groups

    Returns:
    res -- numpy array
    """

    res = np.zeros((ngroups, nrow))
    for c in range(ncol):
        for j in range(int(p[c]), int(p[c + 1])):
            res[y[c], i[j]] += 1

    return res


def cpp_nnzero_groups_dense(data: np.ndarray, y: List, ngroups: int) -> np.ndarray:
    """
    Arguments:
    data -- numpy array (not from a dgCMatrix)
    y -- list of encoded group types
    ngroups -- number of groups

    Returns:
    res -- numpy array
    """

    res = np.zeros((ngroups, data.shape[1]))

    for c in range(data.shape[1]):
        for r in range(data.shape[0]):
            if data[r, c] != 0:
                res[y[r], c] += 1

    return res


def cpp_nnzero_groups_dense_t(data: np.ndarray, y: List, ngroups: int) -> np.ndarray:
    """
    Arguments:
    data -- numpy array (not from a dgCMatrix)
    y -- list of encoded group types
    ngroups -- number of groups

    Returns:
    res -- numpy array
    """
    res = np.zeros((ngroups, data.shape[0]))

    for c in range(data.shape[1]):
        for r in range(data.shape[0]):
            if data[r, c] != 0:
                res[y[c], r] += 1

    return res
