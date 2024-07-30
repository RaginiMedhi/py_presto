# pylint: disable=too-many-arguments,invalid-name,broad-exception-raised

"""
File for util functions of Presto.
"""

# inbuilt

from typing import Tuple, List, Any
from statsmodels.stats.multitest import multipletests

# pip

import numpy as np
import pandas as pd

from scipy.stats import norm
from scipy.sparse import csr_matrix, isspmatrix_csr

# local imports

from python_presto.rcpp import *


def _get_sparse_matrix_vectors(
    data: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts the i, x and p vectors of the dgCMatrix

    Arguments:
    data -- the numpy array of the original sparse matrix

    Returns:
    i -- numpy array of row index of non-zero elements
    p -- numpy array of number of non-zero elements
    x -- numpy array of non-zero elements of X

    """
    data = np.array(data)
    non_zero_indices = np.nonzero(data.T)

    i = non_zero_indices[1]
    x = data.T[non_zero_indices]

    p = np.zeros(data.shape[1] + 1, dtype=int)
    current_position = 0

    for col in range(data.shape[1]):
        p[col] = current_position
        current_position += np.count_nonzero(data[:, col])
    p[data.shape[1]] = len(i)

    return i, p, x


def get_margin(data: csr_matrix | np.ndarray, y: List) -> int:
    """
    Arguments:
    data -- csr_matrix or numpy array
    y -- list of encoded group types

    Returns:
    m -- MARGIN argument for functions
    """
    if data.shape[0] == len(y):
        return 2

    if data.shape[1] == len(y):
        return 1

    raise Exception("nrow(data) or ncol(data) != length(group labels)")


def rank_matrix(data: csr_matrix | np.ndarray) -> dict:
    """
    Arguments:
    data -- csr_matrix or numpy array
    """
    if isspmatrix_csr(data):
        return _rank_matrix_dgcmatrix(data)

    if isinstance(data, np.ndarray):
        return _rank_matrix_matrix(data)

    raise ValueError("Unsupported matrix type")


def _rank_matrix_dgcmatrix(data: csr_matrix) -> dict:
    """
    Arguments:
    data -- csr_matrix

    Returns:
    named dictionary
    """
    ranked_data = csr_matrix(data)
    xr, ties = _rank_sparse_matrix(ranked_data)

    return {"ranked_data": xr, "ties": ties}


def _rank_sparse_matrix(data: csr_matrix) -> Tuple[np.ndarray, List]:
    """
    Arguments:
    data -- csr_matrix

    Returns:
    xr -- numpy array
    ties -- list
    """
    data_array = data.toarray()

    _, p, x = _get_sparse_matrix_vectors(data_array)
    ncol = data_array.shape[1]
    nrow = data_array.shape[0]

    xr, ties = cpp_in_place_rank_mean(x, p, ncol)

    for i in range(ncol):
        if p[i + 1] == p[i]:
            n_zero = nrow - (p[i + 1] - p[i])
            ties[i].append(n_zero)
            xr[p[i] : p[i + 1]] += n_zero

    return xr, ties


def _rank_matrix_matrix(data: np.ndarray) -> dict:
    """
    Arguments:
    data -- numpy array

    Returns:
    dictionary where X is a numpy array and ties is a list
    """
    x = data.T

    ties = [[] for _ in range(x.shape[1])]

    for c in range(x.shape[1]):
        v_sort = [(x[i, c], i) for i in range(x.shape[0])]
        v_sort.sort()

        rank_sum = 0
        n = 1
        i = 1

        while i < len(v_sort):
            if v_sort[i][0] != v_sort[i - 1][0]:
                for j in range(n):
                    x[v_sort[i - 1 - j][1], c] = (rank_sum / n) + 1
                rank_sum = i
                if n > 1:
                    ties[c].append(n)
                n = 1
            else:
                rank_sum += i
                n += 1
            i += 1

        for j in range(n):
            x[v_sort[i - 1 - j][1], c] = (rank_sum / n) + 1

    return {"ranked_data": x, "ties": ties}


def sum_groups(data: csr_matrix | np.ndarray, y: List, margin: int) -> np.ndarray:
    """
    Arguments:
    data -- csr_matrix or numpy array
    y -- list of encoded group types
    margin -- 1 or 2
    """
    if margin == 2 and data.shape[0] != len(y):
        raise ValueError("nrow(data) != length(group labels)")

    if margin == 1 and data.shape[1] != len(y):
        raise ValueError("ncol(data) != length(group labels)")

    if isspmatrix_csr(data):
        return _sum_groups_dgcmatrix(data, y, margin)

    if isinstance(data, np.ndarray):
        return _sum_groups_matrix(data, y, margin)

    raise ValueError("Unsupported matrix type")


def _sum_groups_dgcmatrix(data: csr_matrix, y: List, margin: int) -> np.ndarray:
    """
    Arguments:
    data -- csr_matrix
    y -- list of encoded group labels
    margin-- 1 or 2
    """

    data = data.toarray()

    i, p, x = _get_sparse_matrix_vectors(data)

    ncol = data.shape[1]
    nrow = data.shape[0]

    ngroups = len(np.unique(y))

    if margin == 1:
        return cpp_sum_groups_sparse_t(x, p, i, ncol, nrow, y, ngroups)

    return cpp_sum_groups_sparse(x, p, i, ncol, y, ngroups)


def _sum_groups_matrix(data: np.ndarray, y: List, margin: int) -> np.ndarray:
    """
    Arguments:
    data -- numpy array
    y -- list of encoded group types
    margin -- 1 or 2
    """
    unique_y = np.unique(y)
    ngroups = len(unique_y)

    if margin == 1:
        return cpp_sum_groups_dense_t(data, y, ngroups)

    return cpp_sum_groups_dense(data, y, ngroups)


def nnzero_groups(data: csr_matrix | np.ndarray, y: List, margin: int) -> np.ndarray:
    """
    Arguments:
    data -- csr_matrix or numpy array
    y -- list of encoded group types
    MARGIN -- 1 or 2
    """
    if margin == 2 and data.shape[0] != len(y):
        raise ValueError("nrow(data) != length(group labels)")
    if margin == 1 and data.shape[1] != len(y):
        raise ValueError("ncol(data) != length(group labels)")

    if isspmatrix_csr(data):
        return _nnzero_groups_dgcmatrix(data, y, margin)

    if isinstance(data, np.ndarray):
        return _nnzero_groups_matrix(data, y, margin)

    raise ValueError("Unsupported matrix type")


def _nnzero_groups_dgcmatrix(data: csr_matrix, y: List, margin: int) -> np.ndarray:
    """
    Arguments:
    data -- csr_matrix
    y -- list of encoded group types
    MARGIN -- 1 or 2
    """
    data = data.toarray()

    i, p, _ = _get_sparse_matrix_vectors(data)

    ncol = data.shape[1]
    nrow = data.shape[0]

    ngroups = len(np.unique(y))

    if margin == 1:
        res = cpp_nnzero_groups_sparse_t(p, i, ncol, nrow, y, ngroups)
    elif margin == 2:
        res = cpp_nnzero_groups_sparse(p, i, ncol, y, ngroups)

    return res


def _nnzero_groups_matrix(data: np.ndarray, y: List, margin: int) -> np.ndarray:
    """
    Arguments:
    data -- numpy array
    y -- list of encoded group types
    MARGIN -- 1 or 2
    """
    unique_y = np.unique(y)
    ngroups = len(unique_y)

    if margin == 1:
        return cpp_nnzero_groups_dense_t(data, y, ngroups)

    return cpp_nnzero_groups_dense(data, y, ngroups)


def compute_ustat(
    data: np.ndarray, xr: np.ndarray, y: List, group_size: np.ndarray
) -> np.ndarray:
    """
    Arguments:
    data -- csr_matrix or numpy array
    xr -- numpy array output of function 'rank_matrix'
    y -- list of encoded group types
    group_size -- counts in each group type

    Returns:
    ustat -- numpy array
    """
    m = get_margin(xr, y)

    if isspmatrix_csr(data):
        grs = sum_groups(csr_matrix(xr), y, m)
        nn = nnzero_groups(csr_matrix(xr), y, m)
        gnz = (group_size - nn.T).T
        _, p, _ = _get_sparse_matrix_vectors(xr)
        zero_ranks = (xr.shape[0] - np.diff(p) + 1) / 2
        ustat_t = (gnz * zero_ranks).T + grs.T - group_size * (group_size + 1) / 2
        ustat = ustat_t.T

    elif isinstance(data, np.ndarray):
        grs = sum_groups(xr, y, m)
        ustat = (grs.T - group_size * (group_size + 1) / 2).T

    return ustat


def compute_pval(
    ustat: np.ndarray, ties: list, length: int, n1n2: np.ndarray
) -> np.ndarray:
    """
    Arguments:
    ustat -- numpy array from function 'compute_ustat'
    ties -- numpy array from function 'rank_matrix'
    length -- length of the list for encoded group types
    n1n2 -- numpy array

    Returns:
    pvals -- numpy array
    """
    z = (ustat.T - (n1n2 / 2)).T
    z = z - np.sign(z) * 0.5
    x1 = np.power(length, 3) - length
    x2 = 1 / (12 * (np.square(length) - length))
    rhs = [(x1 - np.sum(np.array(tvals) ** 3 - np.array(tvals))) * x2 for tvals in ties]
    n1n2_matrix = np.array(n1n2).reshape(-1, 1)
    rhs_matrix = np.array(rhs).reshape(1, -1)
    product_matrix = np.dot(n1n2_matrix, rhs_matrix)
    usigma = np.sqrt(product_matrix)
    z = np.divide(z, usigma).T
    pvals = 2 * norm.cdf(-np.abs(z))

    return pvals


def adjust_pvalues(pvals: np.ndarray):
    """
    Arguments:
    ustat -- numpy array from function 'compute_pval'
    """
    return multipletests(pvals, method="fdr_bh")[1]


def tidy_results(
    res_list: dict[str, Any], features: np.ndarray, group_types: List[np.str_]
) -> pd.DataFrame:
    """
    Arguments:
    res -- results dictionary
    features -- gene list
    group_types -- types of groups

    Returns:
    res -- pandas dataframe
    """
    array_flat = {}
    for key, value in res_list.items():
        array_flat[key] = value.flatten()
    res = pd.DataFrame(array_flat)

    res["feature"] = np.tile(features, len(group_types))
    res["group"] = np.repeat(group_types, len(features))
    columns_order = [
        "feature",
        "group",
        "avgExpr",
        "logFC",
        "statistic",
        "auc",
        "pval",
        "padj",
        "pct_in",
        "pct_out",
    ]

    res = res.reindex(columns=columns_order)

    return res
