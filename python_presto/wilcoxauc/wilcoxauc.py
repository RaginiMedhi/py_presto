# pylint: disable=too-many-arguments,invalid-name,broad-exception-raised

""" 

Python compatible wilcoxauc function from Presto. 
"""
# inbuilt

import time

# pip

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, isspmatrix_csr

# local imports

from python_presto.utils import *


def py_presto_wilcoxauc(
    data: np.ndarray,
    y: List,
    features: np.ndarray,
    groups_id: dict,
    groups_use: List = None,
    verbose=True,
) -> pd.DataFrame:
    """
    Arguments:
    data -- the numpy array of the original sparse matrix
    y -- list of encoded group labels
    features -- gene list
    group_id -- dictionary of group types and labels
    group_use -- list of labels to use

    Returns:
    results -- dataframe of results
    """

    start_time = time.time()

    group_types = list(groups_id.keys())
    index = features.copy()

    if groups_use is not None:
        idx_use = [i for i, label in enumerate(y) if label in groups_use]
        y = [y[i] for i in idx_use]
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy(dtype="float")
            data = data[:, idx_use]
        elif isspmatrix_csr(data):
            data = csr_matrix(data)
            data = data[:, idx_use]

    if data.shape[1] != len(y):
        raise ValueError(
            "Number of columns of data does not match length of group labels"
        )

    if isspmatrix_csr(data):
        data = csr_matrix(data)

    elif isinstance(data, pd.DataFrame):
        data = data.to_numpy(dtype="float")

    select = True if np.isnan(np.array(y)).any() else False
    y = pd.Categorical(y)

    if select:
        idx_use = [i for i in range(len(y)) if not pd.isna(y[i])]
        if len(idx_use) < len(y):
            y = y[idx_use]
            data = data[:, idx_use]
            if verbose:
                print("Removing NA values from labels")
    else:
        print("Calculating stats!")

    _, group_size = np.unique(y, return_counts=True)
    if (group_size > 0).sum() < 2:
        raise ValueError("Must have at least 2 groups defined.")

    n1n2 = group_size * (data.shape[1] - group_size)

    if isspmatrix_csr(data):
        xt = csr_matrix(csr_matrix(data.copy()).transpose())
        rank_res = rank_matrix(xt)
        Xr = rank_res["ranked_data"]
    elif isinstance(data, np.ndarray):
        rank_res = rank_matrix(data.copy())
        Xr = rank_res["ranked_data"]
        xt = data.copy()

    ustat = compute_ustat(xt, Xr, y, group_size)
    auc = ustat.T / n1n2

    ties = rank_res["ties"]
    length = len(y)

    pval = compute_pval(ustat, ties, length, n1n2)

    fdr = np.apply_along_axis(adjust_pvalues, axis=0, arr=pval)

    m = get_margin(data, y)

    group_sums = sum_groups(data, y, m)
    group_nnz = nnzero_groups(data, y, m)

    col_sums_nnz = np.sum(group_nnz, axis=0)
    group_pct = group_nnz.T / group_size

    group_pct_out = (-group_nnz + col_sums_nnz) / (len(y) - group_size).reshape(-1, 1)
    group_pct_out = group_pct_out.T

    group_means = group_sums.T / group_size
    cs = np.sum(group_sums, axis=0)
    gs = group_size

    lfc = np.column_stack(
        [
            group_means[:, g] - (cs - group_sums[g, :]) / (len(y) - gs[g])
            for g in range(len(gs))
        ]
    )

    res_list = {
        "auc": auc,
        "pval": pval,
        "padj": fdr,
        "pct_in": 100 * group_pct,
        "pct_out": 100 * group_pct_out,
        "avgExpr": group_means,
        "statistic": ustat.T,
        "logFC": lfc,
    }

    results = tidy_results(res_list, index, group_types)
    print(f"Total run time: {(time.time() - start_time)} seconds")

    return results
