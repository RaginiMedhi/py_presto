import pandas as pd
from python_presto.utils import *


def test_encode_groups(get_ysm):
    cell_types = pd.Series(get_ysm)
    group_labels, groups_id = encode_groups(cell_types)
    assert group_labels[1] == 1
    assert groups_id[0] == "jurkat"


def test_rank_sparse_matrix(get_sm):
    xt = csr_matrix(csr_matrix(get_sm).transpose())
    ranked_sm = rank_matrix(xt)
    assert ranked_sm["ranked_data"][4] == 102.0
    assert len(ranked_sm["ties"]) == 20


def test_rank_matrix_matrix(get_mm):
    ranked_mm = rank_matrix(get_mm)
    assert ranked_mm["ranked_data"].shape[0] == 150
    assert ranked_mm["ranked_data"][0, 0] == 27.0


def test_compute_ustat(get_sm, get_ysm):
    cell_types = pd.Series(get_ysm)
    group_labels, _ = encode_groups(cell_types)
    _, group_size = np.unique(group_labels, return_counts=True)
    xt = csr_matrix(csr_matrix(get_sm).transpose())
    rank_res = rank_matrix(xt)
    Xr = rank_res["ranked_data"]
    ustat = compute_ustat(xt, Xr, group_labels, group_size)
    assert ustat[0, 2] == ustat[0, 2]
    assert ustat.shape == (2, 20)


def test_compute_pval(get_sm, get_ysm):
    cell_types = pd.Series(get_ysm)
    group_labels, _ = encode_groups(cell_types)
    _, group_size = np.unique(group_labels, return_counts=True)
    n1n2 = group_size * (get_sm.shape[1] - group_size)
    xt = csr_matrix(csr_matrix(get_sm).transpose())
    rank_res = rank_matrix(xt)
    Xr = rank_res["ranked_data"]
    ustat = compute_ustat(xt, Xr, group_labels, group_size)
    ties = rank_res["ties"]
    length = len(group_labels)
    pval = compute_pval(ustat, ties, length, n1n2)
    assert pval.shape == (20, 2)
    assert pval[0, 1] == 0.8531802254150416
