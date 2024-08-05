import pandas as pd
from python_presto import encode_groups, py_presto_wilcoxauc


def test_wilcoxauc(get_sm, get_ysm, get_fsm):
    cell_types = pd.Series(get_ysm)
    group_labels, groups_id = encode_groups(cell_types)
    features = get_fsm
    results = py_presto_wilcoxauc(get_sm, group_labels, features, groups_id)
    assert results.shape == (40, 10)
    assert results.avgExpr[5] == 0.5008675548479302
    assert results.pval[4] == 0.9416262271822518
