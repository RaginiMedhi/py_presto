import pytest
import numpy as np
import pandas as pd
from scipy import io
from scipy.sparse import csr_matrix


@pytest.fixture
def get_ym():
    return np.genfromtxt("/usr/py_presto/test_data/matrix/labels.txt", dtype=str)


@pytest.fixture
def get_ysm():
    return np.genfromtxt(
        "/usr/py_presto/test_data/sparse_matrix/celltype.txt", dtype=str
    )


@pytest.fixture
def get_sm():
    sparsematrix = io.mmread("/usr/py_presto/test_data/sparse_matrix/sparsematrix.mtx")
    data = csr_matrix(sparsematrix, dtype=np.float64)
    return data


@pytest.fixture
def get_mm():
    data = pd.read_csv("/usr/py_presto/data/matrix/exprs.csv", index_col=0)
    data = data.to_numpy(dtype="float")
    return data
