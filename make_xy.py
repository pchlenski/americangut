import numpy as np
import pandas as pd
import anndata
from scipy.sparse import spmatrix

_neg_master = ["no", "false", "i do not have this condition", "never"]

_pos_master = [
    "yes",
    "true",
    "diagnosed by a medical professional (doctor, physician assistant)",
    "self-diagnosed",
    "diagnosed by an alternative medicine practitioner",
    "rarely (less than once/week)",
    "rarely (a few times/month)",
    "occasionally (1-2 times/week)",
    "regularly (3-5 times/week)",
    "daily",
]

_nan_master = [
    "nan",
    "not provided",
    "not sure",
    "not collected",
    "unspecified",
]


def make_anndata_X_y(
    adata: anndata.AnnData = "/home/phil/americangut/data/big_table_with_embeddings.h5ad",
    column: str = "ibd",
    pos_vals: list = _pos_master,
    neg_vals: list = _neg_master,
    nan_vals: list = _nan_master,
    balance: bool = False,
    embedding: str = None,
    seed: int = None,
    to_dense: bool = True,
) -> (np.ndarray, np.ndarray):
    """
    Turn anndata object into X and y matrices for classification based on a column.

    Args:
    ----
    adata (anndata.AnnData): Annotated data matrix.
    column (str): Name of metadata column to use for classification.
    pos_vals (list): List of strings that indicate positive class.
    neg_vals (list): List of strings that indicate negative class.
    nan_vals (list): List of strings that indicate missing data.
    balance (bool): Whether to balance classes.
    embedding (str): Name of embedding to use for X values.
    seed (int): Random seed. Only used when balancing classes.
    to_dense (bool): Whether to convert sparse matrices to dense.

    Returns:
    -------
    X (np.ndarray): Data matrix.
    y (np.ndarray): Class labels.
    """

    # Input validation
    if isinstance(adata, str):
        adata = anndata.read_h5ad(adata)
    assert isinstance(adata, anndata.AnnData)

    # Body site hack
    if column == "host_body_habitat":
        adata = adata[
            adata.obs.host_body_habitat.isin(
                ["uberon:skin", "uberon:oral cavity"]
            )
        ]
        adata.obs.host_body_habitat = adata.obs.host_body_habitat.replace(
            {"uberon:skin": "yes", "uberon:oral cavity": "no"}
        )

    # Get X values for starting
    if embedding is None:
        data = adata.X.todense()
    else:
        data = adata.obsm[embedding]

    # Get y values and clean them up (pandas operations on metadata)
    vals = adata.obs[column].str.lower()
    nan_vals = vals.isna() | vals.isin(nan_vals)
    pos_vals = vals.isin(pos_vals)
    neg_vals = vals.isin(neg_vals)
    assert not (pos_vals & neg_vals).any()
    try:
        assert (pos_vals | neg_vals | nan_vals).all()
    except AssertionError:
        print(column)
        print(vals[~(pos_vals | neg_vals | nan_vals)])

    # Initialize X and y arrays as numpy arrays; drop NaNs
    X = np.array(data[~nan_vals.values, :])
    y = np.array(pos_vals[~nan_vals])

    # Balance the classes
    n_pos = y.sum()
    n_neg = (~y).sum()
    if balance:
        if seed is not None:
            np.random.seed(seed)
        if n_pos > n_neg:
            drop = np.random.choice(
                np.where(y)[0], n_pos - n_neg, replace=False
            )
        elif n_neg > n_pos:
            drop = np.random.choice(
                np.where(~y)[0], n_neg - n_pos, replace=False
            )
        X = np.delete(X, drop, axis=0)
        y = np.delete(y, drop, axis=0)

    return X, y
