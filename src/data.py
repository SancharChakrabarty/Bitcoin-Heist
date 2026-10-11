"""
Data loading, preprocessing, and graph construction for Bitcoin Heist dataset.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.model_selection import train_test_split
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

# Standard feature columns in the UCI BitcoinHeist dataset
FEATURE_COLS = ["year", "day", "length", "weight", "count", "looped", "neighbors", "income"]


def load_dataset(file_path: str) -> pd.DataFrame:
    """Load the BitcoinHeist CSV dataset from disk."""
    df = pd.read_csv(file_path)
    print(f"Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print("Columns:", df.columns.tolist())
    return df


def preprocess(
    df: pd.DataFrame,
    max_samples: int = 50_000,
    random_state: int = 42,
) -> tuple[np.ndarray, pd.Series]:
    """
    Encode labels, subsample, and scale features.

    Args:
        df: Raw dataframe with feature columns and a 'label' column.
        max_samples: Maximum number of rows to keep (stratified).
        random_state: Random seed for reproducibility.

    Returns:
        X_scaled (np.ndarray): Scaled feature matrix.
        y_binary (pd.Series): Binary labels — 0 = benign, 1 = ransomware.
    """
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    print("Using features:", available_features)

    X_raw = df[available_features].copy()
    y_raw = df["label"].copy()

    # Binary encoding: 'white' (benign) = 0, everything else = 1
    y_binary = (y_raw != "white").astype(int)
    print(f"Benign (0): {(y_binary == 0).sum():,}  |  Ransomware (1): {(y_binary == 1).sum():,}")

    # Stratified subsample to keep memory usage manageable
    if len(X_raw) > max_samples:
        idx = (
            y_binary.groupby(y_binary, group_keys=False)
            .apply(lambda x: x.sample(min(len(x), max_samples // 2), random_state=random_state))
            .index
        )
        X_raw = X_raw.loc[idx].reset_index(drop=True)
        y_binary = y_binary.loc[idx].reset_index(drop=True)
        print(f"Subsampled to {len(X_raw):,} rows")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    print("Preprocessing complete.")
    return X_scaled, y_binary


def build_graph(
    X_scaled: np.ndarray,
    y_binary: pd.Series,
    k_neighbors: int = 5,
    test_size: float = 0.2,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Data:
    """
    Build a k-NN similarity graph and return a PyG Data object with masks.

    Args:
        X_scaled: Scaled node feature matrix.
        y_binary: Binary node labels.
        k_neighbors: Number of neighbours for the k-NN graph.
        test_size: Fraction of nodes reserved for testing.
        val_size: Fraction of training nodes reserved for validation.
        random_state: Random seed.

    Returns:
        PyG Data object with x, edge_index, y, train_mask, val_mask, test_mask.
    """
    n = len(X_scaled)
    print(f"Building {k_neighbors}-NN graph for {n:,} nodes …")

    A = kneighbors_graph(X_scaled, k_neighbors, mode="connectivity", include_self=False, n_jobs=-1)
    A_coo = sp.coo_matrix(A)
    edge_index = torch.tensor(np.vstack([A_coo.row, A_coo.col]), dtype=torch.long)

    x = torch.tensor(X_scaled, dtype=torch.float)
    y = torch.tensor(y_binary.values, dtype=torch.long)

    # Train / Val / Test splits
    idx_all = np.arange(n)
    idx_train, idx_test = train_test_split(
        idx_all, test_size=test_size, stratify=y_binary.values, random_state=random_state
    )
    idx_train, idx_val = train_test_split(
        idx_train, test_size=val_size, stratify=y_binary.values[idx_train], random_state=random_state
    )

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True

    data = Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    print(data)
    print(f"Nodes — train: {train_mask.sum():,}  val: {val_mask.sum():,}  test: {test_mask.sum():,}")
    print("Graph construction complete.")
    return data
