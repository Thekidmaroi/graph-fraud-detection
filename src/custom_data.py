"""
Generic graph builder for the "Train on Your Data" module.

Everything else in this project (data_prep.py, demo_data.py) builds a
torch_geometric Data object shaped for one specific schema (Elliptic, or the
illustrative demo fixture). This module builds the *same shaped* Data object
-- x / y / edge_index / directed_edge_index / tx_id / train_mask / val_mask /
test_mask -- from an arbitrary pair of (nodes, edges) tables a visitor
uploads, so any organization's transaction/entity network can be trained
with the exact same GCN (model_gnn.train_one) and evaluated with the exact
same metrics (metrics.illicit_metrics) as the Elliptic benchmark, with no
code changes required downstream.

Design choices, and why:
  - Non-numeric feature columns are one-hot encoded automatically rather
    than requiring the visitor to pre-encode them -- most real transaction
    tables mix numeric amounts with categorical fields (merchant category,
    country, channel).
  - Missing values are imputed (median for numeric, a "missing" indicator
    category for categorical) rather than dropped, since dropping rows
    would silently shrink the graph the visitor thinks they uploaded.
  - If a time column is supplied, the split is temporal (by time
    percentile), matching this project's central methodological point that
    a random split leaks future graph structure into training. Without a
    time column, the split falls back to a stratified random split, same
    as demo_data.py.
  - A hard node/edge cap keeps live, in-browser training responsive; beyond
    the cap we take a random subsample rather than rejecting the upload
    outright, and say so in the returned report.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

MAX_NODES = 8000
MAX_EDGES = 150_000
MIN_LABELED_PER_CLASS = 6


class GraphBuildError(ValueError):
    """Raised when the uploaded tables can't be turned into a trainable
    graph -- caught in app.py and shown as a plain-language st.error,
    never a raw traceback."""


@dataclass
class BuildReport:
    n_nodes: int = 0
    n_edges_kept: int = 0
    n_edges_dropped: int = 0
    n_features: int = 0
    n_labeled: int = 0
    n_positive: int = 0
    n_negative: int = 0
    n_duplicate_ids_dropped: int = 0
    subsampled: bool = False
    split_kind: str = "random"
    warnings: list = field(default_factory=list)


def _encode_features(df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, int]:
    work = df[feature_cols].copy()
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(work[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    parts = []
    if numeric_cols:
        num = work[numeric_cols].apply(pd.to_numeric, errors="coerce")
        num = num.fillna(num.median(numeric_only=True))
        num = num.fillna(0.0)  # an all-NaN column has no median -> fill 0
        parts.append(num.to_numpy(dtype=np.float64))
    if categorical_cols:
        cat = work[categorical_cols].astype("object").fillna("__missing__").astype(str)
        cat_dummies = pd.get_dummies(cat, columns=categorical_cols)
        # cap one-hot blow-up from high-cardinality free-text columns
        if cat_dummies.shape[1] > 200:
            cat_dummies = cat_dummies.iloc[:, :200]
        parts.append(cat_dummies.to_numpy(dtype=np.float64))

    if not parts:
        raise GraphBuildError("No usable feature columns were selected.")
    x = np.concatenate(parts, axis=1)
    return x, x.shape[1]


def build_custom_graph(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    id_col: str,
    label_col: str | None,
    positive_values: list,
    feature_cols: list[str],
    src_col: str,
    dst_col: str,
    time_col: str | None = None,
    seed: int = 0,
) -> tuple[Data, BuildReport]:
    report = BuildReport()
    rng = np.random.default_rng(seed)

    nodes_df = nodes_df.copy()
    before = len(nodes_df)
    nodes_df = nodes_df.drop_duplicates(subset=[id_col], keep="first").reset_index(drop=True)
    report.n_duplicate_ids_dropped = before - len(nodes_df)

    # ---- optional subsample for responsiveness --------------------------
    if len(nodes_df) > MAX_NODES:
        nodes_df = nodes_df.sample(n=MAX_NODES, random_state=seed).reset_index(drop=True)
        report.subsampled = True
        report.warnings.append(
            f"Your upload had more than {MAX_NODES:,} rows; a random sample of {MAX_NODES:,} "
            "was used so training stays fast in the browser."
        )

    if not feature_cols:
        raise GraphBuildError("Select at least one feature column.")

    x, n_features = _encode_features(nodes_df, feature_cols)
    report.n_features = n_features

    # ---- labels -----------------------------------------------------------
    if label_col is not None and label_col in nodes_df.columns:
        raw = nodes_df[label_col]
        is_positive = raw.isin(positive_values)
        is_missing = raw.isna() | (raw.astype(str).str.strip() == "")
        y = np.where(is_missing, -1, np.where(is_positive, 1, 0)).astype(np.int64)
    else:
        y = np.full(len(nodes_df), -1, dtype=np.int64)

    report.n_positive = int((y == 1).sum())
    report.n_negative = int((y == 0).sum())
    report.n_labeled = report.n_positive + report.n_negative

    if report.n_positive < MIN_LABELED_PER_CLASS or report.n_negative < MIN_LABELED_PER_CLASS:
        raise GraphBuildError(
            f"Not enough labeled examples to train on: found {report.n_positive} positive and "
            f"{report.n_negative} negative rows, need at least {MIN_LABELED_PER_CLASS} of each. "
            "Check that you picked the right label column and the right 'positive' value(s)."
        )

    # ---- edges --------------------------------------------------------------
    id_to_idx = {v: i for i, v in enumerate(nodes_df[id_col].tolist())}
    src_raw = edges_df[src_col]
    dst_raw = edges_df[dst_col]
    src_idx = src_raw.map(id_to_idx)
    dst_idx = dst_raw.map(id_to_idx)
    valid = src_idx.notna() & dst_idx.notna()
    n_dropped = int((~valid).sum())
    src_idx = src_idx[valid].astype(int).to_numpy()
    dst_idx = dst_idx[valid].astype(int).to_numpy()
    keep_self = src_idx != dst_idx
    n_dropped += int((~keep_self).sum())
    src_idx, dst_idx = src_idx[keep_self], dst_idx[keep_self]

    if len(src_idx) > MAX_EDGES:
        sel = rng.choice(len(src_idx), size=MAX_EDGES, replace=False)
        src_idx, dst_idx = src_idx[sel], dst_idx[sel]
        report.subsampled = True
        report.warnings.append(f"More than {MAX_EDGES:,} edges were found; a random subset was kept.")

    if len(src_idx) == 0:
        raise GraphBuildError(
            "None of the edges could be matched to a node ID. Check that the edge file's "
            "source/destination columns use the same ID values as the node file's ID column."
        )

    report.n_edges_kept = len(src_idx)
    report.n_edges_dropped = n_dropped

    directed_edge_index = torch.tensor(np.vstack([src_idx, dst_idx]), dtype=torch.long)
    edge_index = torch.cat([directed_edge_index, directed_edge_index.flip(0)], dim=1)

    data = Data(x=torch.tensor(x, dtype=torch.float32), edge_index=edge_index)
    data.y = torch.tensor(y, dtype=torch.long)
    data.directed_edge_index = directed_edge_index
    data.tx_id = torch.arange(len(nodes_df), dtype=torch.long)
    data.node_label_lookup = nodes_df[id_col].astype(str).tolist()  # for display only

    n_nodes = len(nodes_df)
    labeled_idx = np.where(y != -1)[0]

    if time_col is not None and time_col in nodes_df.columns:
        report.split_kind = "temporal"
        times = pd.to_numeric(nodes_df[time_col], errors="coerce").to_numpy()
        times_labeled = times[labeled_idx]
        finite = np.isfinite(times_labeled)
        if finite.sum() < len(labeled_idx):
            report.warnings.append("Some rows had a non-numeric time value; they were excluded from the split.")
        labeled_idx = labeled_idx[finite]
        times_labeled = times_labeled[finite]
        order = np.argsort(times_labeled, kind="stable")
        labeled_idx = labeled_idx[order]
        n = len(labeled_idx)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        train_sel = labeled_idx[:n_train]
        val_sel = labeled_idx[n_train:n_train + n_val]
        test_sel = labeled_idx[n_train + n_val:]
    else:
        report.split_kind = "random"
        from sklearn.model_selection import train_test_split

        train_sel, rest = train_test_split(
            labeled_idx, train_size=0.6, random_state=seed, stratify=y[labeled_idx]
        )
        val_sel, test_sel = train_test_split(
            rest, train_size=0.5, random_state=seed, stratify=y[rest]
        )

    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[train_sel] = True
    val_mask[val_sel] = True
    test_mask[test_sel] = True
    data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask
    report.n_nodes = n_nodes

    return data, report
