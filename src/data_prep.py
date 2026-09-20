"""
Elliptic Bitcoin transaction graph -- loading and the (mandatory) temporal
split.

Format (Weber et al. 2019, "Anti-Money Laundering in Bitcoin: Experimenting
with Graph Convolutional Networks for Financial Forensics"):
  - elliptic_txs_features.csv : 203,769 rows x 167 cols, NO HEADER.
        col 0  = txId
        col 1  = time step (1..49)
        cols 2..167 = 165 features (94 "local" tx features + 72 features
                      aggregated from 1-hop neighbours, as released by
                      Elliptic; the exact semantics of most columns were
                      never disclosed by Elliptic for confidentiality).
  - elliptic_txs_classes.csv  : header "txId,class"; class in {"1","2","unknown"}
        1 = illicit, 2 = licit, "unknown" = unlabeled (77% of nodes).
  - elliptic_txs_edgelist.csv : header "txId1,txId2"; directed money-flow edge.

THE CRITICAL METHODOLOGICAL POINT of this whole project: this dataset has a
built-in temporal structure (49 time steps), so a random train/test split
leaks future graph topology and future illicit patterns into training. Every
serious benchmark on this dataset (the original paper, Elliptic++, the 2025
graph-transformer AML papers) splits BY TIME STEP. We do the same:
  - train: steps 1-34
  - val:   steps 35-39
  - test:  steps 40-49
This mirrors known illicit concentration dynamics (the illicit rate spikes
around step 43, a "dark market shutdown" event in the original paper) and is
also why per-time-step performance (not just an aggregate score) is reported
in eval.py: a model that only looks good on average can still collapse
exactly on the hardest, most operationally relevant time steps.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROC_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_STEPS = set(range(1, 35))
VAL_STEPS = set(range(35, 40))
TEST_STEPS = set(range(40, 50))


def load_raw():
    feat_path = RAW_DIR / "elliptic_txs_features.csv"
    class_path = RAW_DIR / "elliptic_txs_classes.csv"
    edge_path = RAW_DIR / "elliptic_txs_edgelist.csv"

    features = pd.read_csv(feat_path, header=None)
    n_feat_cols = features.shape[1] - 2
    features.columns = ["txId", "timeStep"] + [f"feat_{i}" for i in range(n_feat_cols)]

    classes = pd.read_csv(class_path)
    classes.columns = [c.strip() for c in classes.columns]

    edges = pd.read_csv(edge_path)
    edges.columns = [c.strip() for c in edges.columns]

    return features, classes, edges


def build_graph() -> Data:
    features, classes, edges = load_raw()

    df = features.merge(classes, on="txId", how="left")
    cls = df["class"].astype(str).replace({"1": "illicit", "2": "licit", "unknown": "unknown"})
    label = cls.map({"illicit": 1, "licit": 0, "unknown": -1})
    df = pd.concat([df.drop(columns=["class"]), cls.rename("class"), label.rename("label")], axis=1)

    # Stable node ordering: txId -> contiguous integer node index (what PyG needs)
    df = df.reset_index(drop=True)
    txid_to_idx = {tx: i for i, tx in enumerate(df["txId"])}

    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    x = torch.tensor(df[feat_cols].values, dtype=torch.float32)
    y = torch.tensor(df["label"].values, dtype=torch.long)
    time_step = torch.tensor(df["timeStep"].values, dtype=torch.long)

    src = edges["txId1"].map(txid_to_idx)
    dst = edges["txId2"].map(txid_to_idx)
    valid = src.notna() & dst.notna()
    edge_index = torch.tensor(np.vstack([src[valid].values, dst[valid].values]), dtype=torch.long)

    # Make the graph undirected for message passing (GCN/SAGE/GAT all assume
    # symmetric neighborhoods); the original directed edge is still what a
    # money-flow investigator cares about, so we keep the direction available
    # separately for the Streamlit subgraph viewer.
    edge_index_undirected = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    data = Data(x=x, edge_index=edge_index_undirected, y=y)
    data.time_step = time_step
    data.directed_edge_index = edge_index
    data.tx_id = torch.tensor(df["txId"].values, dtype=torch.long)

    train_mask = torch.tensor(df["timeStep"].isin(TRAIN_STEPS).values) & (y != -1)
    val_mask = torch.tensor(df["timeStep"].isin(VAL_STEPS).values) & (y != -1)
    test_mask = torch.tensor(df["timeStep"].isin(TEST_STEPS).values) & (y != -1)

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data


def summarize(data: Data):
    n_nodes = data.num_nodes
    n_edges = data.directed_edge_index.shape[1]
    n_labeled = (data.y != -1).sum().item()
    n_illicit = (data.y == 1).sum().item()
    print(f"Nodes: {n_nodes:,} | directed edges: {n_edges:,} | labeled: {n_labeled:,} ({n_labeled/n_nodes:.1%})")
    print(f"Illicit: {n_illicit:,} ({n_illicit/n_labeled:.1%} of labeled)")
    for name, mask in [("train", data.train_mask), ("val", data.val_mask), ("test", data.test_mask)]:
        n = mask.sum().item()
        n_ill = ((data.y == 1) & mask).sum().item()
        print(f"  {name}: {n:,} labeled nodes, {n_ill:,} illicit ({n_ill/max(n,1):.1%})")


if __name__ == "__main__":
    data = build_graph()
    summarize(data)
    torch.save(data, PROC_DIR / "elliptic_graph.pt")
    print(f"Saved graph to {PROC_DIR / 'elliptic_graph.pt'}")
