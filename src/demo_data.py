"""
A small, *structured* illustrative graph used only when the real Elliptic
CSVs are not yet present in data/raw/ (they are Kaggle-gated -- see README).

Unlike the pure-noise fixtures in tests/test_pipeline.py (which exist only
to exercise code paths), this one is deliberately built to LOOK like a
believable transaction network: a few tight "fraud rings" (densely
connected among themselves, feature vectors drawn from a shifted
distribution), a larger loosely-connected "legitimate" population, and some
unlabeled nodes -- so a visitor sees a coherent, explainable demo rather
than a wall of random noise, while a banner makes unmistakably clear this
is illustrative, not real Elliptic data.
"""
import numpy as np
import torch
from torch_geometric.data import Data

N_FEATURES = 32  # smaller than Elliptic's 165 -- this is a toy illustration, not a benchmark
N_TIME_STEPS = 20


def build_demo_graph(seed: int = 7) -> Data:
    rng = np.random.default_rng(seed)

    n_rings, ring_size = 4, rng.integers(5, 9, size=4)
    n_illicit = int(ring_size.sum())
    n_licit = 180
    n_unknown = 60
    n_nodes = n_illicit + n_licit + n_unknown

    x = np.zeros((n_nodes, N_FEATURES))
    y = np.full(n_nodes, -1, dtype=int)
    time_step = rng.integers(1, N_TIME_STEPS + 1, size=n_nodes)

    edges = []
    node_cursor = 0
    ring_members = []
    for size in ring_size:
        members = list(range(node_cursor, node_cursor + size))
        ring_members.append(members)
        x[members] = rng.normal(loc=2.5, scale=0.6, size=(size, N_FEATURES))
        y[members] = 1
        # dense intra-ring connectivity (fraud rings transact with each other a lot)
        for i in members:
            for j in members:
                if i != j and rng.random() < 0.55:
                    edges.append((i, j))
        node_cursor += size

    licit_start = node_cursor
    licit_members = list(range(licit_start, licit_start + n_licit))
    x[licit_members] = rng.normal(loc=0.0, scale=1.0, size=(n_licit, N_FEATURES))
    y[licit_members] = 0
    node_cursor += n_licit

    unknown_start = node_cursor
    unknown_members = list(range(unknown_start, unknown_start + n_unknown))
    x[unknown_members] = rng.normal(loc=0.3, scale=1.1, size=(n_unknown, N_FEATURES))
    y[unknown_members] = -1

    # sparse random licit-licit and licit-unknown edges (normal transaction noise)
    all_normal = licit_members + unknown_members
    n_random_edges = int(len(all_normal) * 1.8)
    src = rng.choice(all_normal, n_random_edges)
    dst = rng.choice(all_normal, n_random_edges)
    edges += list(zip(src.tolist(), dst.tolist()))

    # a few "bridge" edges: rings occasionally transact with a handful of
    # unsuspecting licit/unknown nodes -- this is what makes a fraud ring
    # detectable via its neighborhood even before looking at its own features
    for members in ring_members:
        bridges = rng.choice(all_normal, size=3, replace=False)
        for b in bridges:
            edges.append((int(rng.choice(members)), int(b)))

    edges = [(i, j) for i, j in edges if i != j]
    edge_index = torch.tensor(np.array(edges).T, dtype=torch.long)
    edge_index_undirected = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    data = Data(x=torch.tensor(x, dtype=torch.float32), edge_index=edge_index_undirected)
    data.y = torch.tensor(y, dtype=torch.long)
    data.time_step = torch.tensor(time_step, dtype=torch.long)
    data.directed_edge_index = edge_index
    data.tx_id = torch.arange(n_nodes, dtype=torch.long) + 900000  # fake, obviously-not-real IDs

    labeled = data.y != -1
    n_labeled = labeled.sum().item()
    perm = torch.randperm(n_labeled)
    labeled_idx = torch.where(labeled)[0][perm]
    n_train = int(n_labeled * 0.6)
    n_val = int(n_labeled * 0.2)

    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[labeled_idx[:n_train]] = True
    val_mask[labeled_idx[n_train:n_train + n_val]] = True
    test_mask[labeled_idx[n_train + n_val:]] = True

    data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask
    return data
