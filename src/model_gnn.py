"""
Graph neural network benchmark: GCN (Kipf & Welling 2017), GraphSAGE
(Hamilton et al. 2017), GAT (Velickovic et al. 2018) -- the three
architectures every follow-up paper on this dataset benchmarks against,
trained full-batch and transductively over the whole (undirected,
time-stamped) transaction graph, with the mandatory temporal train/val/test
masks from data_prep.py.

Class imbalance (illicit ~10% of labeled nodes) is handled with a weighted
BCE loss rather than resampling, which would require constructing a
different graph per epoch -- not sensible for a GNN, unlike for tabular
models.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

from data_prep import build_graph
from metrics import illicit_metrics, per_timestep_metrics, best_threshold_by_f1

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(exist_ok=True)
torch.manual_seed(0)


class GNN(nn.Module):
    def __init__(self, arch: str, in_dim: int, hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        conv_cls = {"gcn": GCNConv, "sage": SAGEConv, "gat": GATConv}[arch]
        if arch == "gat":
            heads = 4
            self.conv1 = conv_cls(in_dim, hidden // heads, heads=heads)
            self.conv2 = conv_cls(hidden, hidden // heads, heads=heads)
            self.conv3 = conv_cls(hidden, 1, heads=1, concat=False)
        else:
            self.conv1 = conv_cls(in_dim, hidden)
            self.conv2 = conv_cls(hidden, hidden)
            self.conv3 = conv_cls(hidden, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv3(x, edge_index).squeeze(-1)  # logit


def train_one(arch: str, data, epochs=150, lr=0.01, weight_decay=5e-4, patience=15):
    x = data.x
    # Standardize features (helps GAT/GCN optimization a lot on raw Elliptic
    # features, which are already roughly standardized by Elliptic but not
    # exactly zero-mean/unit-var after our own preprocessing).
    mean, std = x[data.train_mask].mean(0, keepdim=True), x[data.train_mask].std(0, keepdim=True) + 1e-6
    x = (x - mean) / std

    model = GNN(arch, in_dim=x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    y = data.y.float()
    pos_weight = torch.tensor(
        (data.y[data.train_mask] == 0).sum().item() / max((data.y[data.train_mask] == 1).sum().item(), 1)
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_f1, best_state, bad = -1.0, None, 0
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(x, data.edge_index)
        loss = criterion(logits[data.train_mask], y[data.train_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(x, data.edge_index)
            val_score = torch.sigmoid(logits[data.val_mask]).numpy()
            val_true = data.y[data.val_mask].numpy()
            thr = best_threshold_by_f1(val_true, val_score)
            val_pred = (val_score >= thr).astype(int)
            val_f1 = illicit_metrics(val_true, val_pred, val_score)["illicit_f1"]

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  [{arch}] epoch {epoch}: train_loss={loss.item():.4f} val_illicit_f1={val_f1:.4f}", flush=True)

        if val_f1 > best_val_f1 + 1e-4:
            best_val_f1, best_state, bad = val_f1, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    return model, x


def evaluate(model, x, data, arch_name):
    model.eval()
    with torch.no_grad():
        logits = model(x, data.edge_index)
        val_score = torch.sigmoid(logits[data.val_mask]).numpy()
        val_true = data.y[data.val_mask].numpy()
        thr = best_threshold_by_f1(val_true, val_score)

        results = {}
        for name, mask in [("val", data.val_mask), ("test", data.test_mask)]:
            score = torch.sigmoid(logits[mask]).numpy()
            true = data.y[mask].numpy()
            pred = (score >= thr).astype(int)
            results[name] = illicit_metrics(true, pred, score)

        test_score = torch.sigmoid(logits[data.test_mask]).numpy()
        test_pred = (test_score >= thr).astype(int)
        per_ts = per_timestep_metrics(
            data.y[data.test_mask].numpy(), test_pred, test_score, data.time_step[data.test_mask].numpy()
        )
        per_ts.to_csv(MODELS_DIR / f"{arch_name}_per_timestep.csv", index=False)

    print(f"=== {arch_name} ===")
    print(json.dumps(results, indent=2))
    with open(MODELS_DIR / f"{arch_name}_metrics.json", "w") as f:
        json.dump({"threshold": thr, **results}, f, indent=2)
    return results


def main():
    data = build_graph()
    for arch in ["gcn", "sage", "gat"]:
        model, x_std = train_one(arch, data)
        evaluate(model, x_std, data, arch)
        torch.save(model.state_dict(), MODELS_DIR / f"{arch}.pt")


if __name__ == "__main__":
    main()
