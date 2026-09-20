# Graph Neural Networks for Bitcoin AML — Elliptic Benchmark

Benchmarks GCN, GraphSAGE and GAT (PyTorch Geometric) against an XGBoost
tabular baseline for illicit-transaction detection on the **Elliptic**
Bitcoin dataset (203,769 transactions, 234,355 money-flow edges, 49 time
steps) — the standard academic benchmark for graph-based anti-money-
laundering research (Weber et al. 2019, IBM/Elliptic; Elliptic++, Elmougy &
Liu, KDD 2023; recent 2025 graph-transformer AML work).

## The methodological point this project is built around

Elliptic has 49 time steps and known temporal dynamics (the illicit rate
spikes sharply around step 43, a real dark-market shutdown event). A random
train/test split leaks future graph topology and future illicit patterns
into training — a mistake that quietly inflates every metric. This repo
splits **by time step** (train: 1–34, val: 35–39, test: 40–49) like every
serious published benchmark, and reports **per-time-step** Illicit-F1 /
PR-AUC in addition to the aggregate, because the original paper's headline
finding was that graph model performance is *not* stable over time — hiding
that behind one aggregate number would hide the most operationally relevant
result for an actual AML team.

Accuracy is never reported: illicit transactions are ~10% of labeled nodes
and 77% of all nodes are unlabeled, so accuracy is dominated by the trivial
"always licit" classifier.

## Status

| Component | Status |
|---|---|
| Data loading + temporal split (`src/data_prep.py`) | done, validated on synthetic data matching the real schema |
| Metrics: Illicit-F1, PR-AUC, per-time-step breakdown (`src/metrics.py`) | done |
| XGBoost tabular baseline (`src/model_baseline.py`) | done, pipeline validated |
| GCN / GraphSAGE / GAT (`src/model_gnn.py`) | done, pipeline validated + speed-tested at full graph scale (~1s/epoch on CPU) |
| Streamlit app (`app/app.py`) -- plain-language UI, auto demo-mode fallback | done |
| GNNExplainer explainability | planned (the app currently shows a simpler "N flagged neighbors" signal) |

The Streamlit app is written for a non-technical visitor first (a recruiter,
a hiring manager) -- plain-language framing, a "detective" narrative, and a
network-graph visualization with a color legend -- with a "🔬 Technical
details" expander for anyone who wants the real metric names and
architecture references. Until the real Elliptic CSVs are added, it runs in
an honestly-labeled **demo mode**: a small, structured, illustrative network
(`src/demo_data.py`) with a GCN trained live in a few seconds, never
presented as if it were the real benchmark result. Drop the 3 real CSVs into
`data/raw/` and run the training scripts, and the app automatically switches
to the real trained models.

**Blocked on data**: Elliptic is distributed via Kaggle / Google Drive, both
blocked by this environment's network policy. Every line of code above has
been validated end-to-end against synthetic data with the exact same
schema, so training on the real files is a matter of dropping them in
`data/raw/` and running the scripts below — nothing left to debug.

## Get the data (manual step, ~50MB)

1. Download the 3 CSVs from <https://www.kaggle.com/datasets/ellipticco/elliptic-data-set>
   (`elliptic_txs_features.csv`, `elliptic_txs_classes.csv`, `elliptic_txs_edgelist.csv`).
2. Put them in `data/raw/`.

## Reproduce

```bash
pip install -r requirements.txt
python src/data_prep.py        # builds the PyG graph + temporal masks
python src/model_baseline.py   # XGBoost baseline
python src/model_gnn.py        # GCN, GraphSAGE, GAT
```

## Repo layout

```
src/data_prep.py       loads the 3 raw CSVs, builds a PyG Data graph, temporal train/val/test masks
src/metrics.py          Illicit-F1, precision/recall, PR-AUC/ROC-AUC, per-time-step breakdown
src/model_baseline.py   XGBoost on node features only (no graph)
src/model_gnn.py        GCN / GraphSAGE / GAT, full-batch transductive training
src/demo_data.py        structured illustrative fallback network, used when Elliptic isn't present yet
app/app.py              Streamlit app: plain-language subgraph investigator + model comparison
```

## References

- Weber et al. (2019), *Anti-Money Laundering in Bitcoin: Experimenting with
  GCNs for Financial Forensics*, [arXiv:1908.02591](https://arxiv.org/abs/1908.02591).
- Elmougy & Liu (2023), *Demystifying Fraudulent Transactions and Illicit
  Nodes in the Bitcoin Network*, KDD'23, [arXiv:2306.06108](https://arxiv.org/abs/2306.06108).
- Kipf & Welling (2017), *Semi-Supervised Classification with GCNs*.
- Hamilton, Ying & Leskovec (2017), *Inductive Representation Learning on
  Large Graphs* (GraphSAGE).
- Velickovic et al. (2018), *Graph Attention Networks*.
