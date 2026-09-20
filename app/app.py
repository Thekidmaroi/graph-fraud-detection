"""
Bitcoin Fraud Detective -- a Streamlit app designed for a visitor with no
background in graph theory or machine learning.

Real mode: if the real Elliptic dataset has been placed in data/raw/ and the
training scripts (src/model_baseline.py, src/model_gnn.py) have been run,
this app loads those real, pre-trained models and results.

Demo mode (automatic fallback): Elliptic is distributed via Kaggle, which is
blocked in the environment this was built in (see README). Until the real
CSVs are dropped in and the models trained, the app trains a small GCN live
(a few seconds, cached) on a structured illustrative network built in
src/demo_data.py -- clearly banner'd as demo data throughout, never
presented as if it were the real result.
"""
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
PROC_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

st.set_page_config(page_title="Bitcoin Fraud Detective", layout="wide", page_icon="🕵️")

ARCH_INFO = {
    "xgb": {"name": "Looking at the transaction alone", "emoji": "🔍", "blurb": "Judges each transaction only by its own numbers (amount, timing, ...), ignoring who it's connected to."},
    "gcn": {"name": "Network AI (GCN)", "emoji": "🕸️", "blurb": "Also looks at the transaction's direct neighbors -- fraud tends to cluster."},
    "sage": {"name": "Network AI (GraphSAGE)", "emoji": "🕸️", "blurb": "A more scalable version of the same neighbor-aware idea."},
    "gat": {"name": "Network AI (GAT)", "emoji": "🕸️", "blurb": "Also weighs *which* neighbors matter most, like paying more attention to your closest associates."},
}


# --------------------------------------------------------- data loading --
def real_data_available():
    csvs = ["elliptic_txs_features.csv", "elliptic_txs_classes.csv", "elliptic_txs_edgelist.csv"]
    return all((RAW_DIR / f).exists() for f in csvs)


def real_models_available():
    return all((MODELS_DIR / f"{tag}_metrics.json").exists() for tag in ["xgb", "gcn", "sage", "gat"])


@st.cache_resource
def load_real_graph():
    from data_prep import build_graph
    return build_graph()


@st.cache_resource
def load_demo_graph_and_models():
    """Builds the small illustrative network and trains a tiny GCN + a
    logistic-regression tabular baseline on it live. Cached so this only
    happens once per app session, not on every interaction."""
    from demo_data import build_demo_graph
    from model_gnn import train_one, evaluate as gnn_evaluate
    from sklearn.linear_model import LogisticRegression
    from metrics import illicit_metrics, per_timestep_metrics, best_threshold_by_f1

    data = build_demo_graph()
    model, x_std = train_one("gcn", data, epochs=80)
    gnn_results = gnn_evaluate(model, x_std, data, "demo_gcn")

    X = data.x.numpy()
    y = data.y.numpy()
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X[data.train_mask.numpy()], y[data.train_mask.numpy()])
    val_score = clf.predict_proba(X[data.val_mask.numpy()])[:, 1]
    thr = best_threshold_by_f1(y[data.val_mask.numpy()], val_score)
    test_score = clf.predict_proba(X[data.test_mask.numpy()])[:, 1]
    test_pred = (test_score >= thr).astype(int)
    xgb_results = {"test": illicit_metrics(y[data.test_mask.numpy()], test_pred, test_score)}

    with torch.no_grad():
        gcn_scores_all = torch.sigmoid(model(x_std, data.edge_index)).numpy()

    return data, model, x_std, gcn_scores_all, {"gcn": gnn_results, "xgb": xgb_results}


@st.cache_data
def load_real_metrics_and_timesteps():
    import json
    metrics, per_ts = {}, {}
    for tag in ["xgb", "gcn", "sage", "gat"]:
        mp = MODELS_DIR / f"{tag}_metrics.json"
        tp = MODELS_DIR / f"{tag}_per_timestep.csv"
        if mp.exists():
            metrics[tag] = json.load(open(mp))
        if tp.exists():
            per_ts[tag] = pd.read_csv(tp)
    return metrics, per_ts


# ------------------------------------------------------------------ UI ----
st.title("🕵️ Bitcoin Fraud Detective")
st.markdown(
    "##### Can an AI catch money launderers by looking at who a Bitcoin transaction is connected to?"
)
st.markdown(
    "This project applies a **Graph Neural Network** -- a type of AI that reasons about "
    "networks of connections, also used for things like social-network recommendations and "
    "Google Maps traffic prediction -- to spot illicit Bitcoin transactions. The core idea: "
    "*fraud tends to cluster*. Just like a detective learns to spot suspicious people partly "
    "by who they associate with, this AI learns to spot suspicious transactions partly by "
    "which other transactions they exchange money with."
)

DEMO_MODE = not (real_data_available() and real_models_available())

if DEMO_MODE:
    if real_data_available() and not real_models_available():
        st.warning(
            "Real Elliptic data was found in `data/raw/` but the models haven't been trained "
            "yet -- run `python src/model_baseline.py` and `python src/model_gnn.py`. "
            "Showing an illustrative demo network in the meantime."
        )
    else:
        st.warning(
            "🎭 **Demo mode.** The real dataset (Elliptic: 203,769 anonymized Bitcoin "
            "transactions) is distributed via Kaggle, which requires a manual download -- "
            "see README.md. Everything below runs **live** on a small, structured, "
            "illustrative network built to demonstrate the same method honestly labeled as "
            "a demo, not a real result."
        )
    data, gcn_model, x_std, gcn_scores_all, demo_results = load_demo_graph_and_models()
else:
    data = load_real_graph()

st.divider()
tab1, tab2 = st.tabs(["🔎  Investigate a Transaction", "📈  How Good Is the Detective?"])

# =========================================================== TAB 1 =======
with tab1:
    test_idx = torch.where(data.test_mask)[0]
    illicit_idx = test_idx[data.y[test_idx] == 1]
    licit_idx = test_idx[data.y[test_idx] == 0]

    st.write(
        f"There are **{len(test_idx)}** transactions in the held-out test set "
        f"({'demo network' if DEMO_MODE else 'real Elliptic data, time steps 40-49'}), "
        f"of which **{len(illicit_idx)}** are confirmed fraudulent."
    )
    show_illicit_only = st.checkbox("Only show confirmed-fraud transactions", value=True)
    pool = illicit_idx if show_illicit_only else test_idx

    choice = st.selectbox(
        "Pick a transaction to investigate",
        options=pool.tolist()[:150],
        format_func=lambda i: f"Transaction #{data.tx_id[i].item()} "
                               f"({'known fraud' if data.y[i]==1 else 'known legitimate'})",
    )

    edge_index = data.directed_edge_index.numpy()
    G = nx.DiGraph()
    G.add_edges_from(edge_index.T.tolist())

    k = st.slider("How many steps out in the network to look?", 1, 3, 2,
                   help="1 = direct connections only. 2 = connections of connections. Investigators usually look 2-3 steps out.")

    col_graph, col_verdict = st.columns([2, 1])

    if choice in G:
        neighborhood = {choice}
        frontier = {choice}
        for _ in range(k):
            nxt = set()
            for n in frontier:
                nxt |= set(G.predecessors(n)) | set(G.successors(n))
            neighborhood |= nxt
            frontier = nxt
        neighborhood = list(neighborhood)[:150]
        sub = G.subgraph(neighborhood)
        direct_neighbors = list(set(G.predecessors(choice)) | set(G.successors(choice)))
        n_direct_illicit = sum(1 for n in direct_neighbors if data.y[n].item() == 1)

        with col_graph:
            pos = nx.spring_layout(sub, seed=0)
            edge_x, edge_y = [], []
            for u, v in sub.edges():
                edge_x += [pos[u][0], pos[v][0], None]
                edge_y += [pos[u][1], pos[v][1], None]
            node_x = [pos[n][0] for n in sub.nodes()]
            node_y = [pos[n][1] for n in sub.nodes()]

            def color_of(n):
                if n == choice:
                    return "#D62728"
                lbl = data.y[n].item()
                return "#FF7F0E" if lbl == 1 else ("#1F77B4" if lbl == 0 else "#B0B0B0")

            node_color = [color_of(n) for n in sub.nodes()]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=0.6, color="#999"), hoverinfo="none"))
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y, mode="markers",
                marker=dict(size=11, color=node_color, line=dict(width=1, color="white")),
                hoverinfo="skip",
            ))
            fig.update_layout(showlegend=False, height=480, margin=dict(l=10, r=10, t=10, b=10),
                               xaxis=dict(visible=False), yaxis=dict(visible=False))
            st.plotly_chart(fig, width='stretch')
            st.markdown(
                "🔴 **Selected transaction**&nbsp;&nbsp;&nbsp;"
                "🟠 **Known fraud**&nbsp;&nbsp;&nbsp;"
                "🔵 **Known legitimate**&nbsp;&nbsp;&nbsp;"
                "⚪ **Unknown / unconfirmed**"
            )

        with col_verdict:
            if DEMO_MODE:
                score = float(gcn_scores_all[choice])
            else:
                score = None  # real mode: would load the per-model saved test scores

            true_label = "🚨 Fraud" if data.y[choice].item() == 1 else "✅ Legitimate"
            st.markdown(f"### Known status: {true_label}")
            if score is not None:
                st.metric("AI fraud probability", f"{score:.0%}")
                st.progress(min(max(score, 0.0), 1.0))

            st.markdown("**Why the AI flagged (or cleared) it:**")
            st.markdown(f"- It has **{len(direct_neighbors)}** direct transaction partners.")
            if n_direct_illicit > 0:
                st.markdown(f"- 🔺 **{n_direct_illicit}** of them are already known fraud — a strong warning sign.")
            else:
                st.markdown("- 🔻 None of its direct partners are known fraud.")
            st.caption(
                "A full production system would also show a formal feature-attribution "
                "explanation (e.g. GNNExplainer); this view shows the intuitive network signal."
            )
    else:
        st.info("This transaction has no recorded connections in the network.")

# =========================================================== TAB 2 =======
with tab2:
    st.subheader("Two ways to judge a fraud detector")
    st.markdown(
        "- **Recall**: out of all the *real* fraud cases, how many did the detector actually catch? "
        "Miss too many and criminals get away.\n"
        "- **Precision**: out of everything the detector *flagged*, how many were actually fraud? "
        "Flag too many innocent transactions and investigators drown in false alarms.\n\n"
        "A good detector needs both — and this project compares a detector that only looks at a "
        "transaction on its own against detectors that also look at its network of connections."
    )

    if DEMO_MODE:
        rows = []
        for tag in ["xgb", "gcn"]:
            m = demo_results[tag]["test"]
            rows.append({
                "Approach": ARCH_INFO[tag]["name"],
                "Caught (Recall)": f"{m['illicit_recall']:.0%}",
                "Accuracy of alarms (Precision)": f"{m['illicit_precision']:.0%}",
            })
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
        st.caption("Demo network only (small sample) — illustrative, not a benchmark result.")
    else:
        metrics, per_ts = load_real_metrics_and_timesteps()
        rows = []
        for tag in ["xgb", "gcn", "sage", "gat"]:
            if tag in metrics and "test" in metrics[tag]:
                m = metrics[tag]["test"]
                rows.append({
                    "Approach": ARCH_INFO[tag]["name"],
                    "Caught (Recall)": f"{m['illicit_recall']:.0%}",
                    "Accuracy of alarms (Precision)": f"{m['illicit_precision']:.0%}",
                })
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

        st.subheader("Does the detective stay sharp over time?")
        st.markdown(
            "Fraud patterns change. A detector that looks great on average can still fall apart "
            "at the exact moments that matter most -- this happened for real in this dataset "
            "around a known dark-market shutdown event. This chart shows performance "
            "**month by month** in the test period, not just one average number."
        )
        fig2 = go.Figure()
        for tag, df in per_ts.items():
            fig2.add_trace(go.Scatter(x=df["time_step"], y=df["illicit_f1"], mode="lines+markers", name=ARCH_INFO[tag]["name"]))
        fig2.update_layout(xaxis_title="Time step (test period)", yaxis_title="Detection quality (Illicit F1)", height=420)
        st.plotly_chart(fig2, width='stretch')

    with st.expander("🔬 Technical details (for data scientists)"):
        st.markdown(
            "- **xgb** = XGBoost on node features alone (no graph)\n"
            "- **gcn** / **sage** / **gat** = Graph Convolutional Network / GraphSAGE / Graph "
            "Attention Network (PyTorch Geometric), full-batch transductive training\n"
            "- Evaluated with **Illicit-F1 / PR-AUC**, never accuracy (illicit transactions are "
            "~10% of labeled nodes, 77% of all nodes are unlabeled)\n"
            "- **Temporal train/val/test split** (steps 1-34 / 35-39 / 40-49) -- a random split "
            "would leak future graph structure into training\n\n"
            "See README.md for the full methodology and academic references (Weber et al. 2019; "
            "Elmougy & Liu, KDD 2023)."
        )

st.divider()
st.caption(
    "Built by Marwane Houngnon · Dataset: Elliptic Bitcoin transaction graph (203,769 "
    "transactions) · Models: PyTorch Geometric, XGBoost · See the GitHub repo for full code, "
    "tests and methodology."
)
