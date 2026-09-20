"""
Fraud Detective -- a Streamlit app designed for a visitor with no
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

Visual design: adapted from fraud.net (an enterprise fraud/risk AI vendor)
-- a dark navy canvas, a single bright blue accent with a pink secondary
accent, Poppins typeface, rounded dark cards with a colored top accent, and
no emoji anywhere, in place of the default Streamlit look. The underlying
graph-ML pipeline is untouched; only the presentation layer changed.
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

st.set_page_config(page_title="Fraud Detective -- Graph AI for Bitcoin", layout="wide")

# ---------------------------------------------------------------- palette --
# Adapted from fraud.net: near-black navy canvas, a single bright blue
# accent (their primary CTA color), a pink secondary accent (their
# wordmark), Poppins typeface. No emoji anywhere in this app -- symbols are
# small colored square/dot chips instead, exactly like a real SaaS product.
BG = "#08031F"
CARD = "#140E36"
CARD_BORDER = "rgba(255,255,255,0.09)"
BLUE = "#0085FF"
BLUE_SOFT = "#48A8FF"
PINK = "#E91E8C"
INK = "#F2F1FB"
MUTED = "#A6A4C7"
GREEN = "#22C55E"
RED = "#F04452"
AMBER = "#F5A524"
GRAY = "#7C7B9C"

ARCH_INFO = {
    "xgb": {
        "name": "Tabular Baseline",
        "tag": "NO GRAPH",
        "color": GRAY,
        "blurb": "Judges each transaction only by its own numbers (amount, timing, ...), ignoring who it's connected to.",
    },
    "gcn": {
        "name": "Graph AI -- GCN",
        "tag": "GRAPH NEURAL NET · FEATURED",
        "color": BLUE,
        "blurb": "Also looks at the transaction's direct neighbors in the network -- fraud tends to cluster.",
    },
    "sage": {
        "name": "Graph AI -- GraphSAGE",
        "tag": "GRAPH NEURAL NET",
        "color": BLUE_SOFT,
        "blurb": "A more scalable version of the same neighbor-aware idea, built for larger networks.",
    },
    "gat": {
        "name": "Graph AI -- GAT",
        "tag": "GRAPH NEURAL NET",
        "color": PINK,
        "blurb": "Also weighs *which* neighbors matter most, like paying more attention to your closest associates.",
    },
}
ARCH_ORDER = ["xgb", "gcn", "sage", "gat"]


# ------------------------------------------------------------------- CSS --
def inject_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {{ font-family: 'Poppins', sans-serif; }}

        .stApp {{ background: {BG}; }}
        .block-container {{ padding-top: 2rem; max-width: 1120px; }}

        /* ---- hero ---------------------------------------------------- */
        .hero {{
            background: radial-gradient(120% 160% at 100% 0%, #1B1666 0%, {BG} 60%), {BG};
            border: 1px solid {CARD_BORDER};
            border-radius: 20px;
            padding: 2.75rem 2.75rem 2.25rem 2.75rem;
            margin-bottom: 1.75rem;
        }}
        .hero .eyebrow {{
            display: flex;
            align-items: center;
            gap: .5rem;
            text-transform: uppercase;
            letter-spacing: .08em;
            font-size: .78rem;
            font-weight: 700;
            color: {BLUE_SOFT};
            margin-bottom: .9rem;
        }}
        .hero .eyebrow .dot {{
            width: 8px; height: 8px; border-radius: 50%; background: {PINK}; display: inline-block;
        }}
        .hero h1 {{
            font-size: 2.35rem;
            font-weight: 700;
            line-height: 1.18;
            margin: 0 0 .9rem 0;
            color: white;
        }}
        .hero p {{
            font-size: 1.02rem;
            color: {MUTED};
            max-width: 700px;
            margin-bottom: 1.4rem;
        }}
        .checklist {{ list-style: none; padding: 0; margin: 0; }}
        .checklist li {{
            color: {INK};
            font-size: .96rem;
            padding: .35rem 0;
            padding-left: 1.6rem;
            position: relative;
        }}
        .checklist li::before {{
            content: "\\2713";
            position: absolute;
            left: 0;
            color: {BLUE_SOFT};
            font-weight: 800;
        }}

        /* ---- stat strip ------------------------------------------------ */
        .stat-strip {{ display: flex; gap: 1rem; margin-bottom: 1.75rem; flex-wrap: wrap; }}
        .stat-card {{
            flex: 1 1 200px;
            background: {CARD};
            border: 1px solid {CARD_BORDER};
            border-radius: 14px;
            padding: 1.1rem 1.3rem;
        }}
        .stat-card .num {{ font-size: 1.5rem; font-weight: 800; color: white; }}
        .stat-card .lbl {{ font-size: .85rem; color: {MUTED}; margin-top: .15rem; }}

        /* ---- section headings ------------------------------------------ */
        h2, h3 {{ color: white !important; font-weight: 700; }}
        .section-note {{ color: {MUTED}; font-size: .95rem; }}
        .stMarkdown p, .stMarkdown li {{ color: {INK}; }}
        [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {{ color: {MUTED} !important; }}

        /* ---- capability cards (the 4 detection approaches) -------------- */
        .cap-card {{
            background: {CARD};
            border: 1px solid {CARD_BORDER};
            border-radius: 16px;
            padding: 1.25rem 1.3rem 1.1rem 1.3rem;
            height: 100%;
            border-top: 3px solid var(--accent, {BLUE});
        }}
        .cap-card .tag {{
            display: inline-block;
            font-size: .66rem;
            font-weight: 800;
            letter-spacing: .04em;
            padding: .2rem .55rem;
            border-radius: 999px;
            color: {BG};
            background: var(--accent, {BLUE});
            margin-bottom: .7rem;
        }}
        .cap-card .name {{ font-size: 1.0rem; font-weight: 700; color: white; margin-bottom: .35rem; }}
        .cap-card .blurb {{ font-size: .85rem; color: {MUTED}; line-height: 1.5; }}

        /* ---- callout (demo-mode banner) ---------------------------------- */
        .callout {{
            background: rgba(0, 133, 255, 0.10);
            border: 1px solid rgba(0, 133, 255, 0.35);
            border-radius: 14px;
            padding: 1rem 1.2rem;
            color: {INK};
            font-size: .92rem;
        }}
        .callout .pill {{
            display: inline-block;
            font-size: .66rem;
            font-weight: 800;
            letter-spacing: .04em;
            padding: .2rem .55rem;
            border-radius: 999px;
            color: {BG};
            background: {AMBER};
            margin-right: .5rem;
        }}

        /* ---- verdict badges ------------------------------------------------ */
        .badge {{
            display: inline-block;
            padding: .3rem .8rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: .85rem;
            color: white;
        }}
        .badge-red {{ background: {RED}; }}
        .badge-green {{ background: {GREEN}; }}
        .badge-gray {{ background: {GRAY}; }}

        /* ---- factor rows (why flagged) --------------------------------------- */
        .factor-row {{ padding: .35rem 0; font-size: .95rem; color: {INK}; }}
        .factor-tag {{
            display: inline-block;
            min-width: 108px;
            text-align: center;
            font-size: .72rem;
            font-weight: 800;
            letter-spacing: .03em;
            padding: .15rem .5rem;
            border-radius: 6px;
            color: white;
            margin-right: .5rem;
        }}
        .factor-warn {{ background: {RED}; }}
        .factor-clear {{ background: {GREEN}; }}

        /* ---- graph legend ------------------------------------------------------ */
        .legend-row {{ display: flex; gap: 1.3rem; flex-wrap: wrap; margin-top: .6rem; }}
        .legend-item {{ display: flex; align-items: center; gap: .45rem; font-size: .85rem; color: {MUTED}; }}
        .legend-chip {{ width: 12px; height: 12px; border-radius: 3px; display: inline-block; }}

        /* ---- comparison cards (tab 2) ------------------------------------------- */
        .metric-card {{
            background: {CARD};
            border: 1px solid {CARD_BORDER};
            border-radius: 16px;
            padding: 1.2rem 1.3rem;
            height: 100%;
        }}
        .metric-card .tag {{
            display: inline-block;
            font-size: .66rem;
            font-weight: 800;
            letter-spacing: .04em;
            padding: .2rem .55rem;
            border-radius: 999px;
            color: {BG};
            margin-bottom: .6rem;
        }}
        .metric-card .name {{ font-size: .95rem; font-weight: 700; color: white; margin-bottom: .7rem; }}
        .metric-row {{ display: flex; justify-content: space-between; align-items: baseline; padding: .25rem 0; }}
        .metric-row .lbl {{ font-size: .82rem; color: {MUTED}; }}
        .metric-row .val {{ font-size: 1.15rem; font-weight: 800; color: white; }}

        /* ---- streamlit widget overrides ------------------------------------- */
        .stTabs [data-baseweb="tab"] {{ font-weight: 700; color: {MUTED}; }}
        .stTabs [aria-selected="true"] {{ color: white !important; }}
        .stTabs [data-baseweb="tab-highlight"] {{ background-color: {BLUE} !important; }}
        [data-testid="stTab"] .react-aria-SelectionIndicator {{ background-color: {BLUE} !important; }}
        div[data-testid="stMetricValue"] {{ color: white; }}
        [data-testid="stProgressBarTrack"] > div {{ background-color: {BLUE} !important; }}
        [data-testid="stSliderThumbValue"] {{ color: {BLUE_SOFT} !important; }}
        div[data-baseweb="slider"] div[role="slider"] {{ background-color: {BLUE} !important; border-color: {BLUE} !important; }}
        [data-testid="stPlotlyChart"] > div {{ border-radius: 16px; overflow: hidden; }}
        [data-testid="stDataFrame"] {{ border: 1px solid {CARD_BORDER}; border-radius: 12px; overflow: hidden; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


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
inject_css()

DEMO_MODE = not (real_data_available() and real_models_available())

# ---- hero ---------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
        <div class="eyebrow"><span class="dot"></span>Personal data science project</div>
        <h1>Can an AI catch money launderers by looking at who a Bitcoin transaction talks to?</h1>
        <p>This project applies a <strong>Graph Neural Network</strong> -- a type of AI that reasons
        about networks of connections, also used for social-network recommendations and Google Maps
        traffic prediction -- to spot illicit Bitcoin transactions. The core idea: <strong>fraud
        tends to cluster</strong>. Just like a detective learns to spot suspicious people partly by
        who they associate with, this AI learns to spot suspicious transactions partly by which
        other transactions they exchange money with.</p>
        <ul class="checklist">
            <li>Elliptic dataset: 203,769 real, anonymized Bitcoin transactions</li>
            <li>Four detection approaches compared head-to-head, from tabular to graph AI</li>
            <li>Every verdict explained in plain English, not just a score</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- capability cards: the four detection approaches ---------------------
st.subheader("Four ways to catch a fraudulent transaction")
st.markdown('<p class="section-note">From "look at the transaction alone" to "read its whole neighborhood".</p>', unsafe_allow_html=True)
cap_cols = st.columns(4)
for col, tag in zip(cap_cols, ARCH_ORDER):
    info = ARCH_INFO[tag]
    with col:
        st.markdown(
            f"""
            <div class="cap-card" style="--accent:{info['color']}">
                <span class="tag">{info['tag']}</span>
                <div class="name">{info['name']}</div>
                <div class="blurb">{info['blurb']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
st.write("")

if DEMO_MODE:
    if real_data_available() and not real_models_available():
        st.markdown(
            '<div class="callout"><span class="pill">SETUP NEEDED</span>Real Elliptic data was found '
            'in <code>data/raw/</code> but the models haven\'t been trained yet -- run '
            '<code>python src/model_baseline.py</code> and <code>python src/model_gnn.py</code>. '
            'Showing an illustrative demo network in the meantime.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="callout"><span class="pill">DEMO MODE</span>The real dataset (Elliptic: '
            '203,769 anonymized Bitcoin transactions) is distributed via Kaggle, which requires a '
            'manual download -- see README.md. Everything below runs <strong>live</strong> on a '
            'small, structured, illustrative network built to demonstrate the same method honestly '
            '-- labeled as a demo, never presented as a real result.</div>',
            unsafe_allow_html=True,
        )
    data, gcn_model, x_std, gcn_scores_all, demo_results = load_demo_graph_and_models()
else:
    data = load_real_graph()

st.write("")
tab1, tab2 = st.tabs(["Investigate a Transaction", "How Good Is the Detective?"])

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
                    return PINK
                lbl = data.y[n].item()
                return RED if lbl == 1 else (BLUE if lbl == 0 else GRAY)

            node_color = [color_of(n) for n in sub.nodes()]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=0.6, color="rgba(166,164,199,0.35)"), hoverinfo="none"))
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y, mode="markers",
                marker=dict(size=11, color=node_color, line=dict(width=1, color=BG)),
                hoverinfo="skip",
            ))
            fig.update_layout(
                showlegend=False, height=480, margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(visible=False), yaxis=dict(visible=False),
                paper_bgcolor=CARD, plot_bgcolor=CARD,
                font=dict(family="Poppins, sans-serif", color=INK),
            )
            st.plotly_chart(fig, width='stretch')
            st.markdown(
                f"""
                <div class="legend-row">
                    <div class="legend-item"><span class="legend-chip" style="background:{PINK}"></span>Selected transaction</div>
                    <div class="legend-item"><span class="legend-chip" style="background:{RED}"></span>Known fraud</div>
                    <div class="legend-item"><span class="legend-chip" style="background:{BLUE}"></span>Known legitimate</div>
                    <div class="legend-item"><span class="legend-chip" style="background:{GRAY}"></span>Unknown / unconfirmed</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_verdict:
            if DEMO_MODE:
                score = float(gcn_scores_all[choice])
            else:
                score = None  # real mode: would load the per-model saved test scores

            is_fraud = data.y[choice].item() == 1
            badge_label = "Fraud" if is_fraud else "Legitimate"
            badge_class = "red" if is_fraud else "green"
            st.markdown(f'<p class="section-note">Known status</p><span class="badge badge-{badge_class}">{badge_label}</span>', unsafe_allow_html=True)
            st.write("")
            if score is not None:
                st.metric("AI fraud probability", f"{score:.0%}")
                st.progress(min(max(score, 0.0), 1.0))

            st.markdown('<p style="font-weight:700; color:white; margin-top:1rem;">Why the AI flagged (or cleared) it</p>', unsafe_allow_html=True)
            st.markdown(f'<div class="factor-row"><span class="factor-tag factor-clear">CONTEXT</span>It has <strong>{len(direct_neighbors)}</strong> direct transaction partners.</div>', unsafe_allow_html=True)
            if n_direct_illicit > 0:
                st.markdown(f'<div class="factor-row"><span class="factor-tag factor-warn">WARNING SIGN</span><strong>{n_direct_illicit}</strong> of them are already known fraud.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="factor-row"><span class="factor-tag factor-clear">CLEAR SIGNAL</span>None of its direct partners are known fraud.</div>', unsafe_allow_html=True)
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
        "A good detector needs both -- and this project compares a detector that only looks at a "
        "transaction on its own against detectors that also look at its network of connections."
    )
    st.write("")

    if DEMO_MODE:
        active_tags = [t for t in ["xgb", "gcn"] if t in demo_results]
        cols = st.columns(len(active_tags))
        for col, tag in zip(cols, active_tags):
            m = demo_results[tag]["test"]
            info = ARCH_INFO[tag]
            with col:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <span class="tag" style="background:{info['color']}">{info['tag']}</span>
                        <div class="name">{info['name']}</div>
                        <div class="metric-row"><span class="lbl">Caught (Recall)</span><span class="val">{m['illicit_recall']:.0%}</span></div>
                        <div class="metric-row"><span class="lbl">Accuracy of alarms (Precision)</span><span class="val">{m['illicit_precision']:.0%}</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.caption("Demo network only (small sample) -- illustrative, not a benchmark result.")
    else:
        metrics, per_ts = load_real_metrics_and_timesteps()
        active_tags = [t for t in ARCH_ORDER if t in metrics and "test" in metrics[t]]
        cols = st.columns(len(active_tags))
        for col, tag in zip(cols, active_tags):
            m = metrics[tag]["test"]
            info = ARCH_INFO[tag]
            with col:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <span class="tag" style="background:{info['color']}">{info['tag']}</span>
                        <div class="name">{info['name']}</div>
                        <div class="metric-row"><span class="lbl">Caught (Recall)</span><span class="val">{m['illicit_recall']:.0%}</span></div>
                        <div class="metric-row"><span class="lbl">Accuracy of alarms (Precision)</span><span class="val">{m['illicit_precision']:.0%}</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.write("")
        st.subheader("Does the detective stay sharp over time?")
        st.markdown(
            "Fraud patterns change. A detector that looks great on average can still fall apart "
            "at the exact moments that matter most -- this happened for real in this dataset "
            "around a known dark-market shutdown event. This chart shows performance "
            "**month by month** in the test period, not just one average number."
        )
        fig2 = go.Figure()
        for tag, df in per_ts.items():
            fig2.add_trace(go.Scatter(
                x=df["time_step"], y=df["illicit_f1"], mode="lines+markers",
                name=ARCH_INFO[tag]["name"], line=dict(color=ARCH_INFO[tag]["color"], width=3),
            ))
        fig2.update_layout(
            xaxis_title="Time step (test period)", yaxis_title="Detection quality (Illicit F1)", height=420,
            paper_bgcolor=CARD, plot_bgcolor=CARD, font=dict(family="Poppins, sans-serif", color=INK),
            legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig2, width='stretch')

    with st.expander("Technical details (for data scientists)"):
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
