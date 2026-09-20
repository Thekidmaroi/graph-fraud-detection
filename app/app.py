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

Visual design: adapted end-to-end from fraud.net (an enterprise fraud/risk
AI vendor) -- a dark navy canvas, a bright blue accent with a pink
secondary accent, Poppins typeface, a top nav bar, rounded dark cards with
a colored top accent and a hover-lift, a node-diagram "how it works"
section, animated stat counters, a closing gradient CTA banner with
floating shapes, and a multi-column footer -- and no emoji anywhere.
Motion is CSS-only (entrance fade-ins, hover transitions, a pulsing status
dot, animated counters) rather than fraud.net's scroll-linked JS, which
Streamlit's layout can't safely reproduce.

Two modules go beyond the original project, both requested to make the
demo a genuinely general-purpose tool rather than an Elliptic-only toy:

  - "Train on Your Data" (tab 3, src/custom_data.py): upload any
    organization's own node/edge tables and train the *same* GCN live,
    with the same metrics -- generalizing the pipeline beyond Elliptic.
  - The interpretation layer (src/explain.py), wired into every
    investigation view: an edge-ablation explanation (in the spirit of
    GNNExplainer) plus two plain-language narratives -- one for the
    investigator deciding what to do next, one for the account holder
    whose record was flagged.
"""
import hashlib
import io
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import torch

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
PROC_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

from custom_data import build_custom_graph, GraphBuildError, MAX_NODES, MAX_EDGES  # noqa: E402
from explain import neighbor_importance, two_audience_narrative  # noqa: E402

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
        .block-container {{ padding-top: 1.5rem; max-width: 1120px; }}

        @keyframes fadeInUp {{ from {{ opacity: 0; transform: translateY(14px); }} to {{ opacity: 1; transform: none; }} }}
        @keyframes pulseDot {{
            0% {{ box-shadow: 0 0 0 0 rgba(233,30,140,.55); }}
            70% {{ box-shadow: 0 0 0 8px rgba(233,30,140,0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(233,30,140,0); }}
        }}
        @keyframes floatShape {{
            0%, 100% {{ transform: translateY(0) rotate(0deg); }}
            50% {{ transform: translateY(-14px) rotate(12deg); }}
        }}

        /* ---- top nav ------------------------------------------------- */
        .topnav {{ display: flex; justify-content: space-between; align-items: center; padding: .25rem 0 1.4rem 0; animation: fadeInUp .5s ease both; }}
        .topnav .brand {{ font-weight: 800; font-size: 1.05rem; color: white; display: flex; align-items: center; gap: .55rem; }}
        .topnav .brand .mark {{ width: 11px; height: 11px; border-radius: 3px; background: linear-gradient(135deg, {PINK}, {BLUE}); display: inline-block; }}
        .topnav nav {{ display: flex; gap: 1.6rem; }}
        .topnav nav a {{ color: {MUTED}; font-size: .86rem; font-weight: 600; text-decoration: none !important; border-bottom: 1px solid transparent; padding-bottom: 2px; transition: color .2s ease, border-color .2s ease; }}
        .topnav nav a:hover {{ color: white !important; border-bottom-color: {BLUE}; }}

        /* ---- hero ---------------------------------------------------- */
        .hero {{
            background: radial-gradient(120% 160% at 100% 0%, #1B1666 0%, {BG} 60%), {BG};
            border: 1px solid {CARD_BORDER};
            border-radius: 20px;
            padding: 2.75rem 2.75rem 2.25rem 2.75rem;
            margin-bottom: 1.75rem;
            animation: fadeInUp .6s ease both;
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
            animation: pulseDot 2.2s infinite;
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

        /* ---- section headings ------------------------------------------ */
        h2, h3 {{ color: white !important; font-weight: 700; }}
        .section-note {{ color: {MUTED}; font-size: .95rem; }}
        .stMarkdown p, .stMarkdown li {{ color: {INK}; }}
        [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {{ color: {MUTED} !important; }}

        /* ---- capability cards (the 4 detection approaches) -------------- */
        .cap-card {{
            background: {CARD};
            border-top: 3px solid var(--accent, {BLUE});
            border-radius: 16px;
            padding: 1.25rem 1.3rem 1.1rem 1.3rem;
            height: 100%;
            animation: fadeInUp .6s ease both;
            transition: transform .2s ease, box-shadow .2s ease;
        }}
        .cap-card:hover {{ transform: translateY(-4px); box-shadow: 0 16px 32px rgba(0,0,0,.4), 0 0 0 1px var(--accent, {BLUE}) inset; }}
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

        /* ---- how-it-works flow diagram ------------------------------------ */
        .flow-row {{ display: flex; align-items: stretch; gap: .6rem; flex-wrap: wrap; margin-bottom: 1.6rem; }}
        .flow-step {{
            flex: 1 1 170px; background: {CARD}; border: 1px dashed {CARD_BORDER}; border-radius: 14px;
            padding: 1rem 1.1rem; transition: transform .2s ease, border-color .2s ease; animation: fadeInUp .6s ease both;
        }}
        .flow-step:hover {{ transform: translateY(-3px); border-color: {BLUE}; }}
        .flow-num {{
            display: inline-flex; align-items: center; justify-content: center; width: 22px; height: 22px;
            border-radius: 50%; background: {BLUE}; color: {BG}; font-size: .72rem; font-weight: 800; margin-bottom: .5rem;
        }}
        .flow-title {{ font-weight: 700; color: white; font-size: .92rem; margin-bottom: .3rem; }}
        .flow-desc {{ font-size: .78rem; color: {MUTED}; line-height: 1.45; }}
        .flow-arrow {{ display: flex; align-items: center; color: {MUTED}; font-size: 1.2rem; font-weight: 800; flex: 0 0 auto; }}

        /* ---- callout (demo-mode / status banner) ---------------------------------- */
        .callout {{
            background: rgba(0, 133, 255, 0.10);
            border: 1px solid rgba(0, 133, 255, 0.35);
            border-radius: 14px;
            padding: 1rem 1.2rem;
            color: {INK};
            font-size: .92rem;
            animation: fadeInUp .5s ease both;
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

        /* ---- comparison cards (tab 2 / tab 3) ------------------------------------------- */
        .metric-card {{
            background: {CARD};
            border: 1px solid {CARD_BORDER};
            border-radius: 16px;
            padding: 1.2rem 1.3rem;
            height: 100%;
            transition: transform .2s ease, box-shadow .2s ease;
            animation: fadeInUp .6s ease both;
        }}
        .metric-card:hover {{ transform: translateY(-4px); box-shadow: 0 16px 32px rgba(0,0,0,.4); }}
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

        /* ---- interpretation narrative cards ------------------------------------- */
        .narrative-card {{
            background: {CARD}; border-top: 3px solid var(--accent, {BLUE}); border-radius: 16px;
            padding: 1.1rem 1.25rem; height: 100%; animation: fadeInUp .5s ease both;
        }}
        .narrative-card .kicker {{ text-transform: uppercase; letter-spacing: .06em; font-size: .7rem; font-weight: 800; margin-bottom: .5rem; }}
        .narrative-card p {{ font-size: .88rem; color: {INK}; line-height: 1.55; margin: 0; }}

        /* ---- closing CTA banner ------------------------------------------------- */
        .cta-banner {{
            position: relative; overflow: hidden; border-radius: 20px; padding: 3rem 2rem; text-align: center;
            margin: 1.5rem 0; background: linear-gradient(135deg, {PINK} 0%, {BLUE_SOFT} 55%, {BLUE} 100%);
        }}
        .cta-inner {{ position: relative; z-index: 2; }}
        .cta-banner h2 {{ color: white !important; font-size: 1.7rem; margin-bottom: .6rem; }}
        .cta-banner p {{ color: rgba(255,255,255,0.92); font-size: 1rem; margin-bottom: 1.3rem; }}
        .cta-btn-row {{ display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap; }}
        .cta-btn, .cta-btn:visited {{
            background: white; color: {BG} !important; font-weight: 700; padding: .8rem 1.4rem; border-radius: 10px;
            text-decoration: none !important; display: inline-block; transition: transform .2s ease, box-shadow .2s ease;
        }}
        .cta-btn:hover {{ transform: translateY(-2px); box-shadow: 0 10px 24px rgba(0,0,0,.25); }}
        .cta-btn-outline, .cta-btn-outline:visited {{
            background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.6); color: white !important;
            font-weight: 700; padding: .8rem 1.4rem; border-radius: 10px; text-decoration: none !important;
            display: inline-block; transition: background .2s ease, transform .2s ease;
        }}
        .cta-btn-outline:hover {{ background: rgba(255,255,255,0.22); transform: translateY(-2px); }}
        .cta-shape {{ position: absolute; background: rgba(255,255,255,0.14); border-radius: 8px; animation: floatShape 6s ease-in-out infinite; }}
        .cta-shape.s1 {{ width: 70px; height: 70px; top: -20px; left: -20px; }}
        .cta-shape.s2 {{ width: 50px; height: 50px; bottom: 10px; right: 60px; animation-delay: 1.2s; }}
        .cta-shape.s3 {{ width: 34px; height: 34px; bottom: 40px; right: 150px; animation-delay: 2.4s; }}

        /* ---- footer ------------------------------------------------------------- */
        .site-footer {{ display: flex; gap: 2.5rem; flex-wrap: wrap; background: #0B0826; border: 1px solid {CARD_BORDER}; border-radius: 16px; padding: 1.8rem 2rem; margin-top: .5rem; }}
        .footer-col {{ display: flex; flex-direction: column; gap: .45rem; min-width: 180px; }}
        .footer-col h4 {{ color: white !important; font-size: .85rem; margin: 0 0 .3rem 0; }}
        .footer-col a, .footer-col span {{ color: {MUTED}; font-size: .82rem; text-decoration: none !important; transition: color .2s ease; }}
        .footer-col a:hover {{ color: {BLUE_SOFT} !important; }}

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
        button[kind="primary"] {{ background-color: {BLUE} !important; border-color: {BLUE} !important; transition: transform .15s ease, box-shadow .15s ease; }}
        button[kind="primary"]:hover {{ transform: translateY(-1px); box-shadow: 0 8px 18px rgba(0,133,255,.35); }}
        [data-testid="stFileUploaderDropzone"] {{ background: {CARD}; border: 1px dashed {CARD_BORDER}; border-radius: 12px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def nav_bar():
    st.markdown(
        f"""
        <div class="topnav" id="top">
            <div class="brand"><span class="mark"></span>Fraud Detective</div>
            <nav>
                <a href="#how-it-works">How it works</a>
                <a href="#tabs-section">Explore &amp; compare</a>
                <a href="#tabs-section">Train your data</a>
            </nav>
        </div>
        """,
        unsafe_allow_html=True,
    )


def animated_counters(stats: list[tuple[int, str]]):
    """A self-contained HTML/JS stat strip that counts each number up from
    zero once, on load. The only real JavaScript in this app: a one-shot
    count-up doesn't depend on the outer page's scroll position, unlike
    fraud.net's scroll-linked reveals, which Streamlit's layout (no direct
    access to the parent page's scroll from a component iframe) can't
    safely reproduce -- see the module docstring."""
    cards = "".join(
        f'<div class="c"><div class="n" data-target="{t}">0</div><div class="l">{l}</div></div>'
        for t, l in stats
    )
    html = f"""
    <div style="display:flex;gap:16px;flex-wrap:wrap;font-family:'Poppins',sans-serif;">
      <style>
        body {{ margin:0; background:{BG}; }}
        .c {{ flex:1 1 160px; background:{CARD}; border:1px solid {CARD_BORDER}; border-radius:14px; padding:16px 20px; box-sizing:border-box; }}
        .n {{ font-size:1.55rem; font-weight:800; color:#fff; }}
        .l {{ font-size:.8rem; color:{MUTED}; margin-top:4px; }}
      </style>
      {cards}
    </div>
    <script>
      document.querySelectorAll('.n').forEach(function(el) {{
        var target = parseInt(el.getAttribute('data-target'), 10);
        var dur = 1100;
        var start = null;
        function step(ts) {{
          if (!start) start = ts;
          var p = Math.min(1, (ts - start) / dur);
          var eased = 1 - Math.pow(1 - p, 3);
          el.textContent = Math.floor(eased * target).toLocaleString();
          if (p < 1) requestAnimationFrame(step); else el.textContent = target.toLocaleString();
        }}
        requestAnimationFrame(step);
      }});
    </script>
    """
    components.html(html, height=104)


def how_it_works():
    st.markdown('<div id="how-it-works"></div>', unsafe_allow_html=True)
    st.subheader("How this works, end to end")
    st.markdown(
        '<p class="section-note">The same five-step pipeline runs whether the data is the '
        'Elliptic benchmark or your own upload in the "Train on Your Data" tab.</p>',
        unsafe_allow_html=True,
    )
    steps = [
        ("Raw records", "Transactions and who-paid-whom, in whatever shape your systems already log them."),
        ("Graph construction", "Every record becomes a node, every connection an edge -- automatically (src/custom_data.py for a new upload)."),
        ("Graph AI (GCN)", "A neural network reads each node's own numbers AND its neighbors', trained live, in the browser."),
        ("Explanation layer", "Edge-ablation attribution finds which neighbors or features actually moved the score (src/explain.py)."),
        ("Two-sided verdict", "One plain-language readout for the investigator, one for the account holder."),
    ]
    parts = []
    for i, (title, desc) in enumerate(steps):
        parts.append(
            f'<div class="flow-step" style="animation-delay:{i * 0.08:.2f}s">'
            f'<div class="flow-num">{i + 1}</div>'
            f'<div class="flow-title">{title}</div>'
            f'<div class="flow-desc">{desc}</div></div>'
        )
        if i < len(steps) - 1:
            parts.append('<div class="flow-arrow">&rarr;</div>')
    st.markdown(f'<div class="flow-row">{"".join(parts)}</div>', unsafe_allow_html=True)


def closing_cta():
    st.markdown(
        """
        <div class="cta-banner">
            <div class="cta-shape s1"></div>
            <div class="cta-shape s2"></div>
            <div class="cta-shape s3"></div>
            <div class="cta-inner">
                <h2>Explore this project further</h2>
                <p>Full code, tests and methodology -- or train the same graph AI on your own data, above.</p>
                <div class="cta-btn-row">
                    <a class="cta-btn" href="https://github.com/Thekidmaroi/graph-fraud-detection" target="_blank" rel="noreferrer">View the code on GitHub</a>
                    <a class="cta-btn-outline" href="#tabs-section">Train it on your own data</a>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def site_footer():
    st.markdown(
        """
        <div class="site-footer">
            <div class="footer-col">
                <h4>Project</h4>
                <a href="https://github.com/Thekidmaroi/graph-fraud-detection" target="_blank" rel="noreferrer">GitHub repository</a>
                <a href="https://github.com/Thekidmaroi/graph-fraud-detection#readme" target="_blank" rel="noreferrer">Full README &amp; methodology</a>
            </div>
            <div class="footer-col">
                <h4>Benchmarked against</h4>
                <span>Weber et al. 2019, "Anti-Money Laundering in Bitcoin"</span>
                <span>Elmougy &amp; Liu, KDD 2023</span>
                <span>PyTorch Geometric &middot; scikit-learn</span>
            </div>
            <div class="footer-col">
                <h4>Built by</h4>
                <span>Marwane Houngnon</span>
                <a href="https://github.com/Thekidmaroi" target="_blank" rel="noreferrer">github.com/Thekidmaroi</a>
            </div>
        </div>
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
    from metrics import illicit_metrics, best_threshold_by_f1

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

    return data, model, x_std, {"gcn": gnn_results, "xgb": xgb_results}


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


def sample_company_dataset(seed: int = 3):
    """The same illustrative-network recipe as src/demo_data.py (a few
    tight fraud rings plus a larger loosely-connected population), reshaped
    into two plain CSV-style tables -- so a visitor with no data of their
    own can still click through the whole "Train on Your Data" flow and
    see it genuinely work, including a categorical column, to prove the
    one-hot-encoding path is real and not numeric-only."""
    rng = np.random.default_rng(seed)
    ring_sizes = rng.integers(6, 11, size=4)
    n_illicit = int(ring_sizes.sum())
    n_legit, n_unknown = 140, 40
    n = n_illicit + n_legit + n_unknown
    n_feat = 12

    x = np.zeros((n, n_feat))
    label = np.array([""] * n, dtype=object)
    time_step = rng.integers(1, 40, size=n)
    edges = []
    cursor = 0
    ring_members = []
    for size in ring_sizes:
        members = list(range(cursor, cursor + size))
        ring_members.append(members)
        x[members] = rng.normal(2.2, 0.5, size=(size, n_feat))
        label[members] = "yes"
        for i in members:
            for j in members:
                if i != j and rng.random() < 0.5:
                    edges.append((i, j))
        cursor += size

    legit_start = cursor
    legit_members = list(range(legit_start, legit_start + n_legit))
    x[legit_members] = rng.normal(0.0, 1.0, size=(n_legit, n_feat))
    label[legit_members] = "no"
    cursor += n_legit

    unk_members = list(range(cursor, cursor + n_unknown))
    x[unk_members] = rng.normal(0.2, 1.1, size=(n_unknown, n_feat))
    label[unk_members] = ""

    all_normal = legit_members + unk_members
    n_rand = int(len(all_normal) * 1.6)
    src = rng.choice(all_normal, n_rand)
    dst = rng.choice(all_normal, n_rand)
    edges += list(zip(src.tolist(), dst.tolist()))
    for members in ring_members:
        bridges = rng.choice(all_normal, size=3, replace=False)
        for b in bridges:
            edges.append((int(rng.choice(members)), int(b)))
    edges = [(i, j) for i, j in edges if i != j]

    ids = [f"ACCT-{1000 + i}" for i in range(n)]
    nodes_df = pd.DataFrame({
        "record_id": ids,
        **{f"feat_{i}": x[:, i] for i in range(n_feat)},
        "channel": rng.choice(["web", "mobile", "branch"], size=n),
        "is_fraud": label,
        "time_step": time_step,
    })
    edges_df = pd.DataFrame({
        "src_id": [ids[i] for i, j in edges],
        "dst_id": [ids[j] for i, j in edges],
    })
    return nodes_df, edges_df


def render_investigation(data, model, x_std, entity_noun: str, display_id, key_prefix: str, source_note: str):
    """The shared 'pick a record, see its network, get a verdict and a
    full two-audience explanation' view -- used by both the Elliptic/demo
    tab and the Train-on-Your-Data tab, so a visitor's own uploaded data
    gets exactly the same investigation and interpretation experience as
    the reference dataset."""
    test_idx = torch.where(data.test_mask)[0]
    illicit_idx = test_idx[data.y[test_idx] == 1]

    st.write(
        f"There are **{len(test_idx)}** {entity_noun}s in the held-out test set ({source_note}), "
        f"of which **{len(illicit_idx)}** are confirmed fraudulent."
    )
    show_illicit_only = st.checkbox(f"Only show confirmed-fraud {entity_noun}s", value=True, key=f"{key_prefix}_only")
    pool = illicit_idx if show_illicit_only else test_idx

    if len(pool) == 0:
        st.info(f"No {entity_noun}s match this filter.")
        return

    choice = st.selectbox(
        f"Pick a {entity_noun} to investigate",
        options=pool.tolist()[:150],
        format_func=lambda i: f"{entity_noun.capitalize()} {display_id(i)} "
                               f"({'known fraud' if data.y[i] == 1 else 'known legitimate'})",
        key=f"{key_prefix}_pick",
    )

    edge_index = data.directed_edge_index.numpy()
    G = nx.DiGraph()
    G.add_edges_from(edge_index.T.tolist())

    k = st.slider(
        "How many steps out in the network to look?", 1, 3, 2, key=f"{key_prefix}_k",
        help="1 = direct connections only. 2 = connections of connections. Investigators usually look 2-3 steps out.",
    )

    col_graph, col_verdict = st.columns([2, 1])

    if choice not in G:
        st.info(f"This {entity_noun} has no recorded connections in the network.")
        return

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
        st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_network_chart")
        st.markdown(
            f"""
            <div class="legend-row">
                <div class="legend-item"><span class="legend-chip" style="background:{PINK}"></span>Selected {entity_noun}</div>
                <div class="legend-item"><span class="legend-chip" style="background:{RED}"></span>Known fraud</div>
                <div class="legend-item"><span class="legend-chip" style="background:{BLUE}"></span>Known legitimate</div>
                <div class="legend-item"><span class="legend-chip" style="background:{GRAY}"></span>Unknown / unconfirmed</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    scores_all = None
    if model is not None:
        with torch.no_grad():
            scores_all = torch.sigmoid(model(x_std, data.edge_index)).numpy()

    with col_verdict:
        score = float(scores_all[choice]) if scores_all is not None else None

        is_fraud = data.y[choice].item() == 1
        badge_label = "Fraud" if is_fraud else "Legitimate"
        badge_class = "red" if is_fraud else "green"
        st.markdown(f'<p class="section-note">Known status</p><span class="badge badge-{badge_class}">{badge_label}</span>', unsafe_allow_html=True)
        st.write("")
        if score is not None:
            st.metric("AI fraud probability", f"{score:.0%}")
            st.progress(min(max(score, 0.0), 1.0))

        st.markdown('<p style="font-weight:700; color:white; margin-top:1rem;">Quick read</p>', unsafe_allow_html=True)
        st.markdown(f'<div class="factor-row"><span class="factor-tag factor-clear">CONTEXT</span>It has <strong>{len(direct_neighbors)}</strong> direct {entity_noun} partners.</div>', unsafe_allow_html=True)
        if n_direct_illicit > 0:
            st.markdown(f'<div class="factor-row"><span class="factor-tag factor-warn">WARNING SIGN</span><strong>{n_direct_illicit}</strong> of them are already known fraud.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="factor-row"><span class="factor-tag factor-clear">CLEAR SIGNAL</span>None of its direct partners are known fraud.</div>', unsafe_allow_html=True)

    if model is None or score is None:
        st.caption(
            "A full production system would also show a formal feature-attribution explanation "
            "(e.g. GNNExplainer); this view shows the intuitive network signal."
        )
        return

    st.write("")
    st.markdown('<p style="font-weight:700;color:white;font-size:1.05rem;">How the AI explains this decision</p>', unsafe_allow_html=True)
    factors = neighbor_importance(model, x_std, data.edge_index, choice, neighborhood, direct_neighbors)
    for f in factors:
        if f.ref_node is not None:
            f.label = f"its connection to {entity_noun} {display_id(f.ref_node)}"

    if not factors:
        st.caption(f"This {entity_noun} has no direct neighbors in the sampled network, so no neighbor-based explanation is available.")
        return

    ordered = list(reversed(factors))
    fig3 = go.Figure(go.Bar(
        x=[f.delta * 100 for f in ordered],
        y=[f.label for f in ordered],
        orientation="h",
        marker_color=[RED if f.direction == "toward_fraud" else GREEN for f in ordered],
    ))
    fig3.update_layout(
        height=140 + 34 * len(ordered), margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Change in fraud probability if this connection were removed (points)",
        paper_bgcolor=CARD, plot_bgcolor=CARD, font=dict(family="Poppins, sans-serif", color=INK),
    )
    st.plotly_chart(fig3, width="stretch", key=f"{key_prefix}_factors_chart")

    narrative = two_audience_narrative(score, factors, entity_label=f"this {entity_noun}")
    nc1, nc2 = st.columns(2)
    with nc1:
        st.markdown(
            f'<div class="narrative-card" style="--accent:{BLUE}">'
            f'<div class="kicker" style="color:{BLUE_SOFT}">For the investigator / risk team</div>'
            f'<p>{narrative["compliance_text"]}</p></div>',
            unsafe_allow_html=True,
        )
    with nc2:
        st.markdown(
            f'<div class="narrative-card" style="--accent:{PINK}">'
            f'<div class="kicker" style="color:{PINK}">For the account holder</div>'
            f'<p>{narrative["counterparty_text"]}</p></div>',
            unsafe_allow_html=True,
        )


# ------------------------------------------------------------------ UI ----
inject_css()
nav_bar()

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
            <li>Bring your own data and train the same model -- see the third tab below</li>
            <li>Every verdict explained in plain English, for both sides of the decision</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

animated_counters([
    (203769, "Real Bitcoin transactions in the reference dataset"),
    (4, "Detection approaches compared head-to-head"),
    (3, "Interactive modules: investigate, compare, train"),
])
st.write("")

# ---- capability cards: the four detection approaches ---------------------
st.subheader("Four ways to catch a fraudulent transaction")
st.markdown('<p class="section-note">From "look at the transaction alone" to "read its whole neighborhood".</p>', unsafe_allow_html=True)
cap_cols = st.columns(4)
for i, (col, tag) in enumerate(zip(cap_cols, ARCH_ORDER)):
    info = ARCH_INFO[tag]
    with col:
        st.markdown(
            f"""
            <div class="cap-card" style="--accent:{info['color']}; animation-delay:{i * 0.08:.2f}s">
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
    data, gcn_model, x_std, demo_results = load_demo_graph_and_models()
else:
    data = load_real_graph()
    gcn_model, x_std = None, None

st.write("")
how_it_works()

st.markdown('<div id="tabs-section"></div>', unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["Investigate a Transaction", "How Good Is the Detective?", "Train on Your Data"])

# =========================================================== TAB 1 =======
with tab1:
    render_investigation(
        data, gcn_model if DEMO_MODE else None, x_std,
        entity_noun="transaction",
        display_id=lambda i: str(data.tx_id[i].item()),
        key_prefix="t1",
        source_note="demo network" if DEMO_MODE else "real Elliptic data, time steps 40-49",
    )

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
        for i, (col, tag) in enumerate(zip(cols, active_tags)):
            m = demo_results[tag]["test"]
            info = ARCH_INFO[tag]
            with col:
                st.markdown(
                    f"""
                    <div class="metric-card" style="animation-delay:{i * 0.08:.2f}s">
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
        for i, (col, tag) in enumerate(zip(cols, active_tags)):
            m = metrics[tag]["test"]
            info = ARCH_INFO[tag]
            with col:
                st.markdown(
                    f"""
                    <div class="metric-card" style="animation-delay:{i * 0.08:.2f}s">
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
        st.plotly_chart(fig2, width="stretch", key="t2_timeseries_chart")

    with st.expander("Technical details (for data scientists)"):
        st.markdown(
            "- **xgb** = tabular baseline on node features alone (no graph) -- logistic regression in "
            "demo mode / Train-on-Your-Data, XGBoost in real mode (see src/model_baseline.py)\n"
            "- **gcn** / **sage** / **gat** = Graph Convolutional Network / GraphSAGE / Graph "
            "Attention Network (PyTorch Geometric), full-batch transductive training\n"
            "- Evaluated with **Illicit-F1 / PR-AUC**, never accuracy (illicit transactions are "
            "~10% of labeled nodes, 77% of all nodes are unlabeled)\n"
            "- **Temporal train/val/test split** (steps 1-34 / 35-39 / 40-49) -- a random split "
            "would leak future graph structure into training\n\n"
            "See README.md for the full methodology and academic references (Weber et al. 2019; "
            "Elmougy & Liu, KDD 2023)."
        )

# =========================================================== TAB 3 =======
with tab3:
    st.subheader("Bring your own data")
    st.markdown(
        '<p class="section-note">Any organization can drop in its own transaction or entity network '
        "and train the same graph AI live, right here -- no code, no GPU, and nothing leaves this "
        "browser session.</p>",
        unsafe_allow_html=True,
    )

    with st.expander("What format does my data need to be in?"):
        st.markdown(
            "- **A node table** (CSV): one row per transaction, account, claim or any other entity, "
            "an ID column, any number of feature columns (numeric or categorical -- categorical "
            "columns are one-hot encoded automatically), and optionally a label column marking known "
            "fraud cases. Most rows can be blank/unknown, exactly like the real Elliptic data.\n"
            "- **An edge table** (CSV): two columns giving pairs of IDs that transacted with, or are "
            "otherwise connected to, each other.\n"
            f"- Up to **{MAX_NODES:,} rows** and **{MAX_EDGES:,} edges** are used live in the browser; "
            "larger uploads are randomly subsampled, never rejected outright.\n"
            "- Training happens inside this Streamlit session only -- nothing is written to a "
            "database or sent anywhere else."
        )

    col_up1, col_up2 = st.columns(2)
    with col_up1:
        nodes_file = st.file_uploader("Node / entity table (CSV)", type="csv", key="byod_nodes_file")
    with col_up2:
        edges_file = st.file_uploader("Edge / connections table (CSV)", type="csv", key="byod_edges_file")

    use_sample = st.checkbox(
        "...or just try it with a sample illustrative dataset",
        value=(nodes_file is None and edges_file is None),
        key="byod_use_sample",
    )

    nodes_df = edges_df = None
    source_hash = None
    if nodes_file is not None and edges_file is not None:
        nodes_bytes, edges_bytes = nodes_file.getvalue(), edges_file.getvalue()
        nodes_df = pd.read_csv(io.BytesIO(nodes_bytes))
        edges_df = pd.read_csv(io.BytesIO(edges_bytes))
        source_hash = hashlib.md5(nodes_bytes + edges_bytes).hexdigest()
    elif use_sample:
        nodes_df, edges_df = sample_company_dataset()
        source_hash = "sample-v1"

    if nodes_df is None:
        st.info("Upload both files above, or check the sample-dataset box, to get started.")
    else:
        st.markdown("**Preview -- node table**")
        st.dataframe(nodes_df.head(6), width="stretch")

        entity_noun = st.text_input(
            "What do you call one row? (used only to phrase the results in plain English)",
            value="record", key="byod_noun",
        ).strip() or "record"

        # Every widget below is keyed off `source_hash` (sample dataset vs. this exact
        # upload) rather than a fixed string. Streamlit only re-applies a widget's
        # `default`/`index` the first time that key is created -- on later reruns it
        # keeps whatever the widget already holds in session_state. Without this, a
        # visitor who first loads the tab (which auto-selects the sample dataset) and
        # then uploads their own files would keep the sample dataset's column
        # selections, which don't exist in their file's column list -- Streamlit then
        # can't display those stale values, so pickers silently render as if nothing
        # were selected (observed as "Feature columns to train on" showing empty and
        # "Select at least one feature column" on Train). Scoping the key to the data
        # source forces a brand-new widget -- with a correct fresh default -- every
        # time the underlying columns actually change.
        src_tag = source_hash or "none"
        cols = list(nodes_df.columns)
        c1, c2, c3 = st.columns(3)
        with c1:
            id_col = st.selectbox("ID column", cols, index=cols.index("record_id") if "record_id" in cols else 0, key=f"byod_id_col_{src_tag}")
        with c2:
            label_options = ["(none -- unsupervised)"] + cols
            default_label_idx = (cols.index("is_fraud") + 1) if "is_fraud" in cols else 0
            label_choice = st.selectbox("Label column (marks known fraud)", label_options, index=default_label_idx, key=f"byod_label_col_{src_tag}")
            label_col = None if label_choice.startswith("(none") else label_choice
        with c3:
            time_options = ["(none -- random split)"] + cols
            # Default to the random, class-stratified split even when a time column exists:
            # it guarantees both classes appear in the held-out set, which a temporal split
            # cannot promise on a small upload (a short, unlucky window can hold zero fraud
            # cases). The time-based split -- the methodologically stronger choice on a large
            # dataset, see README.md -- stays one click away for anyone who wants it.
            time_choice = st.selectbox("Time column (optional; enables a fair time-based split)", time_options, index=0, key=f"byod_time_col_{src_tag}")
            time_col = None if time_choice.startswith("(none") else time_choice

        positive_values = []
        if label_col:
            uniques = [v for v in nodes_df[label_col].dropna().unique().tolist() if str(v).strip() != ""]
            guessed = [v for v in uniques if str(v).strip().lower() in ("1", "yes", "true", "fraud", "illicit")]
            positive_values = st.multiselect(
                f"Which value(s) of `{label_col}` mean fraud?", uniques,
                default=guessed or uniques[:1], key=f"byod_pos_vals_{src_tag}_{label_col}",
            )

        default_features = [c for c in cols if c not in {id_col, label_col, time_col}]
        feature_cols = st.multiselect(
            "Feature columns to train on", cols, default=default_features,
            key=f"byod_feat_cols_{src_tag}_{id_col}_{label_col}_{time_col}",
        )

        st.markdown("**Preview -- edge table**")
        st.dataframe(edges_df.head(6), width="stretch")
        ecols = list(edges_df.columns)
        ec1, ec2 = st.columns(2)
        with ec1:
            src_col = st.selectbox("Source ID column", ecols, index=ecols.index("src_id") if "src_id" in ecols else 0, key=f"byod_src_col_{src_tag}")
        with ec2:
            dst_default = ecols.index("dst_id") if "dst_id" in ecols else min(1, len(ecols) - 1)
            dst_col = st.selectbox("Destination ID column", ecols, index=dst_default, key=f"byod_dst_col_{src_tag}")

        train_clicked = st.button("Train the graph AI on this data", type="primary", key="byod_train_btn")

        settings_key = (source_hash, id_col, label_col, tuple(sorted(map(str, positive_values))),
                         tuple(feature_cols), time_col, src_col, dst_col)
        cache_key = "byod_result_" + hashlib.md5(str(settings_key).encode()).hexdigest()

        if train_clicked:
            try:
                with st.spinner("Building the graph and training a GCN live -- usually a few seconds..."):
                    from model_gnn import train_one
                    from metrics import illicit_metrics, best_threshold_by_f1
                    from sklearn.linear_model import LogisticRegression

                    g_data, report = build_custom_graph(
                        nodes_df, edges_df, id_col=id_col, label_col=label_col,
                        positive_values=positive_values, feature_cols=feature_cols,
                        src_col=src_col, dst_col=dst_col, time_col=time_col,
                    )
                    model, x_std_c = train_one("gcn", g_data, epochs=60)
                    with torch.no_grad():
                        scores_full = torch.sigmoid(model(x_std_c, g_data.edge_index)).numpy()
                    val_true = g_data.y[g_data.val_mask].numpy()
                    val_score = scores_full[g_data.val_mask.numpy()]
                    thr = best_threshold_by_f1(val_true, val_score)
                    test_true = g_data.y[g_data.test_mask].numpy()
                    test_score = scores_full[g_data.test_mask.numpy()]
                    test_pred = (test_score >= thr).astype(int)
                    gnn_m = illicit_metrics(test_true, test_pred, test_score)

                    X, y_np = g_data.x.numpy(), g_data.y.numpy()
                    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(
                        X[g_data.train_mask.numpy()], y_np[g_data.train_mask.numpy()]
                    )
                    val_score_b = clf.predict_proba(X[g_data.val_mask.numpy()])[:, 1]
                    thr_b = best_threshold_by_f1(val_true, val_score_b)
                    test_score_b = clf.predict_proba(X[g_data.test_mask.numpy()])[:, 1]
                    test_pred_b = (test_score_b >= thr_b).astype(int)
                    base_m = illicit_metrics(test_true, test_pred_b, test_score_b)

                st.session_state[cache_key] = {
                    "data": g_data, "model": model, "x_std": x_std_c, "report": report,
                    "gnn_metrics": gnn_m, "baseline_metrics": base_m, "entity_noun": entity_noun,
                }
                st.session_state["byod_last_key"] = cache_key
            except GraphBuildError as e:
                st.error(str(e))

        active_key = cache_key if cache_key in st.session_state else st.session_state.get("byod_last_key")
        result = st.session_state.get(active_key) if active_key else None

        if result:
            r = result["report"]
            noun = result["entity_noun"]
            st.write("")
            st.markdown(
                f'<div class="callout"><span class="pill" style="background:{GREEN}">TRAINED</span>'
                f'{r.n_nodes:,} {noun}s &middot; {r.n_labeled:,} labeled ({r.n_positive} positive / '
                f'{r.n_negative} negative) &middot; {r.n_edges_kept:,} edges &middot; {r.n_features} '
                f'features &middot; {r.split_kind} split</div>',
                unsafe_allow_html=True,
            )
            for w in r.warnings:
                st.warning(w)
            st.write("")

            mcols = st.columns(2)
            with mcols[0]:
                st.markdown(
                    f'<div class="metric-card"><span class="tag" style="background:{GRAY}">NO GRAPH</span>'
                    f'<div class="name">Tabular Baseline (Logistic Regression)</div>'
                    f'<div class="metric-row"><span class="lbl">Caught (Recall)</span><span class="val">{result["baseline_metrics"]["illicit_recall"]:.0%}</span></div>'
                    f'<div class="metric-row"><span class="lbl">Accuracy of alarms (Precision)</span><span class="val">{result["baseline_metrics"]["illicit_precision"]:.0%}</span></div></div>',
                    unsafe_allow_html=True,
                )
            with mcols[1]:
                st.markdown(
                    f'<div class="metric-card"><span class="tag" style="background:{BLUE}">GRAPH NEURAL NET</span>'
                    f'<div class="name">Graph AI -- GCN (your data)</div>'
                    f'<div class="metric-row"><span class="lbl">Caught (Recall)</span><span class="val">{result["gnn_metrics"]["illicit_recall"]:.0%}</span></div>'
                    f'<div class="metric-row"><span class="lbl">Accuracy of alarms (Precision)</span><span class="val">{result["gnn_metrics"]["illicit_precision"]:.0%}</span></div></div>',
                    unsafe_allow_html=True,
                )
            st.caption("Held-out test split -- small uploads mean these numbers can be noisy; treat them as directional, not a certified benchmark.")

            buf = io.BytesIO()
            torch.save(result["model"].state_dict(), buf)
            st.download_button(
                "Download the trained GCN weights (.pt)", data=buf.getvalue(),
                file_name="trained_gcn.pt", key="byod_download",
            )

            st.divider()
            st.markdown(f"**Investigate a {result['entity_noun']} from your data**")
            g_data = result["data"]
            lookup = getattr(g_data, "node_label_lookup", None)
            display_id = (lambda i: lookup[i]) if lookup is not None else (lambda i: str(i))
            render_investigation(
                g_data, result["model"], result["x_std"],
                entity_noun=result["entity_noun"], display_id=display_id,
                key_prefix="byod", source_note="your uploaded data, held-out split",
            )

st.write("")
closing_cta()
site_footer()
