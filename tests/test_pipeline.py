"""
Pytest suite for the graph fraud-detection pipeline. Runs entirely on a
synthetic fixture matching the exact Elliptic file schema (no header on
features.csv, "1"/"2"/"unknown" classes, txId1/txId2 edgelist) -- so CI can
run green even before the real (Kaggle-gated) data is downloaded, while
still catching real bugs in the graph-construction and splitting logic.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metrics import illicit_metrics, per_timestep_metrics, best_threshold_by_f1


@pytest.fixture()
def synthetic_raw_dir(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    n_nodes = 600
    n_feat = 165
    time_steps = rng.integers(1, 50, size=n_nodes)
    tx_ids = np.arange(1, n_nodes + 1)
    feats = rng.normal(size=(n_nodes, n_feat))

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    pd.DataFrame(np.column_stack([tx_ids, time_steps, feats])).to_csv(
        raw_dir / "elliptic_txs_features.csv", header=False, index=False
    )
    classes = rng.choice(["1", "2", "unknown"], size=n_nodes, p=[0.1, 0.6, 0.3])
    pd.DataFrame({"txId": tx_ids, "class": classes}).to_csv(raw_dir / "elliptic_txs_classes.csv", index=False)
    src = rng.choice(tx_ids, 1500)
    dst = rng.choice(tx_ids, 1500)
    pd.DataFrame({"txId1": src, "txId2": dst}).to_csv(raw_dir / "elliptic_txs_edgelist.csv", index=False)

    import data_prep
    monkeypatch.setattr(data_prep, "RAW_DIR", raw_dir)
    monkeypatch.setattr(data_prep, "PROC_DIR", tmp_path / "processed")
    return data_prep


def test_graph_has_no_self_loop_duplication_bug(synthetic_raw_dir):
    data = synthetic_raw_dir.build_graph()
    assert data.num_nodes == 600
    # undirected edge_index must be exactly 2x the directed edge count
    assert data.edge_index.shape[1] == 2 * data.directed_edge_index.shape[1]


def test_labels_are_mapped_correctly(synthetic_raw_dir):
    data = synthetic_raw_dir.build_graph()
    assert set(data.y.unique().tolist()) <= {-1, 0, 1}
    # "unknown" must map to -1 and be excluded from every mask
    unknown_mask = data.y == -1
    assert not (data.train_mask & unknown_mask).any()
    assert not (data.val_mask & unknown_mask).any()
    assert not (data.test_mask & unknown_mask).any()


def test_temporal_split_is_disjoint_and_ordered(synthetic_raw_dir):
    data = synthetic_raw_dir.build_graph()
    train_steps = data.time_step[data.train_mask]
    val_steps = data.time_step[data.val_mask]
    test_steps = data.time_step[data.test_mask]
    if len(train_steps) and len(val_steps):
        assert train_steps.max() <= val_steps.min() or train_steps.max() < 35
    assert set(train_steps.tolist()) & set(val_steps.tolist()) == set()
    assert set(val_steps.tolist()) & set(test_steps.tolist()) == set()


def test_illicit_metrics_perfect_prediction():
    y = np.array([0, 0, 1, 1, 1, 0])
    m = illicit_metrics(y, y, y.astype(float))
    assert m["illicit_precision"] == 1.0
    assert m["illicit_recall"] == 1.0
    assert m["illicit_f1"] == 1.0


def test_illicit_metrics_ignore_majority_class_by_design():
    # A model that predicts "licit" for everyone must score ZERO illicit-F1,
    # even though it would score >85% accuracy -- this is the whole point
    # of not reporting accuracy anywhere in this project.
    y_true = np.array([0] * 90 + [1] * 10)
    y_pred_all_licit = np.zeros(100)
    m = illicit_metrics(y_true, y_pred_all_licit, y_pred_all_licit)
    assert m["illicit_f1"] == 0.0


def test_per_timestep_metrics_skips_timesteps_with_no_illicit():
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 0])
    score = np.array([0.1, 0.8, 0.1, 0.9, 0.4])
    steps = np.array([1, 1, 2, 3, 3])
    df = per_timestep_metrics(y_true, y_pred, score, steps)
    assert 2 not in df["time_step"].values  # step 2 has zero illicit nodes
    assert set(df["time_step"].values) == {1, 3}


def test_best_threshold_by_f1_beats_naive_half():
    rng = np.random.default_rng(0)
    y = rng.choice([0, 1], size=1000, p=[0.9, 0.1])
    score = np.where(y == 1, rng.uniform(0.3, 0.7, size=1000), rng.uniform(0.0, 0.5, size=1000))
    thr = best_threshold_by_f1(y, score)
    from metrics import illicit_metrics
    f1_tuned = illicit_metrics(y, (score >= thr).astype(int), score)["illicit_f1"]
    f1_naive = illicit_metrics(y, (score >= 0.5).astype(int), score)["illicit_f1"]
    assert f1_tuned >= f1_naive
