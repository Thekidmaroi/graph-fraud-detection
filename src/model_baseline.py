"""
Tabular baseline: XGBoost on node features alone, ignoring the graph
entirely. This is the model every GNN in this repo has to beat -- if a GCN
can't outperform gradient boosting on the same features, the graph
structure isn't earning its keep, and that is a real, reportable possible
outcome (the original Elliptic paper found Random Forest was surprisingly
competitive with GCN on this exact dataset).
"""
import json
from pathlib import Path

import joblib
import numpy as np
import torch
import xgboost as xgb

from data_prep import build_graph, PROC_DIR, TEST_STEPS
from metrics import illicit_metrics, per_timestep_metrics, best_threshold_by_f1

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(exist_ok=True)


def main():
    data = build_graph()
    X = data.x.numpy()
    y = data.y.numpy()

    train_idx = data.train_mask.numpy()
    val_idx = data.val_mask.numpy()
    test_idx = data.test_mask.numpy()

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        scale_pos_weight=(y[train_idx] == 0).sum() / max((y[train_idx] == 1).sum(), 1),
        early_stopping_rounds=30,
        n_jobs=2,
    )
    model.fit(X[train_idx], y[train_idx], eval_set=[(X[val_idx], y[val_idx])], verbose=False)

    val_score = model.predict_proba(X[val_idx])[:, 1]
    thr = best_threshold_by_f1(y[val_idx], val_score)

    results = {}
    for name, idx in [("val", val_idx), ("test", test_idx)]:
        score = model.predict_proba(X[idx])[:, 1]
        pred = (score >= thr).astype(int)
        results[name] = illicit_metrics(y[idx], pred, score)
    print(json.dumps(results, indent=2))

    test_score = model.predict_proba(X[test_idx])[:, 1]
    test_pred = (test_score >= thr).astype(int)
    per_ts = per_timestep_metrics(y[test_idx], test_pred, test_score, data.time_step[test_idx].numpy())
    per_ts.to_csv(MODELS_DIR / "xgb_per_timestep.csv", index=False)

    with open(MODELS_DIR / "xgb_metrics.json", "w") as f:
        json.dump({"threshold": thr, **results}, f, indent=2)
    joblib.dump(model, MODELS_DIR / "xgb.joblib")
    print(f"Saved XGBoost baseline to {MODELS_DIR / 'xgb.joblib'} (decision threshold={thr:.3f})")


if __name__ == "__main__":
    main()
