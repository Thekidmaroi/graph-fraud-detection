"""
Evaluation for the Elliptic illicit-transaction task.

Accuracy is not reported anywhere in this project on purpose: with illicit
transactions at ~10% of labeled nodes (and 77% of all nodes unlabeled),
a model that predicts "licit" for everything scores >85% accuracy while
being useless. The original Elliptic paper (Weber et al. 2019) and every
serious follow-up report precision/recall/F1 on the ILLICIT class
specifically, plus PR-AUC -- that is what we do here, plus a per-time-step
breakdown, because the original paper's headline finding was that GNN
performance is NOT stable over time (it degrades sharply around the dark-
market shutdown at step ~43), and any benchmark that hides that behind a
single aggregate number is hiding the most operationally important result.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, roc_auc_score


def illicit_metrics(y_true, y_pred_label, y_pred_score=None) -> dict:
    y_true = np.asarray(y_true)
    y_pred_label = np.asarray(y_pred_label)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred_label, labels=[1], average=None, zero_division=0
    )
    out = {"illicit_precision": float(p[0]), "illicit_recall": float(r[0]), "illicit_f1": float(f1[0])}
    if y_pred_score is not None:
        out["pr_auc"] = float(average_precision_score(y_true, y_pred_score))
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_pred_score))
        except ValueError:
            out["roc_auc"] = float("nan")
    return out


def per_timestep_metrics(y_true, y_pred_label, y_pred_score, time_steps) -> pd.DataFrame:
    df = pd.DataFrame({
        "time_step": np.asarray(time_steps),
        "y_true": np.asarray(y_true),
        "y_pred": np.asarray(y_pred_label),
        "score": np.asarray(y_pred_score),
    })
    rows = []
    for t, g in df.groupby("time_step"):
        if g["y_true"].sum() == 0:
            continue  # no illicit nodes at this time step -> F1 undefined
        m = illicit_metrics(g["y_true"], g["y_pred"], g["score"])
        m["time_step"] = t
        m["n_illicit"] = int(g["y_true"].sum())
        m["n_labeled"] = len(g)
        rows.append(m)
    return pd.DataFrame(rows).sort_values("time_step")


def best_threshold_by_f1(y_true, y_score) -> float:
    """Pick the decision threshold that maximizes illicit-F1 on a validation
    set -- the standard way to turn a graph model's probability output into
    a hard flag/no-flag decision for an investigator queue, rather than
    defaulting to an arbitrary 0.5."""
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-9, None)
    f1 = f1[:-1]  # thresholds has len(precision)-1 entries
    if len(thresholds) == 0:
        return 0.5
    return float(thresholds[np.argmax(f1)])
