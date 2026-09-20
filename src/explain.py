"""
End-to-end interpretation layer: turns a raw model score into (1) a ranked
list of *why* -- which neighbors or features actually moved the score, and
(2) two plain-language narratives written for the two different audiences
who read a fraud alert in production: the investigator deciding what to do
next, and the account holder whose transaction got flagged.

Two attribution methods, both intentionally simple and auditable rather
than approximate/black-box, matching this project's overall stance that a
production fraud model must be explainable, not just accurate:

  - `neighbor_importance`: an edge-ablation explanation for the graph model
    -- remove one neighbor's connection at a time, on the same k-hop
    subgraph already built for the network visualization, and measure how
    much the predicted probability moves. This is a lightweight instance of
    the same idea as GNNExplainer (Ying et al., NeurIPS 2019): explain a
    GNN's prediction by which edges matter to it, found by perturbation
    rather than a separate trained explainer network.
  - `linear_feature_importance`: for the tabular (no-graph) baseline, an
    exact first-order decomposition coefficient x (value - training mean)
    -- the same idea SHAP's LinearExplainer uses for a linear model, without
    the extra dependency.
"""
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class Factor:
    label: str          # human-readable description, e.g. "its connection to transaction #4821"
    delta: float         # signed change in predicted probability if this factor were absent
    direction: str        # "toward_fraud" or "toward_legit"
    kind: str             # "neighbor" or "feature"
    ref_node: int | None = None  # global node index, for "neighbor" factors -- caller resolves display id


def _local_subgraph_tensors(x_std: torch.Tensor, undirected_edge_index: torch.Tensor, node_ids: list[int]):
    """Slices the full standardized feature matrix and the full undirected
    edge_index down to just `node_ids`, remapping to a compact 0..n-1
    local index space a small GNN forward pass can run on cheaply."""
    node_ids = list(node_ids)
    global_to_local = {g: i for i, g in enumerate(node_ids)}
    x_local = x_std[node_ids]

    src, dst = undirected_edge_index
    src, dst = src.tolist(), dst.tolist()
    keep_src, keep_dst = [], []
    for s, d in zip(src, dst):
        if s in global_to_local and d in global_to_local:
            keep_src.append(global_to_local[s])
            keep_dst.append(global_to_local[d])
    if keep_src:
        edge_index_local = torch.tensor([keep_src, keep_dst], dtype=torch.long)
    else:
        edge_index_local = torch.zeros((2, 0), dtype=torch.long)
    return x_local, edge_index_local, global_to_local


def neighbor_importance(
    model,
    x_std: torch.Tensor,
    undirected_edge_index: torch.Tensor,
    target_node: int,
    neighborhood_nodes: list[int],
    direct_neighbors: list[int],
    max_report: int = 6,
) -> list[Factor]:
    """Ranks `target_node`'s direct neighbors by how much removing each one
    (its edge to the target, in both directions) changes the model's
    predicted fraud probability for the target -- computed on the local
    k-hop subgraph only, so this stays fast regardless of the full graph's
    size."""
    if not direct_neighbors:
        return []

    x_local, edge_index_local, g2l = _local_subgraph_tensors(x_std, undirected_edge_index, neighborhood_nodes)
    if target_node not in g2l:
        return []
    local_target = g2l[target_node]

    model.eval()
    with torch.no_grad():
        baseline = torch.sigmoid(model(x_local, edge_index_local))[local_target].item()

    factors = []
    for nb in direct_neighbors:
        if nb not in g2l:
            continue
        local_nb = g2l[nb]
        src, dst = edge_index_local
        keep = ~(((src == g2l[target_node]) & (dst == local_nb)) | ((src == local_nb) & (dst == g2l[target_node])))
        ablated_edge_index = edge_index_local[:, keep]
        with torch.no_grad():
            ablated = torch.sigmoid(model(x_local, ablated_edge_index))[local_target].item()
        delta = baseline - ablated  # positive: this neighbor was pushing the score UP
        factors.append((nb, delta))

    factors.sort(key=lambda t: -abs(t[1]))
    out = []
    for nb, delta in factors[:max_report]:
        out.append(Factor(
            label=f"its connection to node #{nb}",  # caller may override with a real display id
            delta=float(delta),
            direction="toward_fraud" if delta > 0 else "toward_legit",
            kind="neighbor",
            ref_node=nb,
        ))
    return out


def linear_feature_importance(
    coef: np.ndarray,
    x_row: np.ndarray,
    train_mean: np.ndarray,
    feature_names: list[str],
    top_n: int = 5,
) -> list[Factor]:
    """First-order attribution for a linear model: coefficient x (this
    row's value - the training-set average). Exact for a linear model
    (unlike a GNN, no approximation is needed) and sums to the model's own
    logit shift relative to an "average" input."""
    contributions = coef * (x_row - train_mean)
    order = np.argsort(-np.abs(contributions))[:top_n]
    out = []
    for i in order:
        name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
        out.append(Factor(
            label=f"the value of `{name}`",
            delta=float(contributions[i]),
            direction="toward_fraud" if contributions[i] > 0 else "toward_legit",
            kind="feature",
        ))
    return out


def risk_tier(score: float) -> tuple[str, str]:
    if score >= 0.66:
        return "high", "High risk"
    elif score >= 0.33:
        return "medium", "Medium risk"
    return "low", "Low risk"


def two_audience_narrative(score: float, factors: list[Factor], entity_label: str = "this transaction") -> dict:
    """Builds the two plain-language explanations a real fraud-ops system
    is expected to produce alongside a score: one for the team deciding
    what to do next, one for the person on the other end of the decision
    (the substance regulators call a "right to explanation" under GDPR
    Art. 22 and similar rules). Both are template-generated from the same
    computed numbers -- nothing here is invented per case, only phrased."""
    tier, tier_label = risk_tier(score)
    top = factors[:3]

    if tier == "high":
        action = "escalate for manual investigator review before the funds settle"
    elif tier == "medium":
        action = "queue for a lightweight secondary check, not an automatic block"
    else:
        action = "let it proceed, logged for routine monitoring"

    if top:
        factor_bits = "; ".join(
            f"{f.label} ({'+' if f.delta >= 0 else ''}{f.delta:.0%} probability)" for f in top
        )
    else:
        factor_bits = "no single dominant factor -- the score reflects the overall pattern, not one signal"

    compliance_text = (
        f"<strong>{tier_label}</strong> -- model probability <strong>{score:.0%}</strong>. Recommended "
        f"action: {action}. Top contributing factors: {factor_bits}. As with any statistical flag, "
        f"this is a prioritization signal for a human investigator, not a final determination -- "
        f"precision at this operating point is what tab 2 reports, and it is never 100%."
    )

    if tier == "high":
        counterparty_text = (
            f"{entity_label.capitalize()} was placed under additional automated review. This is a "
            "provisional, precautionary step, not a finding of wrongdoing: it happens when a "
            "transaction's pattern of connections resembles ones that were confirmed fraudulent "
            "in the past. A human investigator reviews flagged cases before any final action is "
            "taken, and the account holder has the right to ask for and receive this explanation."
        )
    elif tier == "medium":
        counterparty_text = (
            f"{entity_label.capitalize()} showed some characteristics the system associates with "
            "higher-risk activity, but not enough to escalate on its own. No action is taken "
            "beyond routine logging unless a further signal appears."
        )
    else:
        counterparty_text = (
            f"{entity_label.capitalize()} did not show meaningful similarity to confirmed fraud "
            "patterns and was not flagged."
        )

    return {
        "tier": tier,
        "tier_label": tier_label,
        "action": action,
        "compliance_text": compliance_text,
        "counterparty_text": counterparty_text,
    }
