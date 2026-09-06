#!/usr/bin/env python3
"""Broader Grok metrics beyond per-Q variance, using existing scored runs."""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
PAPER_QD = Path("/agent/repos/silicon-philosophers-paper/assets/questions_data.json")
PRIV = Path("/tmp/silicon_philosophers_private/merged_human_survey_philosophers_normalized.json")
DEMO = Path("/tmp/grok43_demo_full/grok-4.3_scored.json")
NAME = Path("/tmp/grok43_full/grok-4.3_scored.json")


def load_scored(path: Path):
    g = defaultdict(dict)
    for r in json.load(open(path)):
        if r.get("score") is None:
            continue
        name = r["philosopher"] if isinstance(r["philosopher"], str) else r.get("philosopher_name")
        g[r["question"]][name] = float(r["score"])
    return g


def qkey(obj):
    return f"{obj['base_question']}: {obj['position']}"


def paper_matrix(qd, model_key, names):
    qs, cols = [], []
    for qid in sorted(qd, key=lambda x: int(x[1:]) if x[1:].isdigit() else x):
        obj = qd[qid]
        qs.append(qkey(obj))
        cols.append(np.asarray(obj["models"][model_key]["responses"], dtype=float))
    mat = np.column_stack(cols)
    assert mat.shape[0] == len(names), (mat.shape, len(names))
    return qs, mat


def grok_matrix(gmap, names, qs):
    mat = np.full((len(names), len(qs)), np.nan)
    idx = {n: i for i, n in enumerate(names)}
    for j, q in enumerate(qs):
        for phil, sc in gmap.get(q, {}).items():
            i = idx.get(phil)
            if i is not None:
                mat[i, j] = sc
    return mat


def entropy(vals):
    vals = np.asarray(vals)
    vals = vals[~np.isnan(vals)]
    if len(vals) < 2:
        return 0.0
    _, c = np.unique(vals, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def qmetrics(vals):
    vals = np.asarray(vals, float)
    vals = vals[~np.isnan(vals)]
    if len(vals) < 2:
        return None
    var = float(np.var(vals))
    uniq = len(np.unique(vals))
    ent = entropy(vals)
    maxent = math.log2(uniq) if uniq > 1 else 0.0
    nent = ent / maxent if maxent else 0.0
    conc = Counter(vals).most_common(1)[0][1] / len(vals)
    return {
        "variance": var,
        "unique_values": uniq,
        "entropy": ent,
        "normalized_entropy": nent,
        "concentration": conc,
        "is_zero_var": var < 1e-10,
        "is_low_var": uniq <= 2,
    }


def aggregate(qmets):
    avg_ent = float(np.mean([m["entropy"] for m in qmets]))
    avg_nent = float(np.mean([m["normalized_entropy"] for m in qmets]))
    avg_var = float(np.mean([m["variance"] for m in qmets]))
    avg_uniq = float(np.mean([m["unique_values"] for m in qmets]))
    avg_conc = float(np.mean([m["concentration"] for m in qmets]))
    pct_zero = float(100 * np.mean([m["is_zero_var"] for m in qmets]))
    pct_low = float(100 * np.mean([m["is_low_var"] for m in qmets]))
    pct_prob = float(100 * np.mean([m["is_zero_var"] or m["is_low_var"] for m in qmets]))
    quality = (
        0.30 * (avg_nent * 100)
        + 0.25 * min(avg_var / 0.25 * 100, 100)
        + 0.20 * min((avg_uniq - 1) / 4 * 100, 100)
        + 0.25 * ((1 - avg_conc) * 100)
    )
    return {
        "n_questions": len(qmets),
        "mean_per_q_variance": avg_var,
        "avg_entropy": avg_ent,
        "avg_normalized_entropy": avg_nent,
        "avg_unique_values": avg_uniq,
        "avg_concentration": avg_conc,
        "pct_zero_var": pct_zero,
        "pct_low_var": pct_low,
        "pct_problematic": pct_prob,
        "pct_usable": 100 - pct_prob,
        "quality_score_5a": float(quality),
    }


def pearson(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    r = np.corrcoef(x, y)[0, 1]
    return float(r) if not np.isnan(r) else None


def agreement(mat, human):
    mask = ~np.isnan(mat) & ~np.isnan(human)
    xs, ys = mat[mask], human[mask]
    if len(xs) < 10:
        return {}
    return {
        "n_paired_cells": int(mask.sum()),
        "pearson_r": pearson(xs, ys),
        "mae": float(np.mean(np.abs(xs - ys))),
        "exact_match_rate": float(np.mean(np.isclose(xs, ys))),
        "within_0_25_rate": float(np.mean(np.abs(xs - ys) <= 0.25 + 1e-9)),
        "model_mean": float(xs.mean()),
        "human_mean": float(ys.mean()),
    }


def per_q_corr(mat, human):
    rs = []
    for j in range(mat.shape[1]):
        m, h = mat[:, j], human[:, j]
        mask = ~np.isnan(m) & ~np.isnan(h)
        if mask.sum() < 8:
            continue
        r = pearson(m[mask], h[mask])
        if r is not None:
            rs.append(r)
    return {
        "n_questions_with_corr": len(rs),
        "mean_per_q_pearson": float(np.mean(rs)) if rs else None,
        "median_per_q_pearson": float(np.median(rs)) if rs else None,
    }


def _kl(p, q, eps=1e-12):
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    p = (p + eps) / (p + eps).sum()
    q = (q + eps) / (q + eps).sum()
    return float(np.sum(p * np.log(p / q)))


def _js(p, q):
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def cat_div(mat, human):
    kls, jss = [], []
    for j in range(mat.shape[1]):
        m = mat[:, j]
        h = human[:, j]
        m = m[~np.isnan(m)]
        h = h[~np.isnan(h)]
        if len(m) < 10 or len(h) < 10:
            continue
        vals = sorted(set(np.round(m, 4)).union(set(np.round(h, 4))))
        mc = Counter(np.round(m, 4))
        hc = Counter(np.round(h, 4))
        pv = np.array([hc[v] for v in vals], float)
        qv = np.array([mc[v] for v in vals], float)
        kls.append(_kl(pv, qv))
        jss.append(_js(pv, qv))
    return {
        "avg_kl_human_to_model": float(np.mean(kls)) if kls else None,
        "avg_js": float(np.mean(jss)) if jss else None,
        "n_questions_div": len(kls),
    }


def flat20(mat, human):
    mask = ~np.isnan(mat) & ~np.isnan(human)
    xs, ys = mat[mask], human[mask]
    if len(xs) < 50:
        return {}
    edges = np.linspace(0, 1, 21)

    def hist(a):
        idx = np.clip(np.digitize(a, edges[1:-1], right=False), 0, 19)
        return np.bincount(idx, minlength=20).astype(float)

    return {
        "flattened_pearson_r": pearson(xs, ys),
        "flattened_kl_human_to_model": _kl(hist(ys), hist(xs)),
        "flattened_js": _js(hist(ys), hist(xs)),
        "n_flat_cells": int(len(xs)),
    }


def corr_mat(mat, min_n=30):
    nq = mat.shape[1]
    C = np.full((nq, nq), np.nan)
    for i in range(nq):
        C[i, i] = 1.0
        for j in range(i + 1, nq):
            a, b = mat[:, i], mat[:, j]
            mask = ~np.isnan(a) & ~np.isnan(b)
            if mask.sum() < min_n:
                continue
            if np.std(a[mask]) == 0 or np.std(b[mask]) == 0:
                continue
            r = np.corrcoef(a[mask], b[mask])[0, 1]
            if not np.isnan(r):
                C[i, j] = C[j, i] = r
    return C


def rv(A, B):
    A = np.nan_to_num(A, nan=0.0)
    B = np.nan_to_num(B, nan=0.0)
    num = np.trace(A @ B)
    den = math.sqrt(np.trace(A @ A) * np.trace(B @ B))
    return float(num / den) if den else None


def mantel(A, B):
    iu = np.triu_indices(A.shape[0], 1)
    a, b = A[iu], B[iu]
    mask = ~np.isnan(a) & ~np.isnan(b)
    if mask.sum() < 50:
        return None
    return pearson(a[mask], b[mask])


def upper_stats(C):
    iu = np.triu_indices(C.shape[0], 1)
    v = C[iu]
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return {}
    return {
        "mean_abs_r": float(np.mean(np.abs(v))),
        "median_abs_r": float(np.median(np.abs(v))),
        "std_r": float(np.std(v)),
        "pct_abs_r_gt_0_5": float(100 * np.mean(np.abs(v) > 0.5)),
        "n_pairs": int(len(v)),
    }


def evaluate_matrix(mat, human_mat, is_human=False):
    qmets = []
    for j in range(mat.shape[1]):
        m = qmetrics(mat[:, j])
        if m:
            qmets.append(m)
    out = aggregate(qmets)
    if is_human:
        out.update(
            {
                "n_paired_cells": int(np.sum(~np.isnan(human_mat))),
                "pearson_r": 1.0,
                "mae": 0.0,
                "exact_match_rate": 1.0,
                "within_0_25_rate": 1.0,
                "mean_per_q_pearson": 1.0,
                "avg_kl_human_to_model": 0.0,
                "avg_js": 0.0,
                "flattened_js": 0.0,
                "flattened_kl_human_to_model": 0.0,
                "rv_vs_human": 1.0,
                "mantel_r_vs_human": 1.0,
            }
        )
        out.update(upper_stats(corr_mat(mat)))
        return out

    out.update(agreement(mat, human_mat))
    pq = per_q_corr(mat, human_mat)
    out["mean_per_q_pearson"] = pq.get("mean_per_q_pearson")
    out["median_per_q_pearson"] = pq.get("median_per_q_pearson")
    out["n_questions_with_corr"] = pq.get("n_questions_with_corr")
    out.update(cat_div(mat, human_mat))
    out.update(flat20(mat, human_mat))
    C = corr_mat(mat)
    Ch = corr_mat(human_mat)
    out["rv_vs_human"] = rv(C, Ch)
    out["mantel_r_vs_human"] = mantel(C, Ch)
    out.update(upper_stats(C))
    out["cell_coverage"] = float(np.mean(~np.isnan(mat)))
    return out


def main():
    qd = json.load(open(PAPER_QD))
    priv = json.load(open(PRIV))
    names = [p["name"] for p in priv]
    human_priv = {
        p["name"]: {
            k: float(v)
            for k, v in (p.get("responses") or {}).items()
            if v is not None
        }
        for p in priv
    }
    demo = load_scored(DEMO)
    name_only = load_scored(NAME)

    models = [
        "Human",
        "GPT-5.1",
        "Claude Sonnet 4.5",
        "GPT-4o",
        "Llama 3.1 8B",
        "Llama 3.1 8B (FT)",
        "Mistral 7B",
        "Qwen 3 4B",
    ]

    questions_paper, human_mat = paper_matrix(qd, "Human", names)
    results = {}

    for label in models:
        _, mat = paper_matrix(qd, label, names)
        results[label] = evaluate_matrix(mat, human_mat, is_human=(label == "Human"))

    for label, gmap in [
        ("Grok 4.3 (demographics)", demo),
        ("Grok 4.3 (name-only)", name_only),
    ]:
        mat = grok_matrix(gmap, names, questions_paper)
        results[label] = evaluate_matrix(mat, human_mat)
        priv_mat = np.full_like(human_mat, np.nan)
        for i, n in enumerate(names):
            resp = human_priv.get(n, {})
            for j, q in enumerate(questions_paper):
                if q in resp:
                    priv_mat[i, j] = resp[q]
        results[label]["agreement_vs_private_human"] = agreement(mat, priv_mat)
        results[label]["private_human_nonnull_cells"] = int(np.sum(~np.isnan(priv_mat)))

    key_metrics = [
        "mean_per_q_variance",
        "quality_score_5a",
        "avg_entropy",
        "avg_concentration",
        "pct_zero_var",
        "pct_usable",
        "pearson_r",
        "mae",
        "exact_match_rate",
        "within_0_25_rate",
        "mean_per_q_pearson",
        "avg_kl_human_to_model",
        "avg_js",
        "flattened_js",
        "flattened_kl_human_to_model",
        "rv_vs_human",
        "mantel_r_vs_human",
        "mean_abs_r",
    ]

    rows = []
    for model, m in results.items():
        row = {"model": model}
        for k in key_metrics:
            row[k] = m.get(k)
        rows.append(row)

    with open(OUT / "broader_metrics_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model"] + key_metrics)
        w.writeheader()
        w.writerows(rows)

    higher_better = {
        "mean_per_q_variance",
        "quality_score_5a",
        "avg_entropy",
        "pct_usable",
        "pearson_r",
        "exact_match_rate",
        "within_0_25_rate",
        "mean_per_q_pearson",
        "rv_vs_human",
        "mantel_r_vs_human",
        "mean_abs_r",
    }
    rankings = {}
    for metric in key_metrics:
        vals = [(r["model"], r[metric]) for r in rows if r[metric] is not None]
        if not vals:
            continue
        ordered = sorted(vals, key=lambda x: x[1], reverse=(metric in higher_better))
        rankings[metric] = [
            {"rank": i + 1, "model": m, "value": v} for i, (m, v) in enumerate(ordered)
        ]

    gdemo = results["Grok 4.3 (demographics)"]
    gname = results["Grok 4.3 (name-only)"]
    gpt = results["GPT-5.1"]
    claude = results["Claude Sonnet 4.5"]

    summary = {
        "notes": {
            "prompting_audit": (
                "Grok persona+question prompts match 3_model_eval.py for demographic fields; "
                "human survey responses stripped from prompts; private file not committed."
            ),
            "metric_sources": (
                "5a-style quality/entropy/KL-JS/RV/Mantel; flattened 20-bin KL/JS; "
                "human agreement vs paper questions_data Human vectors."
            ),
            "caveat_variance": (
                "Higher mean per-Q variance is closer to humans on silicon-sampling diversity; "
                "it is not a general capability ranking."
            ),
            "caveat_agreement": (
                "Pearson/MAE vs human survey answers is a better proxy for "
                "'answers like the philosophers'."
            ),
        },
        "models": results,
        "rankings": rankings,
        "takeaways": {
            "prompting": "Prompts match paper demographic template; not a prompt bug.",
            "variance": {
                "grok_demo": gdemo["mean_per_q_variance"],
                "grok_name": gname["mean_per_q_variance"],
                "gpt51": gpt["mean_per_q_variance"],
                "claude": claude["mean_per_q_variance"],
            },
            "human_agreement_pearson": {
                "grok_demo": gdemo.get("pearson_r"),
                "grok_name": gname.get("pearson_r"),
                "gpt51": gpt.get("pearson_r"),
                "claude": claude.get("pearson_r"),
            },
            "quality_score_5a": {
                "grok_demo": gdemo.get("quality_score_5a"),
                "grok_name": gname.get("quality_score_5a"),
                "gpt51": gpt.get("quality_score_5a"),
                "claude": claude.get("quality_score_5a"),
                "human": results["Human"].get("quality_score_5a"),
            },
            "distribution_js_vs_human": {
                "grok_demo": gdemo.get("avg_js"),
                "gpt51": gpt.get("avg_js"),
                "claude": claude.get("avg_js"),
            },
            "corr_structure_rv_vs_human": {
                "grok_demo": gdemo.get("rv_vs_human"),
                "gpt51": gpt.get("rv_vs_human"),
                "claude": claude.get("rv_vs_human"),
            },
            "suggested_improvements": [
                "Re-run with reasoning enabled (Grok 4.3 is a reasoning model).",
                "Strip phd_country==Unknown instead of injecting (Unknown).",
                "Optional hybrid: demographics AND philosopher name.",
                "Treat human-agreement / JS / RV as primary better-than-GPT-5.1 criteria, not variance alone.",
            ],
        },
    }

    with open(OUT / "broader_metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print("model | var | quality | H | conc | zero% | r | mae | JS | RV")
    for r in sorted(rows, key=lambda x: -(x["mean_per_q_variance"] or -1)):

        def fmt(v, nd=3):
            return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "NA"

        print(
            f"{r['model'][:28]:28s} | {fmt(r['mean_per_q_variance'],4)} | {fmt(r['quality_score_5a'],1)} | "
            f"{fmt(r['avg_entropy'])} | {fmt(r['avg_concentration'])} | {fmt(r['pct_zero_var'],1)} | "
            f"{fmt(r['pearson_r'])} | {fmt(r['mae'])} | {fmt(r['avg_js'])} | {fmt(r['rv_vs_human'])}"
        )
    print("\nTakeaways:")
    print(json.dumps(summary["takeaways"], indent=2))


if __name__ == "__main__":
    main()
