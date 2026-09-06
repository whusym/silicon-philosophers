# Silicon Philosophers — Grok 4.3 Executive Summary

**Scope:** Full demographic silicon-sampling grid (277 philosophers × 100 questions) for **Grok 4.3** via OpenRouter, compared to paper baselines using **paper §3.2 metrics only**.

**Not used:** Pipeline-internal `quality_score` / `quality_score_5a` (not a paper metric).

**Grok 4.6:** Skipped (no OpenRouter `:batch` endpoint; mandatory reasoning made a full-grid rerun cost-prohibitive). Results below are **Grok 4.3 only**.

## Setup notes
- Prompts match `3_model_eval.py` (AOS / AOI / PhD / institution).
- Human survey answers stripped from prompts; private demographics file not committed.
- Grok 4.3 run: reasoning disabled (paper-comparable commercial setting), temperature 0.
- Name-only ablation included for sensitivity.

## Table A — Per-question variance (paper Figure 1 style)

| Model | Per-Q Var ↑ | Var Ratio vs Human ↓ |
|------|-------------|----------------------|
| Human | 0.0711 | 1.00× |
| Claude Sonnet 4.5 | 0.0400 | 1.78× |
| Mistral 7B | 0.0278 | 2.56× |
| Qwen 3 4B | 0.0276 | 2.57× |
| Llama 3.1 8B | 0.0255 | 2.79× |
| Llama 3.1 8B (FT) | 0.0207 | 3.44× |
| GPT-4o | 0.0198 | 3.59× |
| GPT-5.1 | 0.0197 | 3.62× |
| **Grok 4.3 (demographics)** | **0.0160** | **4.45×** |
| **Grok 4.3 (name-only)** | **0.0095** | **7.49×** |

Higher Per-Q Var = more persona differentiation (closer to humans on silicon-sampling diversity).

## Table B — Paper §3.2 metrics (extended)

| Model | Per-Q Var ↑ | Entropy ↑ | Zero-var % ↓ | JS ↓ | KL ↓ | Mantel r ↑ | RV ↑ |
|------|-------------|-----------|--------------|------|------|------------|------|
| Human | 0.0711 | 1.089 | 0.0 | 0.000 | 0.00 | 1.000 | 1.000 |
| Claude Sonnet 4.5 | 0.0400 | 0.766 | 8.2 | 0.117 | 2.53 | 0.269 | 0.456 |
| Mistral 7B | 0.0278 | 0.664 | 9.2 | 0.345 | 9.03 | 0.090 | 0.421 |
| Qwen 3 4B | 0.0276 | 0.478 | 30.0 | 0.229 | 10.15 | 0.067 | 0.474 |
| Llama 3.1 8B | 0.0255 | 0.611 | 9.1 | 0.171 | 5.09 | 0.106 | 0.422 |
| Llama 3.1 8B (FT) | 0.0207 | 0.736 | 5.1 | 0.241 | 6.12 | 0.133 | 0.413 |
| GPT-4o | 0.0198 | 0.590 | 10.5 | 0.207 | 6.03 | 0.153 | 0.448 |
| GPT-5.1 | 0.0197 | 0.590 | 7.4 | 0.207 | 6.03 | 0.134 | 0.444 |
| **Grok 4.3 (demographics)** | **0.0160** | **0.710** | **11.0** | **0.251** | **6.62** | **0.184** | **0.475** |
| **Grok 4.3 (name-only)** | **0.0095** | **0.390** | **31.0** | **0.235** | **9.74** | **0.044** | **0.500** |

## Table C — Grok 4.3 (demographics) vs GPT-5.1

| Metric | Grok 4.3 (demo) | GPT-5.1 | Verdict |
|--------|-----------------|---------|---------|
| Per-Q Var ↑ | 0.0160 | 0.0197 | Grok worse |
| Var Ratio ↓ | 4.45 | 3.62 | Grok worse |
| Entropy ↑ | 0.710 | 0.590 | Grok better |
| Zero-var % ↓ | 11.0 | 7.4 | Grok worse |
| JS ↓ | 0.251 | 0.207 | Grok worse |
| KL ↓ | 6.62 | 6.03 | Grok worse |
| Mantel r ↑ | 0.184 | 0.134 | Grok better |
| RV ↑ | 0.475 | 0.444 | Grok better |

## Takeaways
1. **Prompting is correct** — same demographic template as the paper; low diversity is not a prompt bug.
2. **On diversity (Per-Q Var), Grok 4.3 underperforms GPT-5.1** (0.016 vs 0.020) and is far below Claude/Human.
3. **Demographics help vs name-only** (0.016 vs 0.0095) but do not close the GPT-5.1 gap.
4. **Corr-structure Mantel/RV vs humans are relatively strong** for Grok 4.3 demographics (better than GPT-5.1).
5. **Distributional match (JS/KL) is weaker than GPT-5.1/Claude.**

## Files
- Metrics: `eval_results_grok43/paper_metrics_comparison.csv`
- Paper site table update: `silicon-philosophers-paper/assets/figure1_stats_bc.csv` (Grok 4.3 row added)

