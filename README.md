# Knowledge-Graph-Gated Defactualization (DSR) for Style-Controllable and Fact-Preserving Generation in Agentic Conversational AI

[![Paper](https://img.shields.io/badge/IEEE%20TKDE-Special%20Issue%20(Submitted)-blue)](#12-citation)
[![License](https://img.shields.io/badge/license-MIT-green)](#13-license)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#5-requirements)
[![Models](https://img.shields.io/badge/LLaMA-2%20%26%203-orange)](#6-models-evaluated)
[![Generations](https://img.shields.io/badge/evaluated%20on-1%2C200%20generations-success)](#7-results)
[![Reproducible](https://img.shields.io/badge/artifacts-cached%20%26%20released-lightgrey)](#3-repository-structure)

Official code, cached steering vectors, knowledge-graph artifacts, and evaluation scripts for:

> T. K. Shrivastava, D. R. Nandu, and R. K. Mundotiya, **"Knowledge-Graph-Gated Defactualization for Style-Controllable and Fact-Preserving Generation in Agentic Conversational AI,"** *IEEE Transactions on Knowledge and Data Engineering, Special Issue: Data and Knowledge Empowered Generative AI* (submitted).

**Affiliation:** MATRA Lab (Multimodal and Multilingual AI for Translational Research and Applications), Department of Computer Science and Engineering, IIT Bhilai.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Why DSR? Positioning Against Fine-Tuning, RAG, and Steering-Only Methods](#2-why-dsr-positioning-against-fine-tuning-rag-and-steering-only-methods)
3. [Repository Structure](#3-repository-structure)
4. [Method Summary](#4-method-summary)
5. [Requirements](#5-requirements)
6. [Models Evaluated](#6-models-evaluated)
7. [Results](#7-results)
8. [Diagnostics: SAF, HSI, TCI, EC](#8-diagnostics-saf-hsi-tci-ec)
9. [Figures & Tables Map](#9-figures--tables-map)
10. [Running the Pipeline](#10-running-the-pipeline)
11. [Limitations](#11-limitations)
12. [Citation](#12-citation)
13. [License](#13-license)
14. [Acknowledgments](#14-acknowledgments)
15. [Key References](#15-key-references)
16. [Contact](#16-contact)

---

## 1. Overview

Activation steering can change *how* a frozen LLM writes (tone, register) without any fine-tuning [1], [2], but it has no built-in notion of *what must not change* — order IDs, product names, customer names, and other verifiable facts can be paraphrased, dropped, or stylistically corrupted along with everything else in the residual stream. We call this failure mode **semantic leakage**: style and content signals occupy the same residual-stream representation, so a steering vector applied uniformly across token positions can drag factual spans toward the requested register.

This repository implements **Defactualize–Steer–Rehydrate (DSR)**: a typed, salience-weighted knowledge graph (KG) that wraps a steering operator as a **mask-and-verification contract**, without ever modifying the steering vector, layer, or strength itself.

```
Raw message → [1] Input Processing → [2] Knowledge Extraction → [3] KG Construction
            → [4] Defactualization (mask entities → typed placeholders)
            → [5] Steered Generation (activation steering at layer L)
            → [6] Rehydration & Verification (deterministic restore from KG)
            → [7] Output (style-controlled, fact-preserving response)
```

The KG never touches the steering direction `v_s`, layer `L`, or strength `α` — it only controls *which spans the steering operator is ever exposed to*, and *which verified values are deterministically written back after decoding*. This is what makes DSR a **gating** mechanism rather than a competing steering method: it is a neuro-symbolic layer wrapped around an unmodified, off-the-shelf activation-steering operator [1], [2].

### Headline results

| Metric | Result | Source |
|---|---|---|
| Entity-coverage gain, KG-gated DSR vs. activation-only (AO) baseline | Cohen's *d* = 0.225, *p*₍Bonf₎ = 1.0 × 10⁻⁴ (Welch *t*-test, Bonferroni-corrected) | Table III, single-model (LLaMA-2-7B-chat) 100-case paired ablation |
| Absolute entity coverage, AO baseline | **0.0000** (empathetic branch) | Table III |
| Absolute entity coverage, DSR (empathetic branch / pooled both styles, same model+cases) | **0.0056** / **0.0193** | Table III, §VI-D |
| Absolute entity coverage, full 6-model × 600-case study (per-model mean, pooled styles) | 0.012 – 0.037 (peaks at LLaMA-3.1-8B-Instruct) | Table V |
| Placeholder leakage (raw `<TYPE>` token reaching the user) | **0.000** across all 1,200 generations and all 6 models | Table III, Fig. 4 |
| Activation-steering path success rate (fallback never triggered) | **100%** across all six models | §IV, §VII |
| Style-fidelity metrics (empathy, formality, style discrimination, TTR, ROUGE-1 divergence) under KG-gating vs. AO | Statistically unchanged (n.s. after Bonferroni correction) | Table III |
| Sole significant style-side difference, KG-gated vs. AO | Response length — KG-gated responses shorter on average (*p*₍Bonf₎ < 0.05) | Table III |
| KG structural invariance across all 6 host models | Largest cross-model spread < 3% of the mean, for node count, edge count, density, and mean salience | Table II |
| Urgency-level trend on entity coverage | No monotonic trend (Jonckheere–Terpstra trend test, *z* = −1.24, *p* = 0.21) — bottleneck is surface realization, not KG extraction | §VI-D |
| LLaMA-2 vs. LLaMA-3 family comparison (*n* = 300 each) | No significant cross-family difference in coverage, readability, or style discrimination (*p* > 0.17); only Flesch Ease differs significantly | §VII |

> **Read this carefully:** the *d* = 0.225 effect is real and statistically robust (it survives Bonferroni correction across the full metric battery in Table III), but it is a **modest, single-model estimate**, and absolute entity-coverage values across the full study remain in the low single digits to low tens-of-percent range (Table V) — DSR measurably increases verified-entity preservation over a steering-only baseline, it does **not** claim to solve factual grounding outright. See [§11 Limitations](#11-limitations) and [§7.1](#71-entity-coverage--placeholder-integrity) for the full picture, including why coverage and leakage are different (and not equally hard) problems.

---

## 2. Why DSR? Positioning Against Fine-Tuning, RAG, and Steering-Only Methods

| Approach | Modifies model weights? | Controls style | Preserves exact entity values | Mechanism |
|---|---|---|---|---|
| Fine-tuning / LoRA | Yes (one run per style × model) | Yes, baked into weights | No explicit guarantee | Gradient update |
| RLHF | Yes | Indirect (reward-shaped) | No explicit guarantee | Reward-model-guided policy update |
| RAG [6] / GraphRAG | No | No | Improves grounding by conditioning on retrieved evidence, but the generator remains free to paraphrase or drop entities inside it | Retrieval into context |
| Activation steering / representation engineering [1], [2] | No | Yes | Not addressed — steering is content-agnostic, causing semantic leakage | Residual-stream perturbation, `h'_L = h_L + α·v_s` |
| Post-hoc hallucination detection | No | N/A | Detects unsupported claims after the fact; does not shape the generation trajectory | Semantic entropy / self-consistency / atomic fact-checking |
| **DSR (this repository)** | **No** | **Yes** (inherits whichever steering operator is plugged in) | **Deterministic mask before steering + salience-ranked rehydration after decoding** (Eqs. 4, 9) | KG-gated wrapper around an unmodified steering operator |

DSR is complementary to, not a replacement for, knowledge-graph-augmented LLM methods that focus on *retrieval* — i.e., deciding what to condition on — since those methods do not engage with the internal representation at all. It is likewise complementary to RAG [6]: DSR assumes the entities requiring preservation are already present in the input message, and does not introduce external knowledge. Agent-memory frameworks (ReAct, MemGPT, generative agents) motivate the natural extension of DSR's per-request, ephemeral KG into a persistent, indexed store for multi-turn grounding — explicitly named as future work (§VII of the paper).

---

## 3. Repository Structure

> **As of the latest push, this is what's actually on `main`** — six per-model run directories, a dedicated activation-only-vs-KG-gated comparison, a multi-model evaluation bundle, and a supplementary-analysis bundle for the extended ablations. Each `A2A_KG_<model>/` directory is self-contained and follows the same internal layout, populated by running `main.py` (or the model's `build_vectors*.ipynb`) per [Section 10](#10-running-the-pipeline). Note that every top-level model folder currently contains one nested subfolder of the same name (e.g. `A2A_KG_Llama-2-13b-chat-hf/A2A_KG_Llama-2-13b-chat-hf/…`) — a leftover of how the folders were uploaded; the paths below reflect the real, current layout rather than the flattened one.

```
KG-Gated-Defactualization/
├── README.md
│
├── A2A_KG_Llama-2-7b-hf/                                    ← base (non-chat) LLaMA-2 7B run
│   └── A2A_KG_Llama-2-7b-hf/
│       ├── main.py
│       ├── build_vectors.ipynb                              ← estimates & caches v_s per style — Eqs. 5–7
│       ├── a2a_kg_pipeline_complete (1) spacy.ipynb
│       ├── a2a_unified_steering_analysis.ipynb               ← layer-separability + alpha-sweep (Fig. 8, 9)
│       ├── alpha_sweep.png / steering_layer_analysis.png     ← cached diagnostic figures
│       ├── .style_cache/                                    ← cached, reusable steering vectors
│       └── outputs/knowledge_graphs/*.json                  ← one typed KG per case (Fig. 2 artifacts)
│
├── A2A_KG_Llama-2-7b-chat-hf/
│   └── A2A_KG_Llama-2-7b-chat-hf/
│       ├── main_7b.py
│       ├── build_vectors.ipynb
│       ├── a2a_kg_7b_chat.ipynb                              ← interactive single-case walkthrough
│       ├── a2a_unified_steering_analysis.ipynb
│       ├── alpha_sweep.png / steering_layer_analysis.png
│       ├── .style_cache/ , .style_cache_7b/
│       └── outputs/
│
├── A2A_KG_Llama-2-13b-chat-hf/                               ← worked example referenced throughout this README
│   └── A2A_KG_Llama-2-13b-chat-hf/
│       ├── main.py
│       ├── build_vectors.ipynb
│       ├── a2a_kg_fixed_memory.ipynb                         ← the Fig. 1 example (Alex, order ORD-1234 delayed)
│       ├── a2a_unified_steering_analysis.ipynb
│       ├── alpha_sweep.png / steering_layer_analysis.png
│       ├── .style_cache/
│       └── outputs/knowledge_graphs/kg_001_*.json … kg_NNN_*.json, results_<timestamp>.jsonl
│
├── A2A_KG_Llama-3.1-8b-Instruct/
│   └── A2A_KG_Llama-3.1-8b-Instruct/
│       ├── main.py
│       ├── build_vectors_llama31_fixed.ipynb
│       ├── a2a_unified_steering_analysis.ipynb
│       ├── alpha_sweep.png
│       └── outputs/
│
├── A2A_KG_Llama-3.2-1B-Instruct/
│   └── A2A_KG_Llama-3.2-1B-Instruct/
│       ├── main.py
│       ├── A2A_pipeline_3.2-3B-Instruct.ipynb
│       ├── a2a_unified_steering_analysis.ipynb
│       ├── zombiekillRAM.ipynb                                ← memory-management / OOM workaround notebook
│       ├── alpha_sweep.png
│       └── outputs/
│
├── A2A_KG_Llama-3.2-3B-Instruct/
│   └── A2A_KG_Llama-3.2-3B-Instruct/
│       ├── main.py
│       ├── A2A_pipeline_3.2-3B-Instruct.ipynb
│       └── outputs/knowledge_graphs/kg_001_*.json … kg_100_*.json, results_20260524_062711.jsonl
│                                                               ← full 100-case run for this model (4 scenarios × 25)
│
├── Comparision_KG-No-KG/                                     ← head-to-head KG-gated vs. activation-only (AO) ablation
│   └── Comparision_KG-No-KG/
│       ├── outputs/knowledge_graphs/                         ← paired case_NNNN_activation_only.json / case_NNNN_kg_steering.json
│       ├── results_<timestamp>.jsonl                         ← two runs (root level)
│       └── Llama-2-7b-chat-hf/                                ← the 100-case ablation underlying Table III, §VI-D
│           ├── main.py
│           ├── a2a_kg_7b_chat_fixed.ipynb, A2A_KG_vs_AO_Evaluation (1).ipynb
│           ├── kg_vs_ao_FINAL_SUMMARY.csv, kg_vs_ao_statistical_tests.csv
│           ├── metrics_all_100_cases.csv, metrics_all_cases.csv
│           ├── outputs/knowledge_graphs/                     ← case_0000…case_0099, paired activation_only / kg_steering
│           └── eval_A_headtohead.png, eval_B_delta_distributions.png, eval_C_entity_coverage.png,
│               eval_D_breakdown_{scenario,sentiment,urgency}.png
│
├── Evaluation/  (+ Evaluation.zip, a duplicate archive of the same folder)
│   └── Evaluation/
│       ├── A2A_KG_MultiModel_Evaluation (1).ipynb            ← aggregates all six models into the study-wide tables
│       ├── leaderboard_summary.csv, statistical_tests_all_models.csv
│       ├── metrics_<model>.csv                                ← per-model metrics (L2-7B-Base/Chat, L2-13B-Chat, L3.1-8B, L3.2-1B/3B)
│       ├── metrics_all_models_combined.csv
│       ├── eval_1…eval_13_*.png                               ← Figs. 3–6 source charts (readability, style fidelity,
│       │                                                          KG structure, radar, ablation breakdowns, Cohen's d heatmap)
│       └── Tier1 Metrics/
│           ├── main.py, a2a_unified_steering_analysis.ipynb
│           └── alpha_sweep.png, steering_layer_analysis.png
│
├── DSR_Supplementary_Analysis/                               ← extended ablations / reviewer-response material
│   ├── DSR_Supplementary_Analysis.ipynb, DK_GenAI_Extended_Ablation.ipynb
│   ├── DSR_Reviewer_Response_Notebook (1).ipynb
│   ├── figA1_variance_decomposition, figB1_zero_inflation, figC1_urgency_trend,
│   │   figD1_tone_scenario_interaction, figE1_generation_contrast, figF1_kg_efficiency_frontier,
│   │   figG1_style_calibration_error, figH1_bounded_effects, figJ1_extended_ablation_dashboard,
│   │   figS1_kg_structural_invariance, figS3_proxy_diagnostics, figS4_scenario_conditioning,
│   │   figS5_bootstrap_entity_cov, figS6_cohens_d_heatmap_extended, figS8_readability_cdf,
│   │   figS9_alpha_proxy_sweep                                ← each as matching .png / .pdf
│   └── tableA1_variance_decomposition.csv, tableB1_entity_sparsity.csv, tableC1_urgency_trend_test.csv,
│       tableD1_tone_scenario_interaction.csv, tableE1_generation_contrast.csv,
│       tableG1_style_calibration_error.csv, tableH1_bounded_effects.csv, tex_table_variance.tex
│
├── kg/                                                        ← standalone KG-construction module + worked example graphs
│   ├── kg.py                                                  ← Algorithm 1 reference implementation
│   └── kg_001…003_battery_issue_*.json
│
├── figS10_scatter_matrix.png / .pdf                           ← top-level supplementary figure
├── tableS1_pairwise_model_tests.csv
├── tableS2_readability_separability.csv
├── tableS3_correlation_analysis.csv
├── tableS4_final_summary.csv / .png / .pdf
└── kg_image.png                                                ← KG schematic used in Fig. 2 / README
```

> **Not currently in the repo:** `LICENSE`, `requirements.txt`, `.env.example`, and a dedicated `figures/` or `paper/` folder are referenced elsewhere in this README (e.g. [§5](#5-requirements), [§13](#13-license)) as expected project scaffolding, but don't exist on `main` yet — add them alongside the next push so those links resolve. There's also no `.gitignore` yet, so build artifacts (`__pycache__/`, `.style_cache/`, and stray pip-install-generated files like `=1.26.0`) are currently tracked in git; see the note at the end of [§5](#5-requirements) for a suggested fix.

### `outputs/knowledge_graphs/*.json` schema

Each file is the typed, salience-weighted graph `G = (V, E)` extracted for one case (Algorithm 1):

```json
{
  "case_index": 1,
  "scenario": "battery_issue",
  "customer_name": "Priya Sharma",
  "order_id": "ORD-7741",
  "knowledge_graph": {
    "nodes": [
      {"value": "TechPro X200 Laptop", "type": "PRODUCT",       "salience": 0.85},
      {"value": "ORD-7741",            "type": "ORDER_ID",      "salience": 0.95},
      {"value": "...",                 "type": "ISSUE",         "salience": 0.90},
      {"value": "high",                "type": "URGENCY",       "salience": 1.00},
      {"value": "positive",            "type": "SENTIMENT",     "salience": 0.65},
      {"value": "Priya Sharma",        "type": "CUSTOMER_NAME", "salience": 0.65}
    ],
    "edges": [
      {"source": "TechPro X200 Laptop", "target": "...", "relation": "has_issue",   "weight": 0.765},
      {"source": "ORD-7741", "target": "TechPro X200 Laptop", "relation": "about_product", "weight": 0.807},
      {"source": "Priya Sharma", "target": "ORD-7741", "relation": "placed_order",  "weight": 0.617}
    ]
  }
}
```

`type ∈ T = {CUSTOMER_NAME, ORDER_ID, PRODUCT, ISSUE, URGENCY, SENTIMENT}`; edge weight `w(u,v) = sal(u)·sal(v)` (Eq. 3) — a multiplicative form chosen deliberately so a single low-confidence endpoint suppresses an edge's weight more sharply than an averaging rule would. Edge weight feeds only the visualization/ranking in Fig. 2; it does **not** feed defactualization (Eq. 4) or rehydration (Eq. 9), which depend on node salience alone.

---

## 4. Method Summary

| Stage | Component | What it does |
|---|---|---|
| 1–2 | Input Processing → Knowledge Extraction | Layered regex (order IDs, products, currency) → spaCy NER (names, issue spans) → lexicon classifiers (urgency, sentiment); highest-confidence-wins, no model training. |
| 3 | KG Construction | Builds typed graph `G=(V,E)`, `|V|≤10`, salience-scored per Eqs. 1–2, terminates in `O(n+|V|+|E|)`. Ontology design follows established graph-quality-management guidance for keeping ontologies enumerable as they scale. |
| 4 | Defactualization | Case-insensitive, longest-match-first substitution of every node value with a typed placeholder `<TYPE>` (Eq. 4) **before** any activation is read or any steering vector is built. |
| 5 | Steered Generation | Contrastive steering vector `v_s` (PCA estimator, Eq. 5) added to the residual stream at a single layer `L`: `h′_L(t) = h_L(t) + α·v_s` (Eq. 8); greedy decode, repetition penalty 1.3, `max_new_tokens=120`. A prompt-steering fallback fires only on malformed output (never triggered in the 1,200-generation study). |
| 6 | Rehydration & Verification | Each placeholder is deterministically replaced with the highest-salience candidate value of its type (Eq. 9) — a lookup, not a generative step. Unmatched placeholders are stripped, never left dangling. |
| 7 | Output | Style-controlled, fact-preserving response; the same defactualized input branches into both `empathetic` and `formal` variants. |

Three estimators for the steering direction `v_s` are implemented and cached — PCA (Eq. 5), mean-difference (Eq. 6), and logistic-regression (Eq. 7). On the layer-separability probe (LLaMA-3.2-3B-Instruct, layers 0–27), the **PCA estimator gave the lowest mean cosine similarity between the empathetic and formal directions (cos = 0.41)**, vs. cos = 0.53 (mean-difference) and cos = 0.49 (logistic regression) — i.e., the most mutually distinguishable style directions of the three. All paper-reported results use the PCA estimator on this basis; the other two are released as cached supplementary artifacts for independent comparison.

---

## 5. Requirements

```
python >= 3.10
torch
transformers
scikit-learn
spacy           # + python -m spacy download en_core_web_sm
langchain-groq
pydantic
python-dotenv
rich
numpy
```

Install:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Environment variables (`.env`)

```ini
GROQ_API_KEY=gsk_...                              # drives the A2A customer/support-context agents
LOCAL_MODEL_NAME=meta-llama/Llama-2-13b-chat-hf    # host model under test; swap per model directory
STEER_LAYER=28                                     # steering layer L (per-model, see Table in paper)
STEER_ALPHA=12.0                                   # steering strength α (per-model)
HF_TOKEN=hf_...                                    # required for gated LLaMA checkpoints
STYLE_CACHE_DIR=.style_cache                       # optional override
```

> **Never commit `.env` or hardcode `GROQ_API_KEY` / `HF_TOKEN` inside notebooks.** A previous push to this repo was blocked by GitHub's secret-scanning push protection for exactly this reason. Keep both keys in a local, untracked `.env` file and load them with `python-dotenv`. A minimal `.gitignore` is also recommended, since none exists on `main` yet:
> ```
> .env
> __pycache__/
> *.pyc
> .style_cache/
> .style_cache_7b/
> =*            # stray files from an unquoted `pip install pkg>=x.y.z` in bash/zsh
> ```

---

## 6. Models Evaluated

Six causal LLaMA variants — LLaMA 2 [9] and LLaMA 3 [10] — are evaluated without any weight modification, spanning 1B–13B parameters and the base, chat, and instruct training regimes.

| Family | Variant | Params | Regime | Mean Entity Coverage (pooled, Table V) |
|---|---|---|---|---|
| LLaMA 2 | 7B | 7B | Base | 0.016 |
| LLaMA 2 | 7B-chat | 7B | Chat | 0.017 |
| LLaMA 2 | 13B-chat | 13B | Chat | 0.023 |
| LLaMA 3.1 | 8B-Instruct | 8B | Instruct | **0.037** (highest) |
| LLaMA 3.2 | 1B-Instruct | 1B | Instruct | 0.012 (lowest) |
| LLaMA 3.2 | 3B-Instruct | 3B | Instruct | 0.016 |

No weights are modified for any model; `v_s` is estimated once per (style, model) pair and cached (~32 KB total for two styles at `d=4096`, roughly seven orders of magnitude below a single model's bf16 weight footprint), then reused unchanged across every incoming message. Neither stylistic fidelity nor factual grounding scales monotonically with parameter count within the 1B–13B range studied — several metrics peak at intermediate model sizes, and model family / training regime (base vs. chat vs. instruct) appear to matter at least as much as raw scale (Fig. 7).

---

## 7. Results

This section expands the headline table in [§1](#1-overview) with the full battery of analyses from the paper (§VI).

### 7.1 Entity Coverage & Placeholder Integrity

- **Coverage gain is real but modest.** The KG-gating ablation (Table III) shows entity coverage rising from an AO baseline of exactly **0.0000** to **0.0056** (empathetic branch) / **0.0193** (pooled) under DSR, a statistically significant gain (Welch *t*-test, Bonferroni-corrected, *d* = 0.225, *p*₍Bonf₎ = 1.0 × 10⁻⁴) that survives correction across the entire metric battery in Table III — while every style-fidelity metric in the same table (empathy, formality, style discrimination, lexical diversity, readability, clarity, ROUGE-1 divergence) remains statistically unchanged.
- **Leakage is the stronger result.** Placeholder leakage — a raw, unresolved `<TYPE>` token reaching the end user — is **exactly 0.0 across all 1,200 generations and all six architecturally distinct host models** (Fig. 4). Because the activation-steering path never produced a malformed response across the corpus (the prompt-steering fallback was never triggered), this confirms the modest coverage numbers stem from the steered model *omitting* placeholders during generation — not from the deterministic rehydration step (Eq. 9) failing to restore values it was given.
- **No monotonic urgency trend.** A Jonckheere–Terpstra trend test across the three urgency levels found no monotonic relationship with coverage (*z* = −1.24, *p* = 0.21), indicating the coverage bottleneck lies in surface realization during decoding, not in KG extraction quality.

### 7.2 Style Fidelity

Table IV (per-model, per-tone, 1,200 generations) shows lexicon-based empathy and formality scores stay modest across all six models, while **readability (Flesch Ease) is the most consistent empathetic–formal separator**, and low cross-tone ROUGE-1/ROUGE-L overlap confirms the two styles differ lexically rather than superficially. Readability effect sizes are significant across all models after Bonferroni correction (paired *t*-test on per-case empathetic − formal differences), except LLaMA-2-7B-base, which shows the Flesch Ease effect in the opposite direction. Style Calibration Error (SCE) — mean absolute deviation of each generation's style score from the model's own 90th-percentile style ceiling — shows no monotonic scaling with model size (Fig. 3).

### 7.3 Structural Invariance of the Knowledge Graph

| Model | \|V\| | \|E\| | Density | Mean Salience |
|---|---|---|---|---|
| LLaMA-2-7B-Chat | 7.55 | 6.19 | 0.121 | 0.824 |
| LLaMA-2-7B-Base | 7.43 | 5.94 | 0.122 | 0.821 |
| LLaMA-2-13B-Chat | 7.45 | 5.94 | 0.122 | 0.812 |
| LLaMA-3.1-8B | 7.46 | 6.09 | 0.123 | 0.823 |
| LLaMA-3.2-1B | 7.54 | 6.00 | 0.120 | 0.812 |
| LLaMA-3.2-3B | 7.64 | 6.16 | 0.121 | 0.823 |

(Table II.) Node count, edge count, density, and mean salience are near-identical across all six host models, with the largest cross-model spread under 3% of the mean for every statistic — confirming KG construction (Algorithm 1) never queries `fθ`, so its output distribution is a property of the input scenario distribution and the fixed extraction ontology, not of which LLM is later steered.

### 7.4 What Actually Drives Coverage? (Variance Decomposition)

A one-way ANOVA partial-η² decomposition (Fig. 5) finds **Scenario Type (η² = 0.053) explains more entity-coverage variance than Host Model (η² = 0.040)** — input conditions contribute more strongly to factual grounding than model identity does. Collectively, input-condition factors explain more coverage variation than which of the six LLMs is doing the generating.

### 7.5 Per-Entity-Type Recovery

A zero-inflation analysis (Fig. 6) shows **SENTIMENT is never recovered in the empathetic pipeline**, while **URGENCY is the most recoverable entity type** — consistent with the ontology design, since urgency is carried by salient lexical tokens that survive steered decoding, whereas sentiment labels stay entangled with the steered stylistic register itself.

### 7.6 Layer Separability and Steering-Strength Sensitivity

Two diagnostics characterize the steering operator independently of the KG layer (§VI-H):

- **Layer separability** (Fig. 8, LLaMA-3.2-3B-Instruct): maximum empathetic/formal separability occurs around layers 6–10, while the deployed layer (`L = 16`) lies beyond this optimum — the alpha-sweep selection protocol does not necessarily land on the layer of greatest style separability.
- **Alpha sweep** (Fig. 9, same model, `L = 16`): Tone Consistency Index (TCI) varies non-monotonically with `α`, while Hallucination Severity Index (HSI) saturates over much of the sweep. This previously undocumented operating region — where hallucination diagnostics go insensitive to steering strength while tone consistency keeps varying — appears in **two of the six evaluated models** (LLaMA-3.2-3B-Instruct and LLaMA-2-13B-chat). Because the KG is held fixed throughout the sweep, this behavior is attributable to the steering operator itself, not to the knowledge-grounding framework.

### 7.7 Cross-Family Robustness

A LLaMA-2 vs. LLaMA-3 sub-corpus comparison (*n* = 300 cases each) found **no significant cross-family difference in entity coverage, readability, or style discrimination** (*p* > 0.17 in all cases) — only Flesch Ease differed significantly, suggesting a host-family effect rather than a KG-layer effect. This supports treating the entity-coverage gain in §7.1 as broadly representative beyond the single model (LLaMA-2-7B-chat) used for the formal ablation, while still flagging that full six-model ablation replication remains future work (§11).

---

## 8. Diagnostics: SAF, HSI, TCI, EC

- **Steering Activation Fidelity (SAF)** — cosine similarity between a response's pooled activation and `v_s` (Eq. 11).
- **Hallucination Severity Index (HSI)** — fraction of response entities absent from the input (Eq. 12); 0 = fully grounded.
- **Tone Consistency Index (TCI)** — directional alignment with the requested style vs. its opposite (Eq. 13).
- **Entity Coverage (EC)** — fraction of KG node values recovered verbatim in the output (Eq. 10); the paper's primary grounding statistic.

A well-steered, well-grounded generation should jointly satisfy `TCI > 0` and `HSI ≈ 0`; §7.6 shows these two objectives are not always jointly attainable at every `α`, which is why the diagnostic pair is reported jointly rather than collapsed into a single blended score.

---

## 9. Figures & Tables Map

| Paper item | What it shows | Where to find / reproduce it |
|---|---|---|
| Fig. 1 | DSR architecture: PCA-based style-direction extraction (top) + end-to-end inference loop (bottom) | `kg_image.png`, any `build_vectors*.ipynb`, `A2A_KG_Llama-2-13b-chat-hf/.../a2a_kg_fixed_memory.ipynb` |
| Fig. 2 | Worked knowledge graphs for three support cases | `kg/kg_001_*.json`–`kg_003_*.json`, `kg/kg.py` |
| Fig. 3 | SCE per model (style-execution consistency) | `Evaluation/Evaluation/eval_3_style_fidelity.png`, any `a2a_unified_steering_analysis.ipynb` |
| Fig. 4 | Entity coverage & placeholder leakage across all 600 cases | `Evaluation/Evaluation/eval_4_factual_grounding.png`, `Evaluation/Evaluation/metrics_all_models_combined.csv` |
| Fig. 5 | ANOVA partial-η² variance decomposition | `DSR_Supplementary_Analysis/figA1_variance_decomposition.png`, `tableA1_variance_decomposition.csv` |
| Fig. 6 | Zero-inflation analysis, per entity type | `DSR_Supplementary_Analysis/figB1_zero_inflation.png`, `tableB1_entity_sparsity.csv` |
| Fig. 7 | Style/grounding metrics vs. model scale (LLaMA-2 vs. LLaMA-3) | `Evaluation/Evaluation/A2A_KG_MultiModel_Evaluation (1).ipynb`, `leaderboard_summary.csv` |
| Fig. 8 | Layer-wise separability (LLaMA-3.2-3B-Instruct) | `A2A_KG_Llama-3.2-3B-Instruct/.../a2a_unified_steering_analysis.ipynb` |
| Fig. 9 | Steering-strength (`α`) sweep, TCI vs. HSI (LLaMA-3.2-3B-Instruct) | same notebook as Fig. 8, plus `alpha_sweep.png` |
| Table I | Notation reference | paper manuscript, §III (not currently hosted in this repo — see [§16 Contact](#16-contact)) |
| Table II | KG structural statistics (per model) | reproduced in [§7.3](#73-structural-invariance-of-the-knowledge-graph) |
| Table III | KG-gating ablation, LLaMA-2-7B-chat | `Comparision_KG-No-KG/Comparision_KG-No-KG/Llama-2-7b-chat-hf/` (`kg_vs_ao_FINAL_SUMMARY.csv`, `kg_vs_ao_statistical_tests.csv`) |
| Table IV | Per-model, per-tone style/grounding scores | `Evaluation/Evaluation/metrics_<model>.csv` for all six models |
| Table V | Per-model style discrimination, entity coverage, SCE, readability effect sizes | `Evaluation/Evaluation/statistical_tests_all_models.csv`, `tableS4_final_summary.csv` |

---

## 10. Running the Pipeline

Each `A2A_KG_<model>/` directory is self-contained. From inside it:

```bash
# Estimate and cache the per-style steering vectors (PCA estimator) — run once per model
jupyter nbconvert --to notebook --execute build_vectors.ipynb

# Batch mode — runs all cases, writes outputs/results_<timestamp>.jsonl and outputs/knowledge_graphs/*.json
python main.py batch

# Single-scenario mode — one case, printed to console, not persisted
python main.py battery_issue     # | wrong_item | billing_error | delivery_delay
```

`main.py` is organized as six logical modules concatenated into one file for ease of distribution:

| Section | Origin module | Responsibility |
|---|---|---|
| 1 | `a2a_types.py` | A2A message/task/artifact schema (Pydantic) |
| 2 | `ideology_kg.py` | KG construction & salience scoring (Algorithm 1) |
| 3 | `style_vectors.py` | Contrastive pair bank, PCA / mean-diff / logistic-regression estimators (Eqs. 5–7) |
| 4 | `customer_agent.py` | A2A customer-side agent (Groq, temp 0.7) |
| 5 | `support_agent.py` | A2A support-context agent + steered generation hook (Groq temp 0.3 / local steered model) |
| 6 | `orchestrator.py` | End-to-end Defactualize–Steer–Rehydrate orchestration (Algorithm 2), batch runner, JSONL logging |

The 600-case evaluation corpus underlying all tables above is itself generated by an agent-to-agent (A2A) pipeline using two LLaMA-3.3-70B instances (Groq API) as customer and support-context agents, extending the corpus-construction approach of prior work on calendar-driven structured communication [8].

Diagnostics notebooks:

- `a2a_kg_fixed_memory.ipynb` — single worked case end-to-end (the Fig. 1 example: *"Alex, order ORD-1234 delayed"*).
- `a2a_unified_steering_analysis.ipynb` — layer-wise separability probe (Fig. 8) and α-sweep over TCI/HSI (Fig. 9).

---

## 11. Limitations

- **Modest absolute coverage.** Entity-coverage improvement is statistically significant but numerically modest (*d* = 0.225); absolute recovery ranges from low single digits to low tens of percent depending on model (Table V) — DSR improves, but does not guarantee, complete factual grounding.
- **Single-model ablation.** The 100-case ablation effect size is a single-model (LLaMA-2-7B-chat) estimate. A LLaMA-2-vs-LLaMA-3 sub-corpus check found no significant cross-family difference (*p* > 0.17), which is reassuring but is not a substitute for a full six-model ablation, which remains future work.
- **Not a retrieval substitute.** DSR assumes all entities requiring preservation are already present in the input — it is complementary to, not a substitute for, retrieval-augmented methods that introduce external knowledge [6].
- **No persistent memory yet.** The current KG is rebuilt and discarded per request; a persistent, indexed multi-turn agent memory — in the spirit of MemGPT or generative-agent memory streams — is an open extension point.
- **Shallow, closed ontology.** `T` is intentionally shallow (6 types, `|V| ≤ 10`) so that defactualization and rehydration form a closed, enumerable bijection rather than an open-vocabulary problem. This bounds graph size at the cost of richer relational modeling (e.g., multi-hop product–component links), and produces the SENTIMENT-never-recovered / URGENCY-most-recoverable asymmetry documented in [§7.5](#75-per-entity-type-recovery).
- **Steering-operator artifacts are inherited, not fixed.** The non-monotonic TCI / saturating HSI region found in two of six models (§7.6) is a property of the underlying activation-steering operator, which DSR wraps but does not modify — DSR cannot correct steering-level pathologies, only gate which spans the steering operator ever touches.

---

## 12. Citation

```bibtex
@article{shrivastava2026kggated,
  title   = {Knowledge-Graph-Gated Defactualization for Style-Controllable and
             Fact-Preserving Generation in Agentic Conversational AI},
  author  = {Shrivastava, Tanmay Kumar and Nandu, Darsh Rohit and Mundotiya, Rajesh Kumar},
  journal = {IEEE Transactions on Knowledge and Data Engineering, Special Issue:
             Data and Knowledge Empowered Generative AI},
  note    = {Submitted},
  institution = {MATRA Lab, Indian Institute of Technology Bhilai}
}
```

Related prior work (calendar-domain instantiation of the Defactualize–Steer–Rehydrate principle) [8]:

```bibtex
@inproceedings{shrivastava2026disentangling,
  title     = {Disentangling style and semantics for calendar-driven text generation:
               A knowledge graph-guided activation steering approach},
  author    = {Shrivastava, Tanmay Kumar and Bajpai, A. and Mundotiya, Rajesh Kumar},
  booktitle = {Advances in Knowledge Discovery and Data Mining (PAKDD 2026)},
  series    = {Lecture Notes in Computer Science},
  volume    = {16600},
  pages     = {590--602},
  year      = {2026},
  publisher = {Springer, Singapore}
}
```

---

## 13. License

Intended to be released under the MIT License. **A `LICENSE` file has not yet been added to this repository** — add one at the repo root (GitHub: *Add file → Create new file → LICENSE*, choose the MIT template) so this section links correctly and the terms are legally binding.

## 14. Acknowledgments

MATRA Lab (Multimodal and Multilingual AI for Translational Research and Applications), Department of Computer Science and Engineering, IIT Bhilai. Portions of this work benefited from the supervised use of Claude and Gemini in drafting and editing. The authors are grateful to the open-source community for tools enabling this research, and have no competing interests to declare.

## 15. Key References

Just the load-bearing citations — the methods DSR directly builds on or benchmarks against. Numbered to match the paper's own bibliography. The manuscript PDF is not currently hosted in this repository (submitted, under review) — contact the authors ([§16](#16-contact)) for the complete reference list, or check back once an arXiv preprint or camera-ready PDF is added.

1. N. Rimsky, N. Gabrieli, J. Schulz, M. Tong, E. Hubinger, A. Turner, "Steering Llama 2 via contrastive activation addition," *ACL 2024*, pp. 15504–15522.
2. A. Zou et al., "Representation engineering: A top-down approach to AI transparency," *arXiv:2310.01405*, 2023.
6. P. Lewis et al., "Retrieval-augmented generation for knowledge-intensive NLP tasks," *NeurIPS 33*, 2020, pp. 9459–9474.
8. T. K. Shrivastava, A. Bajpai, R. K. Mundotiya, "Disentangling style and semantics for calendar-driven text generation: A knowledge graph-guided activation steering approach," *PAKDD 2026*, LNCS vol. 16600, Springer, pp. 590–602.
9. H. Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models," *arXiv:2307.09288*, 2023.
10. A. Dubey et al., "The Llama 3 Herd of Models," *arXiv:2407.21783*, 2024.

## 16. Contact

- Corresponding author: rmundotiya@iitbhilai.ac.in
- First author: tanmayku@iitbhilai.ac.in
- Repository: [github.com/Tanmay-IITDSAI/KG-Gated-Defactualization](https://github.com/Tanmay-IITDSAI/KG-Gated-Defactualization)
