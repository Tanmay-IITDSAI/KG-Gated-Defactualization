# Knowledge-Graph-Gated Defactualization (DSR) for Style-Controllable and Fact-Preserving Generation in Agentic Conversational AI

[![Paper](https://img.shields.io/badge/IEEE%20TKDE-Special%20Issue%20(Submitted)-blue)](paper/Knowledge-Graph-Gated_Defactualization_for_Style-Controllable_and_Fact-Preserving_Generation_in_Agentic_Conversational_AI.pdf)
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
15. [References](#15-references)
16. [Contact](#16-contact)

---

## 1. Overview

Activation steering can change *how* a frozen LLM writes (tone, register) without any fine-tuning [1]–[4], but it has no built-in notion of *what must not change* — order IDs, product names, customer names, and other verifiable facts can be paraphrased, dropped, or stylistically corrupted along with everything else in the residual stream. We call this failure mode **semantic leakage**: style and content signals occupy the same residual-stream representation, so a steering vector applied uniformly across token positions can drag factual spans toward the requested register [16].

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
| Entity-coverage gain, KG-gated DSR vs. activation-only (AO) baseline | Cohen's *d* = 0.225, *p*₍Bonf₎ = 1.0 × 10⁻⁴ (Welch *t*-test [44], Bonferroni-corrected [42]) | Table III, single-model (LLaMA-2-7B-chat) 100-case paired ablation |
| Absolute entity coverage, AO baseline | **0.0000** (empathetic branch) | Table III |
| Absolute entity coverage, DSR (empathetic branch / pooled both styles, same model+cases) | **0.0056** / **0.0193** | Table III, §VI-D |
| Absolute entity coverage, full 6-model × 600-case study (per-model mean, pooled styles) | 0.012 – 0.037 (peaks at LLaMA-3.1-8B-Instruct) | Table V |
| Placeholder leakage (raw `<TYPE>` token reaching the user) | **0.000** across all 1,200 generations and all 6 models | Table III, Fig. 4 |
| Activation-steering path success rate (fallback never triggered) | **100%** across all six models | §IV, §VII |
| Style-fidelity metrics (empathy, formality, style discrimination, TTR, ROUGE-1 divergence) under KG-gating vs. AO | Statistically unchanged (n.s. after Bonferroni correction) | Table III |
| Sole significant style-side difference, KG-gated vs. AO | Response length — KG-gated responses shorter on average (*p*₍Bonf₎ < 0.05) | Table III |
| KG structural invariance across all 6 host models | Largest cross-model spread < 3% of the mean, for node count, edge count, density, and mean salience | Table II |
| Urgency-level trend on entity coverage | No monotonic trend (Jonckheere–Terpstra [43], *z* = −1.24, *p* = 0.21) — bottleneck is surface realization, not KG extraction | §VI-D |
| LLaMA-2 vs. LLaMA-3 family comparison (*n* = 300 each) | No significant cross-family difference in coverage, readability, or style discrimination (*p* > 0.17); only Flesch Ease differs significantly | §VII |

> **Read this carefully:** the *d* = 0.225 effect is real and statistically robust (it survives Bonferroni correction across the full metric battery in Table III), but it is a **modest, single-model estimate**, and absolute entity-coverage values across the full study remain in the low single digits to low tens-of-percent range (Table V) — DSR measurably increases verified-entity preservation over a steering-only baseline, it does **not** claim to solve factual grounding outright. See [§11 Limitations](#11-limitations) and [§7.1](#71-entity-coverage--placeholder-integrity) for the full picture, including why coverage and leakage are different (and not equally hard) problems.

---

## 2. Why DSR? Positioning Against Fine-Tuning, RAG, and Steering-Only Methods

| Approach | Modifies model weights? | Controls style | Preserves exact entity values | Mechanism |
|---|---|---|---|---|
| Fine-tuning / LoRA [11] | Yes (one run per style × model) | Yes, baked into weights | No explicit guarantee | Gradient update |
| RLHF [12] | Yes | Indirect (reward-shaped) | No explicit guarantee | Reward-model-guided policy update |
| RAG [6] / GraphRAG [19] | No | No | Improves grounding by conditioning on retrieved evidence, but the generator remains free to paraphrase or drop entities inside it [13] | Retrieval into context |
| Activation steering / representation engineering [1]–[4], [22], [23] | No | Yes | Not addressed — steering is content-agnostic, causing semantic leakage [16] | Residual-stream perturbation, `h'_L = h_L + α·v_s` |
| Post-hoc hallucination detection [7], [32], [34] | No | N/A | Detects unsupported claims after the fact; does not shape the generation trajectory | Semantic entropy / self-consistency / atomic fact-checking |
| **DSR (this repository)** | **No** | **Yes** (inherits whichever steering operator is plugged in) | **Deterministic mask before steering + salience-ranked rehydration after decoding** (Eqs. 4, 9) | KG-gated wrapper around an unmodified steering operator |

DSR is complementary to, not a replacement for, knowledge-graph-augmented LLM methods that focus on *retrieval* — i.e., deciding what to condition on [5], [17], [18], [46], [47] — since those methods do not engage with the internal representation at all. It is likewise complementary to RAG [6], [21]: DSR assumes the entities requiring preservation are already present in the input message, and does not introduce external knowledge. Agent-memory frameworks such as ReAct [28], MemGPT [29], and generative agents [30] motivate the natural extension of DSR's per-request, ephemeral KG into a persistent, indexed store for multi-turn grounding — explicitly named as future work (§VII of the paper).

---

## 3. Repository Structure

> This release ships one fully populated model run (`A2A_KG_Llama-2-13b-chat-hf/`) as a worked, inspectable example of every artifact the pipeline produces. The remaining five `A2A_KG_<model>/` directories follow an **identical internal layout** and are populated by re-running `main.py` against each model per [Section 10](#10-running-the-pipeline).

```
KG-Gated-Defactualization/
├── README.md                                  ← this file
├── LICENSE
├── requirements.txt
├── .env.example                                ← template for required environment variables
│
├── A2A_KG_Llama-2-7b-base/                     ← per-model run directory (same layout as below)
├── A2A_KG_Llama-2-7b-chat-hf/
├── A2A_KG_Llama-2-13b-chat-hf/                  ← included worked example
│   └── A2A_KG_Llama-2-13b-chat-hf/
│       ├── main.py                              ← single-file pipeline (see Section 10 for module map)
│       ├── build_vectors.ipynb                  ← estimates & caches v_s per (style, model) — Eqs. 5–7
│       ├── a2a_kg_fixed_memory.ipynb            ← interactive single-case walkthrough (Fig. 1 example)
│       ├── a2a_unified_steering_analysis.ipynb  ← layer-separability + alpha-sweep diagnostics (Fig. 8, 9)
│       ├── alpha_sweep.png                      ← cached diagnostic figure
│       ├── .style_cache/                        ← cached, reusable steering vectors (one per style)
│       │   ├── style_vec_empathetic_pca.pkl
│       │   └── style_vec_formal_pca.pkl
│       └── outputs/
│           ├── knowledge_graphs/                ← one typed KG per case (Stage 2–3 output, Fig. 2)
│           │   ├── kg_001_battery_issue_ORD-7741_Priya_Sharma.json
│           │   ├── kg_002_battery_issue_ORD-7742_Leo_Chen.json
│           │   └── ...                          ← kg_NNN_<scenario>_<order_id>_<customer>.json
│           └── results_<timestamp>.jsonl        ← per-case generation + metric records (batch mode)
│
├── A2A_KG_Llama-3.1-8B-Instruct/
├── A2A_KG_Llama-3.2-1B-Instruct/
├── A2A_KG_Llama-3.2-3B-Instruct/
│
├── ablation/                                    ← 100-case KG-gating ablation (Table III, §VI-D)
│   ├── activation_only/                         ← AO baseline generations (no KG masking)
│   └── kg_gated/                                ← full DSR generations, identical (L, α, v_s)
│
├── figures/                                     ← Fig. 1–9 source notebooks / exported assets
│
└── paper/
    └── Knowledge-Graph-Gated_Defactualization_for_Style-Controllable_and_Fact-Preserving_Generation_in_Agentic_Conversational_AI.pdf
```

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
| 1–2 | Input Processing → Knowledge Extraction | Layered regex (order IDs, products, currency) → spaCy NER [35] (names, issue spans) → lexicon classifiers (urgency, sentiment); highest-confidence-wins, no model training. |
| 3 | KG Construction | Builds typed graph `G=(V,E)`, `|V|≤10`, salience-scored per Eqs. 1–2, terminates in `O(n+|V|+|E|)`. Ontology design follows established graph-quality-management guidance for keeping ontologies enumerable as they scale [45]. |
| 4 | Defactualization | Case-insensitive, longest-match-first substitution of every node value with a typed placeholder `<TYPE>` (Eq. 4) **before** any activation is read or any steering vector is built. |
| 5 | Steered Generation | Contrastive steering vector `v_s` (PCA estimator [27], Eq. 5) added to the residual stream at a single layer `L`: `h′_L(t) = h_L(t) + α·v_s` (Eq. 8); greedy decode [37], repetition penalty 1.3 [38], `max_new_tokens=120`. A prompt-steering fallback fires only on malformed output (never triggered in the 1,200-generation study). |
| 6 | Rehydration & Verification | Each placeholder is deterministically replaced with the highest-salience candidate value of its type (Eq. 9) — a lookup, not a generative step. Unmatched placeholders are stripped, never left dangling. |
| 7 | Output | Style-controlled, fact-preserving response; the same defactualized input branches into both `empathetic` and `formal` variants. |

Three estimators for the steering direction `v_s` are implemented and cached — PCA [27] (Eq. 5), mean-difference (Eq. 6), and logistic-regression [25] (Eq. 7). On the layer-separability probe (LLaMA-3.2-3B-Instruct, layers 0–27), the **PCA estimator gave the lowest mean cosine similarity between the empathetic and formal directions (cos = 0.41)**, vs. cos = 0.53 (mean-difference) and cos = 0.49 (logistic regression) — i.e., the most mutually distinguishable style directions of the three. All paper-reported results use the PCA estimator on this basis; the other two are released as cached supplementary artifacts for independent comparison.

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

---

## 6. Models Evaluated

| Family | Variant | Params | Regime | Mean Entity Coverage (pooled, Table V) |
|---|---|---|---|---|
| LLaMA 2 [9] | 7B | 7B | Base | 0.016 |
| LLaMA 2 [9] | 7B-chat | 7B | Chat | 0.017 |
| LLaMA 2 [9] | 13B-chat | 13B | Chat | 0.023 |
| LLaMA 3.1 [10] | 8B-Instruct | 8B | Instruct | **0.037** (highest) |
| LLaMA 3.2 [10] | 1B-Instruct | 1B | Instruct | 0.012 (lowest) |
| LLaMA 3.2 [10] | 3B-Instruct | 3B | Instruct | 0.016 |

No weights are modified for any model; `v_s` is estimated once per (style, model) pair and cached (~32 KB total for two styles at `d=4096`, roughly seven orders of magnitude below a single model's bf16 weight footprint), then reused unchanged across every incoming message. Neither stylistic fidelity nor factual grounding scales monotonically with parameter count within the 1B–13B range studied — several metrics peak at intermediate model sizes, and model family / training regime (base vs. chat vs. instruct) appear to matter at least as much as raw scale (Fig. 7).

---

## 7. Results

This section expands the headline table in [§1](#1-overview) with the full battery of analyses from the paper (§VI).

### 7.1 Entity Coverage & Placeholder Integrity

- **Coverage gain is real but modest.** The KG-gating ablation (Table III) shows entity coverage rising from an AO baseline of exactly **0.0000** to **0.0056** (empathetic branch) / **0.0193** (pooled) under DSR, a statistically significant gain (Welch *t*-test [44], Bonferroni-corrected [42], *d* = 0.225, *p*₍Bonf₎ = 1.0 × 10⁻⁴) that survives correction across the entire metric battery in Table III — while every style-fidelity metric in the same table (empathy, formality, style discrimination, lexical diversity, readability, clarity, ROUGE-1 divergence [39]) remains statistically unchanged.
- **Leakage is the stronger result.** Placeholder leakage — a raw, unresolved `<TYPE>` token reaching the end user — is **exactly 0.0 across all 1,200 generations and all six architecturally distinct host models** (Fig. 4). Because the activation-steering path never produced a malformed response across the corpus (the prompt-steering fallback was never triggered), this confirms the modest coverage numbers stem from the steered model *omitting* placeholders during generation — not from the deterministic rehydration step (Eq. 9) failing to restore values it was given.
- **No monotonic urgency trend.** A Jonckheere–Terpstra trend test [43] across the three urgency levels found no monotonic relationship with coverage (*z* = −1.24, *p* = 0.21), indicating the coverage bottleneck lies in surface realization during decoding, not in KG extraction quality.

### 7.2 Style Fidelity

Table IV (per-model, per-tone, 1,200 generations) shows lexicon-based empathy and formality scores stay modest across all six models, while **readability (Flesch Ease [40]) is the most consistent empathetic–formal separator**, and low cross-tone ROUGE-1/ROUGE-L overlap [39] confirms the two styles differ lexically rather than superficially. Readability effect sizes are significant across all models after Bonferroni correction [42] (paired *t*-test on per-case empathetic − formal differences), except LLaMA-2-7B-base, which shows the Flesch Ease effect in the opposite direction. Style Calibration Error (SCE) — mean absolute deviation of each generation's style score from the model's own 90th-percentile style ceiling — shows no monotonic scaling with model size (Fig. 3).

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
| Fig. 1 | DSR architecture: PCA-based style-direction extraction (top) + end-to-end inference loop (bottom) | `figures/`, `build_vectors.ipynb`, `a2a_kg_fixed_memory.ipynb` |
| Fig. 2 | Worked knowledge graphs for three support cases | `outputs/knowledge_graphs/kg_001_*.json`–`kg_003_*.json`, `figures/` |
| Fig. 3 | SCE per model (style-execution consistency) | `a2a_unified_steering_analysis.ipynb` |
| Fig. 4 | Entity coverage & placeholder leakage across all 600 cases | `outputs/results_<timestamp>.jsonl` aggregation scripts |
| Fig. 5 | ANOVA partial-η² variance decomposition | `outputs/results_<timestamp>.jsonl` aggregation scripts |
| Fig. 6 | Zero-inflation analysis, per entity type | `outputs/results_<timestamp>.jsonl` aggregation scripts |
| Fig. 7 | Style/grounding metrics vs. model scale (LLaMA-2 vs. LLaMA-3) | cross-model aggregation over all six `A2A_KG_<model>/` directories |
| Fig. 8 | Layer-wise separability (LLaMA-3.2-3B-Instruct) | `a2a_unified_steering_analysis.ipynb` |
| Fig. 9 | Steering-strength (`α`) sweep, TCI vs. HSI (LLaMA-3.2-3B-Instruct) | `a2a_unified_steering_analysis.ipynb`, `alpha_sweep.png` |
| Table I | Notation reference | `paper/` PDF, §III |
| Table II | KG structural statistics (per model) | reproduced in [§7.3](#73-structural-invariance-of-the-knowledge-graph) |
| Table III | KG-gating ablation, LLaMA-2-7B-chat | `ablation/activation_only/`, `ablation/kg_gated/` |
| Table IV | Per-model, per-tone style/grounding scores | full 600-case study, all six `A2A_KG_<model>/` directories |
| Table V | Per-model style discrimination, entity coverage, SCE, readability effect sizes | reproduced in part in [§6](#6-models-evaluated) and [§7](#7-results) |

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
| 3 | `style_vectors.py` | Contrastive pair bank, PCA [27] / mean-diff / logistic-regression [25] estimators (Eqs. 5–7) |
| 4 | `customer_agent.py` | A2A customer-side agent (Groq, temp 0.7) |
| 5 | `support_agent.py` | A2A support-context agent + steered generation hook (Groq temp 0.3 / local steered model) |
| 6 | `orchestrator.py` | End-to-end Defactualize–Steer–Rehydrate orchestration (Algorithm 2), batch runner, JSONL logging |

The 600-case evaluation corpus underlying all tables above is itself generated by an agent-to-agent (A2A) pipeline using two LLaMA-3.3-70B instances (Groq API) as customer and support-context agents [10], extending the corpus-construction approach of prior work on calendar-driven structured communication [8].

Diagnostics notebooks:

- `a2a_kg_fixed_memory.ipynb` — single worked case end-to-end (the Fig. 1 example: *"Alex, order ORD-1234 delayed"*).
- `a2a_unified_steering_analysis.ipynb` — layer-wise separability probe (Fig. 8) and α-sweep over TCI/HSI (Fig. 9).

---

## 11. Limitations

- **Modest absolute coverage.** Entity-coverage improvement is statistically significant but numerically modest (*d* = 0.225); absolute recovery ranges from low single digits to low tens of percent depending on model (Table V) — DSR improves, but does not guarantee, complete factual grounding.
- **Single-model ablation.** The 100-case ablation effect size is a single-model (LLaMA-2-7B-chat) estimate. A LLaMA-2-vs-LLaMA-3 sub-corpus check found no significant cross-family difference (*p* > 0.17), which is reassuring but is not a substitute for a full six-model ablation, which remains future work.
- **Not a retrieval substitute.** DSR assumes all entities requiring preservation are already present in the input — it is complementary to, not a substitute for, retrieval-augmented methods that introduce external knowledge [6], [19], [21].
- **No persistent memory yet.** The current KG is rebuilt and discarded per request; a persistent, indexed multi-turn agent memory — in the spirit of MemGPT [29] or generative-agent memory streams [30] — is an open extension point.
- **Shallow, closed ontology.** `T` is intentionally shallow (6 types, `|V| ≤ 10`) so that defactualization and rehydration form a closed, enumerable bijection rather than an open-vocabulary problem [45]. This bounds graph size at the cost of richer relational modeling (e.g., multi-hop product–component links), and produces the SENTIMENT-never-recovered / URGENCY-most-recoverable asymmetry documented in [§7.5](#75-per-entity-type-recovery).
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

Released under the [MIT License](LICENSE).

## 14. Acknowledgments

MATRA Lab (Multimodal and Multilingual AI for Translational Research and Applications), Department of Computer Science and Engineering, IIT Bhilai. Portions of this work benefited from the supervised use of Claude and Gemini in drafting and editing. The authors are grateful to the open-source community for tools enabling this research, and have no competing interests to declare.

## 15. References

Numbered to match the paper's bibliography; only references cited in this README are listed here. See the [full paper](paper/Knowledge-Graph-Gated_Defactualization_for_Style-Controllable_and_Fact-Preserving_Generation_in_Agentic_Conversational_AI.pdf) for the complete 54-item list.

1. N. Rimsky, N. Gabrieli, J. Schulz, M. Tong, E. Hubinger, A. Turner, "Steering Llama 2 via contrastive activation addition," *ACL 2024*, pp. 15504–15522.
2. A. Zou et al., "Representation engineering: A top-down approach to AI transparency," *arXiv:2310.01405*, 2023.
3. K. Konen et al., "Style vectors for steering generative large language models," *Findings of ACL: EACL 2024*, pp. 782–802.
4. J. Zhang et al., "Personalized text generation with contrastive activation steering," *ACL 2025*, pp. 7128–7141.
5. Q. Wang, Z. Mao, B. Wang, L. Guo, "Knowledge graph embedding: A survey of approaches and applications," *IEEE TKDE*, 29(12), 2017, pp. 2724–2743.
6. P. Lewis et al., "Retrieval-augmented generation for knowledge-intensive NLP tasks," *NeurIPS 33*, 2020, pp. 9459–9474.
7. S. Farquhar, J. Kossen, L. Kuhn, Y. Gal, "Detecting hallucinations in large language models using semantic entropy," *Nature*, 630(8017), 2024, pp. 625–630.
8. T. K. Shrivastava, A. Bajpai, R. K. Mundotiya, "Disentangling style and semantics for calendar-driven text generation: A knowledge graph-guided activation steering approach," *PAKDD 2026*, LNCS vol. 16600, Springer, pp. 590–602.
9. H. Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models," *arXiv:2307.09288*, 2023.
10. A. Dubey et al., "The Llama 3 Herd of Models," *arXiv:2407.21783*, 2024.
11. E. J. Hu et al., "LoRA: Low-rank adaptation of large language models," *ICLR 2022*.
12. L. Ouyang et al., "Training language models to follow instructions with human feedback," *NeurIPS 35*, 2022, pp. 27730–27744.
13. F. Shi et al., "Large language models can be easily distracted by irrelevant context," *ICML 2023*, pp. 31210–31227.
16. E. Sheng, K.-W. Chang, P. Natarajan, N. Peng, "Towards controllable biases in language generation," *Findings of EMNLP 2020*, pp. 3239–3254.
17. S. Pan, L. Luo, Y. Wang, C. Chen, J. Wang, X. Wu, "Unifying large language models and knowledge graphs: A roadmap," *IEEE TKDE*, 36(7), 2024, pp. 3580–3599.
18. J. Sun et al., "Think-on-graph: Deep and responsible reasoning of LLMs on knowledge graphs," *ICLR 2024*.
19. D. Edge et al., "From local to global: A graph RAG approach to query-focused summarization," *arXiv:2404.16130*, 2024.
21. Y. Gao et al., "Retrieval-augmented generation for large language models: A survey," *arXiv:2312.10997*, 2024.
22. K. Park, Y. Choe, V. Veitch, "The linear representation hypothesis and the geometry of large language models," *arXiv:2311.03658*, 2023.
23. N. Elhage et al., "A mathematical framework for transformer circuits," *Transformer Circuits Thread*, 2021.
25. G. Alain, Y. Bengio, "Understanding intermediate layers using linear classifier probes," *arXiv:1610.01644*, 2016.
27. J. Hewitt, C. D. Manning, "A structural probe for finding syntax in word representations," *NAACL 2019*, pp. 4129–4138.
28. S. Yao et al., "ReAct: Synergizing reasoning and acting in language models," *ICLR 2023*.
29. C. Packer et al., "MemGPT: Towards LLMs as operating systems," *arXiv:2310.08560*, 2023.
30. J. S. Park et al., "Generative agents: Interactive simulacra of human behavior," *ACM UIST 2023*, pp. 1–22.
32. Z. Ji et al., "Survey of hallucination in natural language generation," *ACM Comput. Surv.*, 55(12), 2023, pp. 1–38.
34. S. Min et al., "FActScore: Fine-grained atomic evaluation of factual precision in long form text generation," *EMNLP 2023*, pp. 12076–12100.
35. M. Honnibal, I. Montani, S. Van Landeghem, A. Boyd, "spaCy: Industrial-strength natural language processing in Python," *Zenodo*, 2020.
37. A. Holtzman, J. Buys, L. Du, M. Forbes, Y. Choi, "The curious case of neural text degeneration," *ICLR 2020*.
38. N. S. Keskar, B. McCann, L. R. Varshney, C. Xiong, R. Socher, "CTRL: A conditional transformer language model for controllable generation," *arXiv:1909.05858*, 2019.
39. C.-Y. Lin, "ROUGE: A package for automatic evaluation of summaries," *Workshop on Text Summarization Branches Out, ACL 2004*, pp. 74–81.
40. R. Flesch, "A new readability yardstick," *Journal of Applied Psychology*, 32(3), 1948, pp. 221–233.
42. O. J. Dunn, "Multiple comparisons among means," *J. American Statistical Association*, 56(293), 1961, pp. 52–64.
43. A. R. Jonckheere, "A distribution-free k-sample test against ordered alternatives," *Biometrika*, 41(1/2), 1954, pp. 133–145.
44. B. L. Welch, "The generalization of 'Student's' problem when several different population variances are involved," *Biometrika*, 34(1/2), 1947, pp. 28–35.
45. B. Xue, L. Zou, "Knowledge graph quality management: A comprehensive survey," *IEEE TKDE*, 35(5), 2023, pp. 4969–4988.
46. M. Yasunaga, H. Ren, A. Bosselut, P. Liang, J. Leskovec, "QA-GNN: Reasoning with language models and knowledge graphs for question answering," *NAACL 2021*, pp. 535–546.
47. Q. Guo, F. Zhuang, C. Qin, H. Zhu, X. Xie, H. Xiong, Q. He, "A survey on knowledge graph-based recommender systems," *IEEE TKDE*, 34(8), 2022, pp. 3549–3568.

## 16. Contact

- Corresponding author: rmundotiya@iitbhilai.ac.in
- First author: tanmayku@iitbhilai.ac.in
- Repository: [github.com/Tanmay-IITDSAI/KG-Gated-Defactualization](https://github.com/Tanmay-IITDSAI/KG-Gated-Defactualization)
