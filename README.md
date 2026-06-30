# Knowledge-Graph-Gated Defactualization (DSR)

**Style-Controllable and Fact-Preserving Generation in Agentic Conversational AI**

[![Paper](https://img.shields.io/badge/IEEE%20TKDE-Special%20Issue-blue)](#citation)
[![License](https://img.shields.io/badge/license-MIT-green)](#license)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#requirements)
[![Models](https://img.shields.io/badge/LLaMA-2%20%26%203-orange)](#models-evaluated)

Official code, cached steering vectors, knowledge-graph artifacts, and evaluation scripts for:

> T. K. Shrivastava, D. R. Nandu, and R. K. Mundotiya, **"Knowledge-Graph-Gated Defactualization for Style-Controllable and Fact-Preserving Generation in Agentic Conversational AI,"** *IEEE Transactions on Knowledge and Data Engineering, Special Issue: Data and Knowledge Empowered Generative AI* (submitted).

Affiliation: MATRA Lab (Multimodal and Multilingual AI for Translational Research and Applications), Department of Computer Science and Engineering, IIT Bhilai.

---

## 1. Overview

Activation steering can change *how* a frozen LLM writes (tone, register) without any fine-tuning, but it has no built-in notion of *what must not change* — order IDs, product names, customer names, and other verifiable facts can be paraphrased, dropped, or stylistically corrupted along with everything else in the residual stream. This repository implements **Defactualize–Steer–Rehydrate (DSR)**: a typed, salience-weighted knowledge graph (KG) that wraps a steering operator as a **mask-and-verification contract**, without ever modifying the steering vector, layer, or strength itself.

```
Raw message → [1] Input Processing → [2] Knowledge Extraction → [3] KG Construction
            → [4] Defactualization (mask entities → typed placeholders)
            → [5] Steered Generation (activation steering at layer L)
            → [6] Rehydration & Verification (deterministic restore from KG)
            → [7] Output (style-controlled, fact-preserving response)
```

The KG never touches the steering direction `v_s`, layer `L`, or strength `α` — it only controls *which spans the steering operator is ever exposed to*, and *which verified values are deterministically written back after decoding*.

### Key results (full paper for details)

| Metric | Result |
|---|---|
| Entity-coverage gain, KG-gated DSR vs. activation-only baseline | Cohen's *d* = 0.225, *p*₍Bonf₎ = 1.0 × 10⁻⁴ |
| Models evaluated | 6 LLaMA-family models, 1B–13B params |
| Evaluation corpus | 600 A2A-generated support cases → 1,200 generations (+100-case KG ablation) |
| Placeholder leakage (raw `<TYPE>` token reaching the user) | **0.0** across all 1,200 generations |
| Activation-steering path success rate (no fallback triggered) | 100% across all six models |
| Style-fidelity metrics (empathy, formality, ROUGE divergence, TTR) under KG-gating | Statistically unchanged vs. steering-only baseline |

---

## 2. Repository structure

> This release ships one fully populated model run (`A2A_KG_Llama-2-13b-chat-hf/`) as a worked, inspectable example of every artifact the pipeline produces. The remaining five `A2A_KG_<model>/` directories follow an **identical internal layout** and are populated by re-running `main.py` against each model per [Section 4](#4-running-the-pipeline).

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
│       ├── main.py                              ← single-file pipeline (see Section 5 for module map)
│       ├── build_vectors.ipynb                  ← estimates & caches v_s per (style, model)
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
├── ablation/                                    ← 100-case KG-gating ablation (Table III, Section VI-D)
│   ├── activation_only/                         ← AO baseline generations (no KG masking)
│   └── kg_gated/                                ← full DSR generations, same (L, α, v_s)
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

`type ∈ T = {CUSTOMER_NAME, ORDER_ID, PRODUCT, ISSUE, URGENCY, SENTIMENT}`; edge weight `w(u,v) = sal(u)·sal(v)` (Eq. 3).

---

## 3. Method summary

| Stage | Component | What it does |
|---|---|---|
| 1–2 | Input Processing → Knowledge Extraction | Layered regex (order IDs, products, currency) → spaCy NER (names, issue spans) → lexicon classifiers (urgency, sentiment); highest-confidence-wins, no model training. |
| 3 | KG Construction | Builds typed graph `G=(V,E)`, `|V|≤10`, salience-scored per Eqs. 1–2, terminates in `O(n+|V|+|E|)`. |
| 4 | Defactualization | Case-insensitive, longest-match-first substitution of every node value with a typed placeholder `<TYPE>` (Eq. 4) **before** any activation is read or any steering vector is built. |
| 5 | Steered Generation | Contrastive steering vector `v_s` (PCA estimator, Eq. 5) added to the residual stream at a single layer `L`: `h′_L(t) = h_L(t) + α·v_s` (Eq. 8); greedy decode, repetition penalty 1.3, `max_new_tokens=120`. A prompt-steering fallback fires only on malformed output (never triggered in the 1,200-generation study). |
| 6 | Rehydration & Verification | Each placeholder is deterministically replaced with the highest-salience candidate value of its type (Eq. 9) — a lookup, not a generative step. Unmatched placeholders are stripped, never left dangling. |
| 7 | Output | Style-controlled, fact-preserving response, identical defactualized input branching into both `empathetic` and `formal` variants. |

### Diagnostics

- **Steering Activation Fidelity (SAF)** — cosine similarity between a response's pooled activation and `v_s` (Eq. 11).
- **Hallucination Severity Index (HSI)** — fraction of response entities absent from the input (Eq. 12); 0 = fully grounded.
- **Tone Consistency Index (TCI)** — directional alignment with the requested style vs. its opposite (Eq. 13).
- **Entity Coverage (EC)** — fraction of KG node values recovered verbatim in the output (Eq. 10); the paper's primary grounding statistic.

---

## 4. Requirements

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

## 5. Running the pipeline

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

Diagnostics notebooks:

- `a2a_kg_fixed_memory.ipynb` — single worked case end-to-end (the Fig. 1 example: *"Alex, order ORD-1234 delayed"*).
- `a2a_unified_steering_analysis.ipynb` — layer-wise separability probe (Fig. 8) and α-sweep over TCI/HSI (Fig. 9).

---

## 6. Models evaluated

| Family | Variant | Params | Regime |
|---|---|---|---|
| LLaMA 2 | 7B | 7B | Base |
| LLaMA 2 | 7B-chat | 7B | Chat |
| LLaMA 2 | 13B-chat | 13B | Chat |
| LLaMA 3.1 | 8B-Instruct | 8B | Instruct |
| LLaMA 3.2 | 1B-Instruct | 1B | Instruct |
| LLaMA 3.2 | 3B-Instruct | 3B | Instruct |

No weights are modified for any model; `v_s` is estimated once per (style, model) pair and cached (~32 KB total for two styles at `d=4096`), then reused unchanged across every incoming message.

---

## 7. Limitations

- Entity-coverage improvement is statistically significant but numerically modest (absolute EC ranges from low single digits to low tens of percent depending on model — see Table V of the paper); DSR improves, but does not guarantee, complete factual grounding.
- The 100-case ablation effect size is a single-model (LLaMA-2-7B-chat) estimate; full six-model ablation replication is future work.
- DSR assumes all entities requiring preservation are already present in the input — it is complementary to, not a substitute for, retrieval-augmented methods that introduce external knowledge.
- The current KG is rebuilt and discarded per request; persistent, indexed multi-turn agent memory is an open extension point.

---

## 8. Citation

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

Related prior work (calendar-domain instantiation of the Defactualize–Steer–Rehydrate principle):

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

## 9. License

Released under the [MIT License](LICENSE).

## 10. Acknowledgments

MATRA Lab (Multimodal and Multilingual AI for Translational Research and Applications), Department of Computer Science and Engineering, IIT Bhilai. Portions of this work benefited from the supervised use of Claude and Gemini in drafting and editing.

## Contact

- Corresponding author: rmundotiya@iitbhilai.ac.in
- First author: tanmayku@iitbhilai.ac.in
