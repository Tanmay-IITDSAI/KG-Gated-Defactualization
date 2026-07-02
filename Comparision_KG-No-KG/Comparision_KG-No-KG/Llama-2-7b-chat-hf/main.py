"""
main.py — A2A Customer Support Pipeline (Single-File Edition)
=============================================================

Runs the full KG-guided activation steering pipeline in one process.
No separate uvicorn servers — everything executes directly.

Usage:
    python main.py [scenario]
    python main.py battery_issue
    python main.py wrong_item
    python main.py billing_error
    python main.py delivery_delay

Available scenarios: battery_issue | wrong_item | billing_error | delivery_delay

Required environment variables (in .env):
    GROQ_API_KEY=gsk_...
    LOCAL_MODEL_NAME=meta-llama/Llama-2-7b-hf   # optional — falls back to Groq
    STEER_LAYER=20
    STEER_ALPHA=15.0
    HF_TOKEN=hf_...   # required only if using Llama-2 activation steering
"""

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — A2A TYPES  (from a2a_types.py)
# ══════════════════════════════════════════════════════════════════════════════

from pydantic import BaseModel, Field
from typing import Any, Literal, Optional
from enum import Enum
import uuid


class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str

class DataPart(BaseModel):
    type: Literal["data"] = "data"
    data: dict[str, Any]
    mimeType: str = "application/json"

class FilePart(BaseModel):
    type: Literal["file"] = "file"
    mimeType: str
    data: str


class Message(BaseModel):
    role: Literal["user", "agent"]
    parts: list[TextPart | DataPart | FilePart]

    def text(self) -> str:
        return " ".join(p.text for p in self.parts if isinstance(p, TextPart))


class Artifact(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    mimeType: str = "application/json"
    parts: list[TextPart | DataPart | FilePart]


class TaskState(str, Enum):
    SUBMITTED = "submitted"
    WORKING   = "working"
    COMPLETED = "completed"
    FAILED    = "failed"
    CANCELLED = "cancelled"

class Task(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sessionId: Optional[str] = None
    state: TaskState = TaskState.SUBMITTED
    message: Message
    artifacts: list[Artifact] = []
    metadata: dict[str, Any] = {}


class AgentCapabilities(BaseModel):
    streaming: bool = False
    pushNotifications: bool = False
    stateTransitionHistory: bool = True

class AgentCard(BaseModel):
    name: str
    description: str
    url: str
    version: str = "1.0.0"
    capabilities: AgentCapabilities = AgentCapabilities()
    skills: list[dict[str, Any]] = []
    bias: Optional[dict[str, Any]] = None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — KNOWLEDGE GRAPH  (from ideology_kg.py)
# ══════════════════════════════════════════════════════════════════════════════

import re
import logging
from dataclasses import dataclass, field as dc_field

logger = logging.getLogger("main")

ENTITY_TYPES = {"ISSUE", "PRODUCT", "ORDER_ID", "CUSTOMER_NAME", "URGENCY", "SENTIMENT"}

PLACEHOLDERS = {
    "ISSUE":         "<ISSUE>",
    "PRODUCT":       "<PRODUCT>",
    "ORDER_ID":      "<ORDER_ID>",
    "CUSTOMER_NAME": "<CUSTOMER_NAME>",
    "URGENCY":       "<URGENCY>",
    "SENTIMENT":     "<SENTIMENT>",
}


@dataclass
class KGNode:
    value: str
    entity_type: str
    salience: float = 1.0


@dataclass
class KGEdge:
    source: str
    target: str
    relation: str
    weight: float = 1.0


@dataclass
class SupportKG:
    nodes: list[KGNode] = dc_field(default_factory=list)
    edges: list[KGEdge] = dc_field(default_factory=list)

    def entities_of_type(self, t: str) -> list[KGNode]:
        return [n for n in self.nodes if n.entity_type == t]

    def add_node(self, value: str, entity_type: str, salience: float = 1.0):
        if not any(n.value == value for n in self.nodes):
            self.nodes.append(KGNode(value, entity_type, salience))

    def add_edge(self, src: str, tgt: str, relation: str, weight: float = 1.0):
        self.edges.append(KGEdge(src, tgt, relation, weight))

    def to_dict(self) -> dict:
        return {
            "nodes": [{"value": n.value, "type": n.entity_type, "salience": n.salience}
                      for n in self.nodes],
            "edges": [{"source": e.source, "target": e.target,
                       "relation": e.relation, "weight": e.weight}
                      for e in self.edges],
        }


# ── spaCy NER bootstrap ───────────────────────────────────────────────────────
# Loaded once at module level; falls back gracefully if not installed.
_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is not None:
        return _nlp
    try:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm")
            logger.info("[KG-NER] spaCy en_core_web_sm loaded")
        except OSError:
            logger.warning("[KG-NER] en_core_web_sm not found — downloading...")
            from spacy.cli import download as spacy_download
            spacy_download("en_core_web_sm")
            _nlp = spacy.load("en_core_web_sm")
            logger.info("[KG-NER] spaCy en_core_web_sm downloaded and loaded")
    except ImportError:
        logger.warning("[KG-NER] spaCy not installed — NER disabled, regex-only fallback active")
        _nlp = False          # False = tried but unavailable
    return _nlp


# ── Regex extraction patterns ─────────────────────────────────────────────────

# ORDER_ID: ORD-1234, TKT-5678, #90123, "order number 7741", etc.
_PAT_ORDER = re.compile(
    r'\b(?:ORD|TKT|REF|ORDER|TICKET)-\s*(\d{3,10})\b'
    r'|\b(?:order|ticket|reference)\s+(?:number|#|no\.?)?\s*(\d{3,10})\b'
    r'|\b#(\d{3,10})\b',
    re.IGNORECASE,
)

# PRODUCT: Quoted names, brand+model patterns, capitalised product names
# Requires quoted name OR article+multi-word caps OR model-number suffix to fire.
# Stopword filter in _regex_extract_product() prevents false positives like "Need", "Regards".
_PAT_PRODUCT = re.compile(
    r'["\']([\w][\w\s\-]{3,40})["\']'                                    # "Quoted Product"
    r'|(?:(?:my|the|a|an|your|our)\s+)'                                       # article required
    r'([A-Z][A-Za-z0-9]{2,}(?:[\s\-][A-Z][A-Za-z0-9]{2,})+'                # >=2 capitalised words
    r'(?:\s+(?:Pro|Plus|Max|Ultra|Lite|v\d+|X\d+|Series\s*\d+))?)'       # optional suffix
    r'|([A-Z][A-Za-z0-9]{2,}[\s\-](?:[A-Z][A-Za-z0-9]{2,}[\s\-])*'      # CamelCase multi-word
    r'(?:Pro|Plus|Max|Ultra|Lite|v\d+|X\d+|\d{3,}))',                       # MUST end in model suffix
    re.MULTILINE,
)

_PRODUCT_STOPWORDS = {
    "regards", "sincerely", "cheers", "thanks", "hello", "dear", "please",
    "need", "want", "have", "this", "that", "your", "order", "issue", "help",
    "team", "support", "service", "company", "product", "item", "package", "best",
}

# CUSTOMER_NAME: sign-off patterns + "my name is" patterns
_PAT_NAME = re.compile(
    r'(?:my name is|i am|i\'m|signed?|regards?|sincerely|cheers|thanks?|best),?\s*'
    r'([A-Z][a-z]{1,20}(?:\s[A-Z][a-z]{1,20}){0,2})',
    re.IGNORECASE,
)

# ISSUE: complaint-describing clauses
_PAT_ISSUE = re.compile(
    r'(?:my\s+)?(?:issue|problem|complaint|concern|bug|error|defect|fault|trouble|difficulty)'
    r'\s+(?:is\s+(?:that\s+)?|with\s+|about\s+)?([^.!?\n]{8,120})',
    re.IGNORECASE,
)
# Additional issue cues: "it keeps", "it stopped", "it won't", "I cannot"
_PAT_ISSUE2 = re.compile(
    r'(?:it\s+(?:keeps?|stopped?|won\'t|doesn\'t|is\s+not)|i\s+(?:cannot|can\'t|am\s+unable\s+to))'
    r'\s+([^.!?\n]{5,80})',
    re.IGNORECASE,
)

# URGENCY lexicons
_URGENCY_HIGH   = re.compile(
    r'\b(urgent|urgently|immediately|asap|as soon as possible|critical|emergency|'
    r'right now|broken|unusable|cannot (?:use|work|function)|completely dead|'
    r'stop(?:ped)? working|won\'t turn on|not turning on)\b',
    re.IGNORECASE,
)
_URGENCY_MEDIUM = re.compile(
    r'\b(soon|quickly|still not|still haven\'t|keep(?:s)? happening|repeatedly|'
    r'several times|multiple times|again|ongoing|days? now|week now)\b',
    re.IGNORECASE,
)

# SENTIMENT lexicons
_SENT_ANGRY  = re.compile(
    r'\b(angry|furious|disgusted|outraged|infuriated|livid|appalled|'
    r'terrible|awful|hate|ridiculous|unacceptable|absurd|disgraceful|horrible)\b',
    re.IGNORECASE,
)
_SENT_FRUSTR = re.compile(
    r'\b(frustrated|frustrating|annoyed|annoying|disappointed|disappointing|'
    r'unhappy|upset|exhausted|fed up|sick of|tired of|not impressed)\b',
    re.IGNORECASE,
)
_SENT_POLITE = re.compile(
    r'\b(kindly|please|would appreciate|grateful|thank you|thanks|politely|'
    r'if possible|at your convenience|hope you can)\b',
    re.IGNORECASE,
)
_SENT_POS    = re.compile(
    r'\b(great|appreciate|happy|pleased|love|excellent|wonderful|amazing)\b',
    re.IGNORECASE,
)

# spaCy label → our entity type mapping
_SPACY_PERSON_LABELS = {"PERSON"}
_SPACY_PRODUCT_LABELS = {"PRODUCT", "ORG", "WORK_OF_ART"}

# Common false-positive person names to suppress
_NAME_STOPWORDS = {
    "i", "you", "we", "me", "my", "your", "our", "hi", "hello", "dear",
    "sir", "madam", "team", "support", "help", "customer", "service",
}


# ── NER pipeline ──────────────────────────────────────────────────────────────

def _ner_extract(text: str) -> dict[str, list[tuple[str, float]]]:
    """
    Run spaCy NER on text. Returns a dict mapping our entity types to
    a list of (value, confidence) tuples. Confidence is approximated from
    entity position and label match quality.
    """
    results: dict[str, list[tuple[str, float]]] = {t: [] for t in ENTITY_TYPES}
    nlp = _get_nlp()
    if not nlp:
        return results   # spaCy unavailable — caller uses regex only

    doc = nlp(text)

    for ent in doc.ents:
        val = ent.text.strip()
        if not val or len(val) < 2:
            continue

        if ent.label_ in _SPACY_PERSON_LABELS:
            if val.lower() not in _NAME_STOPWORDS and val[0].isupper():
                # Salience: higher if name appears near the start or a sign-off
                pos_ratio = ent.start_char / max(len(text), 1)
                salience  = round(0.9 - 0.1 * pos_ratio, 3)
                results["CUSTOMER_NAME"].append((val, salience))

        elif ent.label_ in _SPACY_PRODUCT_LABELS:
            # Filter out short or all-lowercase ents
            if len(val) > 3 and any(c.isupper() for c in val):
                results["PRODUCT"].append((val, 0.85))

    return results


def _regex_extract_order(text: str) -> list[tuple[str, float]]:
    found = []
    for m in _PAT_ORDER.finditer(text):
        raw = next((g for g in m.groups() if g), None)
        if raw:
            # Normalise: if it already starts with ORD- keep it; else prefix
            val = raw if re.match(r'^[A-Z]+-', raw, re.IGNORECASE) else f"ORD-{raw}"
            found.append((val, 0.95))
    return found


def _regex_extract_product(text: str) -> list[tuple[str, float]]:
    found = []
    seen  = set()
    for m in _PAT_PRODUCT.finditer(text):
        val = next((g.strip() for g in m.groups() if g and g.strip()), None)
        if not val or len(val) <= 3:
            continue
        if val.lower() in _PRODUCT_STOPWORDS:
            continue
        words = val.split()
        if len(words) == 1 and not re.search(r'\d', val):
            continue  # single cap word with no digits — too ambiguous
        if val.lower() not in seen:
            seen.add(val.lower())
            found.append((val, 0.80))
    return found


def _regex_extract_name(text: str) -> list[tuple[str, float]]:
    found = []
    for m in _PAT_NAME.finditer(text):
        val = m.group(1).strip()
        if val and val.lower() not in _NAME_STOPWORDS and val[0].isupper():
            found.append((val, 0.75))
    return found


def _regex_extract_issue(text: str) -> list[tuple[str, float]]:
    found = []
    for pat in [_PAT_ISSUE, _PAT_ISSUE2]:
        for m in pat.finditer(text):
            val = m.group(1).strip().rstrip(".,;")
            if val and len(val) > 8:
                found.append((val, 0.90))
    return found


def _classify_urgency(text: str) -> tuple[str, float]:
    if _URGENCY_HIGH.search(text):
        return "high",   0.90
    if _URGENCY_MEDIUM.search(text):
        return "medium", 0.75
    return "low", 0.55


def _classify_sentiment(text: str) -> tuple[str, float]:
    if _SENT_ANGRY.search(text):
        return "angry",      0.92
    if _SENT_FRUSTR.search(text):
        return "frustrated", 0.85
    if _SENT_POLITE.search(text):
        return "polite",     0.70
    if _SENT_POS.search(text):
        return "positive",   0.65
    return "neutral", 0.55


# ── Main KG builder ───────────────────────────────────────────────────────────

def parse_context(text: str, extra: Optional[dict] = None) -> SupportKG:
    """
    Build a SupportKG by running genuine NER on `text` (the raw customer message).

    Strategy (layered, highest-confidence wins):
      1. spaCy NER  — for PERSON and PRODUCT entities
      2. Regex NER  — for ORDER_ID, PRODUCT (fallback), CUSTOMER_NAME (fallback), ISSUE
      3. Lexicon    — for URGENCY and SENTIMENT (keyword classification)
      4. extra dict — structured metadata used ONLY to fill gaps NER missed,
                      and to verify/boost salience of NER-found entities.

    The `extra` dict is intentionally consulted LAST so that NER results
    derived from the raw text are the primary source of truth.
    """
    kg = SupportKG()

    # ── Step 1: spaCy NER on raw text ────────────────────────────────────────
    ner_results = _ner_extract(text)

    for val, sal in ner_results["CUSTOMER_NAME"]:
        kg.add_node(val, "CUSTOMER_NAME", salience=sal)
    for val, sal in ner_results["PRODUCT"]:
        kg.add_node(val, "PRODUCT", salience=sal)

    # ── Step 2: Regex NER on raw text ────────────────────────────────────────

    # ORDER_ID — regex is very reliable here (structured pattern)
    for val, sal in _regex_extract_order(text):
        kg.add_node(val, "ORDER_ID", salience=sal)

    # PRODUCT — supplement spaCy with regex (catches model numbers spaCy misses)
    for val, sal in _regex_extract_product(text):
        kg.add_node(val, "PRODUCT", salience=sal)

    # CUSTOMER_NAME — supplement spaCy with sign-off regex
    for val, sal in _regex_extract_name(text):
        kg.add_node(val, "CUSTOMER_NAME", salience=sal)

    # ISSUE — regex only (spaCy has no ISSUE label; this is domain-specific)
    for val, sal in _regex_extract_issue(text):
        kg.add_node(val, "ISSUE", salience=sal)

    # ── Step 3: Lexicon classification ───────────────────────────────────────
    urg_val, urg_sal = _classify_urgency(text)
    kg.add_node(urg_val, "URGENCY", salience=urg_sal)

    sent_val, sent_sal = _classify_sentiment(text)
    kg.add_node(sent_val, "SENTIMENT", salience=sent_sal)

    # ── Step 4: Gap-fill from structured metadata (extra) ────────────────────
    # Only add if NER found nothing for that type. This makes NER the primary
    # source while ensuring the KG is never empty due to NER misses.
    if extra:
        populated = {n.entity_type for n in kg.nodes}

        if "ORDER_ID" not in populated and (oid := extra.get("order_id")):
            logger.debug(f"[KG] ORDER_ID gap-filled from metadata: {oid}")
            kg.add_node(oid, "ORDER_ID", salience=0.70)    # lower sal = metadata fallback

        if "PRODUCT" not in populated and (prod := extra.get("product")):
            logger.debug(f"[KG] PRODUCT gap-filled from metadata: {prod}")
            kg.add_node(prod, "PRODUCT", salience=0.70)

        if "CUSTOMER_NAME" not in populated and (cust := extra.get("customer")):
            logger.debug(f"[KG] CUSTOMER_NAME gap-filled from metadata: {cust}")
            kg.add_node(cust, "CUSTOMER_NAME", salience=0.65)

        if "ISSUE" not in populated and (issue := extra.get("issue")):
            logger.debug(f"[KG] ISSUE gap-filled from metadata: {issue}")
            kg.add_node(issue, "ISSUE", salience=0.65)

        # URGENCY / SENTIMENT: upgrade salience if NER-derived value matches metadata
        if (meta_urg := extra.get("urgency")):
            for node in kg.entities_of_type("URGENCY"):
                if node.value == meta_urg:
                    node.salience = min(node.salience + 0.10, 1.0)

        if (meta_sent := extra.get("sentiment")):
            for node in kg.entities_of_type("SENTIMENT"):
                if node.value == meta_sent:
                    node.salience = min(node.salience + 0.10, 1.0)

    # ── Step 5: Build edges ───────────────────────────────────────────────────
    products  = kg.entities_of_type("PRODUCT")
    issues    = kg.entities_of_type("ISSUE")
    orders    = kg.entities_of_type("ORDER_ID")
    customers = kg.entities_of_type("CUSTOMER_NAME")
    urgencies = kg.entities_of_type("URGENCY")

    for prod in products:
        for issue in issues:
            kg.add_edge(prod.value, issue.value, "has_issue",
                        weight=round(prod.salience * issue.salience, 3))
        for order in orders:
            kg.add_edge(order.value, prod.value, "about_product",
                        weight=round(order.salience * prod.salience, 3))
    for cust in customers:
        for order in orders:
            kg.add_edge(cust.value, order.value, "placed_order",
                        weight=round(cust.salience * order.salience, 3))
    for urg in urgencies:
        for issue in issues:
            kg.add_edge(issue.value, urg.value, "has_urgency",
                        weight=round(urg.salience * issue.salience, 3))

    logger.info(
        f"[KG] Built: {len(kg.nodes)} nodes "
        f"({sum(1 for n in kg.nodes if n.entity_type=='CUSTOMER_NAME')} NER-PERSON, "
        f"{sum(1 for n in kg.nodes if n.entity_type=='PRODUCT')} NER/regex-PRODUCT, "
        f"{sum(1 for n in kg.nodes if n.entity_type=='ORDER_ID')} regex-ORDER), "
        f"{len(kg.edges)} edges"
    )
    return kg


def defactualize(text: str, kg: SupportKG) -> str:
    result = text
    replacements: list[tuple[str, str]] = []
    for node in kg.nodes:
        if node.entity_type in PLACEHOLDERS and node.value:
            replacements.append((node.value, PLACEHOLDERS[node.entity_type]))
    replacements.sort(key=lambda x: len(x[0]), reverse=True)
    for original, placeholder in replacements:
        pattern = re.compile(re.escape(original), re.IGNORECASE)
        result = pattern.sub(placeholder, result)
    return result


def rehydrate(text: str, kg: SupportKG) -> str:
    result = text
    for entity_type, placeholder in PLACEHOLDERS.items():
        nodes = kg.entities_of_type(entity_type)
        if nodes:
            best = max(nodes, key=lambda n: n.salience)
            result = result.replace(placeholder, best.value)
        else:
            result = result.replace(placeholder, "")
    remaining = re.findall(r'<[A-Z_]+>', result)
    if remaining:
        logger.warning(f"[Rehydrate] Unrehydrated placeholders: {remaining} — stripping.")
        result = re.sub(r'<[A-Z_]+>', '', result)
    result = re.sub(r'  +', ' ', result).strip()
    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — STYLE VECTORS  (from style_vectors.py)
# ══════════════════════════════════════════════════════════════════════════════

import os
import pickle
from pathlib import Path

LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "meta-llama/Llama-2-7b-hf")
STEER_LAYER      = int(os.getenv("STEER_LAYER", "20"))
STEER_ALPHA      = float(os.getenv("STEER_ALPHA", "15.0"))
STYLE_CACHE_DIR  = Path(os.getenv("STYLE_CACHE_DIR", ".style_cache"))
STYLE_CACHE_DIR.mkdir(exist_ok=True)

BEST_METHOD = "pca"

CONTRAST_PAIRS: dict[str, list[tuple[str, str]]] = {
    "empathetic": [
        # Arc 1: Opening acknowledgment
        (
            "I'm so sorry to hear that your <PRODUCT> is experiencing <ISSUE> — "
            "that must be incredibly frustrating, especially when you depend on it every day. "
            "I completely understand why you're <SENTIMENT> about this, and I want you to know "
            "that I'm taking this seriously and will personally make sure we fix it today.",
            "We acknowledge receipt of your complaint regarding <ISSUE> with your <PRODUCT> "
            "under order <ORDER_ID>. "
            "The matter has been logged in our system and assigned to the relevant technical team. "
            "You will receive a formal update within the standard SLA timeframe.",
        ),
        (
            "Oh no, <CUSTOMER_NAME> — I'm genuinely mortified that you received the wrong item. "
            "That is completely unacceptable and I am so sorry this happened to you. "
            "I can only imagine how disappointing it must have been to open the package and find "
            "something entirely different from what you ordered, and I'm going to make this right immediately.",
            "We have been notified that order <ORDER_ID> was fulfilled incorrectly. "
            "A formal investigation has been initiated in accordance with our fulfilment policy. "
            "The incorrect item will be collected and the correct <PRODUCT> dispatched "
            "upon completion of the review process.",
        ),
        (
            "I completely understand how alarming it is to see an unexpected charge on your account, "
            "<CUSTOMER_NAME>, and I sincerely apologise for this error on our part. "
            "You should never have been put in this position, and I want you to know that resolving "
            "this for you is my top priority right now.",
            "Your complaint regarding a billing discrepancy on order <ORDER_ID> has been received. "
            "Our finance team will conduct an audit of the transaction records within 2 business days. "
            "A formal refund confirmation will be issued to your registered email address upon completion.",
        ),
        (
            "I hear you, <CUSTOMER_NAME> — waiting this long for your <PRODUCT> is not okay, "
            "and I'm truly sorry we've left you without a clear update for so long. "
            "You've been incredibly patient and that patience deserves a real answer, "
            "not more waiting, so let me find out exactly what's happening right now.",
            "Your enquiry regarding the delayed delivery of order <ORDER_ID> has been received. "
            "Our logistics team has been asked to investigate the current status of your shipment. "
            "You will be notified of any updates via your registered contact details.",
        ),
        # Arc 2: Ownership statement
        (
            "I want to take full personal ownership of your case, <CUSTOMER_NAME> — "
            "this is not how your experience with us should have gone, and that's on us. "
            "I'm not going to pass you around or make you explain this again to someone else; "
            "I will be your single point of contact until <ISSUE> is fully resolved.",
            "Case <ORDER_ID> has been assigned to a dedicated case manager from our technical department. "
            "The case manager will review all prior interaction history and the reported <ISSUE>. "
            "No further action is required from you at this stage.",
        ),
        (
            "This mistake is entirely on us, <CUSTOMER_NAME>, and I want to personally own it. "
            "I'm escalating your order <ORDER_ID> to our fulfilment team right now with an urgent flag "
            "so that your correct <PRODUCT> is prioritised above everything else in our queue.",
            "Responsibility for the fulfilment error on order <ORDER_ID> has been acknowledged. "
            "The case has been escalated to our Senior Fulfilment Manager for urgent review. "
            "Corrective action will be initiated in line with our standard error-resolution protocol.",
        ),
        (
            "I want to be completely transparent with you, <CUSTOMER_NAME> — this billing error "
            "was our mistake, and it's not something you should have had to chase us about. "
            "I'm flagging your account right now so that our finance team treats your refund "
            "as the highest priority and processes it without further delay.",
            "The billing discrepancy related to <ORDER_ID> has been attributed to a system processing error. "
            "Our Finance and Billing department has been formally notified and will take corrective action. "
            "A resolution timeline will be communicated to you in writing.",
        ),
        (
            "I'm not going to make excuses for how long this has taken, <CUSTOMER_NAME> — "
            "we let you down, and I own that completely. "
            "I'm personally contacting our logistics partner right now to get a firm answer "
            "on where your <PRODUCT> is and when it will arrive.",
            "The delay in delivering order <ORDER_ID> has been noted and escalated. "
            "Our logistics partner has been formally contacted and asked to provide an updated "
            "estimated delivery date, which will be relayed to you upon receipt.",
        ),
        # Arc 3: Resolution commitment
        (
            "Here is exactly what I'm doing for you right now, <CUSTOMER_NAME>: "
            "I've raised an urgent replacement request for your <PRODUCT> under order <ORDER_ID>, "
            "and you will receive a shipping confirmation within 24 hours. "
            "You won't need to return the faulty unit until your replacement is safely in your hands.",
            "A replacement unit for <PRODUCT> under order <ORDER_ID> has been requested. "
            "Dispatch is subject to stock availability and standard processing timelines. "
            "Return instructions for the defective unit will be included with the replacement shipment.",
        ),
        (
            "I've already arranged an urgent re-dispatch of your <PRODUCT> for order <ORDER_ID>, "
            "<CUSTOMER_NAME>, and our courier will collect the incorrect item from you at no cost. "
            "You'll receive a tracking link within the next two hours — I wanted to make sure "
            "this was actioned before the end of today given how long you've already waited.",
            "An urgent re-dispatch of the correct <PRODUCT> for order <ORDER_ID> has been initiated. "
            "A courier collection for the incorrect item will be arranged at our expense. "
            "Tracking information will be provided to your registered email address within 24 hours.",
        ),
        (
            "Your full refund for order <ORDER_ID> has been approved and submitted, <CUSTOMER_NAME>. "
            "It will appear back in your account within 3–5 business days depending on your bank, "
            "and I'll send you a confirmation email right now so you have everything in writing. "
            "I'm so sorry again that this happened — it won't happen again.",
            "A refund for the duplicate charge on order <ORDER_ID> has been approved. "
            "The refund will be processed to the original payment method within 5–7 business days. "
            "A formal confirmation will be issued to your registered email address.",
        ),
        (
            "I've just spoken to our logistics team and your <PRODUCT> is expected to arrive "
            "within the next 48 hours, <CUSTOMER_NAME>. "
            "I've also arranged a partial refund of your shipping cost as an apology for the delay — "
            "you'll see that credited automatically within the next day or two.",
            "Our logistics partner has confirmed that order <ORDER_ID> will be delivered "
            "within the next 2 business days. "
            "A partial refund of shipping costs will be processed in accordance with our "
            "delayed delivery compensation policy.",
        ),
        # Arc 4: Closing
        (
            "Thank you so much for your patience through all of this, <CUSTOMER_NAME> — "
            "I know it hasn't been easy, and I really appreciate you giving us the chance to make it right. "
            "Please don't hesitate to reach out to me directly if anything else comes up; "
            "I genuinely want to make sure your experience with us ends on a much better note.",
            "We thank you for bringing <ISSUE> to our attention and for your patience "
            "during the resolution process. "
            "Should you require any further assistance regarding order <ORDER_ID>, "
            "please contact our support desk quoting your case reference number.",
        ),
        (
            "Again, I'm truly sorry for what happened with your order, <CUSTOMER_NAME>. "
            "You deserved to receive exactly what you paid for, without any of this hassle, "
            "and I hope the steps we've taken today go some way to restoring your trust in us. "
            "We're lucky to have customers like you who give us the opportunity to improve.",
            "We apologise for the inconvenience caused by the fulfilment error on order <ORDER_ID>. "
            "This matter has now been escalated for resolution in accordance with our standard procedures. "
            "We appreciate your continued patience and understanding.",
        ),
        (
            "I'm really glad we could get this resolved for you today, <CUSTOMER_NAME>. "
            "Two years of loyalty means a great deal to us, and the last thing we ever want "
            "is for a billing error to undermine that relationship. "
            "If you ever have any concerns in the future, please come straight to us — we're here for you.",
            "We appreciate your loyalty as a long-standing customer and regret any inconvenience "
            "caused by the billing discrepancy on order <ORDER_ID>. "
            "The matter has been resolved and appropriate steps have been taken to prevent recurrence. "
            "Thank you for your patience.",
        ),
        (
            "I really hope the next time you shop with us it's a completely different experience, "
            "<CUSTOMER_NAME> — you've been so understanding throughout this, and that means a lot. "
            "Your <PRODUCT> is on its way and I'll personally make sure the tracking is updated "
            "so you always know exactly where it is. Take care!",
            "We acknowledge the delay experienced with order <ORDER_ID> and thank you for "
            "your patience during this period. "
            "Your order has now been prioritised and tracking information will be provided "
            "via your registered contact details. "
            "We appreciate your continued custom.",
        ),
    ],
    "formal": [
        # Arc 1: Opening acknowledgment (formal positive, empathetic negative)
        (
            "We acknowledge receipt of your complaint regarding <ISSUE> with your <PRODUCT> "
            "under order <ORDER_ID>. "
            "The matter has been logged in our system and assigned to the relevant technical team. "
            "You will receive a formal update within the standard SLA timeframe.",
            "I'm so sorry to hear that your <PRODUCT> is experiencing <ISSUE> — "
            "that must be incredibly frustrating, especially when you depend on it every day. "
            "I completely understand why you're <SENTIMENT> about this, and I want you to know "
            "that I'm taking this seriously and will personally make sure we fix it today.",
        ),
        (
            "We have been notified that order <ORDER_ID> was fulfilled incorrectly. "
            "A formal investigation has been initiated in accordance with our fulfilment policy. "
            "The incorrect item will be collected and the correct <PRODUCT> dispatched "
            "upon completion of the review process.",
            "Oh no, <CUSTOMER_NAME> — I'm genuinely mortified that you received the wrong item. "
            "That is completely unacceptable and I am so sorry this happened to you. "
            "I'm going to make this right immediately.",
        ),
        (
            "Your complaint regarding a billing discrepancy on order <ORDER_ID> has been received. "
            "Our finance team will conduct an audit of the transaction records within 2 business days. "
            "A formal refund confirmation will be issued to your registered email address upon completion.",
            "I completely understand how alarming it is to see an unexpected charge on your account, "
            "<CUSTOMER_NAME>, and I sincerely apologise for this error on our part. "
            "Resolving this for you is my top priority right now.",
        ),
        (
            "Your enquiry regarding the delayed delivery of order <ORDER_ID> has been received. "
            "Our logistics team has been asked to investigate the current status of your shipment. "
            "You will be notified of any updates via your registered contact details.",
            "I hear you, <CUSTOMER_NAME> — waiting this long for your <PRODUCT> is not okay, "
            "and I'm truly sorry we've left you without a clear update for so long. "
            "Let me find out exactly what's happening right now.",
        ),
        # Arc 2: Ownership
        (
            "Case <ORDER_ID> has been assigned to a dedicated case manager from our technical department. "
            "The case manager will review all prior interaction history and the reported <ISSUE>. "
            "No further action is required from you at this stage.",
            "I want to take full personal ownership of your case, <CUSTOMER_NAME> — "
            "this is not how your experience with us should have gone, and that's on us. "
            "I will be your single point of contact until <ISSUE> is fully resolved.",
        ),
        (
            "Responsibility for the fulfilment error on order <ORDER_ID> has been acknowledged. "
            "The case has been escalated to our Senior Fulfilment Manager for urgent review. "
            "Corrective action will be initiated in line with our standard error-resolution protocol.",
            "This mistake is entirely on us, <CUSTOMER_NAME>, and I want to personally own it. "
            "I'm escalating your order <ORDER_ID> to our fulfilment team right now with an urgent flag "
            "so that your correct <PRODUCT> is prioritised above everything else in our queue.",
        ),
        (
            "The billing discrepancy related to <ORDER_ID> has been attributed to a system processing error. "
            "Our Finance and Billing department has been formally notified and will take corrective action. "
            "A resolution timeline will be communicated to you in writing.",
            "I want to be completely transparent with you, <CUSTOMER_NAME> — this billing error "
            "was our mistake, and it's not something you should have had to chase us about. "
            "I'm flagging your account right now so that our finance team treats your refund "
            "as the highest priority and processes it without further delay.",
        ),
        (
            "The delay in delivering order <ORDER_ID> has been noted and escalated. "
            "Our logistics partner has been formally contacted and asked to provide an updated "
            "estimated delivery date, which will be relayed to you upon receipt.",
            "I'm not going to make excuses for how long this has taken, <CUSTOMER_NAME> — "
            "we let you down, and I own that completely. "
            "I'm personally contacting our logistics partner right now to get a firm answer "
            "on where your <PRODUCT> is and when it will arrive.",
        ),
        # Arc 3: Resolution commitment
        (
            "A replacement unit for <PRODUCT> under order <ORDER_ID> has been requested. "
            "Dispatch is subject to stock availability and standard processing timelines. "
            "Return instructions for the defective unit will be included with the replacement shipment.",
            "Here is exactly what I'm doing for you right now, <CUSTOMER_NAME>: "
            "I've raised an urgent replacement request for your <PRODUCT> under order <ORDER_ID>, "
            "and you will receive a shipping confirmation within 24 hours. "
            "You won't need to return the faulty unit until your replacement is safely in your hands.",
        ),
        (
            "An urgent re-dispatch of the correct <PRODUCT> for order <ORDER_ID> has been initiated. "
            "A courier collection for the incorrect item will be arranged at our expense. "
            "Tracking information will be provided to your registered email address within 24 hours.",
            "I've already arranged an urgent re-dispatch of your <PRODUCT> for order <ORDER_ID>, "
            "<CUSTOMER_NAME>, and our courier will collect the incorrect item from you at no cost. "
            "You'll receive a tracking link within the next two hours.",
        ),
        (
            "A refund for the duplicate charge on order <ORDER_ID> has been approved. "
            "The refund will be processed to the original payment method within 5–7 business days. "
            "A formal confirmation will be issued to your registered email address.",
            "Your full refund for order <ORDER_ID> has been approved and submitted, <CUSTOMER_NAME>. "
            "It will appear back in your account within 3–5 business days depending on your bank, "
            "and I'll send you a confirmation email right now so you have everything in writing.",
        ),
        (
            "Our logistics partner has confirmed that order <ORDER_ID> will be delivered "
            "within the next 2 business days. "
            "A partial refund of shipping costs will be processed in accordance with our "
            "delayed delivery compensation policy.",
            "I've just spoken to our logistics team and your <PRODUCT> is expected to arrive "
            "within the next 48 hours, <CUSTOMER_NAME>. "
            "I've also arranged a partial refund of your shipping cost as an apology for the delay.",
        ),
        # Arc 4: Closing
        (
            "We thank you for bringing <ISSUE> to our attention and for your patience "
            "during the resolution process. "
            "Should you require any further assistance regarding order <ORDER_ID>, "
            "please contact our support desk quoting your case reference number.",
            "Thank you so much for your patience through all of this, <CUSTOMER_NAME> — "
            "I know it hasn't been easy, and I really appreciate you giving us the chance to make it right. "
            "Please don't hesitate to reach out to me directly if anything else comes up.",
        ),
        (
            "We apologise for the inconvenience caused by the fulfilment error on order <ORDER_ID>. "
            "This matter has now been escalated for resolution in accordance with our standard procedures. "
            "We appreciate your continued patience and understanding.",
            "Again, I'm truly sorry for what happened with your order, <CUSTOMER_NAME>. "
            "You deserved to receive exactly what you paid for, without any of this hassle, "
            "and I hope the steps we've taken today go some way to restoring your trust in us.",
        ),
        (
            "We appreciate your loyalty as a long-standing customer and regret any inconvenience "
            "caused by the billing discrepancy on order <ORDER_ID>. "
            "The matter has been resolved and appropriate steps have been taken to prevent recurrence. "
            "Thank you for your patience.",
            "I'm really glad we could get this resolved for you today, <CUSTOMER_NAME>. "
            "Two years of loyalty means a great deal to us, and the last thing we ever want "
            "is for a billing error to undermine that relationship.",
        ),
        (
            "We acknowledge the delay experienced with order <ORDER_ID> and thank you for "
            "your patience during this period. "
            "Your order has now been prioritised and tracking information will be provided "
            "via your registered contact details. "
            "We appreciate your continued custom.",
            "I really hope the next time you shop with us it's a completely different experience, "
            "<CUSTOMER_NAME> — you've been so understanding throughout this, and that means a lot. "
            "Your <PRODUCT> is on its way. Take care!",
        ),
    ],
}

assert len(CONTRAST_PAIRS["empathetic"]) == 16, "Expected 16 empathetic pairs"
assert len(CONTRAST_PAIRS["formal"])    == 16, "Expected 16 formal pairs"


def _cache_path(style: str, method: str) -> Path:
    return STYLE_CACHE_DIR / f"style_vec_{style}_{method}.pkl"


def _save_vector(vec, style: str, method: str):
    path = _cache_path(style, method)
    with open(path, "wb") as f:
        pickle.dump(vec.cpu(), f)
    logger.info(f"[StyleVec] Saved to {path}")


def _load_vector(style: str, method: str):
    path = _cache_path(style, method)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        vec = pickle.load(f)
    logger.info(f"[StyleVec] Loaded from cache: {path}")
    return vec


def _get_layer(model):
    import torch.nn as nn
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        n = len(model.model.layers)
        assert STEER_LAYER < n
        return model.model.layers[STEER_LAYER]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        n = len(model.transformer.h)
        assert STEER_LAYER < n
        return model.transformer.h[STEER_LAYER]
    raise RuntimeError(f"Unsupported architecture: {type(model).__name__}")


def _get_activation(model, tokenizer, text: str):
    import torch
    captured: dict = {}

    def _hook(module, inp, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["act"] = hidden.detach().cpu().float()

    layer  = _get_layer(model)
    handle = layer.register_forward_hook(_hook)
    try:
        device = next(model.parameters()).device
        inputs = tokenizer(text, return_tensors="pt", padding=False, return_attention_mask=True).to(device)
        with torch.no_grad():
            model(**inputs)
    finally:
        handle.remove()

    act = captured["act"]
    return act[0, -1, :]


def _pca(pos_acts, neg_acts):
    import torch
    deltas     = [p - n for p, n in zip(pos_acts, neg_acts)]
    all_deltas = torch.stack(deltas + [-d for d in deltas])
    _, _, Vt   = torch.linalg.svd(all_deltas, full_matrices=False)
    vec = Vt[0]
    return vec / (vec.norm() + 1e-8)


def _mean_difference(pos_acts, neg_acts):
    import torch
    diffs = [p - n for p, n in zip(pos_acts, neg_acts)]
    vec = torch.stack(diffs).mean(dim=0)
    return vec / (vec.norm() + 1e-8)


def _logistic_regression(pos_acts, neg_acts):
    import torch, numpy as np
    from sklearn.linear_model import LogisticRegression
    X = torch.stack(pos_acts + neg_acts).numpy()
    y = np.array([1] * len(pos_acts) + [0] * len(neg_acts))
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X, y)
    w = torch.tensor(clf.coef_[0], dtype=torch.float32)
    return w / (w.norm() + 1e-8)


def build_style_vector(style: str, method: str = BEST_METHOD):
    try:
        import torch
    except ImportError as e:
        logger.warning(f"[StyleVec] torch not available ({e})")
        return None

    pairs = CONTRAST_PAIRS.get(style)
    if not pairs:
        return None

    # Reuse the shared model cache to avoid loading a second copy of the model
    # into memory (which causes OOM on most consumer / Colab GPU setups).
    try:
        model, tokenizer = _get_model_and_tokenizer()
    except Exception as exc:
        logger.warning(f"[StyleVec] Could not obtain model for vector build: {exc}")
        return None

    pos_acts, neg_acts = [], []
    for i, (pos_text, neg_text) in enumerate(pairs):
        logger.info(f"[StyleVec] Pair {i+1}/{len(pairs)}")
        pos_acts.append(_get_activation(model, tokenizer, pos_text))
        neg_acts.append(_get_activation(model, tokenizer, neg_text))

    method_fn = {"pca": _pca, "mean": _mean_difference, "logistic": _logistic_regression}
    vec = method_fn.get(method, _pca)(pos_acts, neg_acts)
    _save_vector(vec, style, method)
    return vec


def get_style_vector(style: str, method: str = BEST_METHOD):
    vec = _load_vector(style, method)
    if vec is not None:
        return vec
    return build_style_vector(style, method)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CUSTOMER AGENT  (from customer_agent.py)
# ══════════════════════════════════════════════════════════════════════════════

SCENARIOS = {
    "battery_issue": {
        "order_id":  "ORD-7741",
        "product":   "TechPro X200 Laptop",
        "customer":  "Priya Sharma",
        "issue":     "battery draining completely within 2 hours of a full charge",
        "urgency":   "high",
        "sentiment": "frustrated",
        "backstory": (
            "Priya bought this laptop 3 weeks ago for her freelance design work. "
            "She already contacted support once last week and was told to 'update drivers', "
            "which she did — but the problem persists. She has a client deadline tomorrow."
        ),
    },
    "wrong_item": {
        "order_id":  "ORD-4492",
        "product":   "SmartHome Hub Pro",
        "customer":  "James Okafor",
        "issue":     "received completely wrong item — got a coffee maker instead",
        "urgency":   "high",
        "sentiment": "angry",
        "backstory": (
            "James ordered the SmartHome Hub Pro as a birthday gift for his wife. "
            "The birthday is in two days. He paid for express shipping. "
            "He opened the box in front of his family and was humiliated."
        ),
    },
    "billing_error": {
        "order_id":  "ORD-5523",
        "product":   "Premium Subscription",
        "customer":  "Sofia Reyes",
        "issue":     "charged twice for the same monthly subscription",
        "urgency":   "medium",
        "sentiment": "frustrated",
        "backstory": (
            "Sofia noticed two identical charges of $49.99 on her credit card statement. "
            "She's been a loyal customer for 2 years. She's not panicking but wants "
            "a clear explanation and a refund promptly."
        ),
    },
    "delivery_delay": {
        "order_id":  "ORD-8834",
        "product":   "Ergonomic Office Chair",
        "customer":  "Tom Huang",
        "issue":     "delivery now 3 weeks overdue with no update from courier",
        "urgency":   "medium",
        "sentiment": "neutral",
        "backstory": (
            "Tom ordered the chair for his home office. He's been working on the floor "
            "for 3 weeks. He's not furious, just wants a clear timeline or a refund "
            "so he can buy locally."
        ),
    },
}

TONE_MAP = {
    "frustrated": (
        "You are frustrated but trying to stay composed. "
        "You've already tried one 'fix' that didn't work. "
        "You mention that this is your second time contacting support. "
        "You are politely firm — you want this solved today."
    ),
    "angry": (
        "You are visibly angry. You use CAPS for emphasis occasionally. "
        "You are on the verge of demanding a full refund and leaving a public review. "
        "You feel embarrassed and let down. You demand to speak to a manager if necessary."
    ),
    "neutral": (
        "You are calm and factual. You explain the problem clearly. "
        "You are reasonable but clearly expect a concrete resolution with a timeline."
    ),
    "polite": (
        "You are polite even though you're upset. You say please and thank you. "
        "You give the agent the benefit of the doubt but clearly need help."
    ),
}


def run_customer_agent(scenario_key: str, llm) -> dict:
    """Generate a realistic customer complaint. Returns complaint_data dict."""
    from langchain_core.messages import SystemMessage, HumanMessage

    scenario = SCENARIOS.get(scenario_key, SCENARIOS["battery_issue"])
    sentiment = scenario["sentiment"]
    issue     = scenario["issue"]
    product   = scenario["product"]
    order_id  = scenario["order_id"]
    customer  = scenario["customer"]
    backstory = scenario["backstory"]

    logger.info(f"[Customer] scenario='{scenario_key}' sentiment={sentiment}")

    ai_msg = llm.invoke([
        SystemMessage(content=(
            f"You are {customer}, a real customer with a genuine complaint.\n\n"
            f"Your situation: {backstory}\n\n"
            f"Your current emotional state: {sentiment.upper()}\n"
            f"Tone guidance: {TONE_MAP.get(sentiment, TONE_MAP['neutral'])}\n\n"
            "Write ONE realistic customer support message (2-5 sentences). "
            "Include the order/reference number naturally. "
            "Do NOT be robotic. Sound like a real person. "
            "Do NOT say you are an AI or a simulation.\n\n"
            "Respond ONLY with the customer message text — no labels, no JSON."
        )),
        HumanMessage(content=(
            f"Write your complaint message about: {issue} "
            f"with product: {product} (order: {order_id})"
        )),
    ])

    customer_message = ai_msg.content.strip()
    logger.info(f"[Customer] Message: {customer_message[:120]}...")

    return {
        "customer_message": customer_message,
        "customer_name":    customer,
        "order_id":         order_id,
        "product":          product,
        "issue":            issue,
        "urgency":          scenario["urgency"],
        "sentiment":        sentiment,
        "scenario":         scenario_key,
        "backstory":        backstory,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — SUPPORT AGENT  (from support_agent.py)
# ══════════════════════════════════════════════════════════════════════════════

import threading

ACTIVATION_STEERING_AVAILABLE = False
try:
    import torch
    import transformers
    # Deps are present — mark as potentially available.
    # Actual model load is attempted lazily in _get_model_and_tokenizer();
    # if that fails the flag is reset to False at runtime.
    ACTIVATION_STEERING_AVAILABLE = True
    logger.info("torch + transformers found — KG activation steering deps OK")
except ImportError as e:
    logger.info(f"Activation steering deps missing ({e}) — prompt steering fallback active")

HF_TOKEN = os.getenv("HF_TOKEN", None)

_model_cache: dict = {}
_model_lock = threading.Lock()


def _get_model_and_tokenizer():
    global _model_cache, ACTIVATION_STEERING_AVAILABLE
    if _model_cache:
        return _model_cache["model"], _model_cache["tokenizer"]

    with _model_lock:
        if _model_cache:
            return _model_cache["model"], _model_cache["tokenizer"]

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"[Support] Loading {LOCAL_MODEL_NAME}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                LOCAL_MODEL_NAME, trust_remote_code=True, token=HF_TOKEN,
            )
            model = AutoModelForCausalLM.from_pretrained(
                LOCAL_MODEL_NAME, trust_remote_code=True,
                torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, token=HF_TOKEN,
            )
        except Exception as exc:
            # Model unavailable (not downloaded, bad HF_TOKEN, OOM, etc.).
            # Disable activation steering permanently for this run so the
            # pipeline falls through to prompt steering without retrying.
            ACTIVATION_STEERING_AVAILABLE = False
            logger.error(
                f"[Support] Failed to load {LOCAL_MODEL_NAME}: {exc}\n"
                "         -> Activation steering DISABLED. Set LOCAL_MODEL_NAME / HF_TOKEN correctly."
            )
            raise

        model.eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        _model_cache["model"]     = model
        _model_cache["tokenizer"] = tokenizer
        logger.info(f"[Support] {LOCAL_MODEL_NAME} loaded and cached — activation steering ENABLED")
        return model, tokenizer


_BAD_PATTERNS = re.compile(
    r'(emotional\s+tone|problem\s+description|\*\*[A-Z]|\bstep\s+\d\b'
    r'|what\s+should\s+you\s+do\s+next|based\s+on\s+the\s+customer'
    r'|<expected_range>|\d\)\s+\*\*)',
    re.IGNORECASE,
)


def _is_response_malformed(text: str) -> bool:
    if not text or len(text.strip()) < 40:
        return True
    if _BAD_PATTERNS.search(text):
        logger.warning("[Support] Malformed response detected")
        return True
    if re.search(r'<[A-Z_]+>', text):
        logger.warning("[Support] Orphaned placeholders remain")
        return True
    return False


def _run_kg_activation_steering(defactualized_prompt: str, style: str, kg: SupportKG) -> str:
    import torch

    steer_vec = get_style_vector(style)
    if steer_vec is None:
        raise RuntimeError(
            f"No style vector for '{style}'. "
            "Run colab_build_vectors_llama2.ipynb and copy .style_cache/ to this folder."
        )

    model, tokenizer = _get_model_and_tokenizer()
    device = next(model.parameters()).device
    vec = steer_vec.to(device=device, dtype=torch.float32)

    style_prefix = {
        "empathetic": (
            "I'm truly sorry to hear about this and I completely understand "
            "how frustrated you must be feeling right now."
        ),
        "formal": (
            "We acknowledge receipt of your complaint and wish to advise you "
            "that your case has been logged and assigned for review."
        ),
    }.get(style, "Thank you for contacting us regarding this matter.")

    full_prompt = (
        f"[INST] You are a customer support agent. "
        f"Write a {style.upper()} support reply to the following customer message. "
        f"Do NOT use headers or bullet points. "
        f"Write ONLY the reply — 3 to 4 sentences maximum. "
        f"Do NOT analyse the message. Just write the reply.\n\n"
        f"Customer message: {defactualized_prompt} [/INST] "
        f"{style_prefix}"
    )

    inputs = tokenizer(full_prompt, return_tensors="pt", padding=False, return_attention_mask=True).to(device)
    input_ids = inputs.input_ids
    target_layer = _get_layer(model)

    def _steer_hook(module, inp, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden_f32 = hidden.float()
        # Apply the steering vector to every position in the sequence.
        # During prefill this covers all prompt tokens; during autoregressive
        # generation each step has seq_len=1 so [:, :, :] == [:, -1, :].
        hidden_f32 = hidden_f32 + STEER_ALPHA * vec
        steered = hidden_f32.to(hidden.dtype)
        return (steered,) + output[1:] if isinstance(output, tuple) else steered

    handle = target_layer.register_forward_hook(_steer_hook)
    try:
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=120,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.3,
                eos_token_id=tokenizer.eos_token_id,
            )
    finally:
        handle.remove()

    new_ids = output_ids[0][input_ids.shape[1]:]
    result  = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    sentences = [s.strip() for s in result.split(".") if s.strip()]
    if len(sentences) > 4:
        result = ". ".join(sentences[:4]) + "."
    return result


def _build_kg_system_prompt(style: str, kg: SupportKG) -> str:
    def _first(entity_type: str, default: str) -> str:
        nodes = kg.entities_of_type(entity_type)
        return nodes[0].value if nodes else default

    urgency   = _first("URGENCY",        "medium")
    sentiment = _first("SENTIMENT",      "neutral")
    product   = _first("PRODUCT",        "<PRODUCT>")
    order     = _first("ORDER_ID",       "<ORDER_ID>")
    customer  = _first("CUSTOMER_NAME",  "<CUSTOMER_NAME>")
    issues    = [n.value for n in kg.entities_of_type("ISSUE")]
    issue_str = "; ".join(issues) if issues else "the reported issue"

    style_instructions = {
        "empathetic": (
            "Write with genuine warmth and personal care.\n"
            "- Lead with an emotional acknowledgment ('I'm so sorry', 'I completely understand')\n"
            "- Use first-person throughout ('I will', 'I'm personally ensuring', 'I've arranged')\n"
            "- Make the customer feel heard as a person, not a ticket number\n"
            "- State one concrete resolution action with a specific timeframe\n"
            "- End with a personal reassurance"
        ),
        "formal": (
            "Write in professional, institutional, procedural language.\n"
            "- Use third-person institutional voice ('Your case has been', 'The team will', 'We acknowledge')\n"
            "- Reference policy, SLAs, or escalation procedures\n"
            "- State exact actions and timeframes without emotional language\n"
            "- Keep tone neutral and factual throughout\n"
            "- End with a formal sign-off"
        ),
    }

    return (
        f"You are a customer support representative writing a reply to a customer complaint.\n\n"
        f"=== KNOWN FACTS (already extracted — do NOT ask for these) ===\n"
        f"Customer name : {customer}\n"
        f"Product       : {product}\n"
        f"Order ID      : {order}\n"
        f"Issue         : {issue_str}\n"
        f"Customer mood : {sentiment}\n"
        f"Urgency level : {urgency}\n\n"
        f"=== TARGET TONE: {style.upper()} ===\n"
        f"{style_instructions.get(style, '')}\n\n"
        f"=== STRICT OUTPUT RULES ===\n"
        f"1. Write EXACTLY 3–4 sentences — no more, no less\n"
        f"2. Use these placeholder tokens literally in your reply "
        f"(the system replaces them automatically):\n"
        f"     <CUSTOMER_NAME>  <PRODUCT>  <ORDER_ID>  <ISSUE>\n"
        f"3. Do NOT write headers, bullet points, or numbered lists\n"
        f"4. Do NOT analyse the customer's email or explain their emotional state\n"
        f"5. Do NOT ask for information already listed above\n"
        f"6. Do NOT invent facts, technical details, or policies not listed above\n"
        f"7. Output ONLY the reply text — no preamble, no JSON, no labels\n"
    )


def _build_kg_user_message(defactualized_message: str, style: str) -> str:
    return (
        f"Customer message (entities masked with placeholders):\n"
        f"{defactualized_message}\n\n"
        f"Write your {style} support reply now (3–4 sentences, placeholder tokens required):"
    )


def run_support_agent(complaint_data: dict, target_style: str, llm, use_kg: bool = True) -> dict:
    """Generate a steered support response. Returns response_data dict."""
    from langchain_core.messages import SystemMessage, HumanMessage

    customer_message = complaint_data.get("customer_message", "")
    scenario         = complaint_data.get("scenario", "unknown")

    logger.info(f"[Support] style='{target_style}' scenario='{scenario}' use_kg={use_kg}")

    # ---------------------------------------------------------
    # A/B TOGGLE: KG Extraction & Defactualization
    # ---------------------------------------------------------
    kg_summary = None
    
    if use_kg:
        # Build KG
        kg = parse_context(
            text=customer_message,
            extra={
                "order_id":  complaint_data.get("order_id"),
                "product":   complaint_data.get("product"),
                "customer":  complaint_data.get("customer_name"),
                "issue":     complaint_data.get("issue"),
                "urgency":   complaint_data.get("urgency"),
                "sentiment": complaint_data.get("sentiment"),
            }
        )
        logger.info(f"[Support] KG: {len(kg.nodes)} nodes, {len(kg.edges)} edges")
        kg_summary = kg.to_dict()

        # Defactualize
        defactualized_message = defactualize(customer_message, kg)
        logger.info(f"[Support] Defactualized: {defactualized_message[:120]}...")
    else:
        # Steering Only (No KG)
        kg = SupportKG() # Pass an empty KG so existing steering functions don't crash
        defactualized_message = customer_message
        logger.info("[Support] KG skipped. Using raw customer message.")


    # Try activation steering
    steering_mode    = "none"
    raw_response     = ""
    already_rehydrated = False   # tracks whether rehydrate() has already been applied

    if ACTIVATION_STEERING_AVAILABLE:
        try:
            mode_log = "KG activation steering" if use_kg else "Standard activation steering"
            logger.info(f"[Support] {mode_log} → style='{target_style}'")
            
            raw_response = _run_kg_activation_steering(defactualized_message, target_style, kg)
            
            # Rehydrate only if using KG
            candidate = rehydrate(raw_response, kg) if use_kg else raw_response
            
            if raw_response and len(raw_response.strip()) > 30 and not _is_response_malformed(candidate):
                steering_mode      = "kg_activation" if use_kg else "standard_activation"
                raw_response       = candidate   # store rehydrated text for the return dict
                already_rehydrated = True
                logger.info(f"[Support] {mode_log} SUCCESS")
            else:
                logger.warning("[Support] Activation output malformed — falling back to prompt steering")
                raw_response = ""
        except Exception as exc:
            logger.warning(f"[Support] Activation steering FAILED: {exc}", exc_info=True)
            raw_response = ""

    # Prompt steering fallback (Groq)
    if not raw_response:
        mode_log = "KG prompt steering fallback" if use_kg else "Standard prompt steering fallback"
        logger.info(f"[Support] {mode_log} → style='{target_style}'")
        
        # Adjust prompt based on whether we are using the KG or just baseline
        if use_kg:
            system_prompt = _build_kg_system_prompt(target_style, kg)
            user_message  = _build_kg_user_message(defactualized_message, target_style)
        else:
            system_prompt = f"You are a customer support agent. Adopt a deeply {target_style} tone in your reply."
            user_message  = f"Respond to this customer message: {customer_message}"

        ai_msg = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ])
        raw_response  = ai_msg.content.strip()
        steering_mode = "kg_prompt_steered" if use_kg else "standard_prompt_steered"
        logger.info(f"[Support] {mode_log} SUCCESS")

        candidate = rehydrate(raw_response, kg) if use_kg else raw_response
        
        if _is_response_malformed(candidate):
            logger.warning("[Support] Prompt-steered output malformed — retrying with stricter prompt")
            retry_msg = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=(
                    f"IMPORTANT: Write ONLY a plain support reply. "
                    f"No headers. No bullets. No analysis. Just 3 sentences.\n\n"
                    f"{user_message}"
                )),
            ])
            raw_response = retry_msg.content.strip()
            candidate    = rehydrate(raw_response, kg) if use_kg else raw_response

        raw_response       = candidate
        already_rehydrated = True

    # Final rehydrate — skipped if already done above to prevent double-substitution.
    if use_kg:
        final_response = raw_response if already_rehydrated else rehydrate(raw_response, kg)
    else:
        final_response = raw_response
        
    logger.info(f"[Support] Final response: {final_response[:120]}...")

    return {
        "support_response":     final_response,
        "defactualized_prompt": defactualized_message, # This acts as the "raw prompt" when use_kg=False
        "raw_steered_output":   raw_response,
        "target_style":         target_style,
        "steering_mode":        steering_mode,
        "kg_summary":           kg_summary,
        "scenario":             scenario,
        "customer_name":        complaint_data.get("customer_name"),
        "order_id":             complaint_data.get("order_id"),
        "product":              complaint_data.get("product"),
        "issue":                complaint_data.get("issue"),
        "sentiment":            complaint_data.get("sentiment"),
        "urgency":              complaint_data.get("urgency"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — ORCHESTRATOR  (from orchestrator.py)
# ══════════════════════════════════════════════════════════════════════════════

import asyncio
import sys
from rich.console import Console
from rich.markup import escape as rich_escape
from rich.panel import Panel
from rich.columns import Columns
from rich.table import Table
from rich.rule import Rule

console = Console()


def make_customer_panel(complaint: dict) -> Panel:
    sentiment = complaint.get("sentiment", "?")
    urgency   = complaint.get("urgency", "?")
    msg       = complaint.get("customer_message", "")
    customer  = complaint.get("customer_name", "?")
    order     = complaint.get("order_id", "?")
    product   = complaint.get("product", "?")
    issue     = complaint.get("issue", "?")

    sent_color = {"angry": "red", "frustrated": "yellow", "neutral": "white", "polite": "green"}.get(sentiment, "white")

    body = (
        f"[bold]Customer:[/bold] {rich_escape(customer)}   [bold]Order:[/bold] {rich_escape(order)}\n"
        f"[bold]Product:[/bold] {rich_escape(product)}\n"
        f"[bold]Issue:[/bold] {rich_escape(issue)}\n"
        f"[bold]Sentiment:[/bold] [{sent_color}]{rich_escape(sentiment.upper())}[/{sent_color}]   "
        f"[bold]Urgency:[/bold] {rich_escape(urgency)}\n\n"
        f"[italic]\"{rich_escape(msg)}\"[/italic]"
    )
    return Panel(body, title="[bold cyan]Customer Message[/bold cyan]", border_style="cyan", width=142)


def make_kg_panel(response_data: dict) -> Panel:
    kg    = response_data.get("kg_summary", {})
    nodes = kg.get("nodes", [])
    edges = kg.get("edges", [])

    node_lines = "\n".join(
        f"  [dim]({n['type']})[/dim] {rich_escape(n['value'])}  [dim](salience: {n['salience']:.1f})[/dim]"
        for n in nodes
    )
    edge_lines = "\n".join(
        f"  {rich_escape(e['source'][:25]):25s} —{rich_escape(e['relation'])}→ {rich_escape(e['target'][:25])}"
        for e in edges[:6]
    )
    defact = response_data.get("defactualized_prompt", "")

    body = (
        f"[bold]Nodes ({len(nodes)}):[/bold]\n{node_lines or '  (none)'}\n\n"
        f"[bold]Edges ({len(edges)}):[/bold]\n{edge_lines or '  (none)'}\n\n"
        f"[bold]Defactualized prompt:[/bold]\n  [dim]{rich_escape(defact[:200])}[/dim]"
    )
    return Panel(body, title="[bold magenta]Knowledge Graph[/bold magenta]", border_style="magenta", width=142)


def make_response_panel(data: dict, label: str, border: str) -> Panel:
    response = data.get("support_response", "")
    style    = data.get("target_style", "?")
    mode     = data.get("steering_mode", "?")
    raw      = data.get("raw_steered_output", "")

    mode_color = {"kg_activation": "green", "kg_prompt_steered": "yellow"}.get(mode, "dim")

    body = (
        f"[bold]Style:[/bold] {rich_escape(style.upper())}   "
        f"[bold]Steering mode:[/bold] [{mode_color}]{rich_escape(mode)}[/{mode_color}]\n\n"
        f"[bold]Final response (rehydrated):[/bold]\n{rich_escape(response)}\n\n"
        f"[bold dim]Defactualized output (pre-rehydration):[/bold dim]\n[dim]{rich_escape(raw[:200])}[/dim]"
    )
    return Panel(body, title=f"[bold {border}]{rich_escape(label)}[/bold {border}]", border_style=border, width=70)


def make_comparison_table(emp_response: dict, form_response: dict) -> Table:
    table = Table(title="Style Comparison — Same Facts, Different Tone", show_header=True, header_style="bold")
    table.add_column("Dimension",          style="dim",   width=22)
    table.add_column("Empathetic Pipeline", style="green", width=52)
    table.add_column("Formal Pipeline",     style="blue",  width=52)

    table.add_row("Target style",    emp_response.get("target_style","?"),    form_response.get("target_style","?"))
    table.add_row("Steering mode",   emp_response.get("steering_mode","?"),   form_response.get("steering_mode","?"))
    table.add_row("KG nodes",
                  str(len((emp_response.get("kg_summary") or {}).get("nodes",[]))),
                  str(len((form_response.get("kg_summary") or {}).get("nodes",[]))))
    table.add_row("KG edges",
                  str(len((emp_response.get("kg_summary") or {}).get("edges",[]))),
                  str(len((form_response.get("kg_summary") or {}).get("edges",[]))))
    table.add_row("Response (first 200 chars)",
                  emp_response.get("support_response","")[:200],
                  form_response.get("support_response","")[:200])
    return table


async def run_pipeline(scenario: str, llm, llm_customer) -> dict:
    """
    Run the full A/B pipeline for a single scenario.
    Runs BOTH modes concurrently in a single call:
      • Mode A — KG + Activation Steering  (use_kg=True)
      • Mode B — Activation Steering Only  (use_kg=False)
    Returns a combined record with both sets of outputs for direct comparison.
    """
    # Step 1: Generate customer complaint (shared input for both modes)
    complaint_data = run_customer_agent(scenario, llm_customer)

    # Step 2: Run all 4 agents concurrently (2 styles × 2 modes)
    loop = asyncio.get_event_loop()
    kg_emp_f   = loop.run_in_executor(None, run_support_agent, complaint_data, "empathetic", llm, True)
    kg_form_f  = loop.run_in_executor(None, run_support_agent, complaint_data, "formal",     llm, True)
    act_emp_f  = loop.run_in_executor(None, run_support_agent, complaint_data, "empathetic", llm, False)
    act_form_f = loop.run_in_executor(None, run_support_agent, complaint_data, "formal",     llm, False)
    kg_emp, kg_form, act_emp, act_form = await asyncio.gather(
        kg_emp_f, kg_form_f, act_emp_f, act_form_f
    )

    # Step 3: Display both modes to console
    console.print(make_customer_panel(complaint_data))

    console.print(Rule("[bold magenta]Mode A — KG + Activation Steering[/bold magenta]"))
    console.print(make_kg_panel(kg_emp))
    console.print(Columns([
        make_response_panel(kg_emp,  "Empathetic — KG+Steering", "green"),
        make_response_panel(kg_form, "Formal — KG+Steering",     "blue"),
    ]))

    console.print(Rule("[bold yellow]Mode B — Activation Steering Only (no KG)[/bold yellow]"))
    console.print(Columns([
        make_response_panel(act_emp,  "Empathetic — Steering Only", "green"),
        make_response_panel(act_form, "Formal — Steering Only",     "blue"),
    ]))

    # Step 4: Build combined A/B record
    def _output_block(r: dict) -> dict:
        return {
            "support_response":     r.get("support_response", ""),
            "defactualized_prompt": r.get("defactualized_prompt", ""),
            "raw_steered_output":   r.get("raw_steered_output", ""),
            "target_style":         r.get("target_style", ""),
            "steering_mode":        r.get("steering_mode", ""),
        }

    record = {
        # ── Shared input ───────────────────────────────────────────────────────
        "input": {
            "scenario":         complaint_data["scenario"],
            "customer_name":    complaint_data["customer_name"],
            "order_id":         complaint_data["order_id"],
            "product":          complaint_data["product"],
            "issue":            complaint_data["issue"],
            "urgency":          complaint_data["urgency"],
            "sentiment":        complaint_data["sentiment"],
            "backstory":        complaint_data.get("backstory", ""),
            "customer_message": complaint_data["customer_message"],
        },
        # ── Mode A: KG + Activation Steering ──────────────────────────────────
        "kg_steering": {
            "knowledge_graph":   kg_emp.get("kg_summary", {}),
            "empathetic_output": _output_block(kg_emp),
            "formal_output":     _output_block(kg_form),
        },
        # ── Mode B: Activation Steering Only (no KG) ──────────────────────────
        "activation_only": {
            "empathetic_output": _output_block(act_emp),
            "formal_output":     _output_block(act_form),
        },
    }
    return record


# ══════════════════════════════════════════════════════════════════════════════
# 100-CASE DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

# fmt: off
BATCH_CASES: list[dict] = [
    # ── battery_issue variants (25 cases) ──────────────────────────────────────
    {"scenario": "battery_issue", "override": {"customer": "Priya Sharma",   "order_id": "ORD-7741", "sentiment": "frustrated", "urgency": "high"}},
    {"scenario": "battery_issue", "override": {"customer": "Leo Chen",       "order_id": "ORD-7742", "sentiment": "angry",      "urgency": "high"}},
    {"scenario": "battery_issue", "override": {"customer": "Aisha Patel",    "order_id": "ORD-7743", "sentiment": "neutral",    "urgency": "medium"}},
    {"scenario": "battery_issue", "override": {"customer": "Marco Rossi",    "order_id": "ORD-7744", "sentiment": "polite",     "urgency": "low"}},
    {"scenario": "battery_issue", "override": {"customer": "Sara Kim",       "order_id": "ORD-7745", "sentiment": "frustrated", "urgency": "high"}},
    {"scenario": "battery_issue", "override": {"customer": "David Nwosu",    "order_id": "ORD-7746", "sentiment": "angry",      "urgency": "high"}},
    {"scenario": "battery_issue", "override": {"customer": "Elena Vasquez",  "order_id": "ORD-7747", "sentiment": "polite",     "urgency": "medium"}},
    {"scenario": "battery_issue", "override": {"customer": "Jake Thompson",  "order_id": "ORD-7748", "sentiment": "neutral",    "urgency": "low"}},
    {"scenario": "battery_issue", "override": {"customer": "Mei Lin",        "order_id": "ORD-7749", "sentiment": "frustrated", "urgency": "high"}},
    {"scenario": "battery_issue", "override": {"customer": "Carlos Diaz",    "order_id": "ORD-7750", "sentiment": "angry",      "urgency": "high"}},
    {"scenario": "battery_issue", "override": {"customer": "Nina Johansson", "order_id": "ORD-7751", "sentiment": "neutral",    "urgency": "medium"}},
    {"scenario": "battery_issue", "override": {"customer": "Omar Hassan",    "order_id": "ORD-7752", "sentiment": "polite",     "urgency": "low"}},
    {"scenario": "battery_issue", "override": {"customer": "Yuki Tanaka",    "order_id": "ORD-7753", "sentiment": "frustrated", "urgency": "high"}},
    {"scenario": "battery_issue", "override": {"customer": "Fatima Al-Amin", "order_id": "ORD-7754", "sentiment": "angry",      "urgency": "high"}},
    {"scenario": "battery_issue", "override": {"customer": "Raj Mehta",      "order_id": "ORD-7755", "sentiment": "neutral",    "urgency": "medium"}},
    {"scenario": "battery_issue", "override": {"customer": "Chloe Martin",   "order_id": "ORD-7756", "sentiment": "polite",     "urgency": "low"}},
    {"scenario": "battery_issue", "override": {"customer": "Andre Dupont",   "order_id": "ORD-7757", "sentiment": "frustrated", "urgency": "high"}},
    {"scenario": "battery_issue", "override": {"customer": "Grace Obi",      "order_id": "ORD-7758", "sentiment": "angry",      "urgency": "high"}},
    {"scenario": "battery_issue", "override": {"customer": "Ivan Petrov",    "order_id": "ORD-7759", "sentiment": "neutral",    "urgency": "medium"}},
    {"scenario": "battery_issue", "override": {"customer": "Layla Nasser",   "order_id": "ORD-7760", "sentiment": "polite",     "urgency": "low"}},
    {"scenario": "battery_issue", "override": {"customer": "Tom Fischer",    "order_id": "ORD-7761", "sentiment": "frustrated", "urgency": "high"}},
    {"scenario": "battery_issue", "override": {"customer": "Amara Diallo",   "order_id": "ORD-7762", "sentiment": "angry",      "urgency": "high"}},
    {"scenario": "battery_issue", "override": {"customer": "Lucas Müller",   "order_id": "ORD-7763", "sentiment": "neutral",    "urgency": "medium"}},
    {"scenario": "battery_issue", "override": {"customer": "Hana Park",      "order_id": "ORD-7764", "sentiment": "polite",     "urgency": "low"}},
    {"scenario": "battery_issue", "override": {"customer": "Ben Adeyemi",    "order_id": "ORD-7765", "sentiment": "frustrated", "urgency": "high"}},
    # ── wrong_item variants (25 cases) ────────────────────────────────────────
    {"scenario": "wrong_item", "override": {"customer": "James Okafor",    "order_id": "ORD-4492", "sentiment": "angry",      "urgency": "high"}},
    {"scenario": "wrong_item", "override": {"customer": "Sophie Bernard",  "order_id": "ORD-4493", "sentiment": "frustrated", "urgency": "high"}},
    {"scenario": "wrong_item", "override": {"customer": "Kwame Boateng",   "order_id": "ORD-4494", "sentiment": "neutral",    "urgency": "medium"}},
    {"scenario": "wrong_item", "override": {"customer": "Maria Santos",    "order_id": "ORD-4495", "sentiment": "polite",     "urgency": "low"}},
    {"scenario": "wrong_item", "override": {"customer": "Alex Turner",     "order_id": "ORD-4496", "sentiment": "angry",      "urgency": "high"}},
    {"scenario": "wrong_item", "override": {"customer": "Preethi Nair",    "order_id": "ORD-4497", "sentiment": "frustrated", "urgency": "high"}},
    {"scenario": "wrong_item", "override": {"customer": "Daniel Weber",    "order_id": "ORD-4498", "sentiment": "neutral",    "urgency": "medium"}},
    {"scenario": "wrong_item", "override": {"customer": "Ling Zhou",       "order_id": "ORD-4499", "sentiment": "polite",     "urgency": "low"}},
    {"scenario": "wrong_item", "override": {"customer": "Tariq Mahmoud",   "order_id": "ORD-4500", "sentiment": "angry",      "urgency": "high"}},
    {"scenario": "wrong_item", "override": {"customer": "Emma Wilson",     "order_id": "ORD-4501", "sentiment": "frustrated", "urgency": "high"}},
    {"scenario": "wrong_item", "override": {"customer": "Ravi Krishnan",   "order_id": "ORD-4502", "sentiment": "neutral",    "urgency": "medium"}},
    {"scenario": "wrong_item", "override": {"customer": "Zara Ahmed",      "order_id": "ORD-4503", "sentiment": "polite",     "urgency": "low"}},
    {"scenario": "wrong_item", "override": {"customer": "Pierre Laurent",  "order_id": "ORD-4504", "sentiment": "angry",      "urgency": "high"}},
    {"scenario": "wrong_item", "override": {"customer": "Naomi Clarke",    "order_id": "ORD-4505", "sentiment": "frustrated", "urgency": "high"}},
    {"scenario": "wrong_item", "override": {"customer": "Hiroshi Kato",    "order_id": "ORD-4506", "sentiment": "neutral",    "urgency": "medium"}},
    {"scenario": "wrong_item", "override": {"customer": "Bianca Ferreira", "order_id": "ORD-4507", "sentiment": "polite",     "urgency": "low"}},
    {"scenario": "wrong_item", "override": {"customer": "Elias Bergman",   "order_id": "ORD-4508", "sentiment": "angry",      "urgency": "high"}},
    {"scenario": "wrong_item", "override": {"customer": "Sunita Rao",      "order_id": "ORD-4509", "sentiment": "frustrated", "urgency": "high"}},
    {"scenario": "wrong_item", "override": {"customer": "Tobias Klein",    "order_id": "ORD-4510", "sentiment": "neutral",    "urgency": "medium"}},
    {"scenario": "wrong_item", "override": {"customer": "Ayana Bekele",    "order_id": "ORD-4511", "sentiment": "polite",     "urgency": "low"}},
    {"scenario": "wrong_item", "override": {"customer": "Viktor Sokolov",  "order_id": "ORD-4512", "sentiment": "angry",      "urgency": "high"}},
    {"scenario": "wrong_item", "override": {"customer": "Ingrid Hansen",   "order_id": "ORD-4513", "sentiment": "frustrated", "urgency": "high"}},
    {"scenario": "wrong_item", "override": {"customer": "Moana Kealoha",   "order_id": "ORD-4514", "sentiment": "neutral",    "urgency": "medium"}},
    {"scenario": "wrong_item", "override": {"customer": "Diego Morales",   "order_id": "ORD-4515", "sentiment": "polite",     "urgency": "low"}},
    {"scenario": "wrong_item", "override": {"customer": "Chioma Osei",     "order_id": "ORD-4516", "sentiment": "angry",      "urgency": "high"}},
    # ── billing_error variants (25 cases) ─────────────────────────────────────
    {"scenario": "billing_error", "override": {"customer": "Sofia Reyes",     "order_id": "ORD-5523", "sentiment": "frustrated", "urgency": "medium"}},
    {"scenario": "billing_error", "override": {"customer": "Noah Scott",      "order_id": "ORD-5524", "sentiment": "angry",      "urgency": "high"}},
    {"scenario": "billing_error", "override": {"customer": "Keiko Yamamoto",  "order_id": "ORD-5525", "sentiment": "neutral",    "urgency": "low"}},
    {"scenario": "billing_error", "override": {"customer": "Luca Bianchi",    "order_id": "ORD-5526", "sentiment": "polite",     "urgency": "medium"}},
    {"scenario": "billing_error", "override": {"customer": "Adaeze Chukwu",   "order_id": "ORD-5527", "sentiment": "frustrated", "urgency": "high"}},
    {"scenario": "billing_error", "override": {"customer": "Samuel Torres",   "order_id": "ORD-5528", "sentiment": "angry",      "urgency": "high"}},
    {"scenario": "billing_error", "override": {"customer": "Mia Andersen",    "order_id": "ORD-5529", "sentiment": "neutral",    "urgency": "medium"}},
    {"scenario": "billing_error", "override": {"customer": "Arjun Gupta",     "order_id": "ORD-5530", "sentiment": "polite",     "urgency": "low"}},
    {"scenario": "billing_error", "override": {"customer": "Valentina Cruz",  "order_id": "ORD-5531", "sentiment": "frustrated", "urgency": "high"}},
    {"scenario": "billing_error", "override": {"customer": "James McCarthy",  "order_id": "ORD-5532", "sentiment": "angry",      "urgency": "high"}},
    {"scenario": "billing_error", "override": {"customer": "Yuna Lee",        "order_id": "ORD-5533", "sentiment": "neutral",    "urgency": "medium"}},
    {"scenario": "billing_error", "override": {"customer": "Emeka Okonkwo",   "order_id": "ORD-5534", "sentiment": "polite",     "urgency": "low"}},
    {"scenario": "billing_error", "override": {"customer": "Klara Novak",     "order_id": "ORD-5535", "sentiment": "frustrated", "urgency": "high"}},
    {"scenario": "billing_error", "override": {"customer": "Hassan Ali",      "order_id": "ORD-5536", "sentiment": "angry",      "urgency": "high"}},
    {"scenario": "billing_error", "override": {"customer": "Anastasia Popov", "order_id": "ORD-5537", "sentiment": "neutral",    "urgency": "medium"}},
    {"scenario": "billing_error", "override": {"customer": "Nadia Leblanc",   "order_id": "ORD-5538", "sentiment": "polite",     "urgency": "low"}},
    {"scenario": "billing_error", "override": {"customer": "Owen Hughes",     "order_id": "ORD-5539", "sentiment": "frustrated", "urgency": "high"}},
    {"scenario": "billing_error", "override": {"customer": "Amina Traoré",    "order_id": "ORD-5540", "sentiment": "angry",      "urgency": "high"}},
    {"scenario": "billing_error", "override": {"customer": "Felix Wagner",    "order_id": "ORD-5541", "sentiment": "neutral",    "urgency": "medium"}},
    {"scenario": "billing_error", "override": {"customer": "Leila Hosseini",  "order_id": "ORD-5542", "sentiment": "polite",     "urgency": "low"}},
    {"scenario": "billing_error", "override": {"customer": "Marcus Brown",    "order_id": "ORD-5543", "sentiment": "frustrated", "urgency": "high"}},
    {"scenario": "billing_error", "override": {"customer": "Soo-Jin Park",    "order_id": "ORD-5544", "sentiment": "angry",      "urgency": "high"}},
    {"scenario": "billing_error", "override": {"customer": "Giulia Romano",   "order_id": "ORD-5545", "sentiment": "neutral",    "urgency": "medium"}},
    {"scenario": "billing_error", "override": {"customer": "Khalid Idris",    "order_id": "ORD-5546", "sentiment": "polite",     "urgency": "low"}},
    {"scenario": "billing_error", "override": {"customer": "Petra Kovacs",    "order_id": "ORD-5547", "sentiment": "frustrated", "urgency": "high"}},
    # ── delivery_delay variants (25 cases) ────────────────────────────────────
    {"scenario": "delivery_delay", "override": {"customer": "Tom Huang",       "order_id": "ORD-8834", "sentiment": "neutral",    "urgency": "medium"}},
    {"scenario": "delivery_delay", "override": {"customer": "Rachel Green",    "order_id": "ORD-8835", "sentiment": "frustrated", "urgency": "high"}},
    {"scenario": "delivery_delay", "override": {"customer": "Ibrahim Jallow",  "order_id": "ORD-8836", "sentiment": "angry",      "urgency": "high"}},
    {"scenario": "delivery_delay", "override": {"customer": "Olga Morozova",   "order_id": "ORD-8837", "sentiment": "polite",     "urgency": "low"}},
    {"scenario": "delivery_delay", "override": {"customer": "Nathan Brooks",   "order_id": "ORD-8838", "sentiment": "neutral",    "urgency": "medium"}},
    {"scenario": "delivery_delay", "override": {"customer": "Pooja Verma",     "order_id": "ORD-8839", "sentiment": "frustrated", "urgency": "high"}},
    {"scenario": "delivery_delay", "override": {"customer": "Christoph Braun", "order_id": "ORD-8840", "sentiment": "angry",      "urgency": "high"}},
    {"scenario": "delivery_delay", "override": {"customer": "Amara Coulibaly", "order_id": "ORD-8841", "sentiment": "polite",     "urgency": "low"}},
    {"scenario": "delivery_delay", "override": {"customer": "Eun-Ji Oh",       "order_id": "ORD-8842", "sentiment": "neutral",    "urgency": "medium"}},
    {"scenario": "delivery_delay", "override": {"customer": "Gabriel Silva",   "order_id": "ORD-8843", "sentiment": "frustrated", "urgency": "high"}},
    {"scenario": "delivery_delay", "override": {"customer": "Miriam Cohen",    "order_id": "ORD-8844", "sentiment": "angry",      "urgency": "high"}},
    {"scenario": "delivery_delay", "override": {"customer": "Takeshi Mori",    "order_id": "ORD-8845", "sentiment": "polite",     "urgency": "low"}},
    {"scenario": "delivery_delay", "override": {"customer": "Zoe Williams",    "order_id": "ORD-8846", "sentiment": "neutral",    "urgency": "medium"}},
    {"scenario": "delivery_delay", "override": {"customer": "Kofi Mensah",     "order_id": "ORD-8847", "sentiment": "frustrated", "urgency": "high"}},
    {"scenario": "delivery_delay", "override": {"customer": "Isabella Ricci",  "order_id": "ORD-8848", "sentiment": "angry",      "urgency": "high"}},
    {"scenario": "delivery_delay", "override": {"customer": "Finn Larsen",     "order_id": "ORD-8849", "sentiment": "polite",     "urgency": "low"}},
    {"scenario": "delivery_delay", "override": {"customer": "Aaliya Khan",     "order_id": "ORD-8850", "sentiment": "neutral",    "urgency": "medium"}},
    {"scenario": "delivery_delay", "override": {"customer": "Dmitri Volkov",   "order_id": "ORD-8851", "sentiment": "frustrated", "urgency": "high"}},
    {"scenario": "delivery_delay", "override": {"customer": "Celine Fontaine", "order_id": "ORD-8852", "sentiment": "angry",      "urgency": "high"}},
    {"scenario": "delivery_delay", "override": {"customer": "Joshua Abara",    "order_id": "ORD-8853", "sentiment": "polite",     "urgency": "low"}},
    {"scenario": "delivery_delay", "override": {"customer": "Serena Nakamura", "order_id": "ORD-8854", "sentiment": "neutral",    "urgency": "medium"}},
    {"scenario": "delivery_delay", "override": {"customer": "Miguel Castro",   "order_id": "ORD-8855", "sentiment": "frustrated", "urgency": "high"}},
    {"scenario": "delivery_delay", "override": {"customer": "Lydia Addo",      "order_id": "ORD-8856", "sentiment": "angry",      "urgency": "high"}},
    {"scenario": "delivery_delay", "override": {"customer": "Patrick Müller",  "order_id": "ORD-8857", "sentiment": "polite",     "urgency": "low"}},
    {"scenario": "delivery_delay", "override": {"customer": "Ximena Flores",   "order_id": "ORD-8858", "sentiment": "neutral",    "urgency": "medium"}},
]
# fmt: on

assert len(BATCH_CASES) == 100, f"Expected 100 cases, got {len(BATCH_CASES)}"


def _apply_override(scenario_key: str, override: dict) -> None:
    """Temporarily patch SCENARIOS[scenario_key] with override values in-place."""
    base = SCENARIOS[scenario_key]
    for k, v in override.items():
        base[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# BATCH RUNNER
# ══════════════════════════════════════════════════════════════════════════════

import json
import datetime
from pathlib import Path


async def run_batch():
    """Run all 100 cases and save results to a timestamped JSONL file."""
    from dotenv import load_dotenv
    load_dotenv()

    from langchain_groq import ChatGroq
    llm          = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
    llm_customer = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

    # Output paths
    timestamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir  = Path("outputs")
    kg_dir      = output_dir / "knowledge_graphs"
    output_dir.mkdir(exist_ok=True)
    kg_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"results_{timestamp}.jsonl"

    total   = len(BATCH_CASES)
    success = 0
    failed  = 0

    console.print()
    console.print(Rule("[bold]A2A Customer Support — 100-Case Batch Run (Both Modes)[/bold]"))
    console.print(
        f"  Output file : [cyan]{output_path}[/cyan]\n"
        f"  Total cases : [bold]{total}[/bold]\n"
        f"  Modes per case: [bold]KG+Steering[/bold] AND [bold]Steering Only[/bold]\n"
        f"  Activation steering: "
        f"[{'green]ENABLED' if ACTIVATION_STEERING_AVAILABLE else 'yellow]DISABLED (prompt steering fallback)'}[/]\n"
    )

    with open(output_path, "w", encoding="utf-8") as fout:
        for idx, case in enumerate(BATCH_CASES, start=1):
            scenario_key = case["scenario"]
            override     = case.get("override", {})

            # Apply override (mutates SCENARIOS dict in-place for this iteration)
            _apply_override(scenario_key, override)

            console.print(Rule(
                f"[bold]Case {idx:>3}/{total}[/bold]  "
                f"scenario=[cyan]{rich_escape(scenario_key)}[/cyan]  "
                f"customer=[yellow]{rich_escape(override.get('customer', '?'))}[/yellow]  "
                f"sentiment=[magenta]{rich_escape(override.get('sentiment', '?'))}[/magenta]"
            ))

            try:
                record = await run_pipeline(scenario_key, llm, llm_customer)
                record["case_index"] = idx
                record["status"]     = "success"
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()

                # Save KG separately
                order_id   = record["input"].get("order_id", f"case{idx}").replace("/", "-")
                customer   = record["input"].get("customer_name", "unknown").replace(" ", "_")
                kg_filename = f"kg_{idx:03d}_{scenario_key}_{order_id}_{customer}.json"
                kg_record  = {
                    "case_index":    idx,
                    "scenario":      scenario_key,
                    "customer_name": record["input"].get("customer_name"),
                    "order_id":      record["input"].get("order_id"),
                    "knowledge_graph": record["kg_steering"]["knowledge_graph"],
                }
                kg_path = kg_dir / kg_filename
                kg_path.write_text(json.dumps(kg_record, indent=2, ensure_ascii=False), encoding="utf-8")

                success += 1
                console.print(f"  [green]✓ Saved case {idx}  |  KG → knowledge_graphs/{rich_escape(kg_filename)}[/green]\n")

            except Exception as exc:
                failed += 1
                error_record = {
                    "case_index":    idx,
                    "status":        "error",
                    "scenario":      scenario_key,
                    "override":      override,
                    "error_message": str(exc),
                }
                fout.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                fout.flush()
                console.print(f"  [red]✗ Case {idx} FAILED: {rich_escape(str(exc))}[/red]\n")

    console.print()
    console.print(Rule("[bold]Batch complete[/bold]"))
    console.print(
        f"  [green]Success:[/green] {success}/{total}\n"
        f"  [red]Failed:[/red]  {failed}/{total}\n"
        f"  [cyan]Results:[/cyan] {output_path}\n"
        f"  [cyan]KGs:[/cyan]     {kg_dir}/  ({success} files)\n"
    )


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    mode = sys.argv[1] if len(sys.argv) > 1 else "batch"

    if mode == "batch":
        # Run all 100 cases and save to outputs/results_<timestamp>.jsonl
        asyncio.run(run_batch())

    elif mode in SCENARIOS:
        # Legacy single-scenario mode: python main.py battery_issue
        # (runs only that one case and prints results; does NOT save to file)
        async def _single():
            from dotenv import load_dotenv
            load_dotenv()
            from langchain_groq import ChatGroq
            llm          = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
            llm_customer = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
            console.print(Rule(f"[bold]Single run — {mode}[/bold]"))
            record = await run_pipeline(mode, llm, llm_customer)
            console.print_json(json.dumps(record, indent=2, ensure_ascii=False))

        asyncio.run(_single())

    else:
        valid = ["batch"] + list(SCENARIOS.keys())
        console.print(f"[red]Unknown mode '{mode}'. Choose from: {valid}[/red]")
        sys.exit(1)