"""
Streamlit ESG Analyzer
======================
Purpose: Upload one or more sustainability reports (any language), analyze content, and output
homogeneous ESG ratings, comparable dashboards, and extracted risk highlights.

How to run:
-----------
1) Create a virtual env and install requirements from the block below.
2) Save this file as app.py
3) `streamlit run app.py`

Suggested requirements.txt (pin or relax as you prefer):
-------------------------------------------------------
streamlit>=1.37
pandas>=2.1
numpy>=1.26
plotly>=5.24
pymupdf>=1.24        # for fast PDF text extraction
langdetect>=1.0.9    # lightweight language detection
transformers>=4.43   # (optional) for zero-shot + translation
sentencepiece>=0.2.0 # if you use MarianMT translation
torch                # if using transformers locally w/ PyTorch backend
scikit-learn>=1.4
python-dotenv>=1.0
rapidfuzz>=3.9
"""

from __future__ import annotations
import os
import io
import re
import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# Optional / lazy-import blocks
# -----------------------------

def lazy_import_transformers():
    try:
        import transformers  # noqa: F401
        return True
    except Exception:
        return False

TRANSFORMERS_OK = lazy_import_transformers()

# -----------------------------
# Configuration
# -----------------------------

st.set_page_config(
    page_title="ESG Analyzer",
    layout="wide",
)

DEFAULT_WEIGHTS = {
    "E": 0.4,
    "S": 0.3,
    "G": 0.3,
}

SUBWEIGHTS = {
    "E": {"Emissions": 0.5, "Resources": 0.3, "Pollution": 0.2},
    "S": {"Labor": 0.4, "Community": 0.3, "Product": 0.3},
    "G": {"Board": 0.4, "Ethics": 0.3, "RiskMgmt": 0.3},
}

RISK_BUCKETS = [
    "Physical climate risk",
    "Transition/regulatory risk",
    "Reputational risk",
    "Litigation/compliance risk",
    "Supply chain risk",
    "Data quality/greenwashing risk",
]

# Simple keyword seeds per sub-topic (expand / fine-tune for your use case)
SEEDS = {
    "Emissions": ["emission", "ghg", "scope 1", "scope 2", "scope 3", "carbon", "co2"],
    "Resources": ["energy", "renewable", "water", "waste", "recycling", "efficiency"],
    "Pollution": ["pollution", "spill", "contamin", "air quality", "toxic"],
    "Labor": ["safety", "injury", "lost time", "union", "wage", "diversity", "inclusion"],
    "Community": ["community", "stakeholder", "philanthropy", "local", "human rights"],
    "Product": ["product safety", "recall", "privacy", "data protection"],
    "Board": ["board", "independent", "chair", "audit committee", "remuneration"],
    "Ethics": ["bribery", "corruption", "whistleblow", "code of conduct"],
    "RiskMgmt": ["risk management", "internal control", "materiality", "scenario", "tcfd"],
}

RISK_SEEDS = {
    "Physical climate risk": ["flood", "heatwave", "wildfire", "storm", "drought", "physical risk"],
    "Transition/regulatory risk": ["carbon tax", "regulation", "legislation", "compliance cost", "cap-and-trade"],
    "Reputational risk": ["boycott", "reputation", "media scrutiny", "controversy"],
    "Litigation/compliance risk": ["fine", "penalty", "investigation", "lawsuit", "non-compliance"],
    "Supply chain risk": ["supplier", "chain disruption", "forced labor", "traceability"],
    "Data quality/greenwashing risk": ["assurance", "restatement", "omission", "selective", "greenwash", "misleading"],
}

LANG_OVERRIDES = {"zh-cn": "zh", "zh-tw": "zh"}

# -----------------------------
# Utilities
# -----------------------------

@dataclass
class ReportResult:
    name: str
    language: str
    translated: bool
    token_count: int
    coverage: Dict[str, float]  # subtopic -> [0,1]
    pillar_scores: Dict[str, float]  # E,S,G -> [0,100]
    overall_score: float  # [0,100]
    confidence: float  # [0,100]
    risk_signals: Dict[str, float]  # risk bucket -> [0,100]
    highlights: List[Tuple[str, str]]  # (subtopic, sentence)


def normalize_lang(code: str) -> str:
    code = (code or "en").lower()
    return LANG_OVERRIDES.get(code, code)


def detect_language(text: str) -> str:
    try:
        from langdetect import detect
        return normalize_lang(detect(text[:2000]))
    except Exception:
        return "en"


@st.cache_data(show_spinner=False)
def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text using PyMuPDF; fall back to a trivial empty string on error."""
    try:
        import fitz  # PyMuPDF
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            parts = []
            for page in doc:
                parts.append(page.get_text("text"))
        text = "\n".join(parts)
        return text
    except Exception as e:
        st.warning(f"PDF extraction failed: {e}")
        return ""


def translate_text(text: str, src_lang: str, tgt_lang: str = "en") -> Tuple[str, bool]:
    """Pluggable translation. Defaults to passthrough if English or transformers unavailable.
    You can set USE_MARIAN_MT=true and a model name via ENV MARIAN_MODEL (e.g., 'Helsinki-NLP/opus-mt-mul-en').
    """
    src_lang = normalize_lang(src_lang)
    if src_lang in ("en", ""):
        return text, False

    if os.getenv("USE_MARIAN_MT", "false").lower() == "true" and TRANSFORMERS_OK:
        try:
            from transformers import MarianMTModel, MarianTokenizer
            model_name = os.getenv("MARIAN_MODEL", "Helsinki-NLP/opus-mt-mul-en")
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            chunks = chunk_text(text, max_tokens=400)
            translated_chunks = []
            for ch in chunks:
                inputs = tokenizer(ch, return_tensors="pt", padding=True, truncation=True)
                outputs = model.generate(**inputs, max_new_tokens=400)
                translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                translated_chunks.extend(translated)
            return "\n".join(translated_chunks), True
        except Exception as e:
            st.warning(f"Translation fallback (pass-through). Error: {e}")
            return text, False
    else:
        # Hook up a commercial API here if desired; keep passthrough as default
        return text, False


def chunk_text(text: str, max_tokens: int = 400) -> List[str]:
    # Rough splitter by sentences; productionize as needed
    sents = re.split(r"(?<=[\.!?])\s+", text)
    chunks, cur = [], []
    count = 0
    for s in sents:
        tokens = len(s.split())
        if count + tokens > max_tokens and cur:
            chunks.append(" ".join(cur))
            cur, count = [s], tokens
        else:
            cur.append(s)
            count += tokens
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def simple_clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\u00A0", " ")
    return text.strip()


def count_seed_hits(text: str, seeds: List[str]) -> int:
    hits = 0
    low = text.lower()
    for kw in seeds:
        hits += len(re.findall(rf"\b{re.escape(kw.lower())}\b", low))
    return hits


def score_subtopics(text: str) -> Tuple[Dict[str, float], List[Tuple[str, str]]]:
    """Return coverage per subtopic in [0,1] based on seed density and top highlights."""
    # Split into sentences for highlighting
    sentences = re.split(r"(?<=[\.!?])\s+", text)
    coverage = {}
    highlights: List[Tuple[str, str]] = []
    for sub, seeds in SEEDS.items():
        hits = [s for s in sentences if any(re.search(rf"\b{re.escape(k)}\b", s, flags=re.I) for k in seeds)]
        # coverage as clipped density proxy
        density = min(1.0, np.log1p(sum(count_seed_hits(" ".join(hits), seeds)) + 1) / 5)
        coverage[sub] = float(density)
        # keep up to 3 highlights per subtopic
        for s in hits[:3]:
            highlights.append((sub, s.strip()))
    return coverage, highlights


def compute_pillar_scores(coverage: Dict[str, float]) -> Dict[str, float]:
    # Weighted subtopic averages -> pillar score on 0..100
    pillar_scores = {}
    for pillar, subs in SUBWEIGHTS.items():
        val = 0.0
        for sub, w in subs.items():
            val += w * coverage.get(sub, 0.0)
        pillar_scores[pillar] = float(round(val * 100, 2))
    return pillar_scores


def compute_risk_signals(text: str) -> Dict[str, float]:
    risks: Dict[str, float] = {}
    for bucket, seeds in RISK_SEEDS.items():
        hits = count_seed_hits(text, seeds)
        risks[bucket] = float(min(100.0, hits * 8.0))  # tune scaling
    return risks


def compute_overall(pillar_scores: Dict[str, float]) -> float:
    return round(
        sum(DEFAULT_WEIGHTS[p] * pillar_scores.get(p, 0.0) for p in ["E", "S", "G"]), 2
    )


def compute_confidence(coverage: Dict[str, float], token_count: int) -> float:
    cov = np.mean(list(coverage.values()) or [0])
    length_factor = min(1.0, np.log1p(max(1, token_count)) / 8)
    return float(round(100 * (0.6 * cov + 0.4 * length_factor), 2))


@st.cache_data(show_spinner=False)
def analyze_report(name: str, file_bytes: bytes) -> ReportResult:
    raw_text = extract_text_from_pdf(file_bytes)
    if not raw_text:
        raw_text = ""  # OCR placeholder if you integrate pytesseract

    lang = detect_language(raw_text)
    text_en, translated = translate_text(raw_text, lang, "en")

    text_en = simple_clean(text_en)
    token_count = len(text_en.split())

    coverage, highlights = score_subtopics(text_en)
    pillar_scores = compute_pillar_scores(coverage)
    overall = compute_overall(pillar_scores)
    risk_signals = compute_risk_signals(text_en)
    confidence = compute_confidence(coverage, token_count)

    return ReportResult(
        name=name,
        language=lang,
        translated=translated,
        token_count=token_count,
        coverage=coverage,
        pillar_scores=pillar_scores,
        overall_score=overall,
        confidence=confidence,
        risk_signals=risk_signals,
        highlights=highlights,
    )


# -----------------------------
# UI Components
# -----------------------------

st.title("🌍 ESG Analyzer: Multilingual Sustainability Report Scoring")

with st.sidebar:
    st.header("Upload reports")
    files = st.file_uploader(
        "Upload one or more PDF sustainability reports",
        type=["pdf"],
        accept_multiple_files=True,
    )
    st.markdown("---")
    st.subheader("Scoring weights")
    wE = st.slider("Weight: Environment (E)", 0.0, 1.0, DEFAULT_WEIGHTS["E"], 0.05)
    wS = st.slider("Weight: Social (S)", 0.0, 1.0, DEFAULT_WEIGHTS["S"], 0.05)
    wG = st.slider("Weight: Governance (G)", 0.0, 1.0, DEFAULT_WEIGHTS["G"], 0.05)
    tot = wE + wS + wG
    if abs(tot - 1.0) > 1e-6:
        st.info("Weights auto-normalized to sum to 1.0")
    total = max(1e-9, wE + wS + wG)
    DEFAULT_WEIGHTS.update({"E": wE / total, "S": wS / total, "G": wG / total})

    st.subheader("Export")
    want_export = st.toggle("Enable CSV/JSON export", value=True)

st.markdown(
    "This demo uses seed-based coverage and risk lexicons for transparency and speed. "
    "You can plug in zero-shot classifiers or domain models under the hood for more nuance."
)

if files:
    results: List[ReportResult] = []
    progress = st.progress(0.0, text="Analyzing...")
    for i, f in enumerate(files):
        progress.progress((i + 1) / len(files), text=f"Analyzing {f.name} ({i+1}/{len(files)})")
        data = f.read()
        res = analyze_report(f.name, data)
        results.append(res)
    progress.empty()

    # ------- Overview table
    st.subheader("Overview & Ratings")
    rows = []
    for r in results:
        rows.append(
            {
                "Report": r.name,
                "Language": r.language + (" → en" if r.translated else ""),
                "Tokens": r.token_count,
                "E": r.pillar_scores["E"],
                "S": r.pillar_scores["S"],
                "G": r.pillar_scores["G"],
                "Overall": r.overall_score,
                "Confidence": r.confidence,
            }
        )
    df = pd.DataFrame(rows).sort_values("Overall", ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ------- Comparison selector
    st.markdown("---")
    st.subheader("Compare reports")
    choices = [r.name for r in results]
    sel = st.multiselect("Pick 2–5 reports to compare", choices, default=choices[: min(3, len(choices))])

    if len(sel) >= 1:
        # Radar per report (E,S,G)
        radar_cols = ["E", "S", "G"]
        radar_df = df[df["Report"].isin(sel)][["Report"] + radar_cols]
        fig = go.Figure()
        categories = radar_cols
        for _, row in radar_df.iterrows():
            fig.add_trace(
                go.Scatterpolar(r=row[radar_cols].tolist(), theta=categories, fill="toself", name=row["Report"]) 
            )
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True, height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Risk heatmap
        risk_rows = []
        for r in results:
            if r.name in sel:
                entry = {"Report": r.name}
                entry.update(r.risk_signals)
                risk_rows.append(entry)
        risk_df = pd.DataFrame(risk_rows).set_index("Report").T
        heat = px.imshow(risk_df, aspect="auto", origin="lower", labels=dict(color="Risk signal (0–100)"))
        st.plotly_chart(heat, use_container_width=True)

    # ------- Drill-down per report
    st.markdown("---")
    st.subheader("Drill-down: coverage & highlights")
    for r in results:
        with st.expander(f"🔎 {r.name}"):
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                st.metric("Overall score", r.overall_score)
                st.metric("Confidence", r.confidence)
                st.write("**Language**:", r.language, "**Translated**:", "Yes" if r.translated else "No")
            with c2:
                cov_df = pd.DataFrame({"Subtopic": list(r.coverage.keys()), "Coverage (0–1)": list(r.coverage.values())})
                st.bar_chart(cov_df.set_index("Subtopic"))
            with c3:
                p_df = pd.DataFrame({"Pillar": ["E", "S", "G"], "Score": [r.pillar_scores["E"], r.pillar_scores["S"], r.pillar_scores["G"]]})
                st.bar_chart(p_df.set_index("Pillar"))

            st.markdown("**Risk signals**")
            r_df = pd.DataFrame({"Bucket": list(r.risk_signals.keys()), "Signal": list(r.risk_signals.values())})
            st.bar_chart(r_df.set_index("Bucket"))

            st.markdown("**Source highlights (by subtopic)**")
            for sub, sent in r.highlights[:30]:
                st.markdown(f"- **{sub}**: {sent}")

    # ------- Export
    if want_export:
        out = {
            "results": [asdict(r) for r in results],
            "weights": DEFAULT_WEIGHTS,
            "timestamp": time.time(),
        }
        json_bytes = json.dumps(out, indent=2).encode("utf-8")
        st.download_button(
            "Download JSON results",
            data=json_bytes,
            file_name="esg_analysis.json",
            mime="application/json",
        )
        flat_rows = []
        for r in results:
            base = {
                "Report": r.name,
                "Language": r.language,
                "Translated": r.translated,
                "Tokens": r.token_count,
                "Overall": r.overall_score,
                "Confidence": r.confidence,
            }
            # Add pillars
            for k, v in r.pillar_scores.items():
                base[f"{k}"] = v
            # Add risks
            for k, v in r.risk_signals.items():
                base[f"Risk::{k}"] = v
            flat_rows.append(base)
        csv_bytes = pd.DataFrame(flat_rows).to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV summary",
            data=csv_bytes,
            file_name="esg_summary.csv",
            mime="text/csv",
        )
else:
    st.info("Upload one or more PDF reports to get started. Supported languages auto-detected; translation to English is optional and pluggable.")

# -----------------------------
# Notes for further improvement (dev checklist)
# -----------------------------
# - Swap seed-based scoring for:
#   (a) Zero-shot topic classification (e.g., BART MNLI) to assign sentences to subtopics.
#   (b) A multilingual sentiment or stance model to weigh positive/negative disclosures.
#   (c) Named entity + event extraction to surface controversies with dates and amounts.
# - Map disclosures to frameworks (CSRD/ESRS, GRI, SASB, TCFD) using a label taxonomy.
# - Add OCR (pytesseract) for scanned PDFs.
# - Add deduplication of boilerplate and tables (e.g., filtering lines with too many digits/symbols).
# - Calibrate scoring with a gold dataset; normalize by sector (GICS/NAICS) using peer medians.
# - Add an explainability panel that shows which sentences most influenced each pillar score.
# - Persist results to a database and allow time-series tracking across years.

