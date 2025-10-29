# === RiskRanger app.py ===
# Place this file at: riskranger/src/app.py

import streamlit as st
import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any

# core helpers you created
from riskranger_core import (
    read_pdf_bytes, detect_lang_safe, get_translator, translate_to_en
)

# ML / viz
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px

# PDF report export
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

# -----------------------------
# Streamlit page setup
# -----------------------------
st.set_page_config(page_title="RiskRanger — AI Risk Intelligence", page_icon="⚡", layout="wide")
st.title("⚡ RiskRanger — AI Risk Intelligence for Filings & ESG Reports")
st.caption("Upload PDF/CSV/TXT → detect language → (auto-translate) → classify risks → score severity → cluster & visualize → export.")

# -----------------------------
# Config / taxonomy
# -----------------------------
RISK_LABELS = [
    "regulatory","climate","cyber","supply chain",
    "litigation","financial","governance","operational","reputational"
]

LEXICON = {
    "regulatory": ["regulation","regulatory","compliance","fine","sanction","license","licence","gdpr","antitrust","cross-border","competition authority"],
    "climate": ["emissions","climate","decarbonization","carbon","scope 1","scope 2","scope 3","renewable","transition risk","physical risk","net zero","hydrogen"],
    "cyber": ["cyber","ransomware","breach","intrusion","malware","ddos","vulnerability","phishing","incident response","soc"],
    "supply chain": ["supplier","logistics","shipping","lead time","shortage","geopolitical","disruption","inventory"],
    "litigation": ["litigation","lawsuit","injunction","damages","settlement","class action","legal proceedings"],
    "financial": ["liquidity","covenant","refinancing","interest rate","fx","currency","credit risk","impairment"],
    "governance": ["board","audit committee","whistleblower","internal control","sox","governance","oversight"],
    "operational": ["outage","downtime","operational","manufacturing","capacity","safety","hazard"],
    "reputational": ["reputation","brand damage","negative publicity","boycott","public perception","media scrutiny"]
}

# -----------------------------
# Cached models
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_zero_shot():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

@st.cache_resource(show_spinner=False)
def get_sentiment():
    return pipeline("text-classification", model="ProsusAI/finbert")

@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# -----------------------------
# Helpers (app scope)
# -----------------------------
def regex_sent_split(txt: str) -> List[str]:
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', txt.strip())
    return [p.strip() for p in parts if p.strip()]

def parse_input(uploaded_files) -> pd.DataFrame:
    """Return unified dataframe with columns: source,page,section,text."""
    rows = []
    for uf in uploaded_files:
        name = uf.name.lower()
        if name.endswith(".pdf"):
            rows.extend(read_pdf_bytes(uf.read(), uf.name))
        elif name.endswith(".csv"):
            df = pd.read_csv(uf)
            if "text" not in df.columns:
                st.error("CSV must include a 'text' column."); st.stop()
            for _, r in df.iterrows():
                rows.append({
                    "source": r.get("source", uf.name),
                    "page": r.get("page", np.nan),
                    "section": r.get("section", ""),
                    "text": str(r["text"])
                })
        else:  # txt
            txt = uf.read().decode("utf-8", errors="ignore")
            for sent in regex_sent_split(txt):
                rows.append({"source": uf.name, "page": np.nan, "section": "", "text": sent})
    return pd.DataFrame(rows)

def lexicon_hits(s: str) -> Dict[str, int]:
    s_low = s.lower()
    return {lab: sum(1 for kw in kws if kw in s_low) for lab, kws in LEXICON.items()}

def score_chunk(zero_shot, sentiment_clf, chunk: str) -> Dict[str, Any]:
    zs = zero_shot(chunk, candidate_labels=RISK_LABELS, multi_label=True)
    sent = sentiment_clf(chunk, truncation=True)[0]
    sign = {"POSITIVE": 1, "NEGATIVE": -1, "NEUTRAL": 0}.get(sent["label"], 0)
    sent_score = sign * float(sent["score"])

    hits = lexicon_hits(chunk)
    zs_scores = dict(zip(zs["labels"], zs["scores"]))
    boosted = {lab: zs_scores.get(lab, 0.0) + 0.05 * hits.get(lab, 0) for lab in RISK_LABELS}
    top_lab = max(boosted, key=boosted.get)
    risk_conf = boosted[top_lab]

    total_hits = sum(hits.values())
    severity = 1/(1+np.exp(-(3*abs(sent_score) + 1.5*risk_conf + 0.5*np.log1p(total_hits))))

    return {
        "pred_label": top_lab,
        "risk_conf": float(risk_conf),
        "sentiment_label": sent["label"],
        "sentiment_score": float(sent_score),
        "lexicon_hits": hits,
        "severity": float(severity),
        "zs_scores": {lab: float(zs_scores.get(lab, 0.0)) for lab in RISK_LABELS},
        "severity_parts": {
            "sent_mag": float(abs(sent_score)),
            "risk_conf": float(risk_conf),
            "lex_density": float(np.log1p(total_hits))
        }
    }

def cluster_embeddings(emb, k=6):
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    return km.fit_predict(emb)

def highlight_lexicon(text: str, label: str) -> str:
    kws = LEXICON.get(label, [])
    out = text
    for kw in sorted(kws, key=len, reverse=True):
        pattern = re.compile(re.escape(kw), flags=re.IGNORECASE)
        out = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", out)
    return out

def build_pdf_report(path: str, summary_df: pd.DataFrame, top_items: pd.DataFrame, title="RiskRanger Report"):
    c = canvas.Canvas(path, pagesize=A4)
    W, H = A4
    y = H - 2*cm
    c.setFont("Helvetica-Bold", 16); c.drawString(2*cm, y, title); y -= 0.8*cm
    c.setFont("Helvetica", 10); c.drawString(2*cm, y, "Automated risk extraction and scoring"); y -= 1.0*cm

    c.setFont("Helvetica-Bold", 12); c.drawString(2*cm, y, "Summary by label"); y -= 0.6*cm
    c.setFont("Helvetica", 9)
    for _, r in summary_df.head(8).iterrows():
        line = f"{r['pred_label']:<15} | n={int(r['n'])} | avg_sev={r['avg_severity']:.2f} | high={int(r['high_count'])}"
        c.drawString(2*cm, y, line); y -= 0.45*cm
        if y < 3*cm: c.showPage(); y = H - 2*cm

    y -= 0.2*cm
    c.setFont("Helvetica-Bold", 12); c.drawString(2*cm, y, "Top risks"); y -= 0.6*cm
    c.setFont("Helvetica", 9)
    for _, r in top_items.head(6).iterrows():
        c.drawString(2*cm, y, f"{r['pred_label']} | sev={r['severity']:.2f} | conf={r['risk_conf']:.2f} | {str(r.get('source'))} p.{str(r.get('page'))}"); y -= 0.45*cm
        txt = (r['text'] or "")[:600].replace("\n"," ")
        for chunk in [txt[i:i+100] for i in range(0, len(txt), 100)]:
            c.drawString(2.5*cm, y, chunk); y -= 0.38*cm
            if y < 3*cm: c.showPage(); y = H - 2*cm
        y -= 0.2*cm
        if y < 3*cm: c.showPage(); y = H - 2*cm
    c.save()

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Upload documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF / CSV (needs 'text') / TXT",
    type=["pdf","csv","txt"], accept_multiple_files=True
)
k_clusters = st.sidebar.slider("Clusters (dedupe similar risks)", 2, 12, 6, 1)
top_n = st.sidebar.slider("Top-N highest severity", 3, 20, 8, 1)

# -----------------------------
# Ingest
# -----------------------------
if uploaded_files:
    df_in = parse_input(uploaded_files)
else:
    st.info("No files uploaded — try with a PDF, CSV (with a 'text' column), or TXT.")
    st.stop()

if df_in.empty:
    st.warning("No text found."); st.stop()

# Language detection + translation to English for analysis
with st.spinner("Detecting language & translating (if needed)..."):
    translator = get_translator()
    langs, texts_en, was_translated = [], [], []
    for t in df_in["text"].astype(str).tolist():
        lang = detect_lang_safe(t)
        langs.append(lang)
        if lang not in ("en","unknown"):
            t_en, flag = translate_to_en(t, translator)
        else:
            t_en, flag = t, False
        texts_en.append(t_en)
        was_translated.append(flag)
    df_in["language"] = langs
    df_in["text_en"] = texts_en
    df_in["translated"] = was_translated

# -----------------------------
# Models
# -----------------------------
with st.spinner("Loading models (first run downloads weights)..."):
    zero_shot = get_zero_shot()
    sentiment = get_sentiment()
    embedder = get_embedder()

# -----------------------------
# Scoring
# -----------------------------
with st.spinner("Scoring paragraphs..."):
    records = []
    texts = df_in["text_en"].astype(str).tolist()
    for i, t in enumerate(texts):
        meta = score_chunk(zero_shot, sentiment, t)
        row = df_in.iloc[i].to_dict(); row.update(meta)
        records.append(row)
    df = pd.DataFrame(records)

# Flatten lexicon hits
lex_cols = sorted({k for d in df["lexicon_hits"] for k in d.keys()})
for c in lex_cols:
    df[f"lex_{c}"] = df["lexicon_hits"].apply(lambda d: d.get(c,0))
df.drop(columns=["lexicon_hits"], inplace=True)

# -----------------------------
# Embeddings & clustering
# -----------------------------
with st.spinner("Embedding & clustering similar risks..."):
    emb = embedder.encode(df["text_en"].tolist(), normalize_embeddings=True, convert_to_numpy=True)
    df["cluster"] = cluster_embeddings(emb, k_clusters)

# -----------------------------
# Dashboard tabs
# -----------------------------
st.subheader("Dashboard")
tab_overview, tab_top, tab_map = st.tabs(["📊 Overview", "🚩 Top Risks", "🗺️ Risk Landscape"])

with tab_overview:
    summary = df.groupby("pred_label").agg(
        n=("text","size"),
        avg_severity=("severity","mean"),
        high_count=("severity", lambda s: (s>0.75).sum())
    ).sort_values(["high_count","avg_severity","n"], ascending=False).reset_index()
    st.dataframe(summary, use_container_width=True)
    st.caption("Counts, average severity, and number of high-severity (>0.75) items by label.")

with tab_top:
    # filters
    flabel = st.multiselect("Filter by label", options=sorted(df["pred_label"].unique()))
    fsrc = st.multiselect("Filter by source", options=sorted(df["source"].astype(str).unique()))
    fmin = st.slider("Min severity", 0.0, 1.0, 0.0, 0.05)
    filtered = df.copy()
    if flabel: filtered = filtered[filtered["pred_label"].isin(flabel)]
    if fsrc:   filtered = filtered[filtered["source"].astype(str).isin(fsrc)]
    filtered = filtered[filtered["severity"] >= fmin]

    top_df = filtered.sort_values("severity", ascending=False).head(top_n)
    st.dataframe(
        top_df[["source","section","page","pred_label","risk_conf","sentiment_label","sentiment_score","severity","text"]],
        use_container_width=True
    )

    # explainability per row
    for row in top_df.to_dict("records"):
        with st.expander(f"{row['pred_label']} | sev={row['severity']:.2f} | {str(row.get('source'))} p.{row.get('page','')}"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Severity breakdown**")
                parts = row["severity_parts"]
                st.write({
                    "sentiment_magnitude": parts["sent_mag"],
                    "risk_confidence": parts["risk_conf"],
                    "lexicon_density_log": parts["lex_density"]
                })
            with col2:
                st.markdown("**Zero-shot scores**")
                zdf = pd.DataFrame(list(row["zs_scores"].items()), columns=["label","score"]).sort_values("score", ascending=False)
                st.bar_chart(zdf.set_index("label"))
            st.markdown("**Highlighted text**")
            st.markdown(highlight_lexicon(row["text"], row["pred_label"]), unsafe_allow_html=True)
            if row.get("translated", False):
                st.caption(f"Translated from {row.get('language','unknown')} → English for analysis.")

with tab_map:
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(emb)
    viz = pd.DataFrame({
        "x": coords[:,0], "y": coords[:,1],
        "label": df["pred_label"], "severity": df["severity"],
        "text": df["text"], "source": df["source"]
    })
    fig = px.scatter(
        viz, x="x", y="y", color="label",
        size=viz["severity"]*10+3,
        hover_data=["source","text","severity"],
        title="Risk landscape (PCA of embeddings)"
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Exports
# -----------------------------
st.subheader("Export")
csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button("Download full results (CSV)", csv_bytes, file_name="riskranger_output.csv", mime="text/csv")

report_path = "riskranger_report.pdf"
# ensure summary/top_df exist
try:
    _ = summary
except NameError:
    summary = df.groupby("pred_label").agg(
        n=("text","size"),
        avg_severity=("severity","mean"),
        high_count=("severity", lambda s: (s>0.75).sum())
    ).sort_values(["high_count","avg_severity","n"], ascending=False).reset_index()
try:
    _ = top_df
except NameError:
    top_df = df.sort_values("severity", ascending=False).head(top_n)

build_pdf_report(report_path, summary, top_df)
with open(report_path, "rb") as f:
    st.download_button("Download press-ready PDF report", f, file_name="riskranger_report.pdf", mime="application/pdf")

st.info("Models: facebook/bart-large-mnli (zero-shot), ProsusAI/finbert (sentiment), all-MiniLM-L6-v2 (embeddings). Reproducible clustering with random_state=42.")

