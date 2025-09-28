# app.py
import os
import tempfile
import re
import faiss
import numpy as np
import pandas as pd
import networkx as nx
import plotly.express as px
import streamlit as st
from pyvis.network import Network
from sentence_transformers import SentenceTransformer
from groq import Groq
import fitz  # PyMuPDF
from io import BytesIO

# -----------------------
# Page config & title
# -----------------------
st.set_page_config(page_title="NASA Bioscience Knowledge Engine", layout="wide")
st.title("🚀 NASA Bioscience Knowledge Engine — Demo (NASA-ready)")
st.markdown(
    "Prototype: ingest a small set of NASA bioscience PDFs (5–10 for live demo), index them, and produce persona-specific, evidence-backed insights, visuals, and recommendations."
)

# -----------------------
# Globals & Models
# -----------------------
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"      # lightweight & fast for demo
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# Groq client (put key into .streamlit/secrets.toml or set env var GROQ_API_KEY)
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.warning("Groq API key not found. Add GROQ_API_KEY to secrets or environment to enable summarization.")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Topics list (adjustable) used for simple heuristic visualizations
TOPICS = ["radiation", "microgravity", "plant", "human", "dna", "microbe", "bacteria", "cell", "growth", "stress"]

# Persona instructions tuned for NASA stakeholders
PERSONA_INSTRUCTIONS = {
    "scientist": "Provide precise experimental findings, methods clarity, cross-paper comparisons, and propose 2 specific follow-up experiments/hypotheses.",
    "manager": "Identify funding priorities, bottlenecks, ROI potential, and concrete areas to allocate resources to accelerate mission readiness.",
    "mission_architect": "Highlight crew health risks, shielding and habitat design implications, mission-level constraints and mitigations for Moon/Mars.",
    "investor": "Focus on applied outcomes, commercialization potential, and one clear investment recommendation with rationale.",
    "learner": "Explain simply for a student, provide 2-3 takeaways and one short quiz question (with answer) to reinforce learning.",
    "researcher": "Give technical insights, likely confounders, and pragmatic next steps for research/experiments."
}

# -----------------------
# Helpers: PDF extraction & chunking
# -----------------------
def extract_text_from_pdf_bytes(b):
    try:
        doc = fitz.open(stream=b, filetype="pdf")
        pages = []
        for p in doc:
            pages.append(p.get_text("text"))
        doc.close()
        return "\n".join(pages)
    except Exception:
        return ""

def chunk_text_words(text, chunk_words=250, overlap_words=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i: i + chunk_words])
        chunks.append(chunk)
        i += chunk_words - overlap_words
    return chunks

# -----------------------
# Caching: process uploaded files once per session
# -----------------------
@st.cache_resource(show_spinner=False)
def process_uploaded_files(uploaded_files, max_files=8, chunk_words=250, overlap_words=50):
    """
    Returns:
      chunks: list[str]
      meta: list[dict] (paper_id, chunk_id)
      vectors: np.ndarray of embeddings
    """
    all_chunks = []
    meta = []
    files = uploaded_files[:max_files]
    for f in files:
        fname = f.name
        txt = extract_text_from_pdf_bytes(f.read())
        chunks = chunk_text_words(txt, chunk_words=chunk_words, overlap_words=overlap_words)
        for idx, ch in enumerate(chunks):
            all_chunks.append(ch)
            meta.append({"paper_id": fname, "chunk_id": f"{fname}_c{idx}"})
    if len(all_chunks) == 0:
        vectors = np.zeros((0, embed_model.get_sentence_embedding_dimension()), dtype="float32")
    else:
        vectors = embed_model.encode(all_chunks, convert_to_numpy=True, show_progress_bar=False)
    return all_chunks, meta, vectors

@st.cache_resource(show_spinner=False)
def build_faiss_index(vectors):
    if vectors.shape[0] == 0:
        return None
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index

# -----------------------
# Retrieval, summarization, heuristics
# -----------------------
def retrieve(query, chunks, meta, index, vectors, k=3):
    if index is None or len(chunks) == 0:
        return []
    qvec = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(qvec)
    D, I = index.search(qvec, k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        results.append({
            "score": float(score),
            "paper_id": meta[idx]["paper_id"],
            "chunk_id": meta[idx]["chunk_id"],
            "text": chunks[idx]
        })
    return results

def build_prompt_for_persona(query, hits, persona):
    instr = PERSONA_INSTRUCTIONS.get(persona, PERSONA_INSTRUCTIONS["scientist"])
    context = "\n\n".join([f"[{i+1}] paper_id:{h['paper_id']} chunk:{h['chunk_id']}\n{h['text'][:1200]}" for i,h in enumerate(hits)])
    prompt = f"""
Persona: {persona}
Instructions: {instr}

User query: "{query}"

Use ONLY the CONTEXT below for evidence. Output EXACTLY:

---SUMMARY---
(two-sentence concise summary)

---DETAILED---
Numbered key points with citations [1],[2]...

---INNOVATIONS---
2 specific innovations or new experiments

---WHAT_TO_STUDY---
3 recommended research directions and why

---WHAT_NOT_TO_STUDY---
2 topics that appear saturated and why

---INVESTMENT_ADVICE---
One short 'invest' and one 'do not invest' recommendation with rationale

---GAPS---
Bullet list of knowledge gaps

CONTEXT:
{context}
"""
    return prompt

def call_groq(prompt, model="meta-llama/llama-4-scout-17b-16e-instruct", max_tokens=800):
    if not client:
        return "Groq API key missing; summarization disabled."
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"user","content":prompt}],
        max_tokens=max_tokens,
        temperature=0.0
    )
    try:
        return resp.choices[0].message.content
    except Exception as e:
        return f"Groq error: {e}"

def knowledge_gap_heuristic(query, chunks, meta, index, vectors, sample_k=20):
    if index is None or len(chunks) == 0:
        return {"coverage_ratio": 0.0, "assessment": "No data"}
    qvec = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(qvec)
    D, I = index.search(qvec, sample_k)
    nonzero = sum(1 for s in D[0] if s > 0.35)
    coverage = nonzero / max(1, sample_k)
    if coverage < 0.05:
        tag = "🚨 Very few relevant chunks — possible knowledge gap"
    elif coverage < 0.25:
        tag = "⚠️ Limited evidence — consider more research"
    else:
        tag = "✅ Sufficient evidence"
    return {"coverage_ratio": coverage, "assessment": tag}

def topics_dataframe(chunks, topics=TOPICS):
    counts = {t:0 for t in topics}
    for txt in chunks:
        txtl = txt.lower()
        for t in topics:
            counts[t] += len(re.findall(r"\b" + re.escape(t.lower()) + r"\b", txtl))
    df = pd.DataFrame([{"topic":k,"count":v} for k,v in counts.items()])
    return df.sort_values("count", ascending=False)

# -----------------------
# Knowledge graph creation (pyvis -> HTML)
# -----------------------
def build_knowledge_graph_html(chunks, meta, top_n_papers=60):
    MISSIONS = ["ISS","apollo","artemis","shuttle","soyuz","leo","lunar","mars"]
    ORGANISMS = ["human","plant","mouse","rat","microbe","bacteria","yeast"]
    G = nx.Graph()
    n = min(len(chunks), top_n_papers)
    for idx in range(n):
        pid = meta[idx]["paper_id"]
        G.add_node(pid, label=pid, color="#1f77b4", type="paper")
        txt = chunks[idx].lower()
        ents = set()
        for m in MISSIONS:
            if re.search(r"\b" + re.escape(m.lower()) + r"\b", txt):
                ents.add(m.upper())
        for o in ORGANISMS:
            if re.search(r"\b" + re.escape(o.lower()) + r"\b", txt):
                ents.add(o.capitalize())
        for t in TOPICS:
            if re.search(r"\b" + re.escape(t.lower()) + r"\b", txt):
                ents.add(t)
        for e in ents:
            if not G.has_node(e):
                G.add_node(e, label=e, color="#39FF14", type="entity")
            G.add_edge(pid, e)
    net = Network(height="600px", width="100%", bgcolor="#0b0d17", font_color="white", notebook=False)
    net.from_nx(G)
    net.repulsion(node_distance=150, central_gravity=0.05)
    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.write_html(tmpfile.name)
    with open(tmpfile.name, "r", encoding="utf-8") as f:
        html = f.read()
    return html

# -----------------------
# UI layout: left controls, right results
# -----------------------
st.sidebar.header("Demo Controls (fast demo)")
with st.sidebar:
    st.info("Tip: Upload 3–8 PDFs for a snappy demo. Full 600+ scaling is explained in slides.")
    max_files = st.slider("Max files to process (demo)", min_value=1, max_value=10, value=8)
    chunk_words = st.slider("Chunk size (words)", min_value=100, max_value=400, value=250)
    overlap_words = st.slider("Overlap (words)", min_value=10, max_value=100, value=50)
    k_retrieve = st.slider("Top-k retrieval", min_value=1, max_value=6, value=3)

col1, col2 = st.columns([1, 2])
with col1:
    uploaded_files = st.file_uploader("Upload NASA PDFs (5–10 for demo)", type="pdf", accept_multiple_files=True)
    persona = st.selectbox("Persona (who is asking?)", options=list(PERSONA_INSTRUCTIONS.keys()), index=0)
    query = st.text_input("Enter a question (e.g., 'radiation effects on human DNA')", value="")
    run_btn = st.button("Run Query")

with col2:
    st.markdown("### Demo quicktips")
    st.markdown("- Upload a few PDFs and press **Run Query**. The processing is cached for the session.")
    st.markdown("- Use prepared queries for smooth demo: 'radiation effects on human DNA', 'plant growth in microgravity', 'Mars landing safety'.")

# -----------------------
# Process uploaded files (cached)
# -----------------------
if uploaded_files:
    with st.spinner("Processing PDFs and building index (cached)..."):
        chunks, meta, vectors = process_uploaded_files(uploaded_files, max_files=max_files, chunk_words=chunk_words, overlap_words=overlap_words)
        index = build_faiss_index(vectors)
    st.success(f"Processed {min(len(uploaded_files), max_files)} files → {len(chunks)} chunks indexed.")
else:
    chunks, meta, vectors, index = [], [], np.zeros((0, embed_model.get_sentence_embedding_dimension())), None

# -----------------------
# Run pipeline on demand
# -----------------------
if run_btn:
    if not query:
        st.warning("Please enter a query.")
    elif len(chunks) == 0:
        st.warning("Upload PDFs first (demo: 3–8 files).")
    else:
        # 1) Retrieval
        with st.spinner("Retrieving relevant chunks..."):
            results = retrieve(query, chunks, meta, index, vectors, k=k_retrieve)
            df_results = pd.DataFrame([{
                "paper_id": r["paper_id"],
                "chunk_id": r["chunk_id"],
                "score": r["score"],
                "snippet": r["text"][:300].replace("\n", " ") + "..."
            } for r in results])
        # color-grade and display
        st.markdown("### 📑 Top Retrieved Chunks")
        # simple style: show dataframe
        st.dataframe(df_results, use_container_width=True)

        # 2) Knowledge gap
        gap = knowledge_gap_heuristic(query, chunks, meta, index, vectors)
        st.markdown(f"### 🧩 Knowledge Gap Assessment: {gap['assessment']} (coverage ratio: {gap['coverage_ratio']:.2f})")

        # 3) Persona summarization via Groq
        st.markdown("### 🤖 Persona-tailored Summary, Innovations & Recommendations")
        with st.spinner("Generating structured summary from Groq (may take a few seconds)..."):
            prompt = build_prompt_for_persona(query, results, persona)
            summary_text = call_groq(prompt)
        st.code(summary_text, language=None)

        # 4) Topic bubble chart
        st.markdown("### 📊 Topic Frequency (proxy for evidence density)")
        df_topics = topics_dataframe(chunks)
        fig = px.scatter(df_topics, x="topic", y="count", size="count", color="count",
                         labels={"count": "mentions", "topic": "topic"}, title="Topic frequency (heuristic)")
        st.plotly_chart(fig, use_container_width=True)

        # 5) Knowledge graph (pyvis)
        st.markdown("### 🌌 Knowledge Graph (interactive)")
        with st.spinner("Building knowledge graph (interactive)..."):
            graph_html = build_knowledge_graph_html(chunks, meta, top_n_papers=60)
        st.components.v1.html(graph_html, height=600, scrolling=True)

        # 6) Download artifacts
        st.markdown("### Downloadable artifacts")
        if st.button("Download knowledge_graph.html"):
            with open("knowledge_graph.html", "w", encoding="utf-8") as f:
                f.write(graph_html)
            with open("knowledge_graph.html", "rb") as f:
                st.download_button("Click to download knowledge_graph.html", data=f, file_name="knowledge_graph.html")
        st.markdown("---")
        st.markdown("**Notes:** This demo uses a small subset for speed. The same pipeline scales to 600+ publications with batch embedding and offline indexing.")
