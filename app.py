# app.py
import os, re, tempfile
from io import BytesIO
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from groq import Groq

# -----------------------
# Page config
# -----------------------
st.set_page_config(page_title="NASA Bioscience Knowledge Engine", layout="wide")
st.title("🚀 NASA Bioscience Knowledge Engine — NASA-ready Demo")
st.markdown("Upload 3–8 NASA bioscience PDFs for a fast demo. The system indexes them, runs retrieval + persona summaries, and produces consensus metrics, visualizations, gap analysis and an action plan.")

# -----------------------
# Globals & clients
# -----------------------
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

TOPICS = ["radiation","microgravity","plant","human","dna","microbe","bacteria","cell","growth","stress"]

PERSONA_INSTRUCTIONS = {
    "scientist": "Provide precise experimental findings and propose two follow-up experiments with rationale.",
    "manager": "Identify funding priorities, bottlenecks, and where investment accelerates readiness.",
    "mission_architect": "Emphasize crew health, shielding/habitat implications and mission-level mitigations.",
    "investor": "Focus on applied outcomes, commercialization potential and one investment recommendation.",
    "learner": "Explain simply: 2 takeaways and one quiz Q&A.",
    "researcher": "Give technical comparisons, confounders, and pragmatic next experiments."
}

# -----------------------
# Utilities: PDF, chunking, TF-IDF
# -----------------------
def extract_text_from_pdf_bytes(b):
    try:
        doc = fitz.open(stream=b, filetype="pdf")
        pages = [p.get_text("text") for p in doc]
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

@st.cache_resource(show_spinner=False)
def process_uploaded_files(uploaded_files, max_files=8, chunk_words=250, overlap_words=50):
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

@st.cache_data(show_spinner=False)
def compute_tfidf_topics(chunks, top_n=15):
    if not chunks:
        return []
    tf = TfidfVectorizer(stop_words="english", max_features=1000)
    X = tf.fit_transform(chunks)
    scores = np.asarray(X.sum(axis=0)).ravel()
    terms = np.array(tf.get_feature_names_out())
    order = np.argsort(-scores)
    top_terms = [(terms[i], float(scores[i])) for i in order[:top_n]]
    return top_terms

# -----------------------
# Retrieval & metrics
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

def consensus_score_for_hits(hits):
    texts = [h["text"] for h in hits]
    if len(texts) <= 1:
        return 1.0, 0.0
    embs = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    faiss.normalize_L2(embs)
    sims = cosine_similarity(embs)
    idx = np.triu_indices_from(sims, k=1)
    vals = sims[idx]
    if vals.size == 0:
        return 1.0, 0.0
    return float(np.mean(vals)), float(np.std(vals))

def compute_confidence(results):
    if not results:
        return 0.0
    avg_score = np.mean([r["score"] for r in results])
    num_sources = len(set([r["paper_id"] for r in results]))
    return float(avg_score * np.log1p(num_sources))

# -----------------------
# Groq summarization helpers
# -----------------------
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

def call_groq(prompt, model="meta-llama/llama-4-scout-17b-16e-instruct", max_tokens=600):
    if not client:
        return "Groq API key missing — summarization disabled."
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

# -----------------------
# Visualizations & exports
# -----------------------
def plot_consensus_heatmap(results):
    if not results:
        st.info("No results to show consensus heatmap.")
        return
    texts = [r["text"] for r in results]
    embs = embed_model.encode(texts, convert_to_numpy=True)
    sims = cosine_similarity(embs)
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(sims, cmap="coolwarm", ax=ax)
    ax.set_title("Consensus / Conflict Map (retrieved chunks)")
    st.pyplot(fig)

def plot_research_timeline(meta):
    years = []
    for m in meta:
        match = re.search(r"(19|20)\d{2}", m["paper_id"])
        if match:
            years.append(int(match.group()))
    if not years:
        st.info("No years found in filenames for timeline.")
        return
    df = pd.DataFrame({"year": years})
    fig = px.histogram(df, x="year", nbins=20, title="Research Volume Over Time")
    st.plotly_chart(fig, use_container_width=True)

def topics_bubble_chart(chunks):
    df = compute_topics_df(chunks)
    fig = px.scatter(df, x="topic", y="count", size="count", color="count", title="Topic Mentions (heuristic)")
    st.plotly_chart(fig, use_container_width=True)

def compute_topics_df(chunks, topics=TOPICS):
    counts = {t:0 for t in topics}
    for txt in chunks:
        txtl = txt.lower()
        for t in topics:
            counts[t] += len(re.findall(r"\b" + re.escape(t.lower()) + r"\b", txtl))
    return pd.DataFrame([{"topic":k,"count":v} for k,v in counts.items()]).sort_values("count", ascending=False)

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

def plot_gap_radar(topic_scores):
    categories = list(topic_scores.keys())
    values = list(topic_scores.values())
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values + values[:1], theta=categories + categories[:1], fill='toself', name="Coverage"))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), title="Knowledge Gap Radar")
    st.plotly_chart(fig, use_container_width=True)

def generate_markdown_report(query, persona, results, summary_text, top_terms, mean_sim, std_sim):
    md = []
    md.append(f"# NASA Bio Engine Report\n\n**Query:** {query}\n**Persona:** {persona}\n\n")
    md.append("## Summary\n")
    md.append(summary_text + "\n\n")
    md.append("## Top retrieved snippets\n")
    for r in results:
        md.append(f"- `{r['paper_id']}` ({r['score']:.2f}): {r['text'][:300].strip()}...\n")
    md.append("\n## Top Keywords (TF-IDF)\n")
    md.append(", ".join([t for t,_ in top_terms[:20]]))
    md.append("\n\n## Consensus\n")
    md.append(f"- consensus score: {mean_sim:.2f}\n- dispersion: {std_sim:.2f}\n")
    return "\n".join(md)

# -----------------------
# UI: controls
# -----------------------
st.sidebar.header("Demo Controls (fast)")
st.sidebar.info("Use 3–8 PDFs for a snappy live demo. Full 600+ scale runs offline with batch embedding.")

max_files = st.sidebar.slider("Max files (demo)", 1, 10, 6)
chunk_words = st.sidebar.slider("Chunk size (words)", 100, 400, 250)
overlap_words = st.sidebar.slider("Overlap words", 10, 100, 50)
k_retrieve = st.sidebar.slider("Top-k retrieval", 1, 6, 3)

col1, col2 = st.columns([1,3])
with col1:
    uploaded_files = st.file_uploader("Upload NASA PDFs (3–8 files recommended)", type="pdf", accept_multiple_files=True)
    persona = st.selectbox("Persona (who is asking?)", list(PERSONA_INSTRUCTIONS.keys()))
    query = st.text_input("Enter a question (e.g., 'radiation effects on human DNA')", value="")
    run_btn = st.button("Run Query")

with col2:
    st.markdown("### Quick demo tips")
    st.markdown("- Upload a curated set of PDFs relevant to your query (radiation, plant growth, shielding).")
    st.markdown("- Use k=2–3 and 3–8 files for fastest responses.")
    if not GROQ_API_KEY:
        st.warning("Groq key not found in secrets — summaries will be disabled.")

# -----------------------
# Process uploads
# -----------------------
if uploaded_files:
    with st.spinner("Processing PDFs and building index (cached for session)..."):
        chunks, meta, vectors = process_uploaded_files(uploaded_files, max_files=max_files, chunk_words=chunk_words, overlap_words=overlap_words)
        index = build_faiss_index(vectors)
    st.success(f"Processed {min(len(uploaded_files), max_files)} files → {len(chunks)} chunks indexed.")
    top_terms = compute_tfidf_topics(chunks, top_n=15)
    st.markdown("**Top corpus keywords (TF-IDF):** " + (", ".join([t for t,_ in top_terms[:12]]) if top_terms else "n/a"))
else:
    chunks, meta, vectors, index, top_terms = [], [], np.zeros((0, embed_model.get_sentence_embedding_dimension())), None, []

# -----------------------
# Run pipeline on demand
# -----------------------
if run_btn:
    if not query:
        st.warning("Enter a query to run the pipeline.")
    elif len(chunks) == 0:
        st.warning("Upload PDFs first.")
    else:
        with st.spinner("Retrieving relevant chunks..."):
            results = retrieve(query, chunks, meta, index, vectors, k=k_retrieve)
        df_results = pd.DataFrame([{"paper_id":r["paper_id"], "chunk_id":r["chunk_id"], "score":r["score"], "snippet":r["text"][:300].replace("\n"," ")+"..."} for r in results])

        # Consensus & confidence
        mean_sim, std_sim = consensus_score_for_hits(results) if results else (0.0,0.0)
        conf = compute_confidence(results)

        # tabs for polished UX
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📑 Evidence", "🤖 Persona Insights", "🧩 Knowledge Gaps", "📊 Visual Explorer", "💡 Action Plan"])

        with tab1:
            st.subheader("Top Retrieved Chunks")
            st.dataframe(df_results, use_container_width=True)
            st.markdown(f"**Consensus score:** {mean_sim:.2f}  •  **Dispersion:** {std_sim:.2f}")
            st.markdown(f"**Evidence confidence:** {conf:.2f}")
            st.caption("Consensus heatmap shows pairwise similarity among retrieved chunks.")
            plot_consensus_heatmap(results)
            st.caption("Research timeline (based on detected years in filenames):")
            plot_research_timeline(meta)

        with tab2:
            st.subheader("Persona-tailored Summary & Recommendations")
            with st.spinner("Generating persona-aware summary (Groq)..."):
                prompt = build_prompt_for_persona(query, results, persona)
                summary_text = call_groq(prompt)
            st.code(summary_text)
            st.markdown("#### Persona Investment Scorecard (example)")
            scorecard = pd.DataFrame({
                "Topic": ["Radiation","Plant Growth","Microbes","Human Physiology"],
                "Scientist": ["Mature","Emerging","Gap","Mature"],
                "Manager": ["Invest","Wait","Invest","Do Not Invest"],
                "Mission Architect": ["Critical","Supportive","Minor","Critical"]
            })
            st.table(scorecard)

        with tab3:
            st.subheader("Knowledge Gaps & Radar")
            # heuristic topic coverage: normalize topic counts to 0..1
            df_topics = compute_topics_df(chunks)
            counts = df_topics.set_index("topic")["count"].to_dict()
            # map to our radar topics
            radar_topics = ["radiation","plant","microbe","human"]
            topic_scores = {t: min(1.0, counts.get(t,0)/max(1,max(counts.values()))) for t in radar_topics}
            # invert (low coverage => gap) for visualization (we want coverage)
            plot_gap_radar(topic_scores)
            st.markdown("**Top TF-IDF keywords**")
            st.write([t for t,_ in top_terms[:15]])

        with tab4:
            st.subheader("Visual Explorer")
            st.markdown("Topic frequency bubble chart (evidence density heuristic):")
            topics_bubble_chart(chunks)
            st.markdown("Interactive knowledge graph (papers → entities → topics)")
            graph_html = build_knowledge_graph_html(chunks, meta, top_n_papers=60)
            st.components.v1.html(graph_html, height=600, scrolling=True)
            st.download_button("Download knowledge_graph.html", data=graph_html.encode("utf-8"), file_name="knowledge_graph.html")

        with tab5:
            st.subheader("Action Plan / Downloadable Report")
            # quick action checklist via short Groq call
            short_prompt = f"User query: {query}\nFrom the provided context (top retrieved snippets), produce a short checklist: WHAT_TO_STUDY (3 bullets), WHAT_NOT_TO_STUDY (2 bullets), WHERE_TO_INVEST (1 bullet), WHERE_NOT_TO_INVEST (1 bullet). Context:\n\n" + "\n\n".join([r["text"][:800] for r in results])
            short_action = call_groq(short_prompt, max_tokens=300) if client else "Groq disabled - no action checklist."
            st.markdown("**Quick Action Checklist (short):**")
            st.code(short_action)
            # downloadable markdown report
            md = generate_markdown_report(query, persona, results, summary_text, top_terms, mean_sim, std_sim)
            st.download_button("Download report (Markdown)", data=md.encode("utf-8"), file_name=f"nasa_report_{persona}.md")

# Footer
st.markdown("---")
st.markdown("Demo designed for NASA Apps Challenge — indexes publications, shows consensus/conflict, gaps, and persona-specific recommendations. For full 600+ ingest, perform offline batch embedding & persistent FAISS index.")
