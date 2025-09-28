import streamlit as st
import fitz
import re
import faiss
import numpy as np
import pandas as pd
import plotly.express as px
import networkx as nx
from pyvis.network import Network
from sentence_transformers import SentenceTransformer
from groq import Groq
from io import BytesIO
import os
# Instead of st.secrets, use:
api_key = os.getenv("GROQ_API_KEY")
# -------------------------
# SETUP
# -------------------------
st.set_page_config(page_title="NASA Bioscience Knowledge Engine", layout="wide")

# Initialize models
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
client = Groq(api_key=st.secrets["GROQ_API_KEY"])  # <-- put key in .streamlit/secrets.toml

# Topics for bubble chart
TOPICS = ["radiation","microgravity","plant","human","dna","microbe","bacteria","cell","growth","stress"]

# Persona instructions
PERSONA_INSTRUCTIONS = {
    "scientist": "Provide detailed experimental findings, methods clarity, and new hypotheses.",
    "manager": "Highlight funding opportunities, priority areas, and bottlenecks.",
    "mission_architect": "Emphasize crew health, spacecraft design, and risks for Moon/Mars.",
    "learner": "Explain simply, add 2–3 takeaways, and 1 quiz question.",
    "investor": "Focus on ROI, applications, and where to fund next.",
    "researcher": "Give technical findings, citations, and follow-up experiments."
}

# -------------------------
# HELPERS
# -------------------------
def extract_text_from_pdf(file_bytes):
    text = ""
    pdf = fitz.open(stream=file_bytes, filetype="pdf")
    for page in pdf:
        text += page.get_text("text")
    return text

def chunk_text(text, size=500, overlap=50):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size-overlap)]

def build_faiss_index(chunks):
    vectors = embed_model.encode(chunks)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.array(vectors))
    return index, vectors

def retrieve(query, chunks, index, vectors, k=5):
    q_vec = embed_model.encode([query])
    scores, idx = index.search(q_vec, k)
    return [(chunks[i], float(scores[0][j])) for j,i in enumerate(idx[0])]

def summarize_with_groq(query, retrieved, persona="scientist"):
    context = "\n\n".join([f"[{i+1}] {txt[:800]}" for i,(txt,_) in enumerate(retrieved)])
    instr = PERSONA_INSTRUCTIONS.get(persona, PERSONA_INSTRUCTIONS["scientist"])
    prompt = f"""
Persona: {persona}
Instructions: {instr}

User query: "{query}"

Use CONTEXT (numbered) below for evidence. Output in this format:

---SUMMARY---
Two-sentence concise summary.

---DETAILED---
Numbered key points with citations [1], [2].

---KEYWORDS---
Comma-separated keywords.

---GAPS---
Bullet list of knowledge gaps.

---RECOMMENDATION---
One actionable suggestion for the {persona}.

CONTEXT:
{context}
"""
    resp = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role":"user","content":prompt}],
        max_tokens=700,
        temperature=0.0
    )
    return resp.choices[0].message.content

def topic_counts(chunks):
    counts = {t:0 for t in TOPICS}
    for txt in chunks:
        txt_l = txt.lower()
        for t in TOPICS:
            counts[t] += len(re.findall(r"\b"+re.escape(t)+r"\b", txt_l))
    df = pd.DataFrame([{"topic":k,"count":v} for k,v in counts.items()])
    return df.sort_values("count", ascending=False)

def build_knowledge_graph(chunks):
    MISSIONS = ["ISS","apollo","artemis","shuttle","soyuz","leo","lunar","mars"]
    ORGANISMS = ["human","plant","mouse","rat","microbe","bacteria","yeast"]

    G = nx.Graph()
    for idx, txt in enumerate(chunks):
        pid = f"chunk_{idx}"
        G.add_node(pid, label=pid, color="#1f77b4", type="paper")
        ents = set()
        txt_l = txt.lower()
        for m in MISSIONS:
            if m.lower() in txt_l: ents.add(m.upper())
        for o in ORGANISMS:
            if o.lower() in txt_l: ents.add(o.capitalize())
        for t in TOPICS:
            if t.lower() in txt_l: ents.add(t)
        for e in ents:
            if not G.has_node(e):
                G.add_node(e, label=e, color="#39FF14", type="entity")
            G.add_edge(pid, e)
    net = Network(height="500px", width="100%", bgcolor="#0b0d17", font_color="white")
    net.from_nx(G)
    net.repulsion(node_distance=200, central_gravity=0.2)
    net.write_html("graph.html")
    return "graph.html"

# -------------------------
# STREAMLIT APP
# -------------------------
st.title("🚀 NASA Bioscience Knowledge Engine")
st.markdown("Summarizing 600+ NASA space bioscience studies with AI, visualizations, and persona-specific insights.")

# Upload PDFs
uploaded_files = st.file_uploader("Upload NASA PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    st.success(f"{len(uploaded_files)} PDFs uploaded.")
    all_chunks = []
    for file in uploaded_files:
        text = extract_text_from_pdf(file.read())
        chunks = chunk_text(text)
        all_chunks.extend(chunks)

    index, vectors = build_faiss_index(all_chunks)

    # Query input
    query = st.text_input("🔍 Enter your query:")
    persona = st.selectbox("Choose persona:", list(PERSONA_INSTRUCTIONS.keys()))

    if query:
        retrieved = retrieve(query, all_chunks, index, vectors, k=6)

        st.subheader("📑 Top Retrieved Snippets")
        df = pd.DataFrame([{"Snippet":txt[:200]+"...", "Score":score} for txt,score in retrieved])
        st.table(df)

        st.subheader("🤖 Persona-based Summary")
        summary = summarize_with_groq(query, retrieved, persona)
        st.markdown(f"```\n{summary}\n```")

        st.subheader("📊 Topic Frequency Bubble Chart")
        df_topics = topic_counts(all_chunks)
        fig = px.scatter(df_topics, x="topic", y="count", size="count", color="count",
                         title="Topic Frequency (proxy for evidence density)")
        st.plotly_chart(fig)

        st.subheader("🌌 Knowledge Graph")
        graph_file = build_knowledge_graph(all_chunks)
        with open(graph_file, "r", encoding="utf-8") as f:
            st.components.v1.html(f.read(), height=600, scrolling=True)
