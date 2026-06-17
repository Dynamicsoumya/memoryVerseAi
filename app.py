"""MemoryVerse AI -- single-command Streamlit app.

Run with:  streamlit run app.py

Implements all 5 modules from the hackathon brief in one place:
  1. AI Data Ingestion      (Upload page)
  2. Intelligent Categorization (Vault page)
  3. Relationship Engine     (Knowledge Graph page)
  4. Digital Journey Timeline (Timeline page)
  5. Smart Retrieval System  (Smart Search page)
"""
import os
import sys
import uuid

import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.storage import (
    init_db,
    add_document,
    get_all_documents,
    clear_all,
    UPLOAD_DIR,
)
from core.extraction import extract_text_from_file, fetch_github_repo_info, fetch_url_text
from core.categorize import classify_document, CATEGORIES
from core.relationships import extract_skills, build_graph, render_graph_html
from core.timeline import extract_dates, build_timeline
from core.retrieval import semantic_search, generate_answer
from core.embeddings import index_document, reset_index

st.set_page_config(page_title="MemoryVerse AI", layout="wide")
init_db()

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_files")


def ingest(filename, filepath, source_type, raw_text):
    """Shared pipeline: classify -> extract skills/dates -> store -> index."""
    result = classify_document(raw_text, filename=filename)
    rule_skills = extract_skills(raw_text)
    skills = sorted(set(result.get("skills", []) + rule_skills))
    dates = result.get("dates") or []
    if not dates:
        dates = extract_dates(raw_text)

    doc_id = add_document(
        filename=filename,
        filepath=filepath,
        source_type=source_type,
        category=result["category"],
        confidence=result["confidence"],
        full_text=raw_text,
        skills=skills,
        dates=dates,
        organization=result.get("organization", ""),
    )
    index_document(
        doc_id,
        raw_text,
        metadata={"category": result["category"], "filename": filename},
    )
    return doc_id, result["category"]


st.title("MemoryVerse AI")
st.caption("Turn your scattered files into an intelligent digital identity.")

page = st.sidebar.radio(
    "Navigate",
    ["Upload", "Vault", "Knowledge Graph", "Timeline", "Smart Search"],
)

if os.getenv("ANTHROPIC_API_KEY"):
    st.sidebar.success("LLM-enhanced mode active (Anthropic API key detected)")
else:
    st.sidebar.info(
        "Running in local mode: embeddings + rule-based extraction only.\n\n"
        "Add ANTHROPIC_API_KEY to a .env file for richer LLM-based "
        "categorization and answers."
    )

st.sidebar.divider()
if st.sidebar.button("Load sample data"):
    count = 0
    for fname in sorted(os.listdir(SAMPLE_DIR)):
        src_path = os.path.join(SAMPLE_DIR, fname)
        with open(src_path, "r", encoding="utf-8") as f:
            text = f.read()
        stored_name = f"{uuid.uuid4().hex[:8]}_{fname}"
        stored_path = os.path.join(UPLOAD_DIR, stored_name)
        with open(stored_path, "w", encoding="utf-8") as f:
            f.write(text)
        ingest(fname, stored_path, "sample", text)
        count += 1
    st.sidebar.success(f"Loaded {count} sample documents.")
    st.rerun()

if st.sidebar.button("Clear all data"):
    clear_all()
    reset_index()
    st.sidebar.warning("All data cleared.")
    st.rerun()


if page == "Upload":
    st.header("Module 1 -- AI Data Ingestion")
    st.write(
        "Upload certificates, resumes, project reports, internship letters, "
        "or any academic/professional document. Original files and formats "
        "are preserved untouched on disk."
    )

    uploaded_files = st.file_uploader(
        "Upload documents",
        accept_multiple_files=True,
        type=["pdf", "docx", "txt", "md", "png", "jpg", "jpeg"],
    )
    if uploaded_files:
        for uf in uploaded_files:
            stored_name = f"{uuid.uuid4().hex[:8]}_{uf.name}"
            stored_path = os.path.join(UPLOAD_DIR, stored_name)
            with open(stored_path, "wb") as f:
                f.write(uf.getbuffer())
            raw_text = extract_text_from_file(stored_path)
            doc_id, category = ingest(uf.name, stored_path, "upload", raw_text)
            st.success(f"'{uf.name}' ingested and categorized as **{category}**.")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        gh_url = st.text_input("GitHub repository URL")
        if st.button("Ingest GitHub repo") and gh_url:
            text = fetch_github_repo_info(gh_url)
            name = gh_url.rstrip("/").split("/")[-1] or gh_url
            doc_id, category = ingest(name, gh_url, "github", text)
            st.success(f"Repo ingested and categorized as **{category}**.")
    with col2:
        portfolio_url = st.text_input("Portfolio / profile link")
        if st.button("Ingest link") and portfolio_url:
            text = fetch_url_text(portfolio_url)
            doc_id, category = ingest(portfolio_url, portfolio_url, "link", text)
            st.success(f"Link ingested and categorized as **{category}**.")

elif page == "Vault":
    st.header("Module 2 -- Intelligent Categorization")
    docs = get_all_documents()
    if not docs:
        st.info("No documents yet. Go to Upload, or click 'Load sample data' in the sidebar.")
    else:
        for cat in CATEGORIES.keys():
            cat_docs = [d for d in docs if d["category"] == cat]
            with st.expander(f"{cat} ({len(cat_docs)})", expanded=len(cat_docs) > 0):
                if not cat_docs:
                    st.caption("No documents in this category yet.")
                for d in cat_docs:
                    st.markdown(
                        f"**{d['filename']}**  ·  confidence {d['confidence']}  ·  "
                        f"skills: {', '.join(d['skills']) or '—'}"
                    )
                    if d["source_type"] in ("upload", "sample") and os.path.exists(d["filepath"]):
                        with open(d["filepath"], "rb") as f:
                            st.download_button(
                                "Download original",
                                f.read(),
                                file_name=d["filename"],
                                key=f"dl_{d['id']}",
                            )
                    st.caption(d["text_snippet"])
                    st.divider()

elif page == "Knowledge Graph":
    st.header("Module 3 -- Relationship Engine")
    st.write("Certification -> Skill -> Project -> Internship -> Career Path")
    docs = get_all_documents()
    if not docs:
        st.info("No documents yet. Upload some files or load sample data first.")
    else:
        graph = build_graph(docs)
        html = render_graph_html(graph)
        components.html(html, height=620, scrolling=True)

elif page == "Timeline":
    st.header("Module 4 -- Digital Journey Timeline")
    docs = get_all_documents()
    if not docs:
        st.info("No documents yet. Upload some files or load sample data first.")
    else:
        grouped = build_timeline(docs)
        for year, entries in grouped.items():
            st.subheader(year)
            for e in entries:
                st.markdown(f"- **{e['category']}** — {e['label']}")

elif page == "Smart Search":
    st.header("Module 5 -- Smart Retrieval System")
    query = st.text_input("Ask in plain English", placeholder="e.g. show my AI projects")
    category_filter = st.selectbox("Filter by category (optional)", ["All"] + list(CATEGORIES.keys()))
    if st.button("Search") and query:
        cat = None if category_filter == "All" else category_filter
        results = semantic_search(query, top_k=5, category=cat)
        answer = generate_answer(query, results)
        st.markdown(f"> {answer}")
        for r in results:
            st.markdown(
                f"**{r['filename']}**  ·  {r['category']}  ·  match score {r['score']}"
            )
            if os.path.exists(r["filepath"]):
                with open(r["filepath"], "rb") as f:
                    st.download_button(
                        "Open original",
                        f.read(),
                        file_name=r["filename"],
                        key=f"search_dl_{r['id']}",
                    )
            st.caption(r["snippet"])
            st.divider()
