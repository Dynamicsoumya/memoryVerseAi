# MemoryVerse AI

An AI-powered Digital Identity System that ingests scattered files (certificates,
resumes, project reports, internship letters, GitHub repos, portfolio links),
automatically categorizes and connects them, and lets you retrieve anything
instantly with plain-English search.

Built for the MemoryVerse AI '26 hackathon brief: *"I never have to search
through folders again."*

## Architecture

```
Upload (certs, resumes, links, repos)
        |
Extraction & parsing (OCR, text + metadata)
        |
    -----------------------
   |                       |
AI categorization     Embeddings & vectors
   |                       |
Knowledge graph + timeline |
   |                       |
    -----------------------
        |
Smart retrieval (RAG)
        |
User dashboard
```

Five modules, mapped directly to the brief:

| Brief module | Implementation |
|---|---|
| 1. AI Data Ingestion | `core/extraction.py` -- PDFs, DOCX, images (OCR), GitHub repos via the GitHub API, portfolio links via web scraping |
| 2. Intelligent Categorization | `core/categorize.py` -- embedding similarity to category prototypes (always works, no key needed), upgraded to an LLM pass when `ANTHROPIC_API_KEY` is set |
| 3. Relationship Engine | `core/relationships.py` -- rule-based skill extraction + a knowledge graph linking Certification -> Skill -> Project -> Internship -> Career Path |
| 4. Digital Journey Timeline | `core/timeline.py` -- extracts years from text, groups documents chronologically |
| 5. Smart Retrieval System | `core/retrieval.py` + `core/embeddings.py` -- semantic search over a local ChromaDB vector index, with an optional LLM-generated natural-language answer |

## Why this design (for your thought-process writeup)

- **Embeddings run locally** (`sentence-transformers`, `all-MiniLM-L6-v2`) so
  categorization and search work offline, with zero risk of an API outage or
  rate limit during a live demo.
- **The LLM layer is optional, not required.** If `ANTHROPIC_API_KEY` is set,
  categorization gets more accurate and search results get a natural-language
  summary. If not, everything still works end to end -- this is a deliberate
  reliability choice, not a missing feature.
- **Original files are never modified.** Every upload is stored as-is in
  `data/uploads/`; only metadata (category, skills, dates) is derived and
  indexed separately. This directly satisfies the brief's "preserving original
  files and formats" requirement.
- **The relationship graph is built from shared skills, not guesswork.**
  Documents that mention the same skill keyword get connected, which is what
  produces the Certification -> Skill -> Project -> Internship chain from
  real data rather than a hardcoded demo path.

## Setup

Requires Python 3.10+.

```bash
cd memoryverse-ai
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

First install will take a few minutes (downloads `sentence-transformers` and
its PyTorch backend). The embedding model itself (~80MB) downloads
automatically the first time the app runs and is cached after that.

**Optional -- enable LLM-enhanced mode:**
```bash
cp .env.example .env
# then edit .env and paste your Anthropic API key
```

**Optional -- enable OCR for scanned certificate images:** install Tesseract
on your system (`brew install tesseract` on macOS, `apt install
tesseract-ocr` on Ubuntu). Without it, image uploads are still stored and
viewable, just not text-searchable.

## Run

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`). Click
**"Load sample data"** in the sidebar to populate the vault instantly with
five sample documents for a quick demo, or upload your own files.

## Demo flow for judges

1. Click **Load sample data** (or upload 3-4 real files live).
2. Go to **Vault** -- show documents auto-sorted into categories with no
   manual tagging.
3. Go to **Knowledge Graph** -- show the skill-based connections between a
   certificate, a project, and an internship.
4. Go to **Timeline** -- show the chronological journey view.
5. Go to **Smart Search** -- type "show my AI projects" or "show my
   certificates" and show the natural-language retrieval pulling the right
   original file back instantly.

## Project structure

```
memoryverse-ai/
├── app.py                  # Streamlit UI, one command to run everything
├── requirements.txt
├── .env.example
├── core/
│   ├── storage.py           # SQLite document records
│   ├── extraction.py        # Module 1: ingestion (files, GitHub, links)
│   ├── embeddings.py        # Local embeddings + ChromaDB vector store
│   ├── categorize.py        # Module 2: categorization
│   ├── relationships.py     # Module 3: skill extraction + knowledge graph
│   ├── timeline.py          # Module 4: chronological timeline
│   └── retrieval.py         # Module 5: semantic search + answers
├── sample_files/             # Sample docs for instant demo
└── data/                      # Created at runtime: uploads, vector store, db
```
