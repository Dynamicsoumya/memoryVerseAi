# Thought process -- MemoryVerse AI

*(Fill in the bracketed parts with your own voice before submitting --
judges weight this at 15%, specifically for clarity of explanation.)*

## The problem we saw

A student's digital footprint -- certificates, resumes, project reports,
internship letters, GitHub repos -- ends up scattered across folders, email
attachments, and cloud drives. None of those tools *understand* the content;
they just store bytes. We wanted to build something that understands a
person's growth, not just their files.

## Why we didn't just build "another Google Drive"

[Your note here: e.g. "Storage is a solved problem. The brief explicitly
says this isn't another cloud storage platform -- so we focused our build
time on the three things storage can't do: classify without manual sorting,
find connections across documents, and answer questions in plain English."]

## Architecture decisions and trade-offs

- **Local embeddings over a pure LLM pipeline.** We use
  `sentence-transformers` for categorization and search so the system works
  completely offline. An LLM call (Claude) is layered on top *only* when an
  API key is available, for richer skill/date extraction and natural-language
  answers. Trade-off: the offline path is less nuanced than a full LLM read of
  each document, but it's zero-cost, has no rate limits, and can't fail mid-demo.
- **A real vector database (ChromaDB), not keyword search.** This is what
  makes "show my AI projects" work as a semantic match instead of requiring
  the exact word "AI" to appear in a filename.
- **Skill-based graph edges instead of a hardcoded relationship map.** The
  knowledge graph connects documents that share an extracted skill keyword,
  so the Certification -> Skill -> Project -> Internship chain emerges from
  the actual uploaded data rather than being faked for the demo.
- **Original files are never altered.** Everything is stored byte-for-byte
  in `data/uploads/`; only derived metadata is processed, satisfying the
  brief's requirement to preserve original formats.

## What we'd build next with more time

[Your note here: e.g. "Multi-document resume parsing that splits one resume
into separate Skill/Project/Academic records instead of a single category;
a proper graph database (Neo4j) instead of an in-memory graph for larger
vaults; richer date parsing beyond bare years."]

## The "wow" moment

[Describe the exact demo beat where this clicks for a viewer -- e.g. "We
upload a certificate, a project report, and an internship letter live. Thirty
seconds later, the knowledge graph shows all three connected through the
shared skill 'Machine Learning,' and a plain-English search for 'show my AI
work' pulls back the right files without anyone touching a folder."]
