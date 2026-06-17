"""Module 5: Smart Retrieval System.

Natural-language search over everything ever ingested. Embeds the
user's query, runs a similarity search against the vector store, pulls
full metadata from SQLite, and -- if an Anthropic API key is configured
-- asks the LLM for a one-line natural-language answer summarizing the
results. Falls back to a simple templated answer if no key is set.
"""
import os
from dotenv import load_dotenv
from core.embeddings import search
from core.storage import get_document

load_dotenv()


def semantic_search(query, top_k=5, category=None):
    raw_results = search(query, top_k=top_k, category=category)
    results = []
    for r in raw_results:
        doc_id = int(r["id"])
        doc = get_document(doc_id)
        if doc:
            results.append(
                {
                    "id": doc_id,
                    "filename": doc["filename"],
                    "filepath": doc["filepath"],
                    "category": doc["category"],
                    "snippet": r["snippet"][:240],
                    "score": round(r["score"], 3),
                }
            )
    return results


def generate_answer(query, results):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not results:
        return "No matching documents found yet -- try uploading some files first."
    if not api_key:
        names = ", ".join(r["filename"] for r in results[:3])
        return f"Found {len(results)} matching document(s): {names}."
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        context = "\n".join(
            f"- {r['filename']} ({r['category']}): {r['snippet']}" for r in results
        )
        prompt = f"""A user searched their personal document vault for: "{query}"
Matching documents:
{context}

Write a single short, friendly sentence (max 25 words) telling them what was found. Do not list every file."""
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception:
        names = ", ".join(r["filename"] for r in results[:3])
        return f"Found {len(results)} matching document(s): {names}."
