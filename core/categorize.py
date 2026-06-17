"""Module 2: Intelligent Categorization.

Two layers, so the app never breaks during a demo:

1. Always-on: embedding similarity against a prototype description of
   each category (a lightweight zero-shot classifier built from the
   same sentence-transformer used for search -- no extra model needed).
2. Optional upgrade: if ANTHROPIC_API_KEY is set, an LLM call classifies
   the document AND extracts structured skills/organization/dates in
   one shot, which is more accurate. If the key is missing or the call
   fails for any reason, it falls back to layer 1 automatically.
"""
import os
import json
import numpy as np
from dotenv import load_dotenv
from core.embeddings import embed_text

load_dotenv()

CATEGORIES = {
    "Projects": "A project report, software project, application built, project documentation, or project demo showcasing technical work",
    "Skills": "A certificate or proof of a technical or soft skill such as a programming language, tool, framework, or competency",
    "Certifications": "A certificate of course completion, professional certification, or training certificate from an online or offline program",
    "Internships": "An internship offer letter, internship completion certificate, or internship experience letter from a company",
    "Achievements": "An award, recognition, competition win, hackathon prize, scholarship, or other notable achievement",
    "Academics": "An academic transcript, marksheet, degree certificate, or school/college academic record",
}

_category_vectors = None


def _get_category_vectors():
    global _category_vectors
    if _category_vectors is None:
        _category_vectors = {
            name: np.array(embed_text(desc)) for name, desc in CATEGORIES.items()
        }
    return _category_vectors


def classify_embedding(text):
    """Assigns a category by cosine similarity to category prototype descriptions."""
    if not text or not text.strip():
        return "Academics", 0.0
    vec = np.array(embed_text(text))
    vectors = _get_category_vectors()
    best_cat, best_score = None, -1.0
    for cat, cvec in vectors.items():
        score = float(np.dot(vec, cvec))  # both normalized -> cosine similarity
        if score > best_score:
            best_cat, best_score = cat, score
    return best_cat, round(best_score, 3)


def classify_with_llm(text, filename=""):
    """Optional richer pass using Claude. Returns a dict, or None if the
    API key is missing or the call fails for any reason."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        prompt = f"""You are classifying a document for a personal digital identity system.
Filename: {filename}
Content (truncated): {text[:3000]}

Return ONLY a JSON object, no other text, with this exact shape:
{{"category": "Projects|Skills|Certifications|Internships|Achievements|Academics",
  "skills": ["skill1", "skill2"],
  "organization": "issuing org or company if mentioned, else empty string",
  "dates": ["any year mentioned, e.g. 2024"]}}"""
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception:
        return None


def classify_document(text, filename=""):
    """Main entry point used by the app. Tries the LLM pass first (if a key
    is configured), and always has the embedding pass as a safety net."""
    llm_result = classify_with_llm(text, filename)
    if llm_result and llm_result.get("category") in CATEGORIES:
        return {
            "category": llm_result["category"],
            "confidence": 0.95,
            "skills": llm_result.get("skills", []),
            "organization": llm_result.get("organization", ""),
            "dates": llm_result.get("dates", []),
            "method": "llm",
        }
    category, score = classify_embedding(text)
    return {
        "category": category,
        "confidence": score,
        "skills": [],
        "organization": "",
        "dates": [],
        "method": "embedding",
    }
