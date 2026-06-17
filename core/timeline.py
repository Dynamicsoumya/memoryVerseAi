"""Module 4: Digital Journey Timeline.

Extracts years mentioned in document text and groups documents
chronologically, matching the brief's example format
(2023 -> Python Certification, 2024 -> Data Science Club Lead, ...).
"""
import re

YEAR_PATTERN = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")


def extract_dates(text):
    if not text:
        return []
    years = YEAR_PATTERN.findall(text)
    return sorted(set(years))


def build_timeline(documents):
    """documents: list of dicts with filename, category, dates (list of year strings).
    Returns an ordered dict-like mapping of year -> list of entries."""
    entries = []
    for doc in documents:
        years = doc.get("dates") or []
        year = years[0] if years else "Undated"
        entries.append(
            {
                "year": year,
                "label": doc["filename"],
                "category": doc["category"],
            }
        )

    def sort_key(e):
        return (e["year"] == "Undated", e["year"])

    entries.sort(key=sort_key)

    grouped = {}
    for e in entries:
        grouped.setdefault(e["year"], []).append(e)
    return grouped
