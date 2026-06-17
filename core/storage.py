"""SQLite-backed storage for MemoryVerse AI document records.

Keeps things simple and dependency-free (sqlite3 is in the standard
library) while still giving structured, queryable storage for every
ingested document's metadata. Vector embeddings live separately in
ChromaDB (see embeddings.py); this module is the source of truth for
everything else (category, skills, dates, original file path, etc).
"""
import sqlite3
import json
import os
import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
DB_PATH = os.path.join(DATA_DIR, "memoryverse.db")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            filepath TEXT,
            source_type TEXT,
            category TEXT,
            confidence REAL,
            text_snippet TEXT,
            full_text TEXT,
            skills TEXT,
            dates TEXT,
            organization TEXT,
            upload_time TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def add_document(filename, filepath, source_type, category, confidence,
                  full_text, skills, dates, organization=""):
    conn = get_connection()
    cur = conn.execute(
        """INSERT INTO documents
           (filename, filepath, source_type, category, confidence,
            text_snippet, full_text, skills, dates, organization, upload_time)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            filename,
            filepath,
            source_type,
            category,
            confidence,
            (full_text or "")[:300],
            full_text or "",
            json.dumps(skills or []),
            json.dumps(dates or []),
            organization or "",
            datetime.datetime.now().isoformat(timespec="seconds"),
        ),
    )
    conn.commit()
    doc_id = cur.lastrowid
    conn.close()
    return doc_id


def _row_to_dict(row):
    d = dict(row)
    d["skills"] = json.loads(d["skills"] or "[]")
    d["dates"] = json.loads(d["dates"] or "[]")
    return d


def get_all_documents():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM documents ORDER BY id DESC").fetchall()
    conn.close()
    return [_row_to_dict(r) for r in rows]


def get_document(doc_id):
    conn = get_connection()
    row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
    conn.close()
    return _row_to_dict(row) if row else None


def clear_all():
    conn = get_connection()
    conn.execute("DELETE FROM documents")
    conn.commit()
    conn.close()
