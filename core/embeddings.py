"""Embeddings + persistent vector store (ChromaDB), used for Module 2
(categorization) and Module 5 (smart retrieval).

Runs entirely locally with sentence-transformers -- no API key and no
network call required at query time, which matters for a live demo
(no risk of an API outage or rate limit mid-presentation). Embeddings
are computed once here and passed explicitly into Chroma, rather than
relying on Chroma's own default embedding function.
"""
import os
from sentence_transformers import SentenceTransformer
import chromadb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.path.join(BASE_DIR, "data", "chroma_db")
os.makedirs(CHROMA_DIR, exist_ok=True)

_model = None
_client = None
_collection = None


def get_embedder():
    global _model
    if _model is None:
        # Small (~80MB), fast on CPU, good general-purpose semantic embedding model.
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed_text(text):
    model = get_embedder()
    return model.encode(text, normalize_embeddings=True).tolist()


def get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = _client.get_or_create_collection(name="memoryverse_docs")
    return _collection


def index_document(doc_id, text, metadata):
    """Add (or overwrite) a document's embedding in the vector store."""
    if not text or not text.strip():
        text = metadata.get("filename", "untitled")
    collection = get_collection()
    vector = embed_text(text)
    collection.upsert(
        ids=[str(doc_id)],
        embeddings=[vector],
        documents=[text[:2000]],
        metadatas=[metadata],
    )


def search(query, top_k=5, category=None):
    collection = get_collection()
    vector = embed_text(query)
    kwargs = {"query_embeddings": [vector], "n_results": top_k}
    if category:
        kwargs["where"] = {"category": category}
    results = collection.query(**kwargs)

    out = []
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    for i in range(len(ids)):
        out.append(
            {
                "id": ids[i],
                "snippet": docs[i],
                "metadata": metas[i],
                "score": 1 - dists[i],  # cosine distance -> similarity
            }
        )
    return out


def reset_index():
    """Used by the 'Clear all data' button so demo runs can be repeated cleanly."""
    global _client, _collection
    _client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        _client.delete_collection("memoryverse_docs")
    except Exception:
        pass
    _collection = _client.get_or_create_collection(name="memoryverse_docs")
