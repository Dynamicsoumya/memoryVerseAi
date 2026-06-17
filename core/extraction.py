"""Module 1: AI Data Ingestion.

Extracts raw text from whatever a user throws at the system: PDFs,
Word docs, plain text, certificate images (via OCR), public GitHub
repos, and portfolio/profile links. Every function fails soft -- it
returns a string (possibly empty or an error note) rather than raising,
so one bad upload never crashes the whole ingestion batch.
"""
import os
import requests
from bs4 import BeautifulSoup

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    import docx
except ImportError:
    docx = None

try:
    import pytesseract
    from PIL import Image
except ImportError:
    pytesseract = None
    Image = None


def extract_text_from_file(filepath):
    """Dispatch text extraction based on file extension. Always returns a string."""
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext == ".pdf":
            return _extract_pdf(filepath)
        elif ext == ".docx":
            return _extract_docx(filepath)
        elif ext in (".txt", ".md"):
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        elif ext in (".png", ".jpg", ".jpeg", ".webp"):
            return _extract_image(filepath)
        else:
            return ""
    except Exception as e:
        return f"[extraction error: {e}]"


def _extract_pdf(filepath):
    if PdfReader is None:
        return ""
    reader = PdfReader(filepath)
    text = []
    for page in reader.pages:
        text.append(page.extract_text() or "")
    return "\n".join(text)


def _extract_docx(filepath):
    if docx is None:
        return ""
    d = docx.Document(filepath)
    return "\n".join(p.text for p in d.paragraphs)


def _extract_image(filepath):
    """OCR a certificate/screenshot. Returns '' silently if Tesseract isn't
    installed on the system -- the file is still stored and viewable, it
    just won't be text-searchable until Tesseract is available."""
    if pytesseract is None or Image is None:
        return ""
    try:
        img = Image.open(filepath)
        return pytesseract.image_to_string(img)
    except Exception:
        return ""


def fetch_github_repo_info(url):
    """Pull README + description + language from a public GitHub repo URL
    using the public GitHub REST API (no auth needed for public repos)."""
    try:
        parts = url.rstrip("/").split("github.com/")[-1].split("/")
        owner, repo = parts[0], parts[1]
    except Exception:
        return f"[invalid GitHub URL: {url}]"

    text_parts = []
    try:
        repo_resp = requests.get(
            f"https://api.github.com/repos/{owner}/{repo}", timeout=10
        )
        if repo_resp.ok:
            data = repo_resp.json()
            text_parts.append(f"Repository: {data.get('full_name', repo)}")
            text_parts.append(f"Description: {data.get('description') or ''}")
            text_parts.append(f"Primary language: {data.get('language') or ''}")
            topics = data.get("topics") or []
            if topics:
                text_parts.append(f"Topics: {', '.join(topics)}")
        else:
            text_parts.append(
                f"[GitHub API returned {repo_resp.status_code}: "
                f"{repo_resp.json().get('message', '')}]"
            )
    except Exception as e:
        text_parts.append(f"[repo metadata error: {e}]")

    try:
        readme_resp = requests.get(
            f"https://api.github.com/repos/{owner}/{repo}/readme",
            headers={"Accept": "application/vnd.github.raw"},
            timeout=10,
        )
        if readme_resp.ok:
            text_parts.append(readme_resp.text[:4000])
    except Exception:
        pass

    return "\n".join(text_parts)


def fetch_url_text(url):
    """Fetch a portfolio/profile link and return its readable text content."""
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "MemoryVerseAI/1.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        return "\n".join(lines)[:6000]
    except Exception as e:
        return f"[link fetch error: {e}]"
