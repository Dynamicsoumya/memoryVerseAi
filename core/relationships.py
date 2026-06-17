"""Module 3: Relationship Engine.

Extracts skill keywords from document text (rule-based, always works
offline) and builds a knowledge graph connecting documents through
shared skills, following the chain described in the brief:
Certification -> Skill -> Project -> Internship -> Career Path.
"""
import os
import re
import tempfile
import networkx as nx
from pyvis.network import Network

SKILLS_VOCAB = [
    "Python", "Java", "JavaScript", "TypeScript", "C++", "C", "SQL", "HTML", "CSS",
    "React", "Node.js", "Next.js", "Django", "Flask", "FastAPI", "Spring Boot",
    "Machine Learning", "Deep Learning", "Data Science", "Data Analysis",
    "Natural Language Processing", "NLP", "Computer Vision", "TensorFlow",
    "PyTorch", "Scikit-learn", "Pandas", "NumPy", "AWS", "Azure", "GCP",
    "Docker", "Kubernetes", "Git", "GitHub", "Linux", "MongoDB", "PostgreSQL",
    "MySQL", "REST API", "GraphQL", "Power BI", "Tableau", "Excel",
    "Communication", "Leadership", "Project Management", "UI/UX Design",
    "Figma", "Android Development", "iOS Development", "Flutter", "Kotlin",
    "Swift", "Blockchain", "Cybersecurity", "DevOps", "Embedded Systems",
    "Internet of Things", "IoT", "Cloud Computing", "Data Structures",
    "Algorithms", "Agile", "Scrum",
]


def extract_skills(text):
    """Word-boundary keyword extraction against a curated vocabulary --
    always available, no model required. Uses lookaround assertions
    (not \\b) so short/symbolic skills like 'C' or 'C++' don't false-match
    inside unrelated words like 'React' or 'Machine'."""
    if not text:
        return []
    found = []
    for skill in SKILLS_VOCAB:
        pattern = r"(?<![A-Za-z0-9])" + re.escape(skill) + r"(?![A-Za-z0-9])"
        if re.search(pattern, text, re.IGNORECASE):
            found.append(skill)
    return sorted(set(found))


def build_graph(documents):
    """documents: list of dicts with keys id, filename, category, skills (list)."""
    g = nx.Graph()
    g.add_node("Career Path", type="terminal")

    for doc in documents:
        doc_label = f"{doc['category']}: {doc['filename'][:22]}"
        g.add_node(doc_label, type="document", category=doc["category"])
        for skill in doc.get("skills", []):
            if not g.has_node(skill):
                g.add_node(skill, type="skill")
            g.add_edge(doc_label, skill)

    by_category = {}
    for doc in documents:
        by_category.setdefault(doc["category"], []).append(doc)

    def shared_skill_edges(cat_a, cat_b):
        for a in by_category.get(cat_a, []):
            for b in by_category.get(cat_b, []):
                if set(a.get("skills", [])) & set(b.get("skills", [])):
                    label_a = f"{a['category']}: {a['filename'][:22]}"
                    label_b = f"{b['category']}: {b['filename'][:22]}"
                    g.add_edge(label_a, label_b)

    # Follows the brief's example chain: Certification -> Skill -> Project -> Internship
    shared_skill_edges("Certifications", "Projects")
    shared_skill_edges("Skills", "Projects")
    shared_skill_edges("Projects", "Internships")

    for doc in by_category.get("Internships", []):
        label = f"{doc['category']}: {doc['filename'][:22]}"
        g.add_edge(label, "Career Path")

    return g


def render_graph_html(graph, height="600px"):
    """Renders the graph to a self-contained HTML string for embedding via
    streamlit.components.v1.html()."""
    net = Network(height=height, width="100%", bgcolor="#ffffff", font_color="#222222", notebook=False)

    color_map = {
        "Projects": "#7F77DD",
        "Skills": "#1D9E75",
        "Certifications": "#378ADD",
        "Internships": "#D85A30",
        "Achievements": "#D4537E",
        "Academics": "#888780",
    }

    for node, attrs in graph.nodes(data=True):
        if attrs.get("type") == "skill":
            net.add_node(node, label=node, color="#F2A623", shape="dot", size=14)
        elif attrs.get("type") == "terminal":
            net.add_node(node, label=node, color="#333333", shape="star", size=22)
        else:
            cat = attrs.get("category", "")
            net.add_node(node, label=node, color=color_map.get(cat, "#999999"), shape="box")

    for a, b in graph.edges():
        net.add_edge(a, b)

    net.set_options(
        """
        var options = {
          "physics": { "stabilization": true, "barnesHut": { "springLength": 120 } },
          "interaction": { "hover": true }
        }
        """
    )

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
        tmp_path = tmp.name
    net.write_html(tmp_path)
    with open(tmp_path, "r", encoding="utf-8") as f:
        html = f.read()
    os.remove(tmp_path)
    return html
