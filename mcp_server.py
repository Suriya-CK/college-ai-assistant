#!/usr/bin/env python3
"""MCP server: lets AI agents (Claude Desktop, Cursor, agent frameworks) use the college knowledge base as tools.

MCP (Model Context Protocol) is a standard way to offer tools to an AI. The agent reads each tool's name,
description and parameters, then decides by itself when to call them. This server exposes the same data
the web app uses: crawled pages (hybrid search), structured facts and the full RAG answer pipeline.

Run:  .venv/bin/python mcp_server.py          (stdio transport, started by the AI client)
"""

import sys
import os
import logging
from pathlib import Path

# Auto-activate .venv if run with system python
venv_dir = Path(__file__).resolve().parent / ".venv"
if venv_dir.exists() and sys.prefix != str(venv_dir):
    venv_python = venv_dir / "bin" / "python"
    os.execv(str(venv_python), [str(venv_python)] + sys.argv)

from typing import Any, Dict, List, Optional

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from app.storage import Database
from app.vector_store import VectorSearchEngine
from app.rag_engine import RAGEngine, merge_similar_facts
from app.extractor import CATEGORIES

# stdio transport uses stdout for protocol messages, so all logging goes to stderr
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

mcp = MCPServer(
    name="college-assistant",
    instructions=(
        "Answers questions about Indian colleges from their crawled official websites. "
        "Call list_colleges first to find a college_id. Use get_facts for complete lists "
        "(courses, recruiters, placement stats, fees), search_college for anything else, "
        "and ask_college for a ready-made answer with sources in the user's language."
    ),
)

db = Database()
vector_engine = VectorSearchEngine(db)
_rag: Optional[RAGEngine] = None


def rag() -> RAGEngine:
    """Created on first use: only ask_college needs an LLM."""
    global _rag
    if _rag is None:
        _rag = RAGEngine(db, vector_engine)
    return _rag


def _require_college(college_id: int) -> Dict[str, Any]:
    college = db.get_college(college_id)
    if not college:
        raise ToolError(f"No college with id {college_id}. Call list_colleges to see the available ids.")
    return college


@mcp.tool()
def list_colleges() -> List[Dict[str, Any]]:
    """List every college in the knowledge base with its id, name, website and how many pages were collected."""
    return [
        {
            "college_id": c["id"],
            "name": c["name"],
            "website": c["url"],
            "status": c["status"],
            "pages": c["page_count"],
            "facts": sum(db.get_fact_counts(c["id"]).values()),
        }
        for c in db.list_colleges()
    ]


@mcp.tool()
def get_facts(college_id: int, category: str) -> Dict[str, Any]:
    """Complete list of structured facts extracted from a college's website for one category.

    Use this for "list all ..." questions, where search would only return a few passages.
    Categories: ug_program, pg_program, doctoral_program, department, recruiter, placement_stat,
    fee, scholarship, admission, facility, accreditation, contact.
    """
    college = _require_college(college_id)
    if category not in CATEGORIES:
        raise ToolError(f"Unknown category '{category}'. Use one of: {', '.join(CATEGORIES)}")
    facts = merge_similar_facts(db.get_facts(college_id, [category]))
    return {
        "college": college["name"],
        "category": category,
        "count": len(facts),
        "facts": [{"name": f["name"], "details": f["details"], "source": f["source_url"]} for f in facts],
    }


@mcp.tool()
def search_college(college_id: int, query: str, top_k: int = 5) -> Dict[str, Any]:
    """Search a college's crawled web pages (hybrid keyword BM25 + semantic search) and return matching passages with source URLs."""
    college = _require_college(college_id)
    top_k = max(1, min(top_k, 10))
    results = vector_engine.hybrid_search(college_id, query, top_k=top_k)
    return {
        "college": college["name"],
        "query": query,
        "results": [
            {"title": r.get("title"), "section": r.get("heading"), "url": r.get("url"), "text": r.get("text", "")[:1500]}
            for r in results
        ],
    }


@mcp.tool()
def ask_college(college_id: int, question: str) -> Dict[str, Any]:
    """Answer a question about a college with the full RAG pipeline (search + facts + LLM), with source links.

    Works in English and Indian languages (Tamil, Hindi, Telugu, ...); the answer is in the question's language.
    Needs an LLM API key in the project's .env; for raw data without an LLM use get_facts or search_college.
    """
    _require_college(college_id)
    result = rag().answer_question(college_id, question)
    if result.get("error"):
        raise ToolError(result["error"])
    return {
        "college": result.get("college_name"),
        "answer": result.get("content"),
        "sources": [{"title": c.get("title"), "url": c.get("url")} for c in result.get("citations", [])],
        "answered_by": result.get("provider"),
    }


if __name__ == "__main__":
    mcp.run()   # stdio: the AI client starts this process and talks to it over stdin/stdout
