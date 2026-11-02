#!/usr/bin/env python3
"""Show what retrieval returns for one question, and which chunks the relevance gate keeps.
Usage: python debug_query.py <college_id> "your question"
"""
import logging, sys
logging.basicConfig(level=logging.WARNING)

from app.config import Config
from app.storage import Database
from app.vector_store import VectorSearchEngine
from app.rag_engine import RAGEngine
from app.multilingual import detect_language, get_english_search_terms

if len(sys.argv) < 3:
    sys.exit('Usage: python debug_query.py <college_id> "your question"')
college_id, question = int(sys.argv[1]), sys.argv[2]

db = Database()
engine = RAGEngine(db, VectorSearchEngine(db))
lang = detect_language(question)
search_query = get_english_search_terms(question, lang) if lang != "en" else question
print(f"language={lang} | search query={search_query!r} | MIN_COSINE={Config.MIN_COSINE}")

chunks = engine.vector_engine.hybrid_search(college_id, search_query, top_k=10)
kept_ids = {c["id"] for c in engine._relevance_filter(chunks)}
print(f"{len(chunks)} candidates, {len(kept_ids)} kept by the gate\n")
for c in chunks:
    cos = c.get("cosine")
    print(f"{'KEEP' if c['id'] in kept_ids else 'DROP'} | cosine={'n/a' if cos is None else f'{cos:.3f}'} | "
          f"bm25_rank={c.get('bm25_rank', '-')} | {c['url'][:55]} | {(c.get('heading') or '')[:30]}")
