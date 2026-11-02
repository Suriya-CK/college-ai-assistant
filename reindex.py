#!/usr/bin/env python3
"""Embed any chunks that still lack embeddings. Safe to re-run, e.g. after a Gemini quota reset."""
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

from app.storage import Database
from app.vector_store import VectorSearchEngine

db = Database()
engine = VectorSearchEngine(db)
with db.get_connection() as conn:
    college_ids = [row[0] for row in conn.execute("SELECT id FROM colleges").fetchall()]

for cid in college_ids:
    pending = len(db.get_chunks_without_embeddings(cid))
    print(f"college {cid}: {pending} chunks pending")
    if pending:
        print(f"  embedded {engine.index_pending_chunks(cid)} of {pending}")
