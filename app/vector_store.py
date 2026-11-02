import logging
import math
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from app.config import Config
from app.storage import Database

logger = logging.getLogger(__name__)

class VectorSearchEngine:
    """Hybrid Retrieval Engine combining Dense Embeddings (Gemini/OpenAI) + Sparse BM25 (SQLite FTS)."""

    def __init__(self, db: Optional[Database] = None):
        self.db = db or Database()
        self._gemini_client = None
        self._openai_client = None
        # In-memory embedding matrix cache: college_id -> (chunk_ids, matrix, row_count)
        # Avoids repeated DB round-trips and enables fast vectorized cosine.
        self._emb_cache: Dict[int, Tuple[List[int], np.ndarray, int]] = {}

    def _get_gemini_client(self):
        if not self._gemini_client and Config.GEMINI_API_KEY:
            try:
                from google import genai
                self._gemini_client = genai.Client(api_key=Config.GEMINI_API_KEY)
            except Exception as e:
                logger.warning(f"Could not initialize Gemini client: {e}")
        return self._gemini_client

    def _get_openai_client(self):
        if not self._openai_client and Config.OPENAI_API_KEY:
            try:
                from openai import OpenAI
                self._openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI client: {e}")
        return self._openai_client

    def _embed_config(self, kind: str):
        """Gemini embeddings are asymmetric: documents and queries use different task types."""
        try:
            from google.genai import types
            return types.EmbedContentConfig(task_type="RETRIEVAL_QUERY" if kind == "query" else "RETRIEVAL_DOCUMENT")
        except Exception:
            return None

    def generate_embedding(self, text: str, kind: str = "document") -> Optional[np.ndarray]:
        """Generate dense vector embedding for a query (kind="query") or a chunk (kind="document")."""
        # 1. Gemini. If a Gemini key is configured we NEVER fall back to another model or the hash:
        #    mixing embedding models in one index breaks similarity search. Return None instead.
        client = self._get_gemini_client()
        if client:
            try:
                resp = client.models.embed_content(
                    model=Config.GEMINI_EMBED_MODEL,
                    contents=text[:6000],
                    config=self._embed_config(kind),
                )
                if resp and resp.embeddings:
                    vec = np.array(resp.embeddings[0].values, dtype=np.float32)
                    return vec / (np.linalg.norm(vec) + 1e-9)
            except Exception as e:
                logger.warning(f"Gemini embedding {Config.GEMINI_EMBED_MODEL} failed: {str(e)[:300]}")
            return None

        # 2. Try OpenAI
        openai_client = self._get_openai_client()
        if openai_client:
            try:
                resp = openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=text[:8000]
                )
                vec = np.array(resp.data[0].embedding, dtype=np.float32)
                norm = np.linalg.norm(vec)
                return vec / (norm + 1e-9)
            except Exception as e:
                logger.debug(f"OpenAI embedding failed: {e}")

        # 3. Deterministic word-hash fallback for zero-dependency vector matching
        return self._fallback_hash_embedding(text)

    def _fallback_hash_embedding(self, text: str, dim: int = 512) -> np.ndarray:
        """Stable, deterministic TF-IDF feature hashing embedding with stop-word removal."""
        import hashlib
        from app.storage import STOP_WORDS

        vec = np.zeros(dim, dtype=np.float32)
        words = [w.lower() for w in "".join(c if c.isalnum() else " " for c in text).split() if len(w) >= 3]
        content_words = [w for w in words if w not in STOP_WORDS]
        tokens = content_words if content_words else words

        from collections import Counter
        counts = Counter(tokens)

        for w, count in counts.items():
            # Stable deterministic hash using md5
            h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16) % dim
            # Sublinear term frequency
            tf = 1.0 + math.log(count)
            vec[h] += float(tf)

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def index_pending_chunks(self, college_id: int, batch_size: int = 50) -> int:
        """Generate and save embeddings for any chunks lacking them."""
        pending = self.db.get_chunks_without_embeddings(college_id)
        if not pending:
            return 0

        indexed = 0
        client = self._get_gemini_client()
        if client:
            # One request per batch instead of per chunk, and stop cleanly on quota/other errors.
            # Chunks that fail stay without an embedding (never a hash vector) and are retried on the next run.
            for i in range(0, len(pending), batch_size):
                batch = pending[i:i + batch_size]
                texts = [f"{c.get('title', '')} {c.get('heading', '')} {c['text']}"[:6000] for c in batch]
                try:
                    resp = client.models.embed_content(
                        model=Config.GEMINI_EMBED_MODEL, contents=texts, config=self._embed_config("document")
                    )
                    vecs = [np.array(e.values, dtype=np.float32) for e in resp.embeddings]
                    if len(vecs) != len(batch):
                        raise RuntimeError(f"expected {len(batch)} embeddings, got {len(vecs)}")
                except Exception as e:
                    logger.error(
                        f"Embedding stopped after {indexed}/{len(pending)} chunks (college {college_id}): {str(e)[:300]}. "
                        "Remaining chunks stay pending; run `python reindex.py` later (e.g. after the quota resets)."
                    )
                    break
                batch_updates = [(chunk["id"], vec / (np.linalg.norm(vec) + 1e-9)) for chunk, vec in zip(batch, vecs)]
                self.db.update_chunk_embeddings_batch(batch_updates)
                indexed += len(batch_updates)
            logger.info(f"Generated Gemini embeddings for {indexed}/{len(pending)} chunks in college {college_id}")
            return indexed

        fallback_updates = []
        for chunk in pending:
            content_to_embed = f"{chunk.get('title', '')} {chunk.get('heading', '')} {chunk['text']}"
            vec = self.generate_embedding(content_to_embed)
            if vec is not None:
                fallback_updates.append((chunk["id"], vec))
        if fallback_updates:
            self.db.update_chunk_embeddings_batch(fallback_updates)
            indexed += len(fallback_updates)

        logger.info(f"Generated embeddings for {indexed} chunks in college {college_id}")
        self._emb_cache.pop(college_id, None)
        return indexed

    BOILERPLATE_MIN_PAGES = 3
    _boiler_cache: Dict[int, Tuple[int, set]] = {}

    def _boilerplate_ids(self, college_id: int) -> set:
        """IDs of chunks whose text repeats on >= BOILERPLATE_MIN_PAGES different pages (menus, footers, banners)."""
        from urllib.parse import urlparse
        with self.db.get_connection() as conn:
            rows = conn.execute("SELECT id, url, text FROM chunks WHERE college_id = ?", (college_id,)).fetchall()
        cached = self._boiler_cache.get(college_id)
        if cached and cached[0] == len(rows):
            return cached[1]
        pages_by_key: Dict[str, set] = {}
        key_of: Dict[int, str] = {}
        for cid, url, text in rows:
            body = text.split("\n", 1)[1] if text.startswith("[") and "\n" in text else text  # ignore the "[title - heading]" line
            key = " ".join(body.lower().split())[:300]
            u = urlparse(url)
            page = (u.netloc + u.path).rstrip("/").lower()  # http/https duplicates count as one page
            pages_by_key.setdefault(key, set()).add(page)
            key_of[cid] = key
        ids = {cid for cid, k in key_of.items() if len(pages_by_key[k]) >= self.BOILERPLATE_MIN_PAGES}
        self._boiler_cache[college_id] = (len(rows), ids)
        return ids

    HASH_DIM = 512  # dimension of the hash fallback; its cosines are NOT calibrated, so never gate on them

    def _get_embedding_matrix(self, college_id: int) -> Tuple[List[int], np.ndarray]:
        """Return (chunk_ids, matrix) for a college, rebuilding from DB only when new chunks appear.
        Uses a vectorized numpy matrix for fast batch cosine similarity."""
        all_vecs = self.db.get_all_chunk_embeddings(college_id)
        cached = self._emb_cache.get(college_id)
        # Same count AND same last chunk id (a re-crawl can produce the same count with new chunks)
        if cached and cached[2] == len(all_vecs) and all_vecs and cached[0][-1] == all_vecs[-1][0]:
            return cached[0], cached[1]

        if not all_vecs:
            return [], np.empty((0,), dtype=np.float32)

        ids = [cid for cid, _ in all_vecs]
        matrix = np.stack([vec for _, vec in all_vecs])  # (N, dim)
        self._emb_cache[college_id] = (ids, matrix, len(all_vecs))
        return ids, matrix

    def _cosine_scores(self, college_id: int, query_vec: np.ndarray) -> Dict[int, float]:
        """Vectorized cosine similarity: single np.dot over the full embedding matrix (no Python loop)."""
        ids, matrix = self._get_embedding_matrix(college_id)
        if matrix.ndim < 2 or matrix.shape[0] == 0:
            return {}

        # Filter rows whose dimension matches the query (skip stale mixed-model embeddings)
        q_dim = query_vec.shape[0]
        if matrix.shape[1] != q_dim:
            logger.error(
                f"Embedding dimension mismatch for college {college_id}: query dim={q_dim}, "
                f"stored dim={matrix.shape[1]}. Re-index this college."
            )
            return {}

        sims = matrix @ query_vec          # (N,) — all cosines in one BLAS call
        return {cid: float(s) for cid, s in zip(ids, sims)}

    def search_vector(self, college_id: int, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search chunks by cosine similarity."""
        query_vec = self.generate_embedding(query, kind="query")
        if query_vec is None:
            return []
        scores = self._cosine_scores(college_id, query_vec)
        results = []
        for chunk_id in sorted(scores, key=scores.get, reverse=True)[:top_k]:
            chunk = self.db.get_chunk_by_id(chunk_id)
            if chunk:
                chunk["score"] = scores[chunk_id]
                results.append(chunk)
        return results

    @staticmethod
    def _dedupe(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Drop chunks whose text is identical (same content crawled under several URLs)."""
        seen, unique = set(), []
        for c in chunks:
            key = " ".join((c.get("text") or "").lower().split())[:400]
            if key in seen:
                continue
            seen.add(key)
            unique.append(c)
        return unique

    def hybrid_search(self, college_id: int, query: str, top_k: int = 8, candidates: int = 30) -> List[Dict[str, Any]]:
        """RRF over BM25 + vector rankings. Every result also carries its absolute cosine similarity."""
        query_vec = self.generate_embedding(query, kind="query")
        cosines = self._cosine_scores(college_id, query_vec) if query_vec is not None else {}
        semantic = query_vec is not None and query_vec.shape[0] != self.HASH_DIM

        bm25_results = self.db.search_bm25(college_id, query, limit=candidates)
        vector_ids = sorted(cosines, key=cosines.get, reverse=True)[:candidates]

        rrf_constant = 60
        scores: Dict[int, float] = {}
        chunk_map: Dict[int, Dict[str, Any]] = {}
        for rank, item in enumerate(bm25_results, 1):
            scores[item["id"]] = scores.get(item["id"], 0.0) + 1.0 / (rrf_constant + rank)
            item["bm25_rank"] = rank
            chunk_map[item["id"]] = item
        for rank, cid in enumerate(vector_ids, 1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_constant + rank)

        if not scores:
            return []

        # Batch-fetch any chunks that came from vector search only (avoids N individual DB queries)
        missing_ids = [cid for cid in vector_ids if cid not in chunk_map]
        if missing_ids:
            for chunk in self.db.get_chunks_by_ids(missing_ids):
                chunk_map[chunk["id"]] = chunk

        boilerplate = self._boilerplate_ids(college_id)
        results = []
        for cid in sorted(scores, key=scores.get, reverse=True):
            if cid not in chunk_map or cid in boilerplate:
                continue
            chunk = dict(chunk_map[cid])
            chunk["rrf_score"] = scores[cid]
            chunk["cosine"] = cosines.get(cid)
            chunk["semantic"] = semantic
            results.append(chunk)
        return self._dedupe(results)[:top_k]
