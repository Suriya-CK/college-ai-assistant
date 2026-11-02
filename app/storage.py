import sqlite3
import json
import logging
import re
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager
import numpy as np

from app.config import Config

logger = logging.getLogger(__name__)

STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing",
    "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
    "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is",
    "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves",
    "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
    "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to",
    "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's",
    "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've",
    "your", "yours", "yourself", "yourselves", "tell", "give", "show", "know", "please", "can", "want"
}

class Database:
    """SQLite Database manager for college knowledge and chat sessions."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Config.DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA cache_size = -64000;")  # 64MB memory cache
        conn.execute("PRAGMA temp_store = MEMORY;")
        try:
            yield conn
        finally:
            conn.close()

    def _init_schema(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode = WAL;")

            # Colleges Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS colleges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT UNIQUE NOT NULL,
                    domain TEXT NOT NULL,
                    name TEXT,
                    description TEXT,
                    status TEXT DEFAULT 'pending', -- pending, crawling, completed, failed
                    page_count INTEGER DEFAULT 0,
                    chunk_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Pages Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    college_id INTEGER NOT NULL,
                    url TEXT NOT NULL,
                    title TEXT,
                    clean_text TEXT,
                    word_count INTEGER DEFAULT 0,
                    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (college_id, url),
                    FOREIGN KEY (college_id) REFERENCES colleges(id) ON DELETE CASCADE
                );
            """)
            self._migrate_page_uniqueness(conn)

            # Chunks Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    college_id INTEGER NOT NULL,
                    page_id INTEGER NOT NULL,
                    url TEXT NOT NULL,
                    title TEXT,
                    heading TEXT,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (college_id) REFERENCES colleges(id) ON DELETE CASCADE,
                    FOREIGN KEY (page_id) REFERENCES pages(id) ON DELETE CASCADE
                );
            """)

            # Full-Text Search (FTS5) for fast BM25 keyword matching
            try:
                cursor.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                        chunk_id UNINDEXED,
                        college_id UNINDEXED,
                        title,
                        heading,
                        text,
                        tokenize='porter unicode61'
                    );
                """)
            except sqlite3.OperationalError:
                # Fallback if standard fts5 tokenizer flags vary
                cursor.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                        chunk_id UNINDEXED,
                        college_id UNINDEXED,
                        title,
                        heading,
                        text
                    );
                """)

            # Conversations Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    college_id INTEGER NOT NULL,
                    title TEXT DEFAULT 'College Discussion',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (college_id) REFERENCES colleges(id) ON DELETE CASCADE
                );
            """)

            # Messages Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL, -- 'user' or 'assistant'
                    content TEXT NOT NULL,
                    spoken_text TEXT,
                    citations TEXT, -- JSON array of cited sources
                    audio_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
            """)

            # Structured facts extracted from pages (courses, recruiters, fees...), used for "list all" questions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    college_id INTEGER NOT NULL,
                    category TEXT NOT NULL,
                    name TEXT NOT NULL,
                    name_key TEXT NOT NULL,
                    details TEXT,
                    source_url TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (college_id, category, name_key),
                    FOREIGN KEY (college_id) REFERENCES colleges(id) ON DELETE CASCADE
                );
            """)

            # Indexing for high speed
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_facts_college ON facts(college_id, category);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pages_college ON pages(college_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_college ON chunks(college_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);")

            conn.commit()

    @staticmethod
    def _migrate_page_uniqueness(conn):
        """Older databases made page URLs unique across ALL colleges, so a page another college had
        already crawled (e.g. ceg.annauniv.edu inside the Anna University crawl) could never belong to a
        second college. Rebuild the table so a URL is unique per college instead. Rows and ids are kept."""
        row = conn.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'pages'").fetchone()
        if not row or "UNIQUE (college_id, url)" in row[0]:
            return
        conn.commit()
        conn.execute("PRAGMA foreign_keys = OFF;")   # dropping the old table must not cascade into chunks
        try:
            conn.execute("BEGIN")
            conn.execute("""
                CREATE TABLE pages_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    college_id INTEGER NOT NULL,
                    url TEXT NOT NULL,
                    title TEXT,
                    clean_text TEXT,
                    word_count INTEGER DEFAULT 0,
                    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (college_id, url),
                    FOREIGN KEY (college_id) REFERENCES colleges(id) ON DELETE CASCADE
                );
            """)
            conn.execute("""INSERT INTO pages_new (id, college_id, url, title, clean_text, word_count, scraped_at)
                            SELECT id, college_id, url, title, clean_text, word_count, scraped_at FROM pages""")
            conn.execute("DROP TABLE pages")
            conn.execute("ALTER TABLE pages_new RENAME TO pages")
            conn.commit()
            logger.info("Migrated pages table: URLs are now unique per college")
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.execute("PRAGMA foreign_keys = ON;")

    # ----------------------------------------------------
    # College Methods
    # ----------------------------------------------------
    def create_or_get_college(self, url: str, domain: str, name: Optional[str] = None) -> Dict[str, Any]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM colleges WHERE url = ?", (url,))
            row = cursor.fetchone()
            if row:
                return dict(row)

            cursor.execute(
                """INSERT INTO colleges (url, domain, name, status, created_at, updated_at)
                   VALUES (?, ?, ?, 'crawling', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)""",
                (url, domain, name or domain)
            )
            conn.commit()
            college_id = cursor.lastrowid
            cursor.execute("SELECT * FROM colleges WHERE id = ?", (college_id,))
            return dict(cursor.fetchone())

    def update_college_status(self, college_id: int, status: str, name: Optional[str] = None):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if name:
                cursor.execute(
                    """UPDATE colleges 
                       SET status = ?, name = ?, updated_at = CURRENT_TIMESTAMP 
                       WHERE id = ?""",
                    (status, name, college_id)
                )
            else:
                cursor.execute(
                    """UPDATE colleges 
                       SET status = ?, updated_at = CURRENT_TIMESTAMP 
                       WHERE id = ?""",
                    (status, college_id)
                )
            conn.commit()

    def update_college_counts(self, college_id: int):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM pages WHERE college_id = ?", (college_id,))
            page_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM chunks WHERE college_id = ?", (college_id,))
            chunk_count = cursor.fetchone()[0]

            cursor.execute(
                "UPDATE colleges SET page_count = ?, chunk_count = ? WHERE id = ?",
                (page_count, chunk_count, college_id)
            )
            conn.commit()

    def get_college(self, college_id: int) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM colleges WHERE id = ?", (college_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_college_by_url(self, url: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM colleges WHERE url = ?", (url,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def list_colleges(self) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM colleges ORDER BY updated_at DESC")
            return [dict(row) for row in cursor.fetchall()]

    def delete_college(self, college_id: int):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Also clear from FTS
            cursor.execute("DELETE FROM chunks_fts WHERE college_id = ?", (college_id,))
            cursor.execute("DELETE FROM colleges WHERE id = ?", (college_id,))
            conn.commit()

    def clear_college_content(self, college_id: int):
        """Remove a college's pages, chunks and facts (before a re-crawl), keeping the college and its chats."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chunks_fts WHERE college_id = ?", (college_id,))
            cursor.execute("DELETE FROM chunks WHERE college_id = ?", (college_id,))
            cursor.execute("DELETE FROM pages WHERE college_id = ?", (college_id,))
            cursor.execute("DELETE FROM facts WHERE college_id = ?", (college_id,))
            conn.commit()

    # ----------------------------------------------------
    # Structured Fact Methods
    # ----------------------------------------------------
    @staticmethod
    def fact_key(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", name.lower())

    def save_facts(self, college_id: int, facts: List[Dict[str, str]], source_url: str) -> int:
        """Insert facts, skipping ones already known (same category + name). Returns how many were new."""
        added = 0
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for f in facts:
                key = self.fact_key(f["name"])
                if not key:
                    continue
                cursor.execute(
                    """INSERT OR IGNORE INTO facts (college_id, category, name, name_key, details, source_url)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (college_id, f["category"], f["name"], key, f.get("details") or "", source_url)
                )
                if cursor.rowcount:
                    added += 1
                elif f.get("details"):
                    # Known item: fill in details if we didn't have any yet
                    cursor.execute(
                        """UPDATE facts SET details = ? WHERE college_id = ? AND category = ? AND name_key = ?
                           AND (details IS NULL OR details = '')""",
                        (f["details"], college_id, f["category"], key)
                    )
            conn.commit()
        return added

    def get_facts(self, college_id: int, categories: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if categories:
                placeholders = ",".join("?" * len(categories))
                cursor.execute(
                    f"SELECT * FROM facts WHERE college_id = ? AND category IN ({placeholders}) ORDER BY category, name",
                    (college_id, *categories)
                )
            else:
                cursor.execute("SELECT * FROM facts WHERE college_id = ? ORDER BY category, name", (college_id,))
            return [dict(row) for row in cursor.fetchall()]

    def get_fact_counts(self, college_id: int) -> Dict[str, int]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT category, COUNT(*) FROM facts WHERE college_id = ? GROUP BY category", (college_id,))
            return {row[0]: row[1] for row in cursor.fetchall()}

    def delete_chunks_for_page(self, page_id: int):
        """Remove a page's chunks before re-saving it, so re-importing a page doesn't duplicate them."""
        with self.get_connection() as conn:
            ids = [r[0] for r in conn.execute("SELECT id FROM chunks WHERE page_id = ?", (page_id,)).fetchall()]
            if ids:
                placeholders = ",".join("?" * len(ids))
                conn.execute(f"DELETE FROM chunks_fts WHERE chunk_id IN ({placeholders})", ids)
                conn.execute("DELETE FROM chunks WHERE page_id = ?", (page_id,))
            conn.commit()

    def get_page_titles(self, urls: List[str]) -> Dict[str, str]:
        if not urls:
            return {}
        placeholders = ",".join("?" * len(urls))
        with self.get_connection() as conn:
            rows = conn.execute(f"SELECT url, title FROM pages WHERE url IN ({placeholders})", urls).fetchall()
            return {row[0]: row[1] for row in rows if row[1]}

    def get_pages_with_text(self, college_id: int) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, url, title, clean_text FROM pages WHERE college_id = ? ORDER BY id", (college_id,))
            return [dict(row) for row in cursor.fetchall()]

    # ----------------------------------------------------
    # Page Methods
    # ----------------------------------------------------
    def save_page(self, college_id: int, url: str, title: str, clean_text: str) -> int:
        word_count = len(clean_text.split())
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO pages (college_id, url, title, clean_text, word_count)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(college_id, url) DO UPDATE SET
                       title = excluded.title,
                       clean_text = excluded.clean_text,
                       word_count = excluded.word_count,
                       scraped_at = CURRENT_TIMESTAMP""",
                (college_id, url, title, clean_text, word_count)
            )
            conn.commit()
            cursor.execute("SELECT id FROM pages WHERE college_id = ? AND url = ?", (college_id, url))
            return cursor.fetchone()[0]

    def get_pages_for_college(self, college_id: int, limit: int = 100) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, url, title, word_count, scraped_at FROM pages WHERE college_id = ? ORDER BY id ASC LIMIT ?",
                (college_id, limit)
            )
            return [dict(row) for row in cursor.fetchall()]

    # ----------------------------------------------------
    # Chunk & Vector Methods
    # ----------------------------------------------------
    def save_chunk(
        self,
        college_id: int,
        page_id: int,
        url: str,
        title: str,
        heading: str,
        chunk_index: int,
        text: str,
        embedding: Optional[np.ndarray] = None
    ) -> int:
        blob = embedding.astype(np.float32).tobytes() if embedding is not None else None
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO chunks (college_id, page_id, url, title, heading, chunk_index, text, embedding)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (college_id, page_id, url, title, heading, chunk_index, text, blob)
            )
            chunk_id = cursor.lastrowid

            # Add to FTS
            cursor.execute(
                """INSERT INTO chunks_fts (chunk_id, college_id, title, heading, text)
                   VALUES (?, ?, ?, ?, ?)""",
                (chunk_id, college_id, title or '', heading or '', text)
            )
            conn.commit()
            return chunk_id

    def save_chunks_batch(
        self,
        college_id: int,
        page_id: int,
        url: str,
        title: str,
        chunks: List[Dict[str, Any]]
    ) -> int:
        """Batch insert multiple chunks for a page in a single transaction (10-20x faster)."""
        if not chunks:
            return 0
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for idx, chunk in enumerate(chunks):
                emb = chunk.get("embedding")
                blob = emb.astype(np.float32).tobytes() if emb is not None else None
                heading = chunk.get("heading", "")
                text = chunk.get("text", "")
                cursor.execute(
                    """INSERT INTO chunks (college_id, page_id, url, title, heading, chunk_index, text, embedding)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (college_id, page_id, url, title, heading, idx, text, blob)
                )
                cid = cursor.lastrowid
                cursor.execute(
                    """INSERT INTO chunks_fts (chunk_id, college_id, title, heading, text)
                       VALUES (?, ?, ?, ?, ?)""",
                    (cid, college_id, title or '', heading or '', text)
                )
            conn.commit()
            return len(chunks)

    def update_chunk_embedding(self, chunk_id: int, embedding: np.ndarray):
        blob = embedding.astype(np.float32).tobytes()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE chunks SET embedding = ? WHERE id = ?", (blob, chunk_id))
            conn.commit()

    def update_chunk_embeddings_batch(self, updates: List[Tuple[int, np.ndarray]]):
        """Batch update chunk embeddings in a single transaction."""
        if not updates:
            return
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                "UPDATE chunks SET embedding = ? WHERE id = ?",
                [(emb.astype(np.float32).tobytes(), cid) for cid, emb in updates]
            )
            conn.commit()

    def get_chunks_without_embeddings(self, college_id: int) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, text, title, heading FROM chunks WHERE college_id = ? AND embedding IS NULL",
                (college_id,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def search_bm25(self, college_id: int, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Full-Text BM25 Search using SQLite FTS5 with stop-word filtering and relevance boosting."""
        all_tokens = [t.lower() for t in "".join(c if c.isalnum() else " " for c in query).split() if len(t) >= 2]
        # Filter stop words
        content_tokens = [t for t in all_tokens if t not in STOP_WORDS]
        tokens_to_use = content_tokens if content_tokens else all_tokens

        if not tokens_to_use:
            return []

        # Get college name info to detect overview queries vs specific topic queries
        college = self.get_college(college_id) or {}
        college_name = (college.get("name") or college.get("domain") or "").lower()
        college_tokens = set("".join(c if c.isalnum() else " " for c in college_name).split()) | {"college", "university", "school", "campus"}
        topic_tokens = [t for t in content_tokens if t not in college_tokens]

        # Check query intent
        q_lower = query.lower()
        is_tuition_query = any(k in q_lower for k in ["fee", "tuition", "cost", "price", "aid", "scholarship", "charges", "expense", "afford"])
        is_academic_query = any(k in q_lower for k in ["course", "degree", "major", "program", "department", "curriculum", "study", "studies", "bachelor", "master", "phd"])
        is_admission_query = any(k in q_lower for k in ["admission", "apply", "application", "deadline", "requirement", "eligibility", "admit", "sat", "act", "gpa"])
        is_housing_query = any(k in q_lower for k in ["hostel", "housing", "dorm", "residence", "dining", "room", "board"])

        # Foundational query check (Overview, History, Founding, Leadership)
        foundation_terms = {
            "about", "overview", "history", "established", "founded", "started",
            "origin", "origins", "founder", "founders", "inception", "background"
        }
        is_placement_query = any(k in q_lower for k in ["placement", "recruit", "career", "job", "salary", "package", "internship", "hiring"])
        filler = foundation_terms | {"tell", "describe", "explain", "give", "show", "who", "what", "when", "where",
                                     "please", "info", "information", "details", "detail"}
        # Words that carry the real topic once filler and the college's own name are removed
        topic_core = [t for t in topic_tokens if t not in filler]
        # Overview shortcut ONLY when nothing topical is left ("tell me about the college", "when was it founded")
        is_foundational = len(topic_core) == 0
        if is_tuition_query or is_academic_query or is_admission_query or is_housing_query or is_placement_query:
            is_foundational = False

        with self.get_connection() as conn:
            cursor = conn.cursor()

            # If it's a foundational or overview query, prioritize Wikipedia, History, and About sections
            if is_foundational:
                sql = """
                    SELECT c.id, c.college_id, c.page_id, c.url, c.title, c.heading, c.text, 10.0 as score
                    FROM chunks c
                    JOIN pages p ON c.page_id = p.id
                    WHERE c.college_id = ? 
                      AND (p.url LIKE '%wikipedia%'
                           OR p.url LIKE '%/history%'
                           OR p.url LIKE '%/about%' 
                           OR p.url = ?
                           OR c.heading LIKE '%History%' 
                           OR c.heading LIKE '%Overview%'
                           OR c.heading LIKE '%Welcome%'
                           OR c.heading LIKE '%About%')
                    ORDER BY c.chunk_index ASC
                    LIMIT 80
                """
                try:
                    cursor.execute(sql, (college_id, college.get("url", "")))
                    overview_results = [dict(r) for r in cursor.fetchall()]
                    if overview_results:
                        def prose_rate(text: str) -> float:
                            body = text.split("\n", 1)[1] if text.startswith("[") and "\n" in text else text
                            n = max(len(body.split()), 1)
                            return len(re.findall(r"[.!?](\s|$)", body)) / (n / 100.0)  # sentences per 100 words

                        def bucket(r: Dict[str, Any]) -> int:
                            u, h = (r.get("url") or "").lower(), (r.get("heading") or "").lower()
                            if "wikipedia" in u: return 1
                            if "/history" in u or "history" in h: return 2
                            if "/about" in u or any(k in h for k in ("about", "overview", "welcome")): return 3
                            return 4
                        # Real descriptive prose first; link-list/menu chunks (few sentences) sink to the bottom.
                        overview_results.sort(key=lambda r: (bucket(r), -min(prose_rate(r["text"]), 8.0), r.get("chunk_index", 0)))
                        return overview_results[:limit]
                except Exception as e:
                    logger.debug(f"Overview query failed: {e}")

            # For specific topic queries, use topic_tokens (without college name pollution)
            search_tokens = topic_core if topic_core else (topic_tokens if topic_tokens else tokens_to_use)
            fts_query_and = " AND ".join(f'"{term}"*' for term in search_tokens)
            fts_query_or = " OR ".join(f'"{term}"*' for term in search_tokens)

            sql = """
                SELECT c.id, c.college_id, c.page_id, c.url, c.title, c.heading, c.text,
                       bm25(chunks_fts) as rank
                FROM chunks_fts
                JOIN chunks c ON chunks_fts.chunk_id = c.id
                WHERE chunks_fts MATCH ? AND chunks_fts.college_id = ?
                ORDER BY rank
                LIMIT ?
            """
            # First try matching all keywords
            results = []
            try:
                cursor.execute(sql, (fts_query_and, college_id, limit))
                for row in cursor.fetchall():
                    d = dict(row)
                    d['score'] = float(-d['rank']) + 2.0 # Bonus for matching all terms
                    results.append(d)
            except Exception as e:
                logger.debug(f"FTS AND query failed: {e}")

            # If not enough results, try OR query
            if len(results) < limit:
                try:
                    cursor.execute(sql, (fts_query_or, college_id, limit * 2))
                    seen_ids = {r['id'] for r in results}
                    for row in cursor.fetchall():
                        d = dict(row)
                        if d['id'] not in seen_ids:
                            d['score'] = float(-d['rank'])
                            results.append(d)
                            seen_ids.add(d['id'])
                            if len(results) >= limit:
                                break
                except Exception as e:
                    logger.warning(f"FTS OR query error: {e}")

            # Category-specific boost based on query intent
            q_lower = query.lower()
            is_tuition_query = any(k in q_lower for k in ["fee", "tuition", "cost", "price", "aid", "scholarship", "charges", "expense"])
            is_academic_query = any(k in q_lower for k in ["course", "degree", "major", "program", "department", "curriculum", "study", "studies", "bachelor", "master", "phd"])
            is_admission_query = any(k in q_lower for k in ["admission", "apply", "application", "deadline", "requirement", "eligibility", "admit", "sat", "act", "gpa"])
            is_housing_query = any(k in q_lower for k in ["hostel", "housing", "dorm", "residence", "dining", "room", "board"])
            is_placement_query = any(k in q_lower for k in ["placement", "recruit", "career", "job", "salary", "package", "internship", "hiring"])

            # Relevance boost: if query terms appear in the title or heading, boost score
            for r in results:
                url_lower = (r.get('url') or '').lower()
                title_lower = (r.get('title') or '').lower()
                heading_lower = (r.get('heading') or '').lower()
                matches = sum(1 for t in search_tokens if t in title_lower or t in heading_lower)
                r['score'] += matches * 2.0

                # Targeted Intent Boosts
                if is_tuition_query and any(k in url_lower or k in title_lower or k in heading_lower for k in ["tuition", "fee", "cost", "finance", "aid"]):
                    r['score'] += 8.0
                if is_academic_query and any(k in url_lower or k in title_lower or k in heading_lower for k in ["academic", "program", "degree", "major", "curriculum", "bulletin", "catalog", "study"]):
                    r['score'] += 8.0
                if is_admission_query and any(k in url_lower or k in title_lower or k in heading_lower for k in ["admission", "apply", "applicant", "first-year", "transfer", "requirement"]):
                    r['score'] += 8.0
                if is_housing_query and any(k in url_lower or k in title_lower or k in heading_lower for k in ["housing", "residence", "dorm", "dining", "campus-life", "hostel"]):
                    r['score'] += 8.0
                if is_placement_query and any(k in url_lower or k in title_lower or k in heading_lower for k in ["placement", "career", "recruit", "training", "internship", "industry"]):
                    r['score'] += 8.0

            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:limit]

    def get_all_chunk_embeddings(self, college_id: int) -> List[Tuple[int, np.ndarray]]:
        """Return list of (chunk_id, embedding_vector) for vector cosine similarity."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, embedding FROM chunks WHERE college_id = ? AND embedding IS NOT NULL",
                (college_id,)
            )
            rows = cursor.fetchall()
            results = []
            for row in rows:
                chunk_id = row[0]
                blob = row[1]
                if blob:
                    vec = np.frombuffer(blob, dtype=np.float32)
                    results.append((chunk_id, vec))
            return results

    def get_chunk_by_id(self, chunk_id: int) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_chunks_by_ids(self, chunk_ids: List[int]) -> List[Dict[str, Any]]:
        """Batch fetch multiple chunks in a single query (avoids N round-trips)."""
        if not chunk_ids:
            return []
        placeholders = ",".join("?" * len(chunk_ids))
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM chunks WHERE id IN ({placeholders})", chunk_ids)
            return [dict(row) for row in cursor.fetchall()]

    # ----------------------------------------------------
    # Conversation & Message Methods
    # ----------------------------------------------------
    def create_conversation(self, conv_id: str, college_id: int, title: str = "College Q&A") -> Dict[str, Any]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO conversations (id, college_id, title) VALUES (?, ?, ?)",
                (conv_id, college_id, title)
            )
            conn.commit()
            cursor.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,))
            return dict(cursor.fetchone())

    def get_conversation(self, conv_id: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def add_message(
        self,
        conv_id: str,
        role: str,
        content: str,
        spoken_text: Optional[str] = None,
        citations: Optional[List[Dict[str, Any]]] = None,
        audio_path: Optional[str] = None
    ) -> Dict[str, Any]:
        citations_json = json.dumps(citations or [])
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO messages (conversation_id, role, content, spoken_text, citations, audio_path)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (conv_id, role, content, spoken_text or content, citations_json, audio_path)
            )
            conn.commit()
            msg_id = cursor.lastrowid
            cursor.execute("SELECT * FROM messages WHERE id = ?", (msg_id,))
            row = cursor.fetchone()
            d = dict(row)
            d['citations'] = json.loads(d['citations'])
            return d

    def get_messages(self, conv_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id ASC LIMIT ?",
                (conv_id, limit)
            )
            results = []
            for row in cursor.fetchall():
                d = dict(row)
                d['citations'] = json.loads(d['citations']) if d.get('citations') else []
                results.append(d)
            return results
