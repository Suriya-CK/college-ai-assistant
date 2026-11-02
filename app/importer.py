import re
import email
import html as html_lib
import hashlib
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.crawler import CollegeCrawler, MIN_PAGE_WORDS
from app.storage import Database

logger = logging.getLogger(__name__)

# Chrome/Edge "Save page as" leaves the page's address in a comment at the top of the file
_SAVED_FROM_RE = re.compile(r"<!--\s*saved from url=\(\d+\)\s*(\S+?)\s*-->", re.I)
_CANONICAL_RE = re.compile(r"""<link[^>]+rel=["']canonical["'][^>]*href=["']([^"']+)["']""", re.I)
_OG_URL_RE = re.compile(r"""<meta[^>]+property=["']og:url["'][^>]*content=["']([^"']+)["']""", re.I)
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)

SUPPORTED_EXTENSIONS = {".html", ".htm", ".mhtml", ".mht", ".pdf", ".txt", ".md"}


class PageImporter:
    """Adds pages a person saved from their own browser (HTML, MHTML, PDF, text) to a college's knowledge base.

    For sites that block automated crawlers: the pages go through the same cleaning, chunking and
    search indexing as crawled pages, so answers and fact extraction work the same way.
    """

    def __init__(self, db: Optional[Database] = None):
        self.db = db or Database()
        self.crawler = CollegeCrawler(self.db)

    # ------------------------------------------------------------------
    # Reading files
    # ------------------------------------------------------------------

    def _html_to_text(self, html: str) -> str:
        from bs4 import BeautifulSoup
        from crawl4ai.html2text import HTML2Text

        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
            tag.decompose()
        converter = HTML2Text()
        converter.ignore_images = True
        converter.body_width = 0
        return self.crawler.clean_markdown(converter.handle(str(soup)))

    @staticmethod
    def _page_url(html: str) -> Optional[str]:
        for regex in (_SAVED_FROM_RE, _CANONICAL_RE, _OG_URL_RE):
            m = regex.search(html[:20000])
            if m and m.group(1).startswith(("http://", "https://")):
                return m.group(1)
        return None

    @staticmethod
    def _html_title(html: str) -> str:
        m = _TITLE_RE.search(html)
        return re.sub(r"\s+", " ", html_lib.unescape(m.group(1))).strip() if m else ""

    @staticmethod
    def _mhtml_parts(data: bytes) -> Tuple[str, Optional[str]]:
        """(html, original url) from a 'Webpage, Single File' (.mhtml) save."""
        msg = email.message_from_bytes(data)
        url = msg.get("Snapshot-Content-Location")
        for part in msg.walk():
            if part.get_content_type() == "text/html":
                payload = part.get_payload(decode=True) or b""
                html = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
                return html, url or part.get("Content-Location")
        return "", url

    def read_file(self, path: Path, base_url: str) -> Optional[Dict[str, str]]:
        """{url, title, text} for one saved file, or None if it has no readable text."""
        ext = path.suffix.lower()
        data = path.read_bytes()
        fallback_url = f"{base_url.rstrip('/')}/imported/{re.sub(r'[^a-zA-Z0-9._-]+', '-', path.stem).strip('-')}"

        if ext == ".pdf":
            text, title = self.crawler.pdf_text(data, path.name)
            if not text:
                logger.warning(f"{path.name}: no readable text (scanned PDF?), skipped")
                return None
            return {"url": fallback_url + ".pdf", "title": f"{title or path.stem} (PDF)", "text": self.crawler.clean_markdown(text)}

        if ext in (".txt", ".md"):
            text = data.decode("utf-8", errors="replace").strip()
            first_line = text.splitlines()[0].strip("# ").strip() if text else ""
            return {"url": fallback_url, "title": first_line[:120] or path.stem, "text": text}

        if ext in (".mhtml", ".mht"):
            html, url = self._mhtml_parts(data)
        else:
            html, url = data.decode("utf-8", errors="replace"), None
        if not html:
            return None
        return {
            "url": url or self._page_url(html) or fallback_url,
            "title": self._html_title(html) or path.stem,
            "text": self._html_to_text(html),
        }

    # ------------------------------------------------------------------
    # Importing a folder
    # ------------------------------------------------------------------

    def import_folder(
        self,
        folder: Path,
        college_url: str,
        college_name: Optional[str] = None,
        replace: bool = False,
        progress: Optional[Callable[[str, str], None]] = None,
    ) -> Dict[str, Any]:
        """Import every supported file in `folder` (not sub-folders like 'Page_files') into one college."""
        folder = Path(folder).expanduser()
        if not folder.is_dir():
            raise FileNotFoundError(f"Folder not found: {folder}")
        files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
        if not files:
            raise FileNotFoundError(f"No .html, .mhtml, .pdf or .txt files in {folder}")

        url = self.crawler.normalize_url(college_url)
        domain = self.crawler.get_domain(url)
        college = self.db.create_or_get_college(url, domain, college_name)
        college_id = college["id"]
        if replace:
            self.db.clear_college_content(college_id)

        imported, skipped, total_chunks = [], [], 0
        name_from_pages = None

        # Read everything first, so menus/footers repeated across the saved pages can be removed from all of them
        pages: List[Tuple[Path, Dict[str, str]]] = []
        for path in files:
            try:
                page = self.read_file(path, url)
            except Exception as exc:
                logger.warning(f"Could not read {path.name}: {exc}")
                page = None
            if page:
                pages.append((path, page))
            else:
                skipped.append((path.name, "no readable text"))

        line_counts: Counter = Counter()
        for _, page in pages:
            line_counts.update({ln.strip() for ln in page["text"].splitlines() if ln.strip()})
        repeated_min = max(3, len(pages) // 2) if len(pages) >= 4 else len(pages) + 1   # too few pages to judge

        seen_content = set()
        kept_once = set()   # one copy of each repeated line survives (a menu may be the only list of departments)

        def keep(line: str) -> bool:
            key = line.strip()
            if not key or line_counts[key] < repeated_min:
                return True
            if key in kept_once:
                return False
            kept_once.add(key)
            return True

        for path, page in pages:
            text = "\n".join(ln for ln in page["text"].splitlines() if keep(ln)).strip()
            if len(text.split()) < MIN_PAGE_WORDS:
                skipped.append((path.name, "too little text"))
                continue
            digest = hashlib.sha1(re.sub(r"\s+", " ", text).encode("utf-8")).hexdigest()
            if digest in seen_content:
                skipped.append((path.name, "duplicate of another file"))
                continue
            seen_content.add(digest)

            title = page["title"]
            if name_from_pages is None and not title.endswith("(PDF)"):
                name_from_pages = self.crawler.college_name_from_title(title)
            page_id = self.db.save_page(college_id=college_id, url=page["url"], title=title, clean_text=text)
            self.db.delete_chunks_for_page(page_id)
            chunks = self.crawler.create_chunks(self.crawler.markdown_to_sections(text), title) or \
                [{"heading": "Overview", "text": f"[{title}]\n{text[:3000]}"}]
            self.db.save_chunks_batch(college_id, page_id, page["url"], title, chunks)
            total_chunks += len(chunks)
            imported.append(path.name)
            if progress:
                progress(path.name, title)

        name = college_name or college.get("name")
        if not name or name == domain:
            name = name_from_pages or domain
        self.db.update_college_counts(college_id)
        self.db.update_college_status(college_id, "completed", name)
        return {
            "college_id": college_id,
            "college_name": name,
            "imported": imported,
            "skipped": skipped,
            "total_chunks": total_chunks,
        }
