import os
import io
import re
import uuid
import heapq
import asyncio
import hashlib
import logging
from collections import Counter
from urllib.parse import urljoin, urlparse, urldefrag, urlencode, parse_qsl
from typing import List, Dict, Set, Callable, Optional, Any, Tuple

import requests
import urllib3

# Suppress noisy unverified SSL warnings for institutions with self-signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ.setdefault("CRAWL4AI_NO_CLOUD_NOTICE", "1")

from app.config import Config
from app.storage import Database

logger = logging.getLogger(__name__)
# pypdf logs a warning for every slightly malformed PDF it still reads fine
logging.getLogger("pypdf").setLevel(logging.ERROR)

# Topic keywords matched against a link's URL + anchor text. Higher weight = crawled sooner.
# Keywords of 3 chars or fewer must match a whole token (so "ug" doesn't match "august").
TOPIC_WEIGHTS: Dict[str, int] = {
    # placements / recruiters
    "placement": 12, "placements": 12, "recruit": 12, "recruiters": 12, "career": 8,
    "training": 6, "tpo": 10, "internship": 6, "companies": 8, "industry": 4,
    # programmes
    "course": 10, "courses": 10, "programme": 10, "program": 10, "degree": 8,
    "undergraduate": 10, "postgraduate": 10, "ug": 10, "pg": 10, "btech": 10,
    "mtech": 10, "mba": 8, "mca": 8, "phd": 6, "syllabus": 4, "curriculum": 4,
    "branch": 6, "intake": 8,
    # departments / admissions / fees
    "department": 8, "departments": 8, "dept": 8, "school": 3, "faculty": 4,
    "admission": 12, "admissions": 12, "eligibility": 8, "apply": 6, "fee": 10,
    "fees": 10, "tuition": 10, "scholarship": 8, "prospectus": 8, "brochure": 8,
    # general
    "about": 6, "overview": 5, "history": 4, "facilities": 5, "infrastructure": 4,
    "hostel": 5, "campus": 3, "library": 2, "contact": 4, "accreditation": 5,
    "naac": 5, "nba": 5, "nirf": 5, "ranking": 4, "research": 2, "alumni": 2,
}

# Pages that are almost never useful for answering college questions
LOW_VALUE_RE = re.compile(
    r"(news|event|gallery|photo|video|tender|circular|notice|archive|press|blog|"
    r"result|timetable|time-table|journal|newsletter|download|form)", re.I
)

# Never crawl these: logins, feeds, admin areas, carts, print/share variants
SKIP_URL_RE = re.compile(
    r"(login|logout|signin|sign-in|signup|sign-up|register|wp-admin|wp-login|/feed|"
    r"replytocom=|/tag/|/author/|calendar|print=|share=|/cart|checkout|/cdn-cgi/|"
    r"mailto:|tel:|javascript:)", re.I
)

SOCIAL_DOMAINS = (
    "facebook.com", "twitter.com", "x.com", "instagram.com", "linkedin.com",
    "youtube.com", "youtu.be", "wa.me", "whatsapp.com", "t.me", "pinterest.com",
    "google.com", "goo.gl", "maps.app.goo.gl", "play.google.com", "apple.com",
)

# File extensions that are binary, media, or documents we can't read — skip entirely (PDFs are handled separately)
IGNORED_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico", ".bmp",
    ".zip", ".tar", ".gz", ".rar", ".7z", ".dmg", ".iso", ".apk",
    ".mp3", ".mp4", ".mov", ".avi", ".wmv", ".wav", ".webm",
    ".exe", ".bin", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".csv", ".xml", ".rss", ".json", ".css", ".js",
)

# Query param keys that are session/tracking noise — strip these
_STRIP_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "sessionid", "sid", "phpsessid", "ref", "source",
    "_ga", "_gid", "mc_cid", "mc_eid", "forceview",
}

_INDEX_PAGE_RE = re.compile(r"/(index|default|home)\.(php|html?|aspx?|jsp)$", re.I)
_NOT_FOUND_TITLE_RE = re.compile(r"\b(404|not found|page not found|error)\b", re.I)
# Firewall / CAPTCHA / "are you human" pages served instead of the real content
_CHALLENGE_RE = re.compile(
    r"(captcha|validation request|user validation required|verify (that )?you are (a )?human|are you a robot|"
    r"just a moment\.\.\.|attention required|checking your browser|ddos protection|access denied|request blocked|"
    r"type the (text|characters) you see)", re.I
)

# Links hidden outside <a href>: meta refresh, framesets/iframes, JS location redirects
_META_REFRESH_RE = re.compile(
    r"""<meta[^>]+http-equiv=["']?refresh["']?[^>]*content=["'][^"']*url=([^"'>\s]+)""", re.I
)
_FRAME_SRC_RE = re.compile(r"""<i?frame[^>]+src=["']([^"']+)["']""", re.I)
_JS_LOCATION_RE = re.compile(
    r"""(?:window\.|document\.)?location(?:\.href)?\s*=\s*["']([^"']+)["']""", re.I
)

# Title fragments that are not a college name
_GENERIC_TITLE_PARTS = {
    "home", "home page", "homepage", "welcome", "index", "main", "default", "official website",
    "official site", "welcome to our website", "untitled", "untitled document",
}

_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\((?:[^()]|\([^)]*\))*\)")

MIN_PAGE_WORDS = 30          # thinner pages are not stored or counted (their links are still followed)
BOILERPLATE_MIN_PAGES = 3    # a line repeated on this many earlier pages is site chrome (menus, footers)
MAX_PDF_BYTES = 15 * 1024 * 1024
MAX_PDF_PAGES = 40


class CollegeCrawler:
    """Best-first college website crawler built on Crawl4AI (a real headless browser).

    - Every page is rendered, so JavaScript menus, redirects and SPA content work.
    - The site scope follows the homepage's redirect target (college.edu -> college.ac.in).
    - Pages are visited in order of how useful they look (placements, courses, admissions first).
    - Soft-404s, duplicate pages and near-empty pages never consume the page budget.
    - PDFs (brochures, placement reports, fee structures) are downloaded and their text indexed.
    - Topic-relevant links to other domains (e.g. a separate placement portal) get one hop.
    """

    def __init__(self, db: Optional[Database] = None, extractor: Any = None):
        self.db = db or Database()
        self.extractor = extractor
        self._anchors: Dict[str, str] = {}   # url -> anchor text seen on linking pages
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": Config.USER_AGENT})

    # ------------------------------------------------------------------
    # URL helpers
    # ------------------------------------------------------------------

    def normalize_url(self, raw_url: str) -> str:
        """Strip fragment and tracking params, lowercase scheme/host, strip trailing slash."""
        raw = raw_url.strip()
        if not raw.startswith(("http://", "https://")):
            raw = "https://" + raw

        url, _ = urldefrag(raw)
        parsed = urlparse(url)
        scheme = parsed.scheme.lower() if parsed.scheme else "https"
        netloc = parsed.netloc.lower()
        path = parsed.path.rstrip("/") if parsed.path != "/" else "/"

        if parsed.query:
            clean_params = sorted(
                (k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=False)
                if k.lower() not in _STRIP_PARAMS
            )
            query = urlencode(clean_params)
        else:
            query = ""

        if query:
            return f"{scheme}://{netloc}{path}?{query}"
        return f"{scheme}://{netloc}{path}"

    def dedupe_key(self, url: str) -> str:
        """Identity of a page regardless of http/https, www., trailing slash or index.php."""
        parsed = urlparse(self.normalize_url(url))
        host = parsed.netloc.removeprefix("www.")
        path = _INDEX_PAGE_RE.sub("", parsed.path).rstrip("/")
        return f"{host}{path}?{parsed.query}" if parsed.query else f"{host}{path}"

    def get_domain(self, url: str) -> str:
        return urlparse(url).netloc.lower()

    def _apex_domain(self, domain: str) -> str:
        """Return the registrable domain, e.g. ceg.annauniv.edu -> annauniv.edu, cs.ssn.edu.in -> ssn.edu.in."""
        domain = domain.split(":")[0]
        parts = domain.split(".")
        if len(parts) >= 3 and parts[-2] in ("edu", "ac", "co", "org", "gov", "net", "res", "nic"):
            return ".".join(parts[-3:])
        return ".".join(parts[-2:]) if len(parts) >= 2 else domain

    def is_same_domain(self, base_domain: str, target_url: str) -> bool:
        """True if target belongs to the site rooted at base_domain.

        An apex base (mit.edu, ssn.edu.in) admits all its subdomains; a specific college
        subdomain (ceg.annauniv.edu) admits only itself and its children, not sibling portals.
        """
        base_clean = base_domain.removeprefix("www.")
        target_clean = self.get_domain(target_url).removeprefix("www.")

        if target_clean == base_clean or target_clean.endswith("." + base_clean):
            return True
        base_apex = self._apex_domain(base_clean)
        return base_clean == base_apex and self._apex_domain(target_clean) == base_apex

    def _in_scope(self, url: str, base_domains: Set[str]) -> bool:
        return any(self.is_same_domain(b, url) for b in base_domains)

    @staticmethod
    def _is_pdf(url: str) -> bool:
        return urlparse(url).path.lower().endswith(".pdf")

    @staticmethod
    def _is_crawlable(url: str) -> bool:
        if not url.startswith(("http://", "https://")) or SKIP_URL_RE.search(url):
            return False
        path = urlparse(url).path.lower()
        return not path.endswith(IGNORED_EXTENSIONS)

    def score_url(self, url: str, depth: int, anchor: str = "") -> int:
        """Higher score = crawled sooner. Topic keywords in URL/anchor push a link up, depth and noise push it down."""
        parsed = urlparse(url)
        text = f"{parsed.path} {parsed.query} {anchor or self._anchors.get(url, '')}".lower()
        tokens = set(re.split(r"[^a-z0-9]+", text))
        score = 0
        for kw, weight in TOPIC_WEIGHTS.items():
            if (kw in tokens) if len(kw) <= 3 else (kw in text):
                score += weight
        score = min(score, 30)
        if LOW_VALUE_RE.search(parsed.path):
            score -= 8
        return score - 3 * depth

    # ------------------------------------------------------------------
    # Content processing
    # ------------------------------------------------------------------

    @staticmethod
    def college_name_from_title(title: str) -> str:
        """'Home | CEG Campus' -> 'CEG Campus'; 'Anna University, Chennai - Official Website' -> 'Anna University, Chennai'."""
        parts = [p.strip() for p in re.split(r"\s[|\-–:]\s", title or "") if p.strip()]
        for part in parts:
            if part.lower() not in _GENERIC_TITLE_PARTS:
                return part
        return parts[0] if parts else ""

    @staticmethod
    def clean_markdown(markdown: str) -> str:
        """Drop images, unwrap links to their text, collapse blank lines."""
        text = _MD_IMAGE_RE.sub("", markdown or "")
        text = _MD_LINK_RE.sub(lambda m: m.group(1), text)
        lines = [ln.rstrip() for ln in text.splitlines()]
        out: List[str] = []
        for ln in lines:
            stripped = ln.strip(" \t*-•|")
            if not stripped:
                if out and out[-1] != "":
                    out.append("")
                continue
            out.append(ln)
        return "\n".join(out).strip()

    @staticmethod
    def markdown_to_sections(markdown: str) -> List[Dict[str, str]]:
        """Split markdown into {heading, text} sections at heading lines."""
        sections: List[Dict[str, str]] = []
        heading = "Overview"
        buf: List[str] = []

        def flush():
            text = "\n".join(buf).strip()
            if len(text) > 20:
                sections.append({"heading": heading, "text": text})

        for line in markdown.splitlines():
            m = re.match(r"^\s{0,3}#{1,6}\s+(.*)", line)
            if m:
                flush()
                buf = []
                heading = m.group(1).strip(" #*")[:200] or heading
            else:
                buf.append(line)
        flush()
        return sections

    def create_chunks(
        self,
        sections: List[Dict[str, str]],
        page_title: str,
        max_chunk_words: int = 400,
    ) -> List[Dict[str, str]]:
        """Split page sections into retrieval chunks with 50-word overlap."""
        chunks: List[Dict[str, str]] = []
        for sec in sections:
            heading = sec["heading"]
            words = sec["text"].split()
            if len(words) <= max_chunk_words:
                chunks.append({"heading": heading, "text": f"[{page_title} - {heading}]\n{sec['text']}"})
                continue
            step = max_chunk_words - 50
            for start in range(0, len(words), step):
                chunks.append({
                    "heading": heading,
                    "text": f"[{page_title} - {heading} (part)]\n{' '.join(words[start:start + max_chunk_words])}",
                })
                if start + max_chunk_words >= len(words):
                    break
        return chunks

    def extract_hidden_links(self, html: str, page_url: str) -> List[str]:
        """Links a browser's <a> list misses: meta refresh, frameset/iframe src, JS location redirects."""
        found: List[str] = []
        for regex in (_META_REFRESH_RE, _FRAME_SRC_RE, _JS_LOCATION_RE):
            for href in regex.findall(html or ""):
                href = href.strip().strip("'\"")
                if href and not href.startswith(("#", "javascript:", "about:")):
                    found.append(urljoin(page_url, href))
        return found

    @staticmethod
    def _word_set(text: str) -> Set[str]:
        return set(re.findall(r"[a-z]{3,}", (text or "").lower()))

    def is_soft_404(self, title: str, text: str, status_code: Optional[int],
                    not_found_words: Set[str], site_words: Optional[Set[str]] = None) -> bool:
        """A page that is really an error page, even if the server said 200.

        not_found_words: words of this site's error message (its 404 page minus the menus shared with
        the homepage); site_words: the homepage's words. A page is the error page if it carries most of
        the error message and almost nothing else.
        """
        if status_code is not None and status_code >= 400:
            return True
        if _NOT_FOUND_TITLE_RE.search(title or "") and len(text.split()) < 300:
            return True
        if not_found_words:
            words = self._word_set(text)
            if len(words & not_found_words) >= 0.8 * len(not_found_words):
                own_words = words - not_found_words - (site_words or set())
                if len(own_words) < 15:
                    return True
        return False

    def _strip_boilerplate(self, text: str, line_counts: Counter) -> str:
        """Remove lines already seen on several earlier pages (menus, footers), then record this page's lines."""
        lines = text.splitlines()
        kept = [ln for ln in lines if not ln.strip() or line_counts[ln.strip()] < BOILERPLATE_MIN_PAGES]
        for ln in set(l.strip() for l in lines if l.strip()):
            line_counts[ln] += 1
        return "\n".join(kept).strip()

    # ------------------------------------------------------------------
    # Fetching
    # ------------------------------------------------------------------

    def discover_sitemap_urls(self, start_url: str, limit: int = 500) -> List[str]:
        """URLs from robots.txt sitemaps and common sitemap locations."""
        parsed = urlparse(start_url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        sitemaps = [f"{origin}/sitemap.xml", f"{origin}/sitemap_index.xml", f"{origin}/wp-sitemap.xml"]
        try:
            robots = self.session.get(f"{origin}/robots.txt", timeout=Config.REQUEST_TIMEOUT, verify=False)
            if robots.status_code == 200:
                for line in robots.text.splitlines():
                    if line.lower().startswith("sitemap:"):
                        sitemaps.append(line.split(":", 1)[1].strip())
        except Exception:
            pass

        found: List[str] = []
        seen: Set[str] = set()
        to_read = list(dict.fromkeys(sitemaps))
        while to_read and len(found) < limit and len(seen) < 15:
            sm = to_read.pop(0)
            if sm in seen:
                continue
            seen.add(sm)
            try:
                resp = self.session.get(sm, timeout=Config.REQUEST_TIMEOUT, verify=False)
                if resp.status_code != 200 or "<loc>" not in resp.text:
                    continue
                locs = re.findall(r"<loc>\s*(.*?)\s*</loc>", resp.text, flags=re.I | re.S)
            except Exception:
                continue
            for loc in locs:
                loc = loc.strip()
                if loc.lower().endswith(".xml"):
                    to_read.append(loc)
                elif loc:
                    found.append(self.normalize_url(loc))
                    if len(found) >= limit:
                        break
        return found

    def fetch_pdf(self, url: str) -> Tuple[Optional[str], str]:
        """Download a PDF and return (text, title). Text is None if it isn't a readable PDF."""
        try:
            from pypdf import PdfReader
        except ImportError:
            logger.warning("pypdf is not installed; skipping PDFs. Run: .venv/bin/pip install pypdf")
            return None, ""
        try:
            try:
                resp = self.session.get(url, timeout=(10, 40), stream=True)
            except requests.exceptions.SSLError:
                resp = self.session.get(url, timeout=(10, 40), stream=True, verify=False)
            if resp.status_code != 200:
                return None, ""
            data = b""
            for block in resp.iter_content(64 * 1024):
                data += block
                if len(data) > MAX_PDF_BYTES:
                    return None, ""
            return self.pdf_text(data, urlparse(url).path.rsplit("/", 1)[-1])
        except Exception as exc:
            logger.debug(f"PDF fetch failed for {url}: {exc}")
            return None, ""

    @staticmethod
    def pdf_text(data: bytes, fallback_title: str = "") -> Tuple[Optional[str], str]:
        """(text, title) of a PDF's bytes; text is None if it isn't a readable PDF (e.g. a scanned image)."""
        from pypdf import PdfReader
        if not data.startswith(b"%PDF"):
            return None, ""
        reader = PdfReader(io.BytesIO(data))
        pages = [(p.extract_text() or "") for p in reader.pages[:MAX_PDF_PAGES]]
        title = ""
        try:
            title = (reader.metadata.title or "") if reader.metadata else ""
        except Exception:
            pass
        text = "\n\n".join(pages).strip()
        return (text or None), (title.strip() or fallback_title)

    # ------------------------------------------------------------------
    # Main crawl entry point
    # ------------------------------------------------------------------

    def crawl(
        self,
        start_url: str,
        max_pages: int = Config.DEFAULT_MAX_PAGES,
        max_depth: int = Config.DEFAULT_MAX_DEPTH,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Crawl the college site and store pages + chunks (+ structured facts if an extractor is set)."""
        if Config.QUICK_TEST:
            max_pages = min(max_pages, Config.QUICK_MAX_PAGES)
            max_depth = min(max_depth, Config.QUICK_MAX_DEPTH)
            logger.info(f"QUICK_TEST is on: crawling at most {max_pages} pages, depth {max_depth}")
        return asyncio.run(self._crawl_async(start_url, max_pages, max_depth, progress_callback))

    async def _crawl_async(
        self,
        start_url: str,
        max_pages: int,
        max_depth: int,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]],
    ) -> Dict[str, Any]:
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

        start_url = self.normalize_url(start_url)
        start_host = self.get_domain(start_url)
        base_domains: Set[str] = {start_host}
        primary_hosts: Set[str] = {start_host.removeprefix("www.")}

        def emit(event: str, data: Dict[str, Any]):
            if progress_callback:
                try:
                    progress_callback(event, data)
                except Exception as exc:
                    logger.warning(f"Progress callback error: {exc}")

        emit("status", {"message": f"Starting crawl for {start_url}..."
                                   + (" (quick test mode)" if Config.QUICK_TEST else ""), "domain": start_host})
        college = self.db.create_or_get_college(start_url, start_host)
        college_id = college["id"]
        college_name = college.get("name") or start_host
        # A re-crawl replaces the old content instead of piling duplicate chunks on top of it
        self.db.clear_college_content(college_id)
        self.db.update_college_status(college_id, "crawling")

        max_pdfs = max(5, max_pages // 5)
        max_external = max(3, max_pages // 10)
        concurrency = max(1, Config.CRAWL_CONCURRENCY)
        delay = Config.CRAWL_DELAY
        blocked_streak = 0          # consecutive "403 / too many requests" answers
        slowed_down = False
        retried: Set[str] = set()
        stopped_reason = ""

        # Frontier: (-score, seq, url, depth, is_external)
        heap: List[Tuple[int, int, str, int, bool]] = []
        enqueued: Set[str] = set()
        seen_content: Set[str] = set()
        line_counts: Counter = Counter()
        seq = 0
        scraped_pages = pdf_count = external_count = total_chunks = 0
        not_found_words: Set[str] = set()
        site_words: Set[str] = set()

        def enqueue(url: str, depth: int, anchor: str = "", is_external: bool = False, score: Optional[int] = None):
            nonlocal seq
            url = self.normalize_url(url)
            key = self.dedupe_key(url)
            if key in enqueued or not self._is_crawlable(url) or depth > max_depth:
                return
            if anchor:
                self._anchors[url] = f"{self._anchors.get(url, '')} {anchor}".strip()[:300]
            s = score if score is not None else self.score_url(url, depth, anchor)
            if self._is_pdf(url) and s < 8:
                return   # only topic-relevant PDFs (brochures, placement reports, fee structures)
            enqueued.add(key)
            seq += 1
            heapq.heappush(heap, (-s, seq, url, depth, is_external))

        def handle_links(links: List[Tuple[str, str]], page_url: str, depth: int):
            nonlocal external_count
            for href, anchor in links:
                try:
                    url = self.normalize_url(urljoin(page_url, href))
                except Exception:
                    continue
                if self._in_scope(url, base_domains):
                    enqueue(url, depth + 1, anchor)
                    continue
                host = self.get_domain(url)
                if any(host == d or host.endswith("." + d) for d in SOCIAL_DOMAINS):
                    continue
                # One hop to clearly relevant external pages (e.g. a separately hosted placement portal)
                if external_count < max_external and self.score_url(url, 0, anchor) >= 10:
                    external_count += 1
                    enqueue(url, depth + 1, anchor, is_external=True)

        def store(url: str, title: str, text: str, depth: int) -> bool:
            """Save a page and its chunks. Returns False if it was too thin or a duplicate."""
            nonlocal scraped_pages, total_chunks
            if len(text.split()) < MIN_PAGE_WORDS:
                logger.debug(f"Skipping thin page {url} ({len(text.split())} words)")
                return False
            digest = hashlib.sha1(re.sub(r"\s+", " ", text).encode("utf-8")).hexdigest()
            if digest in seen_content:
                logger.debug(f"Skipping duplicate content at {url}")
                return False
            seen_content.add(digest)

            page_id = self.db.save_page(college_id=college_id, url=url, title=title, clean_text=text)
            chunks = self.create_chunks(self.markdown_to_sections(text), title)
            if not chunks:
                chunks = [{"heading": "Overview", "text": f"[{title}]\n{text[:3000]}"}]
            self.db.save_chunks_batch(college_id, page_id, url, title, chunks)
            scraped_pages += 1
            total_chunks += len(chunks)
            emit("page_success", {
                "url": url, "title": title, "chunks": len(chunks), "depth": depth,
                "scraped_count": scraped_pages, "total_chunks": total_chunks, "max_pages": max_pages,
            })
            return True

        browser_cfg = BrowserConfig(
            headless=True, verbose=False, user_agent=Config.USER_AGENT,
            ignore_https_errors=True, text_mode=True,
        )
        run_cfg = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            wait_until="domcontentloaded",
            page_timeout=Config.PAGE_TIMEOUT_MS,
            delay_before_return_html=1.0,      # let JS menus mount
            # remove_overlay_elements is deliberately off: it deletes menus pinned to the top of the
            # screen (e.g. ceg.annauniv.edu), taking every navigation link with them
            remove_consent_popups=True,
            process_iframes=True,
            excluded_tags=["script", "style", "noscript"],
            word_count_threshold=1,
            verbose=False,
        )

        try:
            async with AsyncWebCrawler(config=browser_cfg) as crawler:
                # ── Homepage first: learn the real domain after redirects ──────
                emit("page_start", {"url": start_url, "depth": 0, "scraped_count": 0, "max_pages": max_pages})
                home = await crawler.arun(start_url, config=run_cfg)
                if not home.success and not self.is_blocked(home):
                    raise RuntimeError(f"Could not load {start_url}: {home.error_message}")
                if self.is_blocked(home):
                    raise RuntimeError(
                        f"{start_host} is blocking automated visitors (it shows a CAPTCHA / security check instead "
                        f"of its pages), so it can't be crawled. Try again later or from a different network."
                    )
                home_url = self.normalize_url(home.redirected_url or home.url)
                home_host = self.get_domain(home_url)
                if home_host != start_host:
                    base_domains.add(home_host)
                    primary_hosts.add(home_host.removeprefix("www."))
                    emit("status", {"message": f"{start_host} redirects to {home_host}; crawling both."})
                enqueued.add(self.dedupe_key(start_url))
                enqueued.add(self.dedupe_key(home_url))

                # ── Fingerprint this site's "page not found" page ─────────────
                origin = f"{urlparse(home_url).scheme}://{home_host}"
                home_raw = self.clean_markdown(home.markdown.raw_markdown if home.markdown else "")
                site_words = self._word_set(home_raw)
                probe = await crawler.arun(f"{origin}/{uuid.uuid4().hex}-missing-page", config=run_cfg)
                if probe.success and (probe.status_code or 200) < 400 and probe.markdown:
                    not_found_words = self._word_set(self.clean_markdown(probe.markdown.raw_markdown)) - site_words

                page_title = (home.metadata or {}).get("title") or home_host
                candidate = self.college_name_from_title(page_title)
                if candidate and (not college_name or college_name in (start_host, home_host)
                                  or college_name.lower() in _GENERIC_TITLE_PARTS):
                    college_name = candidate
                home_text = self._strip_boilerplate(home_raw, line_counts)
                store(home_url, page_title, home_text, 0)
                handle_links(self._result_links(home), home_url, 0)
                for link in self.extract_hidden_links(home.html, home_url):
                    handle_links([(link, "")], home_url, 0)

                # ── Sitemaps ──────────────────────────────────────────────────
                sitemap_urls = await asyncio.to_thread(self.discover_sitemap_urls, home_url)
                in_scope = [u for u in sitemap_urls if self._in_scope(u, base_domains)]
                for u in in_scope:
                    enqueue(u, 1)
                if in_scope:
                    emit("status", {"message": f"Found {len(in_scope)} URLs in the sitemap."})
                emit("status", {"message": f"Crawling up to {max_pages} pages (depth {max_depth})...",
                                "college_id": college_id, "url": home_url})

                # ── Best-first crawl in parallel batches ──────────────────────
                while heap and scraped_pages < max_pages:
                    if delay:
                        await asyncio.sleep(delay)   # be polite: many college sites rate-limit bursts
                    batch: List[Tuple[str, int, bool]] = []
                    while heap and len(batch) < concurrency:
                        _, _, url, depth, is_external = heapq.heappop(heap)
                        if self._is_pdf(url):
                            if pdf_count >= max_pdfs:
                                continue
                            pdf_count += 1
                        batch.append((url, depth, is_external))
                    if not batch:
                        break

                    html_items = [b for b in batch if not self._is_pdf(b[0])]
                    pdf_items = [b for b in batch if self._is_pdf(b[0])]
                    for url, depth, _ in batch:
                        emit("page_start", {"url": url, "depth": depth,
                                            "scraped_count": scraped_pages, "max_pages": max_pages})

                    html_results: Dict[str, Any] = {}
                    if html_items:
                        try:
                            results = await crawler.arun_many([u for u, _, _ in html_items], config=run_cfg)
                            for r in results:
                                html_results[r.url] = r
                                html_results.setdefault(self.normalize_url(r.url), r)
                        except Exception as exc:
                            logger.warning(f"Batch crawl failed: {exc}")
                    pdf_results = await asyncio.gather(
                        *(asyncio.to_thread(self.fetch_pdf, u) for u, _, _ in pdf_items)
                    )

                    for (url, depth, _), (text, title) in zip(pdf_items, pdf_results):
                        if scraped_pages >= max_pages:
                            break
                        if text:
                            store(url, f"{title} (PDF)", self.clean_markdown(text), depth)
                        else:
                            emit("page_error", {"url": url, "error": "Unreadable or scanned PDF"})

                    for url, depth, is_external in html_items:
                        if scraped_pages >= max_pages:
                            break
                        res = html_results.get(url)
                        if res is not None and self.is_blocked(res):
                            blocked_streak += 1
                            if self.dedupe_key(url) not in retried:
                                retried.add(self.dedupe_key(url))
                                seq += 1
                                heapq.heappush(heap, (-self.score_url(url, depth), seq, url, depth, is_external))
                            continue
                        if res is None or not res.success:
                            err = getattr(res, "error_message", None) or "no response"
                            logger.debug(f"Failed {url}: {err}")
                            emit("page_error", {"url": url, "error": str(err)[:200]})
                            continue
                        final_url = self.normalize_url(res.redirected_url or res.url)
                        final_host = self.get_domain(final_url)
                        if not self._is_crawlable(final_url) or any(
                                final_host == d or final_host.endswith("." + d) for d in SOCIAL_DOMAINS):
                            continue   # redirected to a login page, Google Form, social site...
                        if final_url != url:
                            final_key = self.dedupe_key(final_url)
                            if final_key in enqueued and self.dedupe_key(url) != final_key:
                                continue   # redirected onto a page we already have
                            enqueued.add(final_key)
                        blocked_streak = 0
                        title = (res.metadata or {}).get("title") or final_url
                        raw = self.clean_markdown(res.markdown.raw_markdown if res.markdown else "")
                        if self.is_soft_404(title, raw, res.status_code, not_found_words, site_words):
                            logger.debug(f"Soft 404: {final_url}")
                            continue
                        text = self._strip_boilerplate(raw, line_counts)
                        store(final_url, title, text, depth)

                        # Links are followed even from thin pages (splash pages, framesets, redirectors)
                        if not is_external and self._in_scope(final_url, base_domains) and depth < max_depth:
                            handle_links(self._result_links(res), final_url, depth)
                            for link in self.extract_hidden_links(res.html, final_url):
                                handle_links([(link, "")], final_url, depth)

                    # ── The site is refusing us: slow right down once, then give up ──
                    if blocked_streak >= 3 and not slowed_down:
                        slowed_down = True
                        blocked_streak = 0
                        concurrency, delay = 1, max(delay, 5.0)
                        emit("status", {"message": "The site is refusing requests (rate limit / firewall). "
                                                   "Pausing 60s, then crawling one page at a time..."})
                        await asyncio.sleep(60)
                    elif blocked_streak >= 6 and slowed_down:
                        stopped_reason = (f"{home_host} blocks automated access (HTTP 403/429), so the crawl stopped "
                                          f"after {scraped_pages} pages. Try again later or a lower page limit.")
                        emit("status", {"message": stopped_reason})
                        logger.warning(stopped_reason)
                        break

            # ── Structured fact extraction ──────────────────────────────────
            self.db.update_college_counts(college_id)
            facts_found = 0
            if self.extractor is not None and scraped_pages:
                emit("status", {"message": "Extracting courses, recruiters, fees and other facts..."})
                try:
                    facts_found = await asyncio.to_thread(
                        self.extractor.extract_college, college_id,
                        lambda done, total: emit("status", {"message": f"Extracting facts: page {done} of {total}"}),
                    )
                except Exception as exc:
                    logger.warning(f"Fact extraction failed: {exc}", exc_info=True)
                if getattr(self.extractor, "stopped_reason", ""):
                    stopped_reason = " ".join(filter(None, [stopped_reason, self.extractor.stopped_reason]))

            self.db.update_college_status(college_id, "completed", college_name)
            emit("completed", {
                "college_id": college_id, "college_name": college_name,
                "pages_scraped": scraped_pages, "total_chunks": total_chunks, "facts": facts_found,
                "warning": stopped_reason,
            })
            return {
                "success": True, "college_id": college_id, "college_name": college_name,
                "pages_scraped": scraped_pages, "total_chunks": total_chunks, "facts": facts_found,
                "warning": stopped_reason,
            }

        except Exception as exc:
            logger.error(f"Crawl aborted: {exc}", exc_info=True)
            self.db.update_college_counts(college_id)
            self.db.update_college_status(college_id, "failed" if not scraped_pages else "completed", college_name)
            emit("error", {"error": str(exc)})
            return {"success": False, "error": str(exc), "college_id": college_id}

    @staticmethod
    def is_blocked(result: Any) -> bool:
        """The server refused us (firewall, rate limit, CAPTCHA page) rather than the page being missing."""
        if getattr(result, "status_code", None) in (403, 429, 503):
            return True
        err = (getattr(result, "error_message", None) or "").lower()
        if "anti-bot" in err or "403" in err or "429" in err or "too many requests" in err:
            return True
        title = ((getattr(result, "metadata", None) or {}).get("title") or "")
        md = getattr(result, "markdown", None)
        text = (getattr(md, "raw_markdown", None) or "") if md is not None else ""
        # Body text only counts on short pages: a real content page may mention "access denied" in passing
        return bool(_CHALLENGE_RE.search(title) or (len(text.split()) < 200 and _CHALLENGE_RE.search(text)))

    @staticmethod
    def _result_links(result: Any) -> List[Tuple[str, str]]:
        links = result.links or {}
        return [
            (link.get("href", ""), link.get("text", "") or link.get("title", ""))
            for group in ("internal", "external")
            for link in links.get(group, [])
            if link.get("href")
        ]
