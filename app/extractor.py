import re
import json
import time
import logging
from collections import deque
from typing import List, Dict, Any, Optional, Callable

from app.config import Config
from app.storage import Database

logger = logging.getLogger(__name__)

# Fact categories and what belongs in each (shown to the LLM)
CATEGORIES: Dict[str, str] = {
    "ug_program": "Undergraduate degree programmes with branch/specialisation (B.E., B.Tech, B.Sc, B.Com, BBA, B.Arch...)",
    "pg_program": "Postgraduate programmes with specialisation (M.E., M.Tech, MBA, MCA, M.Sc, M.Com...)",
    "doctoral_program": "PhD / doctoral / research programmes",
    "department": "Academic departments, schools or centres",
    "recruiter": "Companies that recruit or hire students / visited the campus for placements",
    "placement_stat": "Placement statistics: percentage placed, highest/average/median package, number of offers (with year)",
    "fee": "Fee amounts (tuition, hostel, exam, admission) with programme and year",
    "scholarship": "Scholarships offered to students",
    "admission": "Admission facts: eligibility, entrance exams accepted, deadlines, counselling codes, intake",
    "facility": "Campus facilities: hostels, library, labs, transport, sports, medical",
    "accreditation": "Accreditations, rankings and affiliations (NAAC grade, NBA, NIRF rank, affiliating university, AICTE)",
    "contact": "Contact details: phone, email, address, principal / placement officer / admission office",
}

# Pages are only sent to the LLM if they mention at least one of these
_RELEVANCE_RE = re.compile(
    r"(b\.?\s?e\b|b\.?\s?tech|m\.?\s?e\b|m\.?\s?tech|mba|mca|b\.?sc|m\.?sc|ph\.?d|undergraduate|postgraduate|"
    r"programme|program|course|department|placement|placed|recruit|compan|package|lpa|salary|offer|"
    r"fee|tuition|scholarship|admission|eligib|intake|hostel|naac|nba|nirf|accredit|affiliat|contact|phone|email)",
    re.I,
)

# A page whose URL or title names one of these topics is likely to hold the facts we want
_TOPIC_PAGE_RE = re.compile(
    r"(programmes?[-_ ]offered|programmes?|programs?|courses?[-_ ]offered|admission|be[-_]btech|ug|pg|"
    r"placement|recruit|cdc|career|training|fee|scholarship|department|about|hostel|facilit|"
    r"contact|accredit|nirf|naac|nba|ranking|intake)",
    re.I,
)
# Long documents full of course codes and exam logistics: rarely worth an LLM call
_LOW_VALUE_PAGE_RE = re.compile(
    r"(syllabus|syllabi|curriculum|scheduling|scheme[-_ ]of|regulation|exam|question|timetable|time-table|"
    r"circular|notice|minutes|tender|newsletter|journal|annexure)",
    re.I,
)

SYSTEM_PROMPT = (
    "You extract structured facts about a college from the text of one of its web pages or PDFs. "
    "Only record facts explicitly stated in the text. Respond with JSON only."
)

WINDOW_CHARS = 6000
MAX_WINDOWS_PER_PAGE = 2
EST_OUTPUT_TOKENS = 1500        # reply + reasoning tokens counted against per-minute limits
MAX_RETRY_WAIT = 90.0           # a longer "try again in" means a daily quota, not a per-minute one

# "try again in 8.82s", "try again in 1m23.5s", "retry in 30s", "retryDelay": "30s", "450ms"
_RETRY_AFTER_RE = re.compile(
    r"(?:try again in|retry in|retry after|retrydelay[\"':\s]*)\s*((?:\d+h)?(?:\d+m(?!s))?(?:[\d.]+(?:ms|s))?)", re.I
)
_DAILY_LIMIT_RE = re.compile(r"(per day|tokens per day|requests per day|\btpd\b|\brpd\b|daily)", re.I)


class QuotaExhausted(RuntimeError):
    """The LLM provider's quota is used up for a long time (e.g. a daily limit): stop extracting."""


def retry_after_seconds(message: str) -> Optional[float]:
    """Parse a provider's 'try again in ...' hint into seconds."""
    m = _RETRY_AFTER_RE.search(message or "")
    if not m or not m.group(1):
        return None
    total = 0.0
    for value, unit in re.findall(r"([\d.]+)(ms|h|m|s)", m.group(1)):
        total += float(value) * {"h": 3600, "m": 60, "s": 1, "ms": 0.001}[unit]
    return total or None


class FactExtractor:
    """Turns crawled pages into a structured fact table using whichever LLM the RAG engine is configured with.

    `complete(system, prompt) -> str` is any LLM call; RAGEngine.complete fits.
    `tokens_per_min` paces calls under a provider's per-minute token limit (0 = no pacing).
    """

    def __init__(self, db: Database, complete: Callable[[str, str], str], tokens_per_min: int = 0,
                 min_interval: Optional[float] = None):
        self.db = db
        self.complete = complete
        self.tokens_per_min = tokens_per_min
        self.min_interval = Config.EXTRACT_MIN_INTERVAL if min_interval is None else min_interval
        self._last_call = 0.0
        self._sent: deque = deque()     # (time, estimated tokens) of calls in the last minute
        self.stopped_reason = ""

    @staticmethod
    def relevance(text: str) -> int:
        return len(_RELEVANCE_RE.findall(text or ""))

    @staticmethod
    def relevant_text(text: str) -> str:
        """Keep only the sections of a page that mention programmes, placements, fees etc.

        Sections are split at markdown headings; a section counts if its heading or body is relevant,
        so a "Our Recruiters" heading keeps the bare list of company names under it.
        """
        parts = re.split(r"(?m)^(?=\s{0,3}#{1,6}\s)", text or "")
        kept = [p.strip() for p in parts if _RELEVANCE_RE.search(p)]
        return "\n\n".join(kept)

    @staticmethod
    def build_prompt(title: str, url: str, text: str) -> str:
        cats = "\n".join(f'- "{k}": {v}' for k, v in CATEGORIES.items())
        return (
            f"Categories:\n{cats}\n\n"
            'Return {"facts": [{"category": "<one of the categories>", "name": "<short item name>", '
            '"details": "<optional short extra info: duration, intake, year, amount, grade>"}]}.\n'
            "Rules:\n"
            "- One fact per item: every programme, department, company, fee line etc. separately. Do not skip items in long lists.\n"
            '- "name" is the canonical item, e.g. "B.E. Computer Science and Engineering", "Infosys", "Highest package 2024", "NAAC".\n'
            "- Ignore navigation menus, news, events and anything about other institutions.\n"
            "- Programmes are degrees only (B.E., M.Tech, MBA, PhD...). Skip short training courses, workshops, FDPs and certificate events.\n"
            '- Record each programme with its branch ("B.E. Civil Engineering"), never a bare degree like "B.E." or "B.Tech.".\n'
            "- A programme name is the degree and branch only: leave out admission quotas (Aided, Self-supporting, Accredited), "
            "regulation years and document versions. Syllabus or regulation documents are not programmes.\n"
            '- Skip bare document/report links that state no actual value (e.g. "NIRF 2024 Report" with no rank).\n'
            '- If the page has no such facts, return {"facts": []}.\n\n'
            f"PAGE: {title}\nURL: {url}\n---\n{text}\n---"
        )

    @staticmethod
    def parse_facts(raw: str) -> List[Dict[str, str]]:
        """Parse the LLM's JSON, tolerating code fences and stray text around it."""
        if not raw:
            return []
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            return []
        try:
            data = json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            return []
        items = data.get("facts", []) if isinstance(data, dict) else []
        facts = []
        for item in items:
            if not isinstance(item, dict):
                continue
            category = str(item.get("category", "")).strip().lower()
            name = re.sub(r"\s+", " ", str(item.get("name", ""))).strip(" -•*")
            details = re.sub(r"\s+", " ", str(item.get("details") or "")).strip()
            if category in CATEGORIES and 2 <= len(name) <= 150:
                facts.append({"category": category, "name": name, "details": details[:300]})
        return facts

    def _pace(self, est_tokens: int):
        """Wait so calls stay under the per-minute token budget and the minimum interval."""
        wait = self.min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        if not self.tokens_per_min:
            return
        while True:
            now = time.monotonic()
            while self._sent and now - self._sent[0][0] > 60:
                self._sent.popleft()
            used = sum(t for _, t in self._sent)
            if not self._sent or used + est_tokens <= self.tokens_per_min:
                return
            time.sleep(max(0.5, 60 - (now - self._sent[0][0]) + 0.5))

    def _call(self, prompt: str) -> str:
        """LLM call paced under rate limits; waits as long as the provider asks, gives up on daily quotas."""
        est_tokens = len(prompt) // 4 + EST_OUTPUT_TOKENS
        for attempt in range(4):
            self._pace(est_tokens)
            self._last_call = time.monotonic()
            self._sent.append((self._last_call, est_tokens))
            try:
                return self.complete(SYSTEM_PROMPT, prompt)
            except Exception as exc:
                msg = str(exc)
                lower = msg.lower()
                if not any(k in lower for k in ("429", "rate limit", "rate_limit", "quota", "resource_exhausted")):
                    raise
                wait = retry_after_seconds(msg)
                if _DAILY_LIMIT_RE.search(msg) or (wait is not None and wait > MAX_RETRY_WAIT):
                    raise QuotaExhausted(msg[:300])
                wait = (wait or 10.0 * (attempt + 1)) + 1.0
                logger.info(f"LLM rate limit hit; waiting {wait:.0f}s before retrying")
                time.sleep(wait)
        raise RuntimeError("LLM rate limit: gave up after retries")

    @staticmethod
    def page_priority(page: Dict[str, Any]) -> float:
        """How worth an LLM call a page is. Pages *about* programmes, admissions, placements and fees
        come first; syllabus/exam documents last, however many keywords they contain."""
        label = f"{page['url']} {page.get('title') or ''}"
        text = page["clean_text"]
        density = FactExtractor.relevance(text) / max(len(text) / 1000, 1.0)   # keyword hits per 1000 chars
        score = min(density, 20.0) + 25.0 * len(set(m.lower() for m in _TOPIC_PAGE_RE.findall(label)))
        if _LOW_VALUE_PAGE_RE.search(label):
            score -= 100.0
        return score

    def select_pages(self, college_id: int) -> List[Dict[str, Any]]:
        pages = self.db.get_pages_with_text(college_id)
        for p in pages:
            p["clean_text"] = self.relevant_text(p["clean_text"])
        scored = [(self.page_priority(p), p) for p in pages if self.relevance(p["clean_text"]) >= 2]
        scored.sort(key=lambda sp: sp[0], reverse=True)
        limit = min(Config.EXTRACT_MAX_PAGES, Config.QUICK_EXTRACT_PAGES) if Config.QUICK_TEST else Config.EXTRACT_MAX_PAGES
        return [p for _, p in scored[:limit]]

    def extract_college(self, college_id: int, progress: Optional[Callable[[int, int], None]] = None) -> int:
        """Extract facts from the college's most relevant pages. Returns the number of new facts stored."""
        # Resume support: pages that already produced facts are not sent again
        done_urls = {f["source_url"] for f in self.db.get_facts(college_id)}
        pages = [p for p in self.select_pages(college_id) if p["url"] not in done_urls]
        added = 0
        failures = 0
        self.stopped_reason = ""
        started = time.monotonic()
        for i, page in enumerate(pages, 1):
            if progress:
                progress(i, len(pages))
            logger.info(f"Extracting facts: page {i}/{len(pages)} ({added} facts so far) {page['url']}")
            text = page["clean_text"] or ""
            max_windows = 1 if Config.QUICK_TEST else MAX_WINDOWS_PER_PAGE
            windows = [text[s:s + WINDOW_CHARS] for s in range(0, len(text), WINDOW_CHARS)][:max_windows]
            for window in windows:
                if self.relevance(window) < 2:
                    continue
                try:
                    raw = self._call(self.build_prompt(page["title"] or "", page["url"], window))
                except QuotaExhausted as exc:
                    self.stopped_reason = (
                        f"LLM quota used up after {i - 1} of {len(pages)} pages ({added} facts saved). "
                        f"Run 'python cli.py extract {college_id}' later to finish."
                    )
                    logger.warning(f"{self.stopped_reason} Provider said: {exc}")
                    return added
                except Exception as exc:
                    failures += 1
                    logger.warning(f"Fact extraction failed for {page['url']}: {str(exc)[:200]}")
                    if failures >= 5 and added == 0:
                        logger.warning("Stopping fact extraction: the LLM keeps failing (check the API key/quota).")
                        return added
                    continue
                added += self.db.save_facts(college_id, self.parse_facts(raw), page["url"])
        logger.info(f"Extracted {added} new facts for college {college_id} from {len(pages)} pages "
                    f"in {time.monotonic() - started:.0f}s")
        return added


def make_extractor(db: Database, rag: Any) -> Optional[FactExtractor]:
    """Fact extractor using the RAG engine's LLM, or None if extraction is off or no API key is set."""
    if not Config.EXTRACT_FACTS:
        return None
    provider = rag._get_active_provider()
    if provider == "fallback":
        return None
    # Groq's free tier allows ~8000 tokens/minute; stay a little under it
    tpm = Config.EXTRACT_TOKENS_PER_MIN or (7000 if provider == "groq" else 0)
    # OpenRouter's free models allow roughly 20 requests a minute
    interval = max(Config.EXTRACT_MIN_INTERVAL, 4.0) if provider == "openrouter" else None
    return FactExtractor(db, rag.complete, tokens_per_min=tpm, min_interval=interval)
