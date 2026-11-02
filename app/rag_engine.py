import re
import time
import json
import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple
import requests

from app.config import Config
from app.storage import Database
from app.vector_store import VectorSearchEngine
from app.multilingual import (
    SUPPORTED_LANGUAGES,
    detect_language,
    get_english_search_terms,
    translate_to_language
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the official College AI Conversational Assistant.
You represent the college and help prospective students, current students, parents, and visitors with accurate, warm, and helpful answers.

Knowledge base rules:
1. Ground your answers strictly in the provided College Knowledge Context below.
2. If the context does not contain the answer, politely say so based on the indexed college pages. Do not fabricate facts, fees, or deadlines.
3. Be conversational, articulate, encouraging, and clear.
4. Format your written response with clean Markdown (headings, bullet points, bold highlights).
5. When the context contains STRUCTURED FACTS and the student asks for a list (courses, companies, departments, fees...), list EVERY matching item from those facts, not just a few examples.
   Never mention "structured facts", "context", "sources" or "the data provided" in your answer: source links are shown to the student automatically.
6. At the very end of your response, output a special block formatted as:
[SPOKEN_SUMMARY]
A 2 to 4 sentence conversational summary of the answer, written specifically for voice output. It should sound friendly, natural when read aloud, and avoid bullet points, URLs, or markdown symbols.
[/SPOKEN_SUMMARY]
"""

# Localized fallback templates for regional languages
REGIONAL_HEADERS = {
    "en": ("Here is what the records for **{college_name}** state regarding **{query}**:", "Here is what I found for {college_name}."),
    "ta": ("**{college_name}** பற்றிய உங்கள் கேள்விக்கு (**{query}**), அதிகாரப்பூர்வ தகவல்கள்:", "{college_name} பற்றிய தகவல்கள் இதோ."),
    "hi": ("**{college_name}** के बारे में आपके प्रश्न (**{query}**) पर आधिकारिक रिकॉर्ड:", "{college_name} के बारे में जानकारी इस प्रकार है."),
    "te": ("**{college_name}** గురించి మీ ప్రశ్నకు (**{query}**) అధికారిక సమాచారం:", "{college_name} గురించి సమాచారం ఇదిగోండి."),
    "kn": ("**{college_name}** ಕುರಿತು ನಿಮ್ಮ ಪ್ರಶ್ನೆಗೆ (**{query}**) ಅಧಿಕೃತ ಮಾಹಿತಿ:", "{college_name} ಕುರಿತು ವಿವರಗಳು ಇಲ್ಲಿವೆ."),
    "ml": ("**{college_name}** സംബന്ധിച്ച നിങ്ങളുടെ ചോദ്യത്തിന് (**{query}**) ഔദ്യോഗിക വിവരങ്ങൾ:", "{college_name} സംബന്ധിച്ച വിവരങ്ങൾ ഇതാ."),
    "bn": ("**{college_name}** সম্পর্কিত আপনার প্রশ্নের (**{query}**) অফিসিয়াল তথ্য:", "{college_name} সম্পর্কে তথ্য এখানে দেওয়া হলো।"),
    "mr": ("**{college_name}** बद्दल आपल्या प्रश्नावर (**{query}**) अधिकृत माहिती:", "{college_name} बद्दल मिळालेली माहिती खालीलप्रमाणे आहे."),
    "gu": ("**{college_name}** વિશે તમારા પ્રશ્ન (**{query}**) પર સત્તાવાર માહિતી:", "{college_name} વિશેની માહિતી નીચે મુજબ છે.")
}

# Question keywords -> fact categories (see app/extractor.py). Short keywords must match a whole word.
FACT_INTENTS = [
    (("compan", "recruit", "hiring", "hire", "placement", "placed", "package", "salary", "lpa"), ("recruiter", "placement_stat")),
    (("ug", "undergraduate", "btech", "bachelor"), ("ug_program",)),
    (("pg", "postgraduate", "mtech", "master", "mba", "mca"), ("pg_program",)),
    (("phd", "doctoral", "doctorate"), ("doctoral_program",)),
    (("department", "dept", "branch", "school"), ("department", "ug_program")),
    (("fee", "fees", "tuition", "cost"), ("fee",)),
    (("scholarship",), ("scholarship",)),
    (("admission", "eligib", "entrance", "apply", "intake", "cutoff", "counselling"), ("admission",)),
    (("hostel", "facilit", "library", "transport", "lab", "labs", "sports"), ("facility",)),
    (("naac", "nba", "nirf", "rank", "accredit", "affiliat", "aicte"), ("accreditation",)),
    (("contact", "phone", "email", "address", "principal"), ("contact",)),
]
_PROGRAM_CATS = ("ug_program", "pg_program", "doctoral_program")
_GENERIC_PROGRAM_WORDS = ("course", "courses", "programme", "programmes", "program", "programs", "degree", "degrees")

FACT_LABELS = {
    "ug_program": "Undergraduate programmes", "pg_program": "Postgraduate programmes",
    "doctoral_program": "Doctoral programmes", "department": "Departments", "recruiter": "Recruiting companies",
    "placement_stat": "Placement statistics", "fee": "Fees", "scholarship": "Scholarships",
    "admission": "Admissions", "facility": "Facilities", "accreditation": "Accreditation & rankings",
    "contact": "Contact",
}
MAX_FACTS_IN_CONTEXT = 250
CHAT_MAX_RETRY_WAIT = 20.0   # seconds a student will wait for a rate-limited answer before getting the fallback

# Words that only describe the kind of degree or mode of study, not which programme it is
_DEGREE_WORDS = {
    "b", "m", "e", "be", "me", "tech", "btech", "mtech", "sc", "bsc", "msc", "com", "bcom", "mcom", "a", "ba", "ma",
    "hons", "honours", "integrated", "dual", "degree", "degrees", "programme", "programmes", "program", "programs",
    "course", "courses", "ug", "pg", "undergraduate", "postgraduate", "full", "part", "time", "fulltime", "parttime",
    "regular", "lateral", "entry", "and", "or", "in", "of", "the", "with", "affiliated", "institutions", "engineering",
}
_NOISE_RE = re.compile(r"\((full|part)[\s-]*time\)|\b(full|part)[\s-]*time\b|\b(programme|program|degree)s?\b", re.I)

# Programme-name noise: admission categories, regulation labels, syllabus versions and dates
_PROGRAM_TAG_RE = re.compile(
    r"\((?:non[\s-]*)?accredited[^)]*\)|\((?:aided|self[\s-]*supporting|ss|cbcs|cdio|pt|ft|tamil medium)\)", re.I
)
_PROGRAM_VERSION_RE = re.compile(
    r"[\s,-]*(?:\d{4}\s*[-–]\s*\d{2,4}|onwards|appended|absolute grading|regulations?\s*\d{4}|r[-\s]?\d{4}).*$", re.I
)
_AIML_RE = re.compile(r"\b(artificial intelligence (?:and|&) machine learning|ai\s*(?:and|&)\s*ml|aiml)\b", re.I)
# Details that describe a document or admission quota rather than the programme
_JUNK_DETAIL_RE = re.compile(r"^(code\s+\w+|.*semester.*|.*appended.*|.*onwards.*|.*grading.*|\d{4}\s*[-–]\s*\d{2,4})$", re.I)


def clean_program_name(name: str) -> str:
    """'BE-Electronics-and-Communication-Engineering-2023-2024 onwards' -> 'B.E. Electronics and Communication Engineering'."""
    n = re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", " ", name)          # hyphenated file-style names
    n = re.sub(r"^\s*(?:BE|B\.E)\b\.?\s*", "B.E. ", n, flags=re.I)
    n = re.sub(r"^\s*(?:BTech|B\.Tech)\b\.?\s*", "B.Tech. ", n, flags=re.I)
    n = _PROGRAM_TAG_RE.sub("", n)
    n = _PROGRAM_VERSION_RE.sub("", n)
    n = _AIML_RE.sub("AI & ML", n)
    n = n.replace("(AI & ML)", "(AI & ML)").replace(" & ", " and ") if "AI & ML" not in n else n
    return re.sub(r"\s+", " ", n).strip(" -—,/")


def _merge_key(category: str, name: str) -> Tuple[str, str]:
    base = _NOISE_RE.sub("", name)
    key = re.sub(r"[^a-z0-9]+", "", base.lower().replace("&", "and"))
    if category in _PROGRAM_CATS:
        key = re.sub(r"^(b|m)(e|tech|sc|arch|des)", r"\1\2", key)
        key = key.replace("barcharchitecture", "barch")   # "B.Arch. Architecture" == "B.Arch"
        key = key.replace("integrated", "")
    return category, key


# Recruiter rows that aren't one company ("Over 100 recruiters") and alumni-talk notes that aren't hiring info
_NOT_A_COMPANY_RE = re.compile(r"^(over|more than|around|about|\d+\s*\+?)\b", re.I)
_TALK_DETAIL_RE = re.compile(r"speaker|guest|lecture|webinar|talk by|alumni talk", re.I)
# A programme counts as "confirmed" when it carries official numbers (usually from the programmes table)
_CONFIRMED_RE = re.compile(r"intake|seats|duration|\d\s*years?", re.I)
MIN_CONFIRMED_PROGRAMMES = 5
# When pages disagree on numbers, the official programmes / admission pages win over department pages
_OFFICIAL_SOURCE_RE = re.compile(r"(programmes?|programs?|courses?)[-_]?offered|/admissions?(/|$)|/programmes?(/|$)", re.I)


def _clean_detail_parts(details: str, is_program: bool) -> List[str]:
    parts = [p.strip() for p in (details or "").split(";") if p.strip()]
    if is_program:
        parts = [p for p in parts if not _JUNK_DETAIL_RE.match(p) and p.upper() not in ("PT", "FT")]
    return parts


def merge_similar_facts(facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Clean a fact list for display: merge near-duplicates and drop noise.

    - "B.E. Civil (Full Time)", "B.E. Civil (Aided)" and "BE-Civil-2023-2024 onwards" become one item;
      a bare "B.E." is dropped when specific branches are listed.
    - When the official programmes table gave at least 5 programmes with intake/duration, stray
      mentions from other pages (abbreviations like "M.E. CS", old programmes) are left out.
    - Recruiters: "Over 100 recruiters" moves to placement statistics; alumni-talk notes are dropped.
    """
    merged: Dict[Tuple[str, str], Dict[str, Any]] = {}
    facts = sorted(facts, key=lambda f: not _OFFICIAL_SOURCE_RE.search(f.get("source_url") or ""))
    for f in facts:
        category = f["category"]
        if category == "recruiter" and _NOT_A_COMPANY_RE.match(f["name"].strip()):
            category = "placement_stat"
        is_program = category in _PROGRAM_CATS
        name = clean_program_name(f["name"]) if is_program else f["name"]
        base = _NOISE_RE.sub("", name).strip(" -—,/")
        key = _merge_key(category, base)
        parts = _clean_detail_parts(f.get("details") or "", is_program)
        if category == "recruiter":
            parts = [p for p in parts if not _TALK_DETAIL_RE.search(p)]
        clean_name = f["name"].strip() == name

        if key in merged:
            item = merged[key]
            if is_program and clean_name and item["_messy"]:
                # The cleanly named entry (usually the official table) wins, numbers included
                item["name"], item["_messy"] = base, False
                if any(_CONFIRMED_RE.search(p) for p in parts):
                    item["_parts"] = parts
                    continue
            has_numbers = any(_CONFIRMED_RE.search(p) for p in item["_parts"])
            for p in parts:
                if p in item["_parts"] or (is_program and has_numbers and _CONFIRMED_RE.search(p)):
                    continue   # don't stack a second, conflicting intake/duration
                item["_parts"].append(p)
        else:
            merged[key] = {**f, "category": category, "name": base or f["name"], "_parts": parts, "_messy": not clean_name}

    items = []
    for item in merged.values():
        item["details"] = "; ".join(item.pop("_parts"))[:300]
        item.pop("_messy", None)
        items.append(item)

    def is_generic(f: Dict[str, Any]) -> bool:
        words = re.findall(r"[a-z]+", f["name"].lower())
        return f["category"] in _PROGRAM_CATS and all(w in _DEGREE_WORDS for w in words)

    specific_cats = {f["category"] for f in items if not is_generic(f)}
    items = [f for f in items if not (is_generic(f) and f["category"] in specific_cats)]

    confirmed = lambda f: f["category"] in _PROGRAM_CATS and bool(_CONFIRMED_RE.search(f.get("details") or ""))
    for cat in _PROGRAM_CATS:
        if sum(1 for f in items if f["category"] == cat and confirmed(f)) >= MIN_CONFIRMED_PROGRAMMES:
            items = [f for f in items if f["category"] != cat or confirmed(f)]

    category_order = {c: i for i, c in reversed(list(enumerate(f["category"] for f in items)))}
    return sorted(items, key=lambda f: (category_order[f["category"]], not confirmed(f)))


def fact_categories_for(query: str) -> List[str]:
    """Which structured fact categories a question is about (empty if none)."""
    q = query.lower()
    tokens = set(re.split(r"[^a-z0-9]+", q.replace(".", "")))

    def hit(kw: str) -> bool:
        return kw in tokens if len(kw) <= 3 else kw in q

    cats: List[str] = []
    for keywords, categories in FACT_INTENTS:
        if any(hit(k) for k in keywords):
            cats.extend(c for c in categories if c not in cats)
    # "what courses are offered" with no level named -> all programme levels
    if not any(c in cats for c in _PROGRAM_CATS) and any(w in tokens for w in _GENERIC_PROGRAM_WORDS):
        cats.extend(_PROGRAM_CATS)
    return cats


class RAGEngine:
    """Conversational RAG Engine supporting Gemini, OpenAI, and local multilingual fallback."""

    # Gemini model priority: Gemini 3.8 Flash primary (all 2.0 models deprecated)
    GEMINI_MODEL_PRIORITY = [
        "gemini-3.8-flash",       # Primary: latest free model
        "gemini-1.5-flash",       # Stable fallback
        "gemini-1.5-flash-8b",    # Lightweight fallback
        "gemini-1.5-pro",         # Pro fallback
    ]
    _working_gemini_model: Optional[str] = None

    def __init__(self, db: Optional[Database] = None, vector_engine: Optional[VectorSearchEngine] = None):
        self.db = db or Database()
        self.vector_engine = vector_engine or VectorSearchEngine(self.db)
        # Simple in-process answer cache keyed by (college_id, query_lower, lang)
        self._answer_cache: Dict[str, Dict[str, Any]] = {}

    def _get_active_provider(self) -> str:
        gemini = (Config.GEMINI_API_KEY or "").strip()
        openai = (Config.OPENAI_API_KEY or "").strip()
        groq = (Config.GROQ_API_KEY or "").strip()
        openrouter = (Config.OPENROUTER_API_KEY or "").strip()
        preferred = (Config.DEFAULT_LLM_PROVIDER or "gemini").strip().lower()
        # No key-prefix checks: providers change key formats. A wrong key shows up as an API error in the logs.
        if preferred == "openai" and openai:
            return "openai"
        if preferred == "groq" and groq:
            return "groq"
        if preferred == "openrouter" and openrouter:
            return "openrouter"
        if preferred in ("groq", "openrouter"):
            return "fallback"
        if gemini:
            return "gemini"
        if openai:
            return "openai"
        return "fallback"

    def _relevance_filter(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep only chunks with real semantic similarity. Skipped when only the hash fallback is available,
        because its cosines are not calibrated (BM25/RRF ordering is all we have there)."""
        if not chunks or not chunks[0].get("semantic"):
            return chunks

        def keep(c: Dict[str, Any]) -> bool:
            cos = c.get("cosine")
            if cos is None:                      # no vector for this chunk (not embedded yet): can't judge, keep it
                return True
            if cos >= Config.MIN_COSINE:         # semantically close enough
                return True
            return (c.get("bm25_rank") or 99) <= 3   # strong keyword match (also covers overview questions)
        return [c for c in chunks if keep(c)]

    def _build_context_prompt(self, chunks: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, str]]]:
        """Format retrieved chunks into context string and extract citations."""
        citations = []
        seen_urls = set()
        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            url = chunk.get("url", "")
            title = chunk.get("title", "College Page")
            heading = chunk.get("heading", "")
            text = chunk.get("text", "")

            if url and url not in seen_urls:
                citations.append({"title": title, "url": url})
                seen_urls.add(url)

            context_parts.append(
                f"--- SOURCE {i}: {title} ({heading}) [URL: {url}] ---\n{text}"
            )

        context_str = "\n\n".join(context_parts)
        return context_str, citations

    @staticmethod
    def _build_facts_context(facts: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, str]]]:
        """Format structured facts grouped by category, plus citations for their source pages."""
        if not facts:
            return "", []
        groups: Dict[str, List[str]] = {}
        sources: List[str] = []
        for f in facts[:MAX_FACTS_IN_CONTEXT]:
            line = f"- {f['name']}" + (f" — {f['details']}" if f.get("details") else "")
            groups.setdefault(f["category"], []).append(line)
            if f.get("source_url") and f["source_url"] not in sources:
                sources.append(f["source_url"])
        parts = [f"#### {FACT_LABELS.get(cat, cat)} ({len(lines)} items)\n" + "\n".join(lines) for cat, lines in groups.items()]
        context = (
            "--- STRUCTURED FACTS (extracted from the whole college website; this is the complete list we have) ---\n"
            + "\n\n".join(parts)
        )
        return context, [{"title": "College website", "url": u} for u in sources[:5]]

    def complete(self, system_instruction: str, prompt: str) -> str:
        """Single-turn LLM call with the configured provider (used for fact extraction). Raises if none is configured."""
        provider = self._get_active_provider()
        if provider == "gemini":
            return self._call_gemini(system_instruction, [], prompt, max_output_tokens=4096)
        if provider == "openai":
            return self._call_openai(system_instruction, [], prompt)
        if provider == "groq":
            return self._call_groq(system_instruction, [], prompt)
        if provider == "openrouter":
            return self._call_openrouter(system_instruction, [], prompt)
        raise RuntimeError("No LLM API key configured")

    def _call_gemini(self, system_instruction: str, history: List[Dict[str, str]], user_prompt: str,
                     max_output_tokens: int = 1024) -> str:
        """Invoke Gemini via google-genai SDK. Tries fastest free models first."""
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=Config.GEMINI_API_KEY)
        contents = []
        for msg in history:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)]))

        # Build model list: gemini-3.8-flash primary, followed by fallbacks (strictly exclude retired 2.0 models)
        candidate_models = []
        if self._working_gemini_model and "2.0" not in self._working_gemini_model:
            candidate_models.append(self._working_gemini_model)
        for m in self.GEMINI_MODEL_PRIORITY:
            if "2.0" not in m and m not in candidate_models:
                candidate_models.append(m)
        if Config.GEMINI_MODEL and "2.0" not in Config.GEMINI_MODEL and Config.GEMINI_MODEL not in candidate_models:
            candidate_models.append(Config.GEMINI_MODEL)

        for model_name in candidate_models:
            # Try config with temperature, and fall back to minimal config for thinking models
            for config_attempt in [
                types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.3,
                    max_output_tokens=max_output_tokens,
                ),
                types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    max_output_tokens=max_output_tokens,
                ),
                None
            ]:
                try:
                    kwargs = {"model": model_name, "contents": contents}
                    if config_attempt is not None:
                        kwargs["config"] = config_attempt
                    response = client.models.generate_content(**kwargs)
                    if response and response.text:
                        RAGEngine._working_gemini_model = model_name
                        return response.text
                except Exception as e:
                    logger.debug(f"Gemini {model_name} attempt failed: {e}")
                    continue
            logger.warning(f"All attempts with Gemini model {model_name} failed.")

        raise RuntimeError("All Gemini model endpoints failed.")

    def _call_openai(self, system_instruction: str, history: List[Dict[str, str]], user_prompt: str) -> str:
        """Invoke OpenAI GPT-4o-mini via direct HTTP request."""
        headers = {
            "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        messages = [{"role": "system", "content": system_instruction}]
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": Config.OPENAI_MODEL,
            "messages": messages,
            "temperature": 0.3
        }
        resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=25)
        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        raise RuntimeError(f"OpenAI API returned status {resp.status_code}: {resp.text}")

    def _call_groq(self, system_instruction: str, history: List[Dict[str, str]], user_prompt: str) -> str:
        """Invoke Groq's OpenAI-compatible chat completions API."""
        headers = {
            "Authorization": f"Bearer {Config.GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        messages = [{"role": "system", "content": system_instruction}]
        messages.extend({"role": msg["role"], "content": msg["content"]} for msg in history)
        messages.append({"role": "user", "content": user_prompt})
        payload = {"model": Config.GROQ_MODEL, "messages": messages, "temperature": 0.3}
        resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=45)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        raise RuntimeError(f"Groq API returned status {resp.status_code}: {resp.text}")

    def _call_openrouter(self, system_instruction: str, history: List[Dict[str, str]], user_prompt: str) -> str:
        """Invoke OpenRouter's OpenAI-compatible chat completions API (one key, many models)."""
        headers = {
            "Authorization": f"Bearer {Config.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "X-Title": "College AI Assistant",
        }
        messages = [{"role": "system", "content": system_instruction}]
        messages.extend({"role": msg["role"], "content": msg["content"]} for msg in history)
        messages.append({"role": "user", "content": user_prompt})

        # Free models are often busy ("rate-limited upstream"): try the chosen model, then the backups
        models = list(dict.fromkeys([Config.OPENROUTER_MODEL] + Config.OPENROUTER_FALLBACK_MODELS))
        errors = []
        for model in models:
            payload = {"model": model, "messages": messages, "temperature": 0.3}
            try:
                resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=60)
            except requests.RequestException as e:
                errors.append(f"{model}: {e}")
                continue
            if resp.status_code == 401:
                # A bad key fails the same way for every model, so stop here
                raise RuntimeError(f"OpenRouter API returned status 401: {resp.text[:300]}")
            if resp.status_code == 200:
                data = resp.json()
                content = "" if data.get("error") else ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
                if content.strip():
                    if model != Config.OPENROUTER_MODEL:
                        logger.info(f"OpenRouter: {Config.OPENROUTER_MODEL} was busy, answered with {model}")
                    return content
                errors.append(f"{model}: empty answer")
                continue
            errors.append(f"{model}: status {resp.status_code} {resp.text[:150]}")
            logger.info(f"OpenRouter model {model} unavailable ({resp.status_code}); trying the next one")
        raise RuntimeError("OpenRouter: all free models are busy right now (429 rate limit). " + " | ".join(errors)[:600])

    def _call_with_retry(self, provider: str, system_instruction: str, history: List[Dict[str, str]], user_prompt: str) -> str:
        """Call the provider; if it says "try again in a few seconds" (per-minute limit), wait once and retry
        rather than dropping straight to the plain fallback answer."""
        call = {"gemini": self._call_gemini, "openai": self._call_openai, "groq": self._call_groq,
                "openrouter": self._call_openrouter}[provider]
        try:
            return call(system_instruction, history, user_prompt)
        except Exception as e:
            from app.extractor import retry_after_seconds
            msg = str(e)
            wait = retry_after_seconds(msg)
            if "429" not in msg or wait is None or wait > CHAT_MAX_RETRY_WAIT:
                raise
            logger.info(f"{provider} per-minute limit hit; retrying the answer in {wait:.1f}s")
            time.sleep(wait + 0.5)
            return call(system_instruction, history, user_prompt)

    def _fallback_generate(self, college: Dict[str, Any], chunks: List[Dict[str, Any]], query: str, lang: str = "en",
                           facts: Optional[List[Dict[str, Any]]] = None) -> str:
        """Deterministic smart summary if no API key is provided, with multilingual support."""
        college_name = college.get("name") or college.get("domain", "the college")
        header_template, spoken_intro_template = REGIONAL_HEADERS.get(lang, REGIONAL_HEADERS["en"])

        if facts:
            # Structured facts answer "list" questions completely without needing an LLM
            groups: Dict[str, List[str]] = {}
            for f in facts[:MAX_FACTS_IN_CONTEXT]:
                groups.setdefault(f["category"], []).append(
                    f"- {f['name']}" + (f" — {f['details']}" if f.get("details") else ""))
            # Item names (companies, courses) stay as written; headings and the spoken line follow the language
            local = (lambda t: translate_to_language(t, lang)) if lang != "en" else (lambda t: t)
            body = "\n\n".join(f"### {local(FACT_LABELS.get(c, c))}\n" + "\n".join(items) for c, items in groups.items())
            header_text = header_template.format(college_name=college_name, query=query)
            spoken_intro = spoken_intro_template.format(college_name=college_name)
            counts = ", ".join(f"{len(items)} {FACT_LABELS.get(c, c).lower()}" for c, items in groups.items())
            spoken = local(f"I found {counts}.")
            note = local("The AI service was busy, so this list comes straight from the college's website data.")
            return f"{header_text}\n\n{body}\n\n> *ℹ️ {note}*\n\n[SPOKEN_SUMMARY]\n{spoken_intro} {spoken}\n[/SPOKEN_SUMMARY]"

        if not chunks:
            no_match_text = {
                "en": f"I searched through the collected data for **{college_name}**, but could not find direct records for: *'{query}'*.",
                "ta": f"**{college_name}** ஆவணங்களில் *'{query}'* பற்றிய நேரடித் தகவல் கிடைக்கவில்லை.",
                "hi": f"**{college_name}** के रिकॉर्ड में *'{query}'* के बारे में सीधे जानकारी नहीं मिली।",
            }.get(lang, f"I searched through the collected data for **{college_name}**, but could not find direct records for: *'{query}'*.")

            spoken_no_match = {
                "en": f"I could not find records matching that question for {college_name}.",
                "ta": f"{college_name} ஆவணங்களில் அந்த கேள்விக்கான நேரடித் தகவல் கிடைக்கவில்லை.",
                "hi": f"{college_name} के रिकॉर्ड में इस प्रश्न से संबंधित जानकारी नहीं मिली।",
            }.get(lang, f"I could not find records matching that question for {college_name}.")

            return f"{no_match_text}\n\n[SPOKEN_SUMMARY]\n{spoken_no_match}\n[/SPOKEN_SUMMARY]"

        # Filter chunks by relevance score to prevent irrelevant cross-topic contamination
        top_score = chunks[0].get("score", 1.0) if chunks else 1.0
        relevant_chunks = []
        for c in chunks[:4]:
            c_score = c.get("score", 0.0)
            if top_score > 0 and c_score < top_score * 0.7:
                continue
            relevant_chunks.append(c)

        if not relevant_chunks:
            relevant_chunks = chunks[:1]

        highlights = []
        spoken_sentences = []
        for c in relevant_chunks:
            t = c.get("text", "").split("\n")
            body_lines = [line.strip() for line in t if line.strip() and not line.startswith("[")]
            body = " ".join(body_lines[:3])
            title = c.get("title", "College Info")
            heading = c.get("heading", "Section")
            url = c.get("url", "#")
            if body and len(body) > 20:
                snippet = body[:280].rstrip(".") + "..."
                # If target is non-English, attempt translation of the snippet
                if lang != "en":
                    translated_snippet = translate_to_language(snippet, lang)
                else:
                    translated_snippet = snippet

                highlights.append(f"- **[{title} - {heading}]({url})**:\n  {translated_snippet}")
                if len(spoken_sentences) < 2:
                    first_sent = body.split(". ")[0].strip()
                    if len(first_sent) > 15:
                        if lang != "en":
                            first_sent = translate_to_language(first_sent, lang)
                        spoken_sentences.append(first_sent)

        bullet_str = "\n\n".join(highlights) if highlights else "No detailed text available."
        header_text = header_template.format(college_name=college_name, query=query)
        spoken_intro = spoken_intro_template.format(college_name=college_name)

        spoken_body = ". ".join(spoken_sentences)
        if spoken_body and not spoken_body.endswith((".", "।", "?")):
            spoken_body += "."

        spoken = f"{spoken_intro} {spoken_body}".strip()

        if not (Config.GEMINI_API_KEY or Config.GROQ_API_KEY or Config.OPENAI_API_KEY):
            tip_text = "> *💡 Tip: Add an AI API key in Settings to get full conversational answers.*"
        else:
            tip_text = "> *ℹ️ The AI service didn't respond (often its free daily limit), so these are the closest matching passages.*"

        written_response = f"{header_text}\n\n{bullet_str}\n\n{tip_text}".strip()
        return f"{written_response}\n\n[SPOKEN_SUMMARY]\n{spoken}\n[/SPOKEN_SUMMARY]"

    def answer_question(
        self,
        college_id: int,
        query: str,
        conversation_id: Optional[str] = None,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """Retrieve relevant college data, build conversational prompt, query LLM, and return structured answer."""
        college = self.db.get_college(college_id)
        if not college:
            return {"error": f"College with ID {college_id} not found."}

        # 1. Resolve Language (support auto-detection or explicit UI choice)
        detected_lang = detect_language(query)
        target_lang = language if language and language in SUPPORTED_LANGUAGES else detected_lang
        lang_info = SUPPORTED_LANGUAGES.get(target_lang, SUPPORTED_LANGUAGES["en"])

        # 2. Check answer cache for identical (non-conversation) queries
        cache_key = f"{college_id}::{query.strip().lower()}::{target_lang}"
        if not conversation_id and cache_key in self._answer_cache:
            logger.info("Answer cache hit — returning cached response.")
            return self._answer_cache[cache_key]

        # 3. Translate regional query keywords into English search terms for DB retrieval
        search_query = get_english_search_terms(query, target_lang) if target_lang != "en" else query
        chunks = self.vector_engine.hybrid_search(college_id, search_query, top_k=8)
        chunks = self._relevance_filter(chunks)[:5]
        context_str, citations = self._build_context_prompt(chunks)
        fact_cats = fact_categories_for(search_query)
        facts = merge_similar_facts(self.db.get_facts(college_id, fact_cats)) if fact_cats else []
        facts_str, fact_citations = self._build_facts_context(facts)
        if not chunks and not facts:
            context_str = "(No relevant pages were found in the crawled college website for this question.)"
        if facts_str:
            context_str = f"{facts_str}\n\n{context_str}" if chunks else facts_str
            # For list answers the pages the facts came from are the real sources; keep only the top 2 search hits
            titles = self.db.get_page_titles([c["url"] for c in fact_citations])
            for c in fact_citations:
                c["title"] = titles.get(c["url"]) or c["title"]
            fact_urls = {c["url"] for c in fact_citations}
            citations = fact_citations + [c for c in citations if c["url"] not in fact_urls][:2]

        # 4. Get past conversation history
        history = []
        if conversation_id:
            past_msgs = self.db.get_messages(conversation_id, limit=6)
            for m in past_msgs:
                history.append({"role": m["role"], "content": m["content"]})

        # 5. Build system & user prompts with language guidance
        college_name = college.get("name") or college.get("domain", "College")
        college_url = college.get("url") or ""

        language_instruction = ""
        if target_lang != "en":
            language_instruction = (
                f"\n\nCRITICAL LANGUAGE REQUIREMENT:\n"
                f"You MUST answer the student's question in {lang_info['name']} ({lang_info['native']}).\n"
                f"Translate and explain all college facts accurately into natural, fluent {lang_info['name']}.\n"
                f"The [SPOKEN_SUMMARY] block MUST ALSO be written purely in {lang_info['name']} for voice readout."
            )

        system_instruction = f"{SYSTEM_PROMPT}\n\nYou are answering questions about **{college_name}** ({college_url}).{language_instruction}"

        user_prompt = (
            f"College Knowledge Context:\n{context_str}\n\n"
            f"Student Question: {query}\n\n"
            f"Provide a helpful, accurate, conversational response with citations based on the context above."
        )

        provider = self._get_active_provider()
        raw_output = ""

        try:
            if provider == "fallback":
                raw_output = self._fallback_generate(college, chunks, query, target_lang, facts)
            else:
                raw_output = self._call_with_retry(provider, system_instruction, history, user_prompt)
        except Exception as e:
            logger.warning(f"Primary LLM ({provider}) generation failed: {e}. Falling back to local summary.")
            raw_output = self._fallback_generate(college, chunks, query, target_lang, facts)
            provider = "fallback"

        # Extract [SPOKEN_SUMMARY] if present
        spoken_text = ""
        written_content = raw_output
        if "[SPOKEN_SUMMARY]" in raw_output and "[/SPOKEN_SUMMARY]" in raw_output:
            parts = raw_output.split("[SPOKEN_SUMMARY]")
            written_content = parts[0].strip()
            summary_part = parts[1].split("[/SPOKEN_SUMMARY]")[0].strip()
            spoken_text = summary_part
        else:
            spoken_text = written_content.split("\n\n")[0]

        # Models sometimes leave citation markers like 【SOURCE 2】 or 【Structured Facts】 in the text;
        # the sources are already shown as links under the answer
        written_content = re.sub(r"[ \t]*【[^】]{0,60}】", "", written_content)
        written_content = re.sub(r"\n?-{3,}\s*$", "", written_content.rstrip()).rstrip()
        spoken_text = re.sub(r"\s*【[^】]{0,60}】", "", spoken_text)

        # One source link per page title (several chunks of the same page share a title)
        unique_citations, seen_titles = [], set()
        for c in citations:
            title = (c.get("title") or c.get("url") or "").strip()
            if title not in seen_titles:
                seen_titles.add(title)
                unique_citations.append(c)
        citations = unique_citations[:6]

        # Save to database if conversation exists
        if conversation_id:
            self.db.add_message(conversation_id, role="user", content=query)
            self.db.add_message(
                conversation_id,
                role="assistant",
                content=written_content,
                spoken_text=spoken_text,
                citations=citations
            )

        result = {
            "conversation_id": conversation_id,
            "college_id": college_id,
            "college_name": college_name,
            "query": query,
            "content": written_content,
            "spoken_text": spoken_text,
            "citations": citations,
            "provider": provider,
            "language": target_lang,
            "language_name": lang_info["name"],
            "speech_locale": lang_info["speech_locale"],
            "default_voice": lang_info["default_voice"]
        }

        # Cache the result for non-conversation queries (max 200 entries, simple LRU via pop)
        if not conversation_id and provider != "fallback":
            if len(self._answer_cache) >= 200:
                self._answer_cache.pop(next(iter(self._answer_cache)))
            self._answer_cache[cache_key] = result

        return result
