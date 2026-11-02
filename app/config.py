import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables
load_dotenv(BASE_DIR / ".env")

class Config:
    BASE_DIR = BASE_DIR
    DATA_DIR = BASE_DIR / "data"
    AUDIO_CACHE_DIR = BASE_DIR / "data" / "audio_cache"
    DB_PATH = BASE_DIR / os.getenv("DATABASE_PATH", "data/college_knowledge.db")

    # LLM Settings
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip().strip("\"'")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip().strip("\"'")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip().strip("\"'")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip().strip("\"'")
    DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "gemini")
    GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    # OpenRouter: ":free" models cost nothing (they have a daily request limit instead)
    OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")
    # Free models share one pool with every OpenRouter user and are often busy; these are tried in turn after OPENROUTER_MODEL
    OPENROUTER_FALLBACK_MODELS = [m.strip() for m in os.getenv(
        "OPENROUTER_FALLBACK_MODELS",
        "nvidia/nemotron-3-ultra-550b-a55b:free,nvidia/nemotron-3.5-super-vl:free,google/gemma-4-31b-it:free,"
        "google/gemma-4-26b-a4b-it:free,nvidia/nemotron-3.5-lightning:free",
    ).split(",") if m.strip()]
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-large-v3")   # Groq speech-to-text; detects the spoken language

    # Crawler Settings
    DEFAULT_MAX_PAGES = int(os.getenv("DEFAULT_MAX_PAGES", "120"))
    DEFAULT_MAX_DEPTH = int(os.getenv("DEFAULT_MAX_DEPTH", "4"))
    CRAWL_CONCURRENCY = int(os.getenv("CRAWL_CONCURRENCY", "2"))   # browser pages rendered in parallel
    CRAWL_DELAY = float(os.getenv("CRAWL_DELAY", "1.0"))           # seconds between batches; many college sites rate-limit bursts
    PAGE_TIMEOUT_MS = int(os.getenv("PAGE_TIMEOUT_MS", "30000"))
    # Structured fact extraction (courses, recruiters, fees...) after a crawl: one LLM call per relevant page
    EXTRACT_FACTS = os.getenv("EXTRACT_FACTS", "1") == "1"
    EXTRACT_MAX_PAGES = int(os.getenv("EXTRACT_MAX_PAGES", "40"))
    EXTRACT_TOKENS_PER_MIN = int(os.getenv("EXTRACT_TOKENS_PER_MIN", "0"))  # 0 = auto (7000 for Groq's free tier, unpaced otherwise)
    EXTRACT_MIN_INTERVAL = float(os.getenv("EXTRACT_MIN_INTERVAL", "2.0"))  # seconds between LLM calls (free-tier rate limits)
    # Quick test mode: small crawl + short extraction, to try the app in a few minutes. Set QUICK_TEST=0 for full runs.
    QUICK_TEST = os.getenv("QUICK_TEST", "0") == "1"
    QUICK_MAX_PAGES = int(os.getenv("QUICK_MAX_PAGES", "20"))
    QUICK_MAX_DEPTH = int(os.getenv("QUICK_MAX_DEPTH", "2"))
    QUICK_EXTRACT_PAGES = int(os.getenv("QUICK_EXTRACT_PAGES", "8"))
    MIN_COSINE = float(os.getenv("MIN_COSINE", "0.4"))  # relevance gate for real embeddings; tune on your test questions
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "12"))
    USER_AGENT = os.getenv(
        "USER_AGENT",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )

    # Server Settings
    HOST = os.getenv("HOST", "127.0.0.1")
    PORT = int(os.getenv("PORT", "8000"))

# Ensure required directories exist
Config.DATA_DIR.mkdir(parents=True, exist_ok=True)
Config.AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
