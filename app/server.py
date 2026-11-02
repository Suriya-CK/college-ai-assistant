import os
import uuid
import asyncio
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, Request, BackgroundTasks, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import Config
from app.storage import Database
from app.crawler import CollegeCrawler
from app.extractor import make_extractor
from app.vector_store import VectorSearchEngine
from app.rag_engine import RAGEngine
from app.voice_service import VoiceService
from app.multilingual import SUPPORTED_LANGUAGES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("college_assistant")

app = FastAPI(title="College AI Voice Assistant", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static and Templates
app_dir = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(app_dir / "static")), name="static")
templates = Jinja2Templates(directory=str(app_dir / "templates"))

# Initialize core singletons
db = Database()
vector_engine = VectorSearchEngine(db)
rag_engine = RAGEngine(db, vector_engine)
voice_service = VoiceService()

# In-memory progress tracking for live crawls
active_crawls: Dict[str, Dict[str, Any]] = {}
crawl_event_queues: Dict[str, asyncio.Queue] = {}
# Fact extraction running after a crawl: college_id -> {"done": pages processed, "total": pages to process}
extraction_progress: Dict[int, Dict[str, int]] = {}

# ----------------------------------------------------------------
# Pydantic Request Models
# ----------------------------------------------------------------
class CrawlRequest(BaseModel):
    url: str
    max_pages: Optional[int] = Config.DEFAULT_MAX_PAGES
    max_depth: Optional[int] = Config.DEFAULT_MAX_DEPTH

class ChatRequest(BaseModel):
    college_id: int
    query: str
    conversation_id: Optional[str] = None
    voice: Optional[str] = None
    generate_audio: Optional[bool] = True
    language: Optional[str] = "en"

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = "en-IN-NeerjaNeural"
    lang: Optional[str] = "en"

class SettingsUpdateRequest(BaseModel):
    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    openrouter_model: Optional[str] = None
    default_provider: Optional[str] = None

# ----------------------------------------------------------------
# Frontend HTML Route
# ----------------------------------------------------------------
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return HTMLResponse(content="", status_code=204)

@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    colleges = db.list_colleges()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "colleges": colleges,
            "languages": SUPPORTED_LANGUAGES,
            "default_voice": "en-IN-NeerjaNeural",
            "has_gemini": bool(Config.GEMINI_API_KEY),
            "has_openai": bool(Config.OPENAI_API_KEY),
            "has_groq": bool(Config.GROQ_API_KEY),
        }
    )

# ----------------------------------------------------------------
# Crawler & Knowledge Management Routes
# ----------------------------------------------------------------
def run_crawler_task(crawl_id: str, url: str, max_pages: int, max_depth: int):
    # Facts are extracted after the crawl finishes, so the college can be chatted with straight away
    crawler = CollegeCrawler(db)

    def on_progress(event_type: str, data: Dict[str, Any]):
        active_crawls[crawl_id]["last_event"] = {"type": event_type, "data": data}
        if crawl_id in crawl_event_queues:
            try:
                loop = active_crawls[crawl_id].get("loop")
                if loop and loop.is_running():
                    loop.call_soon_threadsafe(
                        crawl_event_queues[crawl_id].put_nowait,
                        {"type": event_type, "data": data}
                    )
            except Exception as e:
                logger.debug(f"Queue error: {e}")

    result = crawler.crawl(
        start_url=url,
        max_pages=max_pages,
        max_depth=max_depth,
        progress_callback=on_progress
    )

    if result.get("success"):
        college_id = result["college_id"]
        # Batch index any missing embeddings
        vector_engine.index_pending_chunks(college_id)

    active_crawls[crawl_id]["status"] = "completed" if result.get("success") else "failed"
    active_crawls[crawl_id]["result"] = result

    extractor = make_extractor(db, rag_engine) if result.get("success") else None
    if extractor is not None:
        college_id = result["college_id"]
        logger.info(f"Crawl done; extracting structured facts for college {college_id} in the background...")
        extraction_progress[college_id] = {"done": 0, "total": 0}

        def on_extract(done: int, total: int):
            extraction_progress[college_id] = {"done": done, "total": total}

        try:
            added = extractor.extract_college(college_id, on_extract)
            logger.info(f"Fact extraction finished: {added} facts ({db.get_fact_counts(college_id)})")
            if extractor.stopped_reason:
                logger.warning(extractor.stopped_reason)
        except Exception as e:
            logger.warning(f"Fact extraction failed: {e}", exc_info=True)
        finally:
            extraction_progress.pop(college_id, None)

@app.post("/api/crawl")
async def start_crawl(req: CrawlRequest, background_tasks: BackgroundTasks):
    raw_url = req.url.strip()
    if not raw_url.startswith(("http://", "https://")):
        raw_url = "https://" + raw_url

    parsed = urlparse(raw_url)
    if parsed.netloc in ["127.0.0.1", "localhost", "0.0.0.0"] or parsed.netloc.startswith(("127.", "localhost:")):
        raise HTTPException(
            status_code=400,
            detail="Cannot scrape localhost! Please enter a real college website URL, e.g. https://www.stanford.edu or https://www.harvard.edu"
        )

    crawl_id = str(uuid.uuid4())
    loop = asyncio.get_running_loop()

    active_crawls[crawl_id] = {
        "id": crawl_id,
        "url": raw_url,
        "status": "running",
        "loop": loop,
        "result": None,
        "last_event": None
    }
    crawl_event_queues[crawl_id] = asyncio.Queue()

    max_pages = req.max_pages or Config.DEFAULT_MAX_PAGES
    max_depth = req.max_depth or Config.DEFAULT_MAX_DEPTH
    if Config.QUICK_TEST:
        max_pages = min(max_pages, Config.QUICK_MAX_PAGES)
        max_depth = min(max_depth, Config.QUICK_MAX_DEPTH)
    background_tasks.add_task(run_crawler_task, crawl_id, raw_url, max_pages, max_depth)

    return {"crawl_id": crawl_id, "url": raw_url, "status": "running",
            "max_pages": max_pages, "quick_test": Config.QUICK_TEST}

@app.get("/api/crawl/stream/{crawl_id}")
async def stream_crawl_progress(crawl_id: str):
    """Server-Sent Events (SSE) stream for real-time crawling feedback."""
    if crawl_id not in active_crawls:
        raise HTTPException(status_code=404, detail="Crawl not found")

    queue = crawl_event_queues.get(crawl_id)

    async def event_generator():
        import json
        while True:
            if active_crawls.get(crawl_id, {}).get("status") in ["completed", "failed"] and queue.empty():
                final_res = active_crawls[crawl_id].get("result") or {}
                yield f"data: {json.dumps({'type': 'done', 'data': final_res})}\n\n"
                break

            try:
                event = await asyncio.wait_for(queue.get(), timeout=1.0)
                yield f"data: {json.dumps(event)}\n\n"
            except asyncio.TimeoutError:
                # Send heartbeat
                yield f": heartbeat\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/colleges")
async def list_colleges():
    return db.list_colleges()

@app.get("/api/colleges/{college_id}")
async def get_college(college_id: int):
    college = db.get_college(college_id)
    if not college:
        raise HTTPException(status_code=404, detail="College not found")
    college["fact_count"] = sum(db.get_fact_counts(college_id).values())
    college["extraction"] = extraction_progress.get(college_id)   # None when not running
    return college

@app.get("/api/colleges/{college_id}/pages")
async def get_college_pages(college_id: int, limit: int = 100):
    return db.get_pages_for_college(college_id, limit=limit)

@app.delete("/api/colleges/{college_id}")
async def delete_college(college_id: int):
    db.delete_college(college_id)
    return {"success": True, "deleted_id": college_id}

# ----------------------------------------------------------------
# Conversational Q&A Routes
# ----------------------------------------------------------------
@app.post("/api/chat")
async def chat(req: ChatRequest):
    college = db.get_college(req.college_id)
    if not college:
        raise HTTPException(status_code=404, detail="College not found")

    conv_id = req.conversation_id or str(uuid.uuid4())
    db.create_conversation(conv_id, req.college_id, title=f"Q&A on {college.get('name')}")

    # Run Multilingual RAG Engine
    answer = rag_engine.answer_question(
        college_id=req.college_id,
        query=req.query,
        conversation_id=conv_id,
        language=req.language
    )

    # Generate Audio via VoiceService with a strict timeout so TTS never delays the chat answer
    audio_url = None
    if req.generate_audio and answer.get("spoken_text"):
        voice_choice = req.voice or answer.get("default_voice") or "en-IN-NeerjaNeural"
        # Never read a Tamil answer with an English voice: the voice must match the answer's language
        if req.voice and req.voice.split("-")[0] != answer.get("language", "en"):
            voice_choice = answer.get("default_voice") or voice_choice
        try:
            audio_url = await asyncio.wait_for(
                voice_service.generate_speech(
                    answer["spoken_text"],
                    voice=voice_choice,
                    lang=answer.get("language", "en")
                ),
                timeout=3.5
            )
        except Exception as te:
            logger.debug(f"Neural TTS skipped or timed out ({te}) - will use browser voice synthesis.")
            audio_url = None

    answer["audio_url"] = audio_url
    return answer

@app.get("/api/conversations/{conv_id}/messages")
async def get_conversation_history(conv_id: str):
    return db.get_messages(conv_id)

# ----------------------------------------------------------------
# Voice & Audio & Language Routes
# ----------------------------------------------------------------
@app.get("/api/languages")
async def get_languages():
    return SUPPORTED_LANGUAGES

@app.get("/api/voice/voices")
async def get_voices():
    return voice_service.get_available_voices()

@app.get("/api/voice/voices-by-language")
async def get_voices_by_language():
    return voice_service.get_voices_by_language()

@app.post("/api/voice/tts")
async def text_to_speech(req: TTSRequest):
    audio_url = await voice_service.generate_speech(
        req.text,
        voice=req.voice or "en-IN-NeerjaNeural",
        lang=req.lang or "en"
    )
    if not audio_url:
        raise HTTPException(status_code=500, detail="Failed to synthesize speech")
    return {"audio_url": audio_url}

@app.get("/api/voice/audio/{filename}")
async def get_cached_audio(filename: str):
    file_path = Config.AUDIO_CACHE_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(path=file_path, media_type="audio/mpeg")

@app.post("/api/voice/transcribe")
async def transcribe_audio(request: Request):
    """Transcribe recorded speech and detect its language (used by the mic in Auto-detect mode)."""
    content = await request.body()
    if not content:
        raise HTTPException(status_code=400, detail="No audio received")
    if len(content) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Recording too long")
    if not (Config.GROQ_API_KEY or Config.GEMINI_API_KEY):
        return {"text": "", "language": None,
                "error": "Voice auto-detect needs a Groq or Gemini API key (Settings)."}
    result = await asyncio.to_thread(
        voice_service.transcribe, content, request.headers.get("content-type", "audio/webm")
    )
    if not result.get("text"):
        result["error"] = "Sorry, I couldn't hear that clearly. Please try again."
    return result

# ----------------------------------------------------------------
# Settings Routes
# ----------------------------------------------------------------
def _mask(key: str, prefix: str = "") -> str:
    return f"{prefix or key[:4]}...{key[-4:]}" if len(key) > 8 else ""


@app.get("/api/settings")
async def get_settings():
    return {
        "has_gemini_key": bool(Config.GEMINI_API_KEY),
        "has_openai_key": bool(Config.OPENAI_API_KEY),
        "has_groq_key": bool(Config.GROQ_API_KEY),
        "has_openrouter_key": bool(Config.OPENROUTER_API_KEY),
        "gemini_masked_key": _mask(Config.GEMINI_API_KEY),
        "openai_masked_key": _mask(Config.OPENAI_API_KEY, "sk-"),
        "groq_masked_key": _mask(Config.GROQ_API_KEY, "gsk_"),
        "openrouter_masked_key": _mask(Config.OPENROUTER_API_KEY, "sk-or-"),
        "default_provider": Config.DEFAULT_LLM_PROVIDER,
        "active_provider": rag_engine._get_active_provider(),
        "gemini_model": Config.GEMINI_MODEL,
        "openai_model": Config.OPENAI_MODEL,
        "groq_model": Config.GROQ_MODEL,
        "openrouter_model": Config.OPENROUTER_MODEL,
        "default_max_pages": Config.DEFAULT_MAX_PAGES,
        "default_max_depth": Config.DEFAULT_MAX_DEPTH,
    }


def _clean_setting(value: str) -> str:
    """Pasted keys often carry spaces, quotes or a trailing newline."""
    return value.strip().strip("\"'").strip()


def _write_env_values(values: Dict[str, str]):
    """Update (or add) KEY=value lines in .env, keeping everything else as it is."""
    import re
    env_path = Config.BASE_DIR / ".env"
    content = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    for name, value in values.items():
        line = f"{name}={value}"
        if re.search(rf"^{name}=.*$", content, flags=re.MULTILINE):
            content = re.sub(rf"^{name}=.*$", lambda _: line, content, flags=re.MULTILINE)
        else:
            content = content.rstrip("\n") + f"\n{line}\n"
    env_path.write_text(content, encoding="utf-8")


def _describe_llm_error(error: str) -> str:
    low = error.lower()
    if "401" in low or "invalid api key" in low or "unauthorized" in low or "invalid_api_key" in low:
        return "The key was rejected. Check that you pasted the whole key."
    if "429" in low or "rate limit" in low or "quota" in low:
        return "The key works, but it's rate-limited right now (free-tier limit). Answers will work again once the limit resets."
    if "404" in low or "not a valid model" in low or "no endpoints found" in low:
        return "The key works, but the model name wasn't found. Check the model name."
    return error[:200]


@app.post("/api/settings")
async def update_settings(req: SettingsUpdateRequest):
    updates: Dict[str, str] = {}
    fields = {
        "gemini_api_key": "GEMINI_API_KEY",
        "openai_api_key": "OPENAI_API_KEY",
        "groq_api_key": "GROQ_API_KEY",
        "openrouter_api_key": "OPENROUTER_API_KEY",
        "openrouter_model": "OPENROUTER_MODEL",
    }
    for field, env_name in fields.items():
        value = getattr(req, field)
        if value is not None and _clean_setting(value):
            value = _clean_setting(value)
            setattr(Config, env_name, value)      # takes effect immediately, no restart needed
            updates[env_name] = value
    if req.default_provider:
        provider = req.default_provider.strip().lower()
        if provider not in ("gemini", "groq", "openai", "openrouter"):
            raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")
        Config.DEFAULT_LLM_PROVIDER = provider
        updates["DEFAULT_LLM_PROVIDER"] = provider

    if "GEMINI_API_KEY" in updates:
        vector_engine._gemini_client = None
    if "OPENAI_API_KEY" in updates:
        vector_engine._openai_client = None
    RAGEngine._working_gemini_model = None
    rag_engine._answer_cache.clear()

    try:
        _write_env_values(updates)
        saved = True
    except Exception as e:
        logger.warning(f"Could not write .env: {e}")
        saved = False

    # Prove the chosen provider and key actually answer, with one tiny request
    active = rag_engine._get_active_provider()
    if active == "fallback":
        check = {"ok": False, "message": f"No API key is set for {Config.DEFAULT_LLM_PROVIDER}. Paste its key above."}
    else:
        try:
            await asyncio.wait_for(
                asyncio.to_thread(rag_engine.complete, "Reply with the single word OK.", "Say OK"), timeout=30
            )
            check = {"ok": True, "message": f"{active.capitalize()} key works. Answers will use {active}."}
        except asyncio.TimeoutError:
            check = {"ok": False, "message": f"{active.capitalize()} didn't answer within 30 seconds. Try again shortly."}
        except Exception as e:
            check = {"ok": False, "message": _describe_llm_error(str(e))}

    return {
        "success": True,
        "saved_to_env": saved,
        "updated": sorted(updates.keys()),
        "active_provider": active,
        "check": check,
    }
