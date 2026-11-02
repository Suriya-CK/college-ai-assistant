# 🎓 College AI Voice Assistant

Give it a college website URL. It crawls the site, builds a searchable knowledge base, and answers students' questions (courses, placements, recruiters, fees, admissions) by text or voice in English and 8 Indian languages, with links to the official source pages.

Built with **Python, FastAPI, Crawl4AI, SQLite (FTS5), LLMs (Groq / Gemini / OpenAI / OpenRouter), Whisper and edge-tts**. Also exposes its data to AI agents as an **MCP server**, and runs in **Docker**.

---

## ✨ Features

- **Website crawler** ([Crawl4AI](https://github.com/unclecode/crawl4ai), headless Chromium)
  - Renders JavaScript menus; follows redirects to the college's real domain.
  - Best-first order: placement, programme, admission and fee pages are crawled before news and galleries.
  - Skips fake "page not found" pages (soft 404s), duplicate URLs and duplicate content.
  - Reads PDFs (brochures, placement reports, fee structures) and strips repeated menus and footers.
  - Polite: 2 pages at a time with a pause; detects blocking and CAPTCHA pages, backs off, and reports them instead of saving them.
  - **Manual import** for sites that block automated crawlers: save pages from your browser (HTML, MHTML, PDF, TXT) and import the folder.
- **Structured fact extraction**: an LLM turns pages into JSON facts in 12 categories (UG/PG/PhD programmes, departments, recruiters, placement stats, fees, scholarships, admissions, facilities, accreditation, contacts). Paced to the provider's rate limits, resumable, and cleaned at answer time (duplicates merged, official pages preferred when numbers conflict).
- **Hybrid search (RAG)**: 400-word chunks with 50-word overlap; BM25 keyword search (SQLite FTS5) plus embedding search, merged with Reciprocal Rank Fusion; a relevance gate drops weak matches.
- **Grounded answers**: the LLM answers only from retrieved passages and facts, lists every item for "list all" questions, and cites source pages. Retries on rate limits, falls back across free models, and answers from stored facts if no LLM is reachable.
- **Voice in 9 languages**: Whisper speech-to-text detects the spoken language; the answer is written and spoken (edge-tts neural voices, gTTS fallback) in that language. Tamil, Hindi, Telugu, Kannada, Malayalam, Bengali, Marathi, Gujarati and English.
- **MCP server** (`mcp_server.py`): AI agents such as Claude Desktop can call the knowledge base as tools.
- **Interfaces**: responsive web app with live crawl progress (Server-Sent Events), a CLI, and a Settings panel that tests an API key when you save it.

---

## 🏗️ How it works

```
 INGESTION (once per college)
 URL ─► Crawler (Crawl4AI) ─► clean + strip boilerplate ─► chunk (400/50 words)
                                                              │
                         ┌────────────────────────────────────┼──────────────────────┐
                         ▼                                    ▼                      ▼
                FTS5 keyword index                    embeddings              LLM fact extraction
                                                                               (12 categories)
                         └──────────────── SQLite knowledge base ─────────────────────┘

 QUESTION (every time)
 text / voice ─► Whisper (language detection) ─► hybrid search (BM25 + vectors, RRF)
             ─► + matching facts ─► LLM with grounding rules ─► answer + sources ─► speech
```

---

## 📁 Project structure

```
app/
  config.py          settings from .env
  crawler.py         Crawl4AI crawler, cleaning, chunking, PDF reading
  importer.py        manual import of saved pages
  extractor.py       LLM fact extraction with rate-limit pacing
  storage.py         SQLite schema: colleges, pages, chunks, FTS5, facts, chats
  vector_store.py    embeddings and hybrid search (BM25 + vectors + RRF)
  rag_engine.py      prompt building, LLM providers, fact clean-up, answers
  multilingual.py    language detection and Indian-language support
  voice_service.py   Whisper speech-to-text, edge-tts / gTTS speech
  server.py          FastAPI REST + SSE endpoints
  templates/, static/  web UI
mcp_server.py        MCP server for AI agents
cli.py               command-line interface
run.py               web server launcher
Dockerfile           container image
tests/               unit tests
```

---

## 🚀 Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
crawl4ai-setup                     # installs the headless browser
cp .env.example .env               # then add at least one API key
```

Free API keys: [Groq](https://console.groq.com/) (recommended, also used for Whisper), [Gemini](https://aistudio.google.com/), [OpenRouter](https://openrouter.ai/). Keys can also be entered in the web app's Settings.

### Run the web app
```bash
python run.py
```
Open http://localhost:8000, add a college URL, wait for the crawl, then ask questions by typing or with the mic.

### CLI
```bash
python cli.py crawl https://www.tce.edu --pages 120 --depth 4
python cli.py import data/imports/ssn --url https://www.ssn.edu.in --name "SSN College of Engineering"
python cli.py extract 1            # (re)run fact extraction for college 1
python cli.py list
python cli.py ask 1 "What UG courses are offered?"
python cli.py chat 1 --voice
```

### Run with Docker
```bash
docker build -t college-assistant .
docker run -p 8000:8000 --env-file .env -v "$(pwd)/data:/app/data" college-assistant
```
- `--env-file .env` passes your API keys at run time (they are never copied into the image).
- `-v .../data:/app/data` keeps the database between runs.

---

## 🤖 MCP server (for AI agents)

[MCP (Model Context Protocol)](https://modelcontextprotocol.io/) is a standard way to give AI assistants tools. `mcp_server.py` exposes the knowledge base as four tools; the AI reads their descriptions and decides when to call them.

| Tool | What it does |
|---|---|
| `list_colleges()` | Colleges in the database with their ids |
| `get_facts(college_id, category)` | Complete fact list for one category (e.g. `ug_program`, `recruiter`) |
| `search_college(college_id, query, top_k)` | Hybrid search over crawled pages, with source URLs |
| `ask_college(college_id, question)` | Full RAG answer with sources, in the question's language |

**Use it from Claude Desktop**: add this to `claude_desktop_config.json` (Settings → Developer → Edit Config), using your own path, then restart Claude Desktop:

```json
{
  "mcpServers": {
    "college-assistant": {
      "command": "/full/path/to/project/.venv/bin/python",
      "args": ["/full/path/to/project/mcp_server.py"]
    }
  }
}
```

Then ask Claude, for example: *"Compare the UG programmes at TCE and SSN."* Claude calls `list_colleges`, then `get_facts` for each college, and compares them.

---

## 🧪 Tests
```bash
python -m unittest discover tests
```

## ⚖️ Responsible crawling
The crawler reads public pages slowly and stops when a site blocks it. It never tries to get around bot protection. For such sites, use manual import or ask the institution for access.

## License
MIT
