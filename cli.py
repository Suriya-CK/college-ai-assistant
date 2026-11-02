import sys
import os
from pathlib import Path

# Auto-activate .venv if run with system python
venv_dir = Path(__file__).resolve().parent / ".venv"
if venv_dir.exists() and sys.prefix != str(venv_dir):
    venv_python = venv_dir / "bin" / "python"
    os.execv(str(venv_python), [str(venv_python)] + sys.argv)

import argparse
import sys
import asyncio
from app.config import Config
from app.storage import Database
from app.crawler import CollegeCrawler
from app.vector_store import VectorSearchEngine
from app.rag_engine import RAGEngine
from app.extractor import make_extractor
from app.importer import PageImporter
from app.voice_service import VoiceService

def cmd_crawl(args):
    print(f"\n🕷️  Starting crawl for: {args.url}")
    print(f"Max Pages: {args.pages} | Max Depth: {args.depth}\n")

    db = Database()
    crawler = CollegeCrawler(db, extractor=make_extractor(db, RAGEngine(db)))

    def on_progress(event_type, data):
        if event_type == "status":
            print(f"  ... {data['message']}")
        elif event_type == "page_success":
            print(f"  [✓] ({data['scraped_count']}/{data['max_pages']}) {data['title'][:50]} ({data['chunks']} chunks)")
        elif event_type == "page_error":
            print(f"  [✗] Error {data['url']}: {data['error']}")
        elif event_type == "completed":
            print(f"\n🎉 Crawl finished! Scraped {data['pages_scraped']} pages, created {data['total_chunks']} chunks, "
                  f"extracted {data.get('facts', 0)} facts.")
            if data.get("warning"):
                print(f"⚠️  {data['warning']}")

    res = crawler.crawl(args.url, max_pages=args.pages, max_depth=args.depth, progress_callback=on_progress)
    if res.get("success"):
        college_id = res["college_id"]
        print(f"College ID: {college_id} ({res['college_name']})")
        print("Generating embeddings...")
        v_engine = VectorSearchEngine(db)
        indexed = v_engine.index_pending_chunks(college_id)
        print(f"Indexed {indexed} embeddings.\n")
    else:
        print(f"Crawl failed: {res.get('error')}")

def cmd_import(args):
    print(f"\n📥 Importing saved pages from: {args.folder}")
    db = Database()
    try:
        result = PageImporter(db).import_folder(
            Path(args.folder), args.url, college_name=args.name, replace=args.replace,
            progress=lambda filename, title: print(f"  [✓] {filename}  →  {title[:60]}"),
        )
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    for filename, reason in result["skipped"]:
        print(f"  [–] {filename}: skipped ({reason})")
    college_id = result["college_id"]
    print(f"\nImported {len(result['imported'])} files into '{result['college_name']}' (ID {college_id}), "
          f"{result['total_chunks']} chunks.")

    print("Indexing for search...")
    VectorSearchEngine(db).index_pending_chunks(college_id)

    if args.no_extract:
        print(f"Skipped fact extraction. Run 'python3 cli.py extract {college_id}' when ready.")
        return
    extractor = make_extractor(db, RAGEngine(db))
    if extractor is None:
        print(f"No AI key set, so no fact lists yet. Add one, then run 'python3 cli.py extract {college_id}'.")
        return
    print("Extracting courses, recruiters, fees and other facts...")
    added = extractor.extract_college(college_id, lambda i, n: print(f"  page {i}/{n}", end="\r"))
    print(f"\nAdded {added} facts. Totals: {db.get_fact_counts(college_id)}")
    if extractor.stopped_reason:
        print(f"⚠️  {extractor.stopped_reason}")

def cmd_extract(args):
    db = Database()
    extractor = make_extractor(db, RAGEngine(db))
    if extractor is None:
        print("Fact extraction needs an LLM API key (GEMINI/GROQ/OPENAI) and EXTRACT_FACTS=1.")
        return
    added = extractor.extract_college(args.college_id, lambda i, n: print(f"  page {i}/{n}", end="\r"))
    print(f"\nAdded {added} facts. Totals: {db.get_fact_counts(args.college_id)}")
    if extractor.stopped_reason:
        print(f"⚠️  {extractor.stopped_reason}")

def cmd_list(args):
    db = Database()
    colleges = db.list_colleges()
    if not colleges:
        print("No colleges indexed yet. Use 'python cli.py crawl <URL>' to index one.")
        return

    print("\n--- Indexed Colleges ---")
    for col in colleges:
        print(f"[{col['id']}] {col['name'] or col['domain']} - {col['url']}")
        print(f"    Pages: {col['page_count']} | Chunks: {col['chunk_count']} | Status: {col['status']}")
    print()

def cmd_delete(args):
    db = Database()
    col = db.get_college(args.college_id)
    if not col:
        print(f"College with ID {args.college_id} not found.")
        return
    db.delete_college(args.college_id)
    print(f"Deleted college ID {args.college_id} ({col.get('name') or col.get('domain')}).")

def cmd_ask(args):
    db = Database()
    rag = RAGEngine(db)
    print(f"\n🤔 Question: {args.question}")
    print(f"Searching college knowledge base (ID: {args.college_id})...\n")

    result = rag.answer_question(args.college_id, args.question)
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print("=" * 60)
    print(result["content"])
    print("=" * 60)

    if result.get("citations"):
        print("\nSources:")
        for c in result["citations"]:
            print(f" - {c['title']}: {c['url']}")

    if args.voice and result.get("spoken_text"):
        print("\n🔊 Playing voice response...")
        voice_service = VoiceService()
        async def play():
            audio_url = await voice_service.generate_speech(result["spoken_text"])
            if audio_url:
                filename = audio_url.split("/")[-1]
                path = Config.AUDIO_CACHE_DIR / filename
                # Use macOS afplay to play directly in terminal
                import subprocess
                subprocess.run(["afplay", str(path)])
        asyncio.run(play())
    print()

def cmd_interactive(args):
    db = Database()
    college = db.get_college(args.college_id)
    if not college:
        print(f"College ID {args.college_id} not found.")
        return

    rag = RAGEngine(db)
    voice_service = VoiceService()
    conv_id = f"cli-{args.college_id}"

    print(f"\n🎓 Interactive Assistant for: {college.get('name') or college.get('domain')}")
    print("Type your questions below. Type 'exit' or 'quit' to end.\n")

    while True:
        try:
            query = input("You > ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                break

            result = rag.answer_question(args.college_id, query, conversation_id=conv_id)
            print("\nAssistant > " + result["content"] + "\n")

            if args.voice and result.get("spoken_text"):
                async def play():
                    audio_url = await voice_service.generate_speech(result["spoken_text"])
                    if audio_url:
                        filename = audio_url.split("/")[-1]
                        path = Config.AUDIO_CACHE_DIR / filename
                        import subprocess
                        subprocess.run(["afplay", str(path)])
                asyncio.run(play())

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

def main():
    parser = argparse.ArgumentParser(description="College AI Voice Assistant CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Crawl
    crawl_p = subparsers.add_parser("crawl", help="Crawl and index a college website")
    crawl_p.add_argument("url", help="College website URL (e.g. https://www.stanford.edu)")
    crawl_p.add_argument("--pages", type=int, default=Config.DEFAULT_MAX_PAGES, help="Max pages to crawl")
    crawl_p.add_argument("--depth", type=int, default=Config.DEFAULT_MAX_DEPTH, help="Max crawl depth")

    # Import pages saved by hand (for sites that block crawlers)
    imp_p = subparsers.add_parser("import", help="Import pages you saved from your browser (HTML, MHTML, PDF, TXT)")
    imp_p.add_argument("folder", help="Folder containing the saved files")
    imp_p.add_argument("--url", required=True, help="The college's website address, e.g. https://www.ssn.edu.in")
    imp_p.add_argument("--name", help='College name, e.g. "SSN College of Engineering"')
    imp_p.add_argument("--replace", action="store_true", help="Remove this college's existing pages first")
    imp_p.add_argument("--no-extract", action="store_true", help="Skip AI fact extraction for now")

    # Extract facts for an already-crawled college
    ext_p = subparsers.add_parser("extract", help="(Re)extract structured facts from a crawled college")
    ext_p.add_argument("college_id", type=int, help="College ID from 'list'")

    # List
    subparsers.add_parser("list", help="List all indexed colleges")

    # Ask
    ask_p = subparsers.add_parser("ask", help="Ask a single question")
    ask_p.add_argument("college_id", type=int, help="College ID from 'list'")
    ask_p.add_argument("question", help="Question string")
    ask_p.add_argument("--voice", action="store_true", help="Play audio response")

    # Interactive Chat
    chat_p = subparsers.add_parser("chat", help="Start conversational interactive chat")
    chat_p.add_argument("college_id", type=int, help="College ID from 'list'")
    chat_p.add_argument("--voice", action="store_true", help="Play voice aloud")

    # Delete
    del_p = subparsers.add_parser("delete", help="Delete a college from the database")
    del_p.add_argument("college_id", type=int, help="College ID from 'list'")

    args = parser.parse_args()

    if args.command == "crawl":
        cmd_crawl(args)
    elif args.command == "extract":
        cmd_extract(args)
    elif args.command == "import":
        cmd_import(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "delete":
        cmd_delete(args)
    elif args.command == "ask":
        cmd_ask(args)
    elif args.command == "chat":
        cmd_interactive(args)

if __name__ == "__main__":
    main()
