#!/usr/bin/env python3
"""Launcher for College AI Voice Assistant Web Application."""

import sys
import os
from pathlib import Path

# Auto-activate .venv if run with system python
venv_dir = Path(__file__).resolve().parent / ".venv"
if venv_dir.exists() and sys.prefix != str(venv_dir):
    venv_python = venv_dir / "bin" / "python"
    os.execv(str(venv_python), [str(venv_python)] + sys.argv)

import uvicorn
from app.config import Config

if __name__ == "__main__":
    print(f"🚀 Starting College AI Voice Assistant on http://{Config.HOST}:{Config.PORT}")
    uvicorn.run("app.server:app", host=Config.HOST, port=Config.PORT, reload=True)
