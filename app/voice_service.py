# Copyright (c) 2026 Suriya C K (github.com/Suriya-CK). All rights reserved.
# Part of College AI Voice Assistant. Unauthorized copying or reuse is prohibited. See LICENSE.
import os
import re
import hashlib
import asyncio
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, AsyncIterator

from app.config import Config
from app.multilingual import SUPPORTED_LANGUAGES, get_all_voices, detect_language

logger = logging.getLogger(__name__)

# Whisper reports the spoken language by name; map the ones the app supports to our codes
WHISPER_LANGUAGES = {
    "english": "en", "tamil": "ta", "hindi": "hi", "telugu": "te", "kannada": "kn",
    "malayalam": "ml", "bengali": "bn", "marathi": "mr", "gujarati": "gu",
}
# Browser recording MIME subtype -> file extension Whisper accepts
_AUDIO_EXTENSIONS = {
    "webm": "webm", "ogg": "ogg", "mp4": "mp4", "x-m4a": "m4a", "m4a": "m4a", "aac": "m4a",
    "mpeg": "mp3", "mp3": "mp3", "wav": "wav", "x-wav": "wav", "flac": "flac",
}


# Written forms that text-to-speech reads badly ("L P A", "rupee symbol 58") -> how a person says them.
# Mostly for English answers; the Indian-language voices read their own scripts fine.
_SPOKEN_FORMS = [
    (re.compile(r"₹\s*([\d.,]+)\s*(?:Cr|crores?)\b", re.I), r"\1 crore rupees"),
    (re.compile(r"₹\s*([\d.,]+)\s*(?:L|lakhs?|LPA)\b", re.I), r"\1 lakh rupees"),
    (re.compile(r"₹\s*([\d.,]+)"), r"\1 rupees"),
    (re.compile(r"\bRs\.?\s*([\d.,]+)", re.I), r"\1 rupees"),
    (re.compile(r"\bLPA\b"), "lakh per annum"),
    (re.compile(r"\bCTC\b"), "C T C"),
    (re.compile(r"(\d)\s*Cr\b"), r"\1 crore"),
    (re.compile(r"\bB\.\s?E\.?(?=\s|$)"), "B E"),
    (re.compile(r"\bB\.\s?Tech\.?", re.I), "B Tech"),
    (re.compile(r"\bM\.\s?E\.?(?=\s|$)"), "M E"),
    (re.compile(r"\bM\.\s?Tech\.?", re.I), "M Tech"),
    (re.compile(r"\bPh\.?\s?D\.?", re.I), "P H D"),
    (re.compile(r"\be\.g\.", re.I), "for example"),
    (re.compile(r"\bi\.e\.", re.I), "that is"),
    (re.compile(r"\bvs\.?(?=\s)", re.I), "versus"),
    (re.compile(r"\s&\s"), " and "),
    # "5-10 lakh" -> "5 to 10 lakh" (only ranges with a unit, so dates like 23-11-2026 stay as they are)
    (re.compile(r"(\d)\s*[–-]\s*(\d+(?:\.\d+)?)\s*(lakh|crore|%|percent|years?|months?)", re.I), r"\1 to \2 \3"),
    (re.compile(r"(\d)\s*%"), r"\1 percent"),
]


def resolve_spoken_language(whisper_language: Optional[str], text: str) -> str:
    """Pick the answer language from Whisper's guess and the script the transcript is written in."""
    reported = WHISPER_LANGUAGES.get((whisper_language or "").strip().lower())
    script = detect_language(text)
    if reported and (script in ("en", reported) or {reported, script} <= {"hi", "mr"}):
        return reported          # e.g. Tamil written in Latin letters is still Tamil; trust Whisper on Hindi vs Marathi
    if script != "en":
        return script            # the transcript's script is unambiguous
    return reported or "en"


class VoiceService:
    """Voice Text-to-Speech and Speech processing engine with full Indian & English language support."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Config.AUDIO_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def clean_text_for_speech(self, text: str) -> str:
        """Strip markdown, URLs, citation codes, and formatting for fluid voice reading."""
        # 1. Convert markdown link syntax [text](url) -> text
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
        # 2. Remove raw URLs
        text = re.sub(r"https?://\S+", "", text)
        # 3. Remove citation brackets like [Source: Admissions] or [Page 1]
        text = re.sub(r"\[.*?\]", "", text)
        # 4. Remove markdown headers and formatting emphasis
        text = re.sub(r"[#*_~`>]", "", text)
        # 5. Remove markdown bullet points
        text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
        # 6. Say abbreviations and amounts the way a person would ("58 LPA" -> "58 lakh per annum")
        for pattern, spoken in _SPOKEN_FORMS:
            text = pattern.sub(spoken, text)
        # 7. Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > 1500:
            text = text[:1500] + "..."
        return text

    async def text_to_speech_edge(self, text: str, voice: str = "en-IN-NeerjaNeural") -> Optional[str]:
        """Convert text to speech using Microsoft Edge TTS neural voices."""
        spoken_text = self.clean_text_for_speech(text)
        if not spoken_text:
            return None

        # Hash for caching
        text_hash = hashlib.md5(f"{spoken_text}_{voice}".encode("utf-8")).hexdigest()
        file_path = self.cache_dir / f"{text_hash}.mp3"

        if file_path.exists() and file_path.stat().st_size > 0:
            return f"/api/voice/audio/{file_path.name}"

        try:
            import edge_tts
            communicate = edge_tts.Communicate(spoken_text, voice)
            await communicate.save(str(file_path))
            return f"/api/voice/audio/{file_path.name}"
        except Exception as e:
            logger.warning(f"Edge TTS synthesis error ({voice}): {e}")
            return None

    def text_to_speech_gtts(self, text: str, lang: str = "en") -> Optional[str]:
        """Fallback TTS using gTTS supporting Indian languages (ta, hi, te, kn, ml, bn, mr, gu, en)."""
        spoken_text = self.clean_text_for_speech(text)
        if not spoken_text:
            return None

        text_hash = hashlib.md5(f"{spoken_text}_gtts_{lang}".encode("utf-8")).hexdigest()
        file_path = self.cache_dir / f"{text_hash}.mp3"

        if file_path.exists() and file_path.stat().st_size > 0:
            return f"/api/voice/audio/{file_path.name}"

        try:
            from gtts import gTTS
            # Map language code to gTTS supported code
            gtts_lang = lang if lang in ["ta", "hi", "te", "kn", "ml", "bn", "mr", "gu", "en"] else "en"
            tts = gTTS(text=spoken_text, lang=gtts_lang, slow=False)
            tts.save(str(file_path))
            return f"/api/voice/audio/{file_path.name}"
        except Exception as e:
            logger.warning(f"gTTS synthesis error ({lang}): {e}")
            return None

    def cache_name(self, text: str, voice: str) -> str:
        """Cache file name for an Edge TTS clip (same scheme as text_to_speech_edge)."""
        spoken_text = self.clean_text_for_speech(text)
        return hashlib.md5(f"{spoken_text}_{voice}".encode("utf-8")).hexdigest() + ".mp3"

    async def stream_speech(self, text: str, voice: str = "en-IN-NeerjaNeural", lang: str = "en") -> AsyncIterator[bytes]:
        """Yield MP3 audio while Edge TTS is still generating it, so playback starts in about a second
        instead of after the whole clip (2-4 s). The finished clip is cached; if Edge fails before
        sending any audio, the whole gTTS clip is sent instead."""
        spoken_text = self.clean_text_for_speech(text)
        if not spoken_text:
            return
        file_path = self.cache_dir / self.cache_name(text, voice)
        if file_path.exists() and file_path.stat().st_size > 0:
            yield file_path.read_bytes()
            return

        tmp_path = file_path.with_suffix(".part")
        sent = False
        try:
            import edge_tts
            with open(tmp_path, "wb") as out:
                async for chunk in edge_tts.Communicate(spoken_text, voice).stream():
                    if chunk.get("type") == "audio" and chunk.get("data"):
                        out.write(chunk["data"])
                        sent = True
                        yield chunk["data"]
            if sent:
                tmp_path.replace(file_path)
                return
        except Exception as e:
            logger.warning(f"Edge TTS streaming error ({voice}): {e}")
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        if sent:
            return   # cut off midway: what was sent has already played
        loop = asyncio.get_event_loop()
        url = await loop.run_in_executor(None, self.text_to_speech_gtts, text, lang)
        if url:
            yield (self.cache_dir / url.rsplit("/", 1)[-1]).read_bytes()

    async def generate_speech(self, text: str, voice: str = "en-IN-NeerjaNeural", lang: str = "en") -> Optional[str]:
        """Generate speech file and return public audio URL, with Edge TTS and gTTS fallback."""
        # 1. Try edge-tts with selected voice
        audio_url = await self.text_to_speech_edge(text, voice)
        if audio_url:
            return audio_url

        # 2. Try gTTS fallback with matching language code
        loop = asyncio.get_event_loop()
        audio_url = await loop.run_in_executor(None, self.text_to_speech_gtts, text, lang)
        return audio_url

    def transcribe(self, audio: bytes, content_type: str = "audio/webm") -> Dict[str, Optional[str]]:
        """Speech-to-text that also detects which language was spoken.

        Returns {"text", "language"} where language is one of the app's codes (en, ta, hi...).
        Uses Groq Whisper, then Gemini, so it works with whichever key is configured.
        """
        mime = (content_type or "audio/webm").split(";")[0].strip().lower()
        ext = _AUDIO_EXTENSIONS.get(mime.split("/")[-1], "webm")

        if Config.GROQ_API_KEY:
            try:
                import requests
                resp = requests.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {Config.GROQ_API_KEY}"},
                    files={"file": (f"speech.{ext}", audio, mime)},
                    data={"model": Config.WHISPER_MODEL, "response_format": "verbose_json"},
                    timeout=45,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    text = (data.get("text") or "").strip()
                    return {"text": text, "language": resolve_spoken_language(data.get("language"), text)}
                logger.warning(f"Groq Whisper returned {resp.status_code}: {resp.text[:300]}")
            except Exception as e:
                logger.warning(f"Groq Whisper transcription failed: {e}")

        if Config.GEMINI_API_KEY:
            try:
                from google import genai
                from google.genai import types
                client = genai.Client(api_key=Config.GEMINI_API_KEY)
                response = client.models.generate_content(
                    model=Config.GEMINI_MODEL,
                    contents=[
                        types.Part.from_bytes(data=audio, mime_type=mime),
                        "Transcribe this voice recording exactly, in the language and script it was spoken in. "
                        "Output only the transcribed text.",
                    ],
                )
                text = (response.text or "").strip() if response else ""
                if text:
                    return {"text": text, "language": detect_language(text)}
            except Exception as e:
                logger.warning(f"Gemini transcription failed: {e}")

        return {"text": "", "language": None}

    def get_available_voices(self) -> List[Dict[str, str]]:
        """Return all available neural voices across supported Indian & English languages."""
        return get_all_voices()

    def get_voices_by_language(self) -> Dict[str, Any]:
        """Return voice hierarchy grouped by language."""
        result = {}
        for code, data in SUPPORTED_LANGUAGES.items():
            result[code] = {
                "name": data["name"],
                "native": data["native"],
                "speech_locale": data["speech_locale"],
                "default_voice": data["default_voice"],
                "voices": data["voices"]
            }
        return result
