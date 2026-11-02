import os
import re
import hashlib
import asyncio
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

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
        # 6. Normalize whitespace
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
