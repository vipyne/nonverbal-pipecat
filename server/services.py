"""Single swap spot for STT / LLM / TTS and the system prompt.

Swap a provider by editing the matching factory below; `bot.py` only calls
these three functions.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Repo-root .env, not server/.env — keys are shared with the client tooling.
REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env", override=False)

# Cartesia "Savannah" — swap for any voice id from Cartesia's voice list.
CARTESIA_VOICE_ID = "78ab82d5-25be-4f7d-82b3-7ad64e5b85b2"

SYSTEM_PROMPT = """You are a warm, concise voice assistant in a live spoken \
conversation.

Speak in short, natural sentences — one or two at a time. No markdown, no \
bullet lists, no emoji; everything you write is read aloud. Ask a follow-up \
question when it keeps the conversation moving."""


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} missing — copy .env.example to .env and fill it in")
    return value


def create_stt():
    """Deepgram streaming STT."""
    from pipecat.services.deepgram.stt import DeepgramSTTService

    return DeepgramSTTService(
        api_key=_require("DEEPGRAM_API_KEY"),
        settings=DeepgramSTTService.Settings(model="nova-3-general", language="en-US"),
    )


def create_llm():
    """Anthropic Claude."""
    from pipecat.services.anthropic.llm import AnthropicLLMService

    return AnthropicLLMService(
        api_key=_require("ANTHROPIC_API_KEY"),
        settings=AnthropicLLMService.Settings(
            model="claude-sonnet-5",
            max_tokens=300,
        ),
    )


def create_tts():
    """Cartesia streaming TTS."""
    from pipecat.services.cartesia.tts import CartesiaTTSService

    return CartesiaTTSService(
        api_key=_require("CARTESIA_API_KEY"),
        settings=CartesiaTTSService.Settings(
            model="sonic-3.6",
            voice=CARTESIA_VOICE_ID,
        ),
    )
