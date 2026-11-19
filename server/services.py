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
question when it keeps the conversation moving.

Primarily focus on asking yes or no questions.

The user is on camera and their browser watches for nonverbal cues. You will \
sometimes see a bracketed note like \
"[nonverbal during your last reply: nodded x2, looked confused at 4.1s]". \
Treat it as a hint about how your last reply landed, and let it shape what you \
say next — slow down and re-explain when they looked confused, keep going when \
they nodded along. Never narrate the cue itself ("I see you nodded") unless it \
genuinely fits the moment.

A turn that reads "(nods)" or "(shakes head)" is the user answering your \
question silently, on camera. Take it as a plain yes or no and carry on.

Always end every turn with a yes or no question. 
Start by asking the user 'Would you like to talk about the weather?'.
"""

def _flag(name: str) -> bool:
    """An env flag; anything but unset/0/false counts as on."""
    return os.getenv(name, "0").strip().lower() not in ("", "0", "false")


# Post-reply check-in after a long distraction; off unless the env flag is set.
ENABLE_DISTRACTION_CHECKIN = _flag("ENABLE_DISTRACTION_CHECKIN")
# Server-side VLM slow lane: camera video, sampled frames, scene notes.
ENABLE_VLM = _flag("ENABLE_VLM")


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} missing — copy env.example to .env and fill it in")
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


def create_vision():
    """Moondream VLM for the slow lane; imported only when ENABLE_VLM is on."""
    try:
        from pipecat.services.moondream.vision import MoondreamService
    except ImportError as e:
        raise RuntimeError(
            "ENABLE_VLM=1 needs the moondream extra — run "
            '`cd server && uv add "pipecat-ai[moondream]"`'
        ) from e

    return MoondreamService()
