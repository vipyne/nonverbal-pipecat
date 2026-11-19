"""Folds browser-detected nonverbal cues into the LLM context.

The browser's fast lane sends one small JSON message per cue over the RTVI
data channel. This processor sits between the user context aggregator and the
LLM: cues collected during the bot's reply become a single system note in
front of the next user turn, and a silent nod right after a yes/no question
answers it. Cues never push interruption frames and never cancel TTS.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    LLMContextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext, standard_message_text
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi.frames import RTVIClientMessageFrame

# RTVI client-message type the browser sends each cue under.
CUE_MESSAGE_TYPE = "nonverbal"
# Note opener; the gate finds its own notes by this prefix.
NOTE_PREFIX = "[nonverbal during your last reply: "
# How long after the bot stops speaking a nod still counts as the answer.
ANSWER_WINDOW_S = 5.0
# How long before the bot stops speaking a nod already counts as the answer.
EARLY_ANSWER_WINDOW_S = 1.5
# Distraction span that earns a check-in, when the flag is on.
DISTRACTION_CHECKIN_S = 6.0

# Past-tense phrasing per cue, read by the LLM inside the note.
VERBS = {
    "nod": "nodded",
    "shake": "shook head",
    "distracted": "looked away",
    "confused": "looked confused",
    "amused": "smiled",
    "yawn": "yawned",
}
# Phrasing for a cue whose span has not closed yet.
ONGOING = {"distracted": "still looking away"}
# Slow-lane scene descriptions are hints, not detections.
SCENE_SUFFIX = " (low confidence)"
# Scene notes carried into one context note; only the most recent matter.
MAX_SCENE_NOTES = 2
# Synthetic user turn a silent cue stands in for.
ANSWERS = {"nod": "(nods)", "shake": "(shakes head)"}

# First word of a question answerable with yes or no.
YES_NO_LEADS = frozenset(
    """am are can could did do does had has have is may might must ready right shall
    should sound sounds want wanna was were will would""".split()
)
# Clause boundaries inside the question sentence.
_CLAUSE_SPLIT = re.compile(r"[—–,;:]|\s-\s")
# Connectives skipped before the lead-word check.
_CLAUSE_OPENERS = frozenset("or and but so then well now".split())


def is_yes_no_question(text: str) -> bool:
    """Whether `text` ends in a question a nod can answer."""
    question = _last_question(text)
    if not question:
        return False
    for clause in _CLAUSE_SPLIT.split(question):
        words = re.findall(r"[a-z']+", clause.lower())
        while words and words[0] in _CLAUSE_OPENERS:
            words.pop(0)
        if words and words[0] in YES_NO_LEADS:
            return True
    return False


def _last_question(text: str) -> str:
    """Final sentence of `text`, or `""` unless it ends in a question mark."""
    stripped = text.strip()
    if not stripped.endswith("?"):
        return ""
    start = max(stripped.rfind(c, 0, len(stripped) - 1) for c in ".!?")
    return stripped[start + 1 :].strip()


@dataclass
class _Cue:
    """One cue as the gate remembers it, timed against the bot's utterance."""

    name: str
    offset_s: float
    phase: str | None = None
    duration_ms: float | None = None


@dataclass
class _EarlyAnswer:
    """Nod/shake that fired while the bot was still speaking its question."""

    name: str
    at: float
    cue: _Cue


class NonverbalContextGate(FrameProcessor):
    """Turns fast-lane cue messages into LLM context, never into interruptions.

    Place between the user context aggregator and the LLM service so both the
    cue messages (pushed downstream by the RTVI processor at the head of the
    pipeline) and the bot-speaking frames (pushed upstream by the output
    transport) pass through it.
    """

    def __init__(
        self,
        *,
        context: LLMContext,
        answer_window_s: float = ANSWER_WINDOW_S,
        early_answer_window_s: float = EARLY_ANSWER_WINDOW_S,
        distraction_checkin: bool = False,
        distraction_checkin_s: float = DISTRACTION_CHECKIN_S,
        time_fn: Callable[[], float] = time.monotonic,
        **kwargs,
    ):
        """Initialize the gate.

        Args:
            context: The pipeline's LLM context, shared with the aggregators.
            answer_window_s: Window after the bot stops speaking in which a
                nod or shake answers its yes/no question.
            early_answer_window_s: Window before the bot stops speaking in
                which a nod or shake already counts as that answer.
            distraction_checkin: Enable the post-reply check-in after a long
                distraction.
            distraction_checkin_s: Distraction span that earns a check-in.
            time_fn: Monotonic clock, injectable for tests.
        """
        super().__init__(**kwargs)
        self._context = context
        self._answer_window_s = answer_window_s
        self._early_answer_window_s = early_answer_window_s
        self._distraction_checkin = distraction_checkin
        self._distraction_checkin_s = distraction_checkin_s
        self._now = time_fn

        self._cues: list[_Cue] = []
        self._scenes: list[str] = []
        self._early_answer: _EarlyAnswer | None = None
        self._bot_speaking = False
        self._user_speaking = False
        self._utterance_start: float | None = None
        self._bot_stopped_at: float | None = None
        # Bot's last utterance ended in a yes/no question still unanswered.
        self._answerable = False
        # A distraction span that opened while the bot was speaking.
        self._distraction_open = False
        # Seconds of distraction owed a check-in once the bot finishes.
        self._checkin_owed: float | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Observe cue messages and turn state; mutate context, pass frames on."""
        await super().process_frame(frame, direction)

        if isinstance(frame, RTVIClientMessageFrame) and frame.type == CUE_MESSAGE_TYPE:
            await self.push_frame(frame, direction)
            await self._handle_cue(frame.data)
            return

        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
            self._utterance_start = self._now()
            self._cues.clear()
            self._early_answer = None
            self._distraction_open = False
            self._checkin_owed = None
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            self._bot_stopped_at = self._now()
            self._answerable = is_yes_no_question(self._last_assistant_text())
            if self._answerable:
                logger.debug(f"{self}: yes/no question armed for {self._answer_window_s:.1f}s")
        elif isinstance(frame, UserStartedSpeakingFrame):
            # VAD alone only pauses the window; a committed user turn
            # (LLMContextFrame) is what consumes the question.
            self._user_speaking = True
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._user_speaking = False
        elif isinstance(frame, LLMContextFrame):
            self._answerable = False
            self._apply_note(frame.context)

        await self.push_frame(frame, direction)

        if isinstance(frame, BotStoppedSpeakingFrame):
            await self._maybe_answer_early()
            await self._maybe_check_in()

    def add_scene_note(self, text: str) -> None:
        """Record one VLM scene description for the next context note."""
        text = (text or "").strip()
        if not text or text in self._scenes:
            return
        self._scenes.append(text)
        del self._scenes[:-MAX_SCENE_NOTES]

    async def _handle_cue(self, data: Any):
        """Record one cue, after letting it answer a pending question."""
        if not isinstance(data, dict):
            return
        name = data.get("cue")
        if name not in VERBS:
            logger.debug(f"{self}: ignoring unknown nonverbal cue {name!r}")
            return

        meta = data.get("meta") or {}
        cue = _Cue(
            name=name,
            offset_s=self._offset(),
            phase=meta.get("phase"),
            duration_ms=meta.get("duration_ms"),
        )
        if name == "distracted":
            self._track_distraction(cue)

        if name in ANSWERS and self._bot_speaking:
            # Users often start nodding before the question ends; hold the
            # cue so it can answer once the bot stops.
            self._early_answer = _EarlyAnswer(name=name, at=self._now(), cue=cue)
            logger.info(f"{self}: {name} during bot speech — held as possible early answer")
        elif await self._maybe_answer(name):
            return

        self._cues.append(cue)
        await self._maybe_check_in()

    async def _maybe_answer(self, name: str) -> bool:
        """Answer a pending yes/no question with a synthetic user turn."""
        if name not in ANSWERS:
            return False
        blocker = self._answer_blocker()
        if blocker:
            logger.info(f"{self}: {name} not taken as answer — {blocker}")
            return False

        self._answerable = False
        self._apply_note(self._context)
        answer = ANSWERS[name]
        self._context.add_message({"role": "user", "content": answer})
        logger.info(f"{self}: silent answer {answer}")
        await self.push_frame(LLMContextFrame(self._context))
        return True

    async def _maybe_answer_early(self):
        """Let a nod/shake from the tail of the bot's question answer it."""
        early = self._early_answer
        self._early_answer = None
        if early is None or self._bot_stopped_at is None:
            return
        age = self._bot_stopped_at - early.at
        if age > self._early_answer_window_s:
            if self._answerable:
                logger.info(
                    f"{self}: {early.name} not taken as answer — fired {age:.1f}s "
                    f"before the question ended (> {self._early_answer_window_s:.1f}s)"
                )
            return
        if early.cue in self._cues:
            self._cues.remove(early.cue)
        if not await self._maybe_answer(early.name):
            self._cues.append(early.cue)

    def _answer_blocker(self) -> str | None:
        """Why a nod/shake cannot answer right now, or None."""
        if not self._answerable:
            return "no unanswered yes/no question"
        if self._bot_speaking:
            return "bot still speaking"
        if self._user_speaking:
            return "user speaking"
        if self._bot_stopped_at is None:
            return "bot has not finished a reply yet"
        elapsed = self._now() - self._bot_stopped_at
        if elapsed > self._answer_window_s:
            return f"answer window expired ({elapsed:.1f}s > {self._answer_window_s:.1f}s)"
        return None

    async def _maybe_check_in(self):
        """Ask for a check-in once the bot is done and the user is back."""
        duration_s = self._checkin_owed
        if duration_s is None or self._bot_speaking or self._user_speaking:
            return
        self._checkin_owed = None
        self._context.add_message(
            {
                "role": "system",
                "content": (
                    f"[the user looked away for {duration_s:.1f}s during your last reply — "
                    "check in briefly with one short question, then wait]"
                ),
            }
        )
        logger.info(f"{self}: distraction check-in after {duration_s:.1f}s away")
        await self.push_frame(LLMContextFrame(self._context))

    def _track_distraction(self, cue: _Cue):
        """Owe a check-in for a long distraction that began during the reply."""
        if cue.phase == "start":
            self._distraction_open = self._bot_speaking
            return
        if cue.phase != "end" or not self._distraction_open:
            return
        self._distraction_open = False
        duration_s = (cue.duration_ms or 0) / 1000
        if self._distraction_checkin and duration_s >= self._distraction_checkin_s:
            self._checkin_owed = duration_s

    def _apply_note(self, context: LLMContext):
        """Prepend one compact note in front of the user turn, then forget the cues."""
        summary = self._summarize()
        self._cues.clear()
        self._scenes.clear()
        if not summary:
            return

        note = {"role": "system", "content": f"{NOTE_PREFIX}{summary}]"}
        messages = context.messages
        if messages and isinstance(messages[-1], dict) and messages[-1].get("role") == "user":
            context.set_messages([*messages[:-1], note, messages[-1]])
        else:
            context.add_message(note)
        logger.info(f"{self}: {note['content']}")

    def _summarize(self) -> str:
        """One line of cue phrases, in the order the cues first appeared."""
        ended = {cue.name for cue in self._cues if cue.phase == "end"}
        groups: dict[str, list[_Cue]] = {}
        for cue in self._cues:
            if cue.phase == "start" and cue.name in ended:
                continue
            groups.setdefault(cue.name, []).append(cue)

        parts = []
        for name, cues in groups.items():
            verb = VERBS[name]
            if len(cues) > 1:
                parts.append(f"{verb} x{len(cues)}")
                continue
            cue = cues[0]
            if cue.duration_ms:
                parts.append(f"{verb} for {cue.duration_ms / 1000:.1f}s")
            elif cue.phase == "start":
                parts.append(ONGOING.get(name, verb))
            else:
                parts.append(f"{verb} at {cue.offset_s:.1f}s")

        parts += [f"{scene}{SCENE_SUFFIX}" for scene in self._scenes]
        return ", ".join(parts)

    def _offset(self) -> float:
        """Seconds into the bot's current utterance."""
        if self._utterance_start is None:
            return 0.0
        return max(0.0, self._now() - self._utterance_start)

    def _last_assistant_text(self) -> str:
        for message in reversed(self._context.messages):
            if isinstance(message, dict) and message.get("role") == "assistant":
                return standard_message_text(message)
        return ""
