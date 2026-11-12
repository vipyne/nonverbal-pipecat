"""Optional slow lane: scene-level notes from a server-side VLM.

Off unless `ENABLE_VLM=1`. When on, the transport receives the camera track at
one frame per second and `SceneSampler` passes one frame every few seconds to a
vision service, turning its description into a low-confidence note on the
`NonverbalContextGate`'s rolling state. The query runs in its own task, so a
slow model never holds up audio, and the frames themselves never travel past
the sampler.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import services
from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InputImageRawFrame,
    UserImageRawFrame,
    VisionTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from nonverbal_gate import NonverbalContextGate

# Camera framerate asked of the transport; the sampler thins it down further.
VLM_FRAMERATE = 1
# Seconds between frames handed to the VLM.
SAMPLE_INTERVAL_S = 4.0
# Longest scene note kept; the model sometimes answers in paragraphs.
MAX_NOTE_CHARS = 140

SCENE_PROMPT = (
    "In one short sentence, describe what the person on camera is doing and "
    "whether anyone else is present."
)


class SceneSampler(FrameProcessor):
    """Samples camera frames into VLM scene notes, out of the audio path.

    Place right after `transport.input()`. Image frames stop here; everything
    else passes through untouched.
    """

    def __init__(
        self,
        *,
        vision,
        note_sink: Callable[[str], None],
        interval_s: float = SAMPLE_INTERVAL_S,
        prompt: str = SCENE_PROMPT,
        time_fn: Callable[[], float] = time.monotonic,
        **kwargs,
    ):
        """Initialize the sampler.

        Args:
            vision: Vision service exposing `run_vision(UserImageRawFrame)`.
            note_sink: Called with each scene description.
            interval_s: Seconds between sampled frames.
            prompt: Question asked about each sampled frame.
            time_fn: Monotonic clock, injectable for tests.
        """
        super().__init__(**kwargs)
        self._vision = vision
        self._note_sink = note_sink
        self._interval_s = interval_s
        self._prompt = prompt
        self._now = time_fn

        self._last_sample_at: float | None = None
        self._query = None
        self._last_note = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Swallow camera frames, sampling one per interval; pass the rest on."""
        await super().process_frame(frame, direction)

        if isinstance(frame, InputImageRawFrame):
            self._maybe_sample(frame)
            return

        if isinstance(frame, (EndFrame, CancelFrame)):
            await self._drop_query()

        await self.push_frame(frame, direction)

    async def cleanup(self):
        """Drop an in-flight query before the processor goes away."""
        await self._drop_query()
        await super().cleanup()

    def _maybe_sample(self, frame: InputImageRawFrame):
        """Start one VLM query if the interval has passed and none is running."""
        now = self._now()
        if self._query is not None and not self._query.done():
            return
        if self._last_sample_at is not None and now - self._last_sample_at < self._interval_s:
            return
        self._last_sample_at = now
        self._query = self.create_task(self._describe(frame))

    async def _describe(self, frame: InputImageRawFrame):
        """Ask the VLM about one frame and hand the answer to the note sink."""
        request = UserImageRawFrame(
            image=frame.image,
            size=frame.size,
            format=frame.format,
            text=self._prompt,
            append_to_context=False,
            run_llm=False,
        )
        parts: list[str] = []
        try:
            async for out in self._vision.run_vision(request):
                if isinstance(out, ErrorFrame):
                    logger.warning(f"{self}: vision error: {out.error}")
                    return
                if isinstance(out, VisionTextFrame):
                    parts.append(out.text)
        except Exception as e:
            logger.warning(f"{self}: scene query failed: {e.__class__.__name__} ({e})")
            return

        note = " ".join("".join(parts).split())[:MAX_NOTE_CHARS]
        if not note or note == self._last_note:
            return
        self._last_note = note
        logger.info(f"{self}: scene: {note}")
        self._note_sink(note)

    async def _drop_query(self):
        if self._query is not None and not self._query.done():
            await self.cancel_task(self._query)
        self._query = None


def create_scene_sampler(
    gate: NonverbalContextGate,
    *,
    enabled: bool | None = None,
    vision_factory: Callable[[], object] | None = None,
) -> SceneSampler | None:
    """The slow lane processor when the flag is on, else None.

    With the flag off nothing here touches the VLM service, so its heavy
    dependency stays unimported.
    """
    if enabled is None:
        enabled = services.ENABLE_VLM
    if not enabled:
        return None
    factory = vision_factory or services.create_vision
    return SceneSampler(vision=factory(), note_sink=gate.add_scene_note)
