"""Tests for the optional VLM slow lane (ENABLE_VLM)."""

import asyncio
import importlib.util
import sys
from collections.abc import AsyncGenerator

import pytest
import services
from pipecat.frames.frames import (
    Frame,
    InputImageRawFrame,
    LLMContextFrame,
    TextFrame,
    UserImageRawFrame,
    VisionTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.tests.utils import SleepFrame, run_test

from bot import transport_params
from nonverbal_gate import NOTE_PREFIX, NonverbalContextGate
from vlm_lane import SCENE_PROMPT, VLM_FRAMERATE, SceneSampler, create_scene_sampler

PIXELS = b"\x10\x20\x30"
# Long enough for the sampler's task to finish between queued frames.
SETTLE = 0.05

HAS_MOONDREAM = importlib.util.find_spec("torch") is not None


def image() -> InputImageRawFrame:
    """One decoded camera frame, as the transport pushes it downstream."""
    return InputImageRawFrame(image=PIXELS, size=(1, 1), format="RGB")


class FakeVision:
    """Stand-in for MoondreamService: records requests, yields canned text."""

    def __init__(self, *texts: str, blocked: asyncio.Event | None = None, error: str = ""):
        self.requests: list[UserImageRawFrame] = []
        self._texts = list(texts) or ["the person is looking at a phone"]
        self._blocked = blocked
        self._error = error

    async def run_vision(self, frame: UserImageRawFrame) -> AsyncGenerator[Frame, None]:
        self.requests.append(frame)
        if self._error:
            raise RuntimeError(self._error)
        if self._blocked is not None:
            await self._blocked.wait()
        yield VisionTextFrame(text=self._texts[min(len(self.requests) - 1, len(self._texts) - 1)])


def sampler(vision: FakeVision, note_sink, *, times: list[float], interval_s: float = 4.0):
    """A sampler on a scripted clock: one timestamp per image frame."""
    clock = iter(times)
    return SceneSampler(
        vision=vision,
        note_sink=note_sink,
        interval_s=interval_s,
        time_fn=lambda: next(clock),
    )


def frames(count: int) -> list[Frame]:
    """`count` image frames, each followed by a wait so its task can finish."""
    out: list[Frame] = []
    for _ in range(count):
        out += [image(), SleepFrame(SETTLE)]
    return out


def gate(context: LLMContext) -> NonverbalContextGate:
    return NonverbalContextGate(context=context)


def conversation() -> LLMContext:
    return LLMContext(
        messages=[
            {"role": "system", "content": "you are a bot"},
            {"role": "assistant", "content": "Here you go."},
            {"role": "user", "content": "thanks"},
        ]
    )


def notes(context: LLMContext) -> list[str]:
    return [
        m["content"]
        for m in context.messages
        if isinstance(m, dict) and str(m.get("content", "")).startswith(NOTE_PREFIX)
    ]


# --- sampling ------------------------------------------------------------


async def test_one_frame_is_sampled_per_interval():
    vision = FakeVision()
    collected: list[str] = []
    processor = sampler(vision, collected.append, times=[0.0, 1.0, 2.0, 4.0, 5.0, 8.0])

    await run_test(processor, frames_to_send=frames(6))

    # Sampled at t=0, t=4 and t=8; the frames between fall inside the interval.
    assert len(vision.requests) == 3


async def test_video_frames_never_reach_the_rest_of_the_pipeline():
    vision = FakeVision()
    processor = sampler(vision, lambda _note: None, times=[0.0])

    down, _up = await run_test(
        processor,
        frames_to_send=[image(), TextFrame("still talking"), SleepFrame(SETTLE)],
    )

    assert not any(isinstance(f, InputImageRawFrame) for f in down)
    assert [f.text for f in down if isinstance(f, TextFrame)] == ["still talking"]


async def test_the_sampled_frame_carries_the_scene_prompt():
    vision = FakeVision()
    processor = sampler(vision, lambda _note: None, times=[0.0])

    await run_test(processor, frames_to_send=frames(1))

    request = vision.requests[0]
    assert request.text == SCENE_PROMPT
    assert request.image == PIXELS
    assert request.size == (1, 1)
    # The slow lane only produces notes; the image itself stays out of context.
    assert request.append_to_context is False
    assert request.run_llm is False


async def test_sampling_is_skipped_while_a_description_is_in_flight():
    blocked = asyncio.Event()
    vision = FakeVision(blocked=blocked)
    processor = sampler(vision, lambda _note: None, times=[0.0, 5.0])

    await run_test(processor, frames_to_send=frames(2))

    # The second frame is due by the clock but the first query has not returned.
    assert len(vision.requests) == 1
    blocked.set()


async def test_a_vision_failure_leaves_no_note_and_keeps_frames_flowing():
    vision = FakeVision(error="model exploded")
    collected: list[str] = []
    processor = sampler(vision, collected.append, times=[0.0])

    down, _up = await run_test(
        processor,
        frames_to_send=[image(), TextFrame("still talking"), SleepFrame(SETTLE)],
    )

    assert collected == []
    assert [f.text for f in down if isinstance(f, TextFrame)] == ["still talking"]


# --- scene notes in the gate ---------------------------------------------


async def test_a_scene_note_becomes_a_low_confidence_context_note():
    vision = FakeVision("the person is looking at a phone")
    context = conversation()
    processor = gate(context)
    await run_test(
        sampler(vision, processor.add_scene_note, times=[0.0]),
        frames_to_send=frames(1),
    )

    await run_test(processor, frames_to_send=[LLMContextFrame(context)])

    assert len(notes(context)) == 1
    note = notes(context)[0]
    assert "looking at a phone" in note
    assert "low confidence" in note


async def test_repeated_identical_scene_text_is_noted_once():
    vision = FakeVision("another person is present", "another person is present")
    context = conversation()
    processor = gate(context)
    await run_test(
        sampler(vision, processor.add_scene_note, times=[0.0, 4.0]),
        frames_to_send=frames(2),
    )

    await run_test(processor, frames_to_send=[LLMContextFrame(context)])

    assert notes(context)[0].count("another person is present") == 1


async def test_no_scene_note_means_no_context_note():
    context = conversation()
    processor = gate(context)

    await run_test(processor, frames_to_send=[LLMContextFrame(context)])

    assert notes(context) == []


# --- the flag ------------------------------------------------------------


def test_flag_off_builds_no_slow_lane(monkeypatch):
    monkeypatch.setattr(services, "ENABLE_VLM", False)
    context = conversation()

    assert create_scene_sampler(gate(context)) is None
    assert "pipecat.services.moondream.vision" not in sys.modules
    assert "torch" not in sys.modules


def test_flag_on_builds_a_sampler_from_the_vision_factory(monkeypatch):
    monkeypatch.setattr(services, "ENABLE_VLM", True)
    vision = FakeVision()

    processor = create_scene_sampler(gate(conversation()), vision_factory=lambda: vision)

    assert isinstance(processor, SceneSampler)


def test_flag_off_builds_an_audio_only_transport():
    params = transport_params(enable_vlm=False)

    assert params.audio_in_enabled is True
    assert params.video_in_enabled is False
    assert params.video_in_sources == {}


def test_flag_on_captures_the_camera_slowly():
    params = transport_params(enable_vlm=True)

    assert params.video_in_enabled is True
    assert params.video_in_sources["camera"].framerate == VLM_FRAMERATE


@pytest.mark.skipif(HAS_MOONDREAM, reason="moondream extra is installed")
def test_create_vision_names_the_missing_extra():
    with pytest.raises(RuntimeError, match="moondream"):
        services.create_vision()
