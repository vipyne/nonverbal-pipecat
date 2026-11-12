"""Tests for the NonverbalContextGate frame processor."""

import pytest
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterruptionFrame,
    LLMContextFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frameworks.rtvi.frames import RTVIClientMessageFrame
from pipecat.tests.utils import SleepFrame, run_test

from nonverbal_gate import (
    CUE_MESSAGE_TYPE,
    NOTE_PREFIX,
    NonverbalContextGate,
    is_yes_no_question,
)

# Short windows keep the real-time tests quick; production defaults are seconds.
ANSWER_WINDOW_S = 0.3
CHECKIN_S = 0.2


def cue(name: str, *, confidence: float = 0.9, t_ms: float = 0.0, **meta) -> RTVIClientMessageFrame:
    """One fast-lane cue message, shaped exactly as the browser sends it."""
    return RTVIClientMessageFrame(
        msg_id=f"{name}-{t_ms}",
        type=CUE_MESSAGE_TYPE,
        data={
            "type": "nonverbal",
            "cue": name,
            "t_ms": t_ms,
            "confidence": confidence,
            **({"meta": meta} if meta else {}),
        },
    )


def conversation(*, bot_text: str = "Sure, here you go.") -> LLMContext:
    return LLMContext(
        messages=[
            {"role": "system", "content": "you are a bot"},
            {"role": "user", "content": "tell me a story"},
            {"role": "assistant", "content": bot_text},
        ]
    )


def gate(context: LLMContext, **kwargs) -> NonverbalContextGate:
    kwargs.setdefault("answer_window_s", ANSWER_WINDOW_S)
    return NonverbalContextGate(context=context, **kwargs)


def notes(context: LLMContext) -> list[str]:
    return [
        m["content"]
        for m in context.messages
        if isinstance(m, dict) and str(m.get("content", "")).startswith(NOTE_PREFIX)
    ]


def user_texts(context: LLMContext) -> list[str]:
    return [m["content"] for m in context.messages if isinstance(m, dict) and m.get("role") == "user"]


# --- aggregation ---------------------------------------------------------


async def test_cues_during_bot_utterance_become_one_note():
    context = conversation()
    processor = gate(context)
    # The user aggregator has already appended the next turn by the time its
    # context frame reaches the gate.
    context.add_message({"role": "user", "content": "yes please"})

    await run_test(
        processor,
        frames_to_send=[
            BotStartedSpeakingFrame(),
            cue("nod", t_ms=100),
            cue("nod", t_ms=1300),
            cue("confused", t_ms=2100),
            BotStoppedSpeakingFrame(),
            LLMContextFrame(context),
        ],
    )

    assert len(notes(context)) == 1
    note = notes(context)[0]
    assert "nodded x2" in note
    assert "confused" in note
    # Prepended: the note sits in front of the user turn it annotates.
    messages = [m for m in context.messages if isinstance(m, dict)]
    assert messages[-1]["content"] == "yes please"
    assert messages[-2]["content"] == note


async def test_no_note_when_there_were_no_cues():
    context = conversation()
    processor = gate(context)
    context.add_message({"role": "user", "content": "go on"})

    await run_test(
        processor,
        frames_to_send=[
            BotStartedSpeakingFrame(),
            BotStoppedSpeakingFrame(),
            LLMContextFrame(context),
        ],
    )

    assert notes(context) == []


async def test_note_is_not_repeated_on_the_next_turn():
    context = conversation()
    processor = gate(context)
    context.add_message({"role": "user", "content": "ha"})

    await run_test(
        processor,
        frames_to_send=[
            BotStartedSpeakingFrame(),
            cue("amused", t_ms=500),
            BotStoppedSpeakingFrame(),
            LLMContextFrame(context),
            # Second user turn, no new cues: the note must not come back.
            LLMContextFrame(context),
        ],
    )

    assert len(notes(context)) == 1


# --- nod as an answer ----------------------------------------------------


async def test_nod_answers_a_yes_no_question():
    context = conversation(bot_text="That is a long list. Want me to continue?")
    processor = gate(context)

    down, _up = await run_test(
        processor,
        frames_to_send=[
            BotStartedSpeakingFrame(),
            BotStoppedSpeakingFrame(),
            cue("nod", t_ms=4000),
        ],
    )

    assert "(nods)" in user_texts(context)
    # A context frame is pushed so the LLM answers without the user speaking.
    assert any(isinstance(f, LLMContextFrame) for f in down)


async def test_shake_answers_a_yes_no_question_as_no():
    context = conversation(bot_text="Should I email you the summary?")
    processor = gate(context)

    down, _up = await run_test(
        processor,
        frames_to_send=[
            BotStartedSpeakingFrame(),
            BotStoppedSpeakingFrame(),
            cue("shake", t_ms=4000),
        ],
    )

    assert "(shakes head)" in user_texts(context)
    assert any(isinstance(f, LLMContextFrame) for f in down)


async def test_nod_is_not_an_answer_without_a_yes_no_question():
    context = conversation(bot_text="So that is how the bridge was built.")
    processor = gate(context)

    down, _up = await run_test(
        processor,
        frames_to_send=[
            BotStartedSpeakingFrame(),
            BotStoppedSpeakingFrame(),
            cue("nod", t_ms=4000),
        ],
    )

    assert "(nods)" not in user_texts(context)
    assert not any(isinstance(f, LLMContextFrame) for f in down)


async def test_nod_after_the_window_is_not_an_answer():
    context = conversation(bot_text="Want me to continue?")
    processor = gate(context)

    down, _up = await run_test(
        processor,
        frames_to_send=[
            BotStoppedSpeakingFrame(),
            SleepFrame(sleep=ANSWER_WINDOW_S + 0.2),
            cue("nod", t_ms=4000),
        ],
    )

    assert "(nods)" not in user_texts(context)
    assert not any(isinstance(f, LLMContextFrame) for f in down)


async def test_nod_is_not_an_answer_once_the_user_speaks():
    context = conversation(bot_text="Want me to continue?")
    processor = gate(context)

    down, _up = await run_test(
        processor,
        frames_to_send=[
            BotStoppedSpeakingFrame(),
            UserStartedSpeakingFrame(),
            cue("nod", t_ms=4000),
        ],
    )

    assert "(nods)" not in user_texts(context)
    assert not any(isinstance(f, LLMContextFrame) for f in down)


async def test_only_the_first_nod_answers():
    context = conversation(bot_text="Want me to continue?")
    processor = gate(context)

    await run_test(
        processor,
        frames_to_send=[
            BotStoppedSpeakingFrame(),
            cue("nod", t_ms=4000),
            cue("nod", t_ms=4200),
        ],
    )

    assert user_texts(context).count("(nods)") == 1


# --- no interruption -----------------------------------------------------


async def test_cues_never_interrupt_the_bot():
    context = conversation()
    processor = gate(context)

    cues = [cue("nod", t_ms=100 * i) for i in range(6)]
    down, up = await run_test(
        processor,
        frames_to_send=[BotStartedSpeakingFrame(), *cues],
    )

    assert not any(isinstance(f, InterruptionFrame) for f in down)
    assert not any(isinstance(f, InterruptionFrame) for f in up)
    # Cue frames pass through untouched.
    assert sum(isinstance(f, RTVIClientMessageFrame) for f in down) == len(cues)


# --- distraction check-in (flag) -----------------------------------------


async def test_distraction_checkin_is_off_by_default():
    context = conversation()
    processor = gate(context)

    down, _up = await run_test(
        processor,
        frames_to_send=[
            BotStartedSpeakingFrame(),
            cue("distracted", t_ms=1000, phase="start"),
            cue("distracted", t_ms=7000, phase="end", duration_ms=6000),
            BotStoppedSpeakingFrame(),
        ],
    )

    assert not any(isinstance(f, LLMContextFrame) for f in down)
    assert not any("check in" in str(m.get("content", "")) for m in context.messages if isinstance(m, dict))


async def test_distraction_checkin_fires_when_enabled():
    context = conversation()
    processor = gate(context, distraction_checkin=True, distraction_checkin_s=CHECKIN_S)

    down, _up = await run_test(
        processor,
        frames_to_send=[
            BotStartedSpeakingFrame(),
            cue("distracted", t_ms=1000, phase="start"),
            cue("distracted", t_ms=7000, phase="end", duration_ms=6000),
            BotStoppedSpeakingFrame(),
        ],
    )

    assert any(isinstance(f, LLMContextFrame) for f in down)
    assert any(
        "check in" in str(m.get("content", "")) for m in context.messages if isinstance(m, dict)
    )


async def test_short_distraction_does_not_trigger_a_checkin():
    context = conversation()
    processor = gate(context, distraction_checkin=True, distraction_checkin_s=10.0)

    down, _up = await run_test(
        processor,
        frames_to_send=[
            BotStartedSpeakingFrame(),
            cue("distracted", t_ms=1000, phase="start"),
            cue("distracted", t_ms=3000, phase="end", duration_ms=2000),
            BotStoppedSpeakingFrame(),
        ],
    )

    assert not any(isinstance(f, LLMContextFrame) for f in down)


# --- yes/no question detection -------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Want me to continue?",
        "Do you want the short version?",
        "Shall I keep going?",
        "Does that make sense?",
        "Ready?",
        "Sound good?",
        "That is the gist. Should I go on?",
        "Is that what you meant?",
    ],
)
def test_is_yes_no_question_true(text):
    assert is_yes_no_question(text)


@pytest.mark.parametrize(
    "text",
    [
        "So that is how the bridge was built.",
        "What do you think?",
        "Which one would you pick?",
        "Tell me more.",
        "How did that feel?",
        "",
    ],
)
def test_is_yes_no_question_false(text):
    assert not is_yes_no_question(text)
