"""Nonverbal Pipecat bot — voice pipeline over SmallWebRTC.

Run with `uv run bot.py`; the dev runner serves the WebRTC signalling endpoint
on http://localhost:7860 and the Vite client at :5173 connects to it.
"""

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.runner.types import RunnerArguments, SmallWebRTCRunnerArguments
from pipecat.transports.base_transport import TransportParams, VideoInSourceParams
from pipecat.transports.smallwebrtc.transport import CAM_VIDEO_SOURCE, SmallWebRTCTransport
from pipecat.workers.runner import WorkerRunner

from nonverbal_gate import CUE_MESSAGE_TYPE, NonverbalContextGate
from services import (
    ENABLE_DISTRACTION_CHECKIN,
    ENABLE_VLM,
    SYSTEM_PROMPT,
    create_llm,
    create_stt,
    create_tts,
)
from vlm_lane import VLM_FRAMERATE, create_scene_sampler


def transport_params(*, enable_vlm: bool = ENABLE_VLM) -> TransportParams:
    """Audio always; the camera track only on the VLM slow lane."""
    return TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_in_enabled=enable_vlm,
        video_in_sources=(
            {CAM_VIDEO_SOURCE: VideoInSourceParams(framerate=VLM_FRAMERATE)}
            if enable_vlm
            else {}
        ),
    )


def build_transport(runner_args: SmallWebRTCRunnerArguments) -> SmallWebRTCTransport:
    """SmallWebRTC transport for one session."""
    return SmallWebRTCTransport(
        webrtc_connection=runner_args.webrtc_connection,
        params=transport_params(),
    )


async def bot(runner_args: RunnerArguments):
    """Entry point the Pipecat dev runner calls per WebRTC session."""
    if not isinstance(runner_args, SmallWebRTCRunnerArguments):
        raise RuntimeError(f"unsupported transport args: {type(runner_args).__name__}")

    transport = build_transport(runner_args)

    context = LLMContext(messages=[{"role": "system", "content": SYSTEM_PROMPT}])
    aggregators = LLMContextAggregatorPair(context)

    # Sits ahead of the LLM: sees cue messages pushed downstream by the RTVI
    # processor and bot-speaking frames pushed upstream by transport.output().
    nonverbal = NonverbalContextGate(
        context=context,
        distraction_checkin=ENABLE_DISTRACTION_CHECKIN,
    )

    # Slow lane, when ENABLE_VLM=1: camera frames stop at the sampler, which
    # feeds scene notes to the gate from its own task.
    scene_sampler = create_scene_sampler(nonverbal)

    pipeline = Pipeline(
        [
            transport.input(),
            *([scene_sampler] if scene_sampler else []),
            VADProcessor(vad_analyzer=SileroVADAnalyzer()),
            create_stt(),
            aggregators.user(),
            nonverbal,
            create_llm(),
            create_tts(),
            transport.output(),
            aggregators.assistant(),
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
    )

    @worker.rtvi.event_handler("on_client_message")
    async def on_client_message(_rtvi, message):
        """Log the fast lane; the gate itself reads the frame inside the pipeline."""
        if message.type == CUE_MESSAGE_TYPE:
            logger.info(f"cue: {message.data}")

    @transport.event_handler("on_client_connected")
    async def on_client_connected(_transport, _client):
        logger.info("client connected — greeting")
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(_transport, _client):
        logger.info("client disconnected — ending pipeline")
        await worker.cancel()

    await WorkerRunner(handle_sigint=runner_args.handle_sigint).run(worker)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
