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
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.workers.runner import WorkerRunner

from services import SYSTEM_PROMPT, create_llm, create_stt, create_tts


def build_transport(runner_args: SmallWebRTCRunnerArguments) -> SmallWebRTCTransport:
    """Audio-only SmallWebRTC transport; video arrives only on the VLM slow lane."""
    return SmallWebRTCTransport(
        webrtc_connection=runner_args.webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    )


async def bot(runner_args: RunnerArguments):
    """Entry point the Pipecat dev runner calls per WebRTC session."""
    if not isinstance(runner_args, SmallWebRTCRunnerArguments):
        raise RuntimeError(f"unsupported transport args: {type(runner_args).__name__}")

    transport = build_transport(runner_args)

    context = LLMContext(messages=[{"role": "system", "content": SYSTEM_PROMPT}])
    aggregators = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),
            VADProcessor(vad_analyzer=SileroVADAnalyzer()),
            create_stt(),
            aggregators.user(),
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
