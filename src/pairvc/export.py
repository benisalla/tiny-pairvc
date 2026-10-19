"""Export the streaming model to ONNX, for real-time use outside Python/PyTorch.

    python -m pairvc.export --checkpoint checkpoints/run/best.pt --output pairvc.onnx

The ONNX graph converts ONE chunk (20 ms by default). Its inputs and outputs:

    audio_in   float32 [1, chunk]       next chunk of microphone audio
    speaker    int64   [1]              target voice id
    state_in_i float32 [1, C_i, L_i]    the model's memory (zeros at start)
 -> audio_out  float32 [1, chunk]       converted chunk (lookahead-delayed)
    state_out_i                         feed back as state_in_i for the next chunk

The file's metadata stores sample_rate, n_speakers and lookahead_samples (read
by pairvc.realtime). `run_onnx_streaming` shows the loop; `--verify` checks
ONNX == PyTorch.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor, nn

from pairvc.convert import load_model
from pairvc.model import PairVC

if TYPE_CHECKING:
    import onnxruntime as ort

logger = logging.getLogger(__name__)


class StreamingWrapper(nn.Module):
    """forward(chunk, speaker, *states) -> (chunk_out, *new_states): flat tensors, ONNX-friendly."""

    def __init__(self, model: PairVC) -> None:
        super().__init__()
        self.model = model

    def forward(self, chunk: Tensor, speaker: Tensor, *states: Tensor) -> tuple[Tensor, ...]:
        out, new_states = self.model.stream_step(chunk, speaker, list(states))
        return (out, *new_states)


def export_onnx(
    model: PairVC, path: Path, chunk_samples: int, opset: int = 18, sample_rate: int = 16_000
) -> list[str]:
    """Write the one-chunk streaming graph to `path`. Returns the state input names."""
    model = model.cpu().eval()
    states = model.init_state(1)
    state_in = [f"state_in_{i}" for i in range(len(states))]
    state_out = [f"state_out_{i}" for i in range(len(states))]
    args = (torch.zeros(1, chunk_samples), torch.zeros(1, dtype=torch.long), *states)
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            StreamingWrapper(model).eval(), args, str(path),
            input_names=["audio_in", "speaker", *state_in],
            output_names=["audio_out", *state_out],
            opset_version=opset, dynamo=True,  # fixed shapes: the chunk size is part of the graph
            external_data=False,  # one self-contained .onnx file (the weights are ~38 MB)
        )  # fmt: skip
    write_metadata(path, sample_rate=sample_rate, n_speakers=model.config.n_speakers,
                   lookahead_samples=model.config.lookahead_samples)  # fmt: skip
    return state_in


def write_metadata(path: Path, **values: int) -> None:
    """Store integers in the ONNX file's metadata (what a player needs to know)."""
    import onnx

    graph = onnx.load(str(path))
    for key, value in values.items():
        entry = graph.metadata_props.add()
        entry.key, entry.value = key, str(value)
    onnx.save(graph, str(path))


def read_metadata(session: ort.InferenceSession) -> dict[str, int]:
    """The integers written by `write_metadata` ({} for an older export)."""
    return {k: int(v) for k, v in session.get_modelmeta().custom_metadata_map.items()}


def run_onnx_streaming(path: Path, audio: np.ndarray, speaker: int) -> np.ndarray:
    """Convert [T] float32 audio with the ONNX graph, chunk by chunk (T padded to whole chunks)."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    chunk = inputs[0].shape[1]
    states = {i.name: np.zeros(i.shape, dtype=np.float32) for i in inputs[2:]}
    speaker_id = np.array([speaker], dtype=np.int64)
    audio = np.pad(audio.astype(np.float32), (0, -len(audio) % chunk))
    outputs = []
    for start in range(0, len(audio), chunk):
        feed = {"audio_in": audio[None, start : start + chunk], "speaker": speaker_id, **states}
        out, *new_states = session.run(None, feed)
        states = dict(zip(states, new_states))
        outputs.append(out[0])
    return np.concatenate(outputs)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export PairVC to a streaming ONNX graph.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("pairvc.onnx"))
    p.add_argument("--chunk-ms", type=float, default=20.0, help="chunk size baked into the graph")
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--verify", action="store_true", help="check ONNX output == PyTorch (needs onnxruntime)")
    return p


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    logger.setLevel(logging.INFO)  # our lines only, not the exporter's internals
    args = build_parser().parse_args(argv)
    model, sample_rate = load_model(args.checkpoint)
    hop = model.config.hop_length
    chunk = max(hop, round(args.chunk_ms * sample_rate / 1000 / hop) * hop)
    states = export_onnx(model, args.output, chunk, args.opset, sample_rate)
    logger.info("wrote %s (chunk %d samples, %d state tensors)", args.output, chunk, len(states))

    if args.verify:
        audio = (0.1 * torch.randn(chunk * 25)).numpy()
        with torch.inference_mode():
            reference = model(torch.from_numpy(audio)[None], torch.tensor([0]))[0].numpy()
        onnx_out = run_onnx_streaming(args.output, audio, 0)
        error = float(np.abs(onnx_out - reference).max())
        logger.info("verify: max |ONNX - PyTorch| = %.2e", error)
        if error > 1e-3:
            raise SystemExit("ONNX output differs from PyTorch")


if __name__ == "__main__":
    main()
