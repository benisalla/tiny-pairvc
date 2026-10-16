"""Use a trained model: speech in, the same speech in a target voice out.

    python -m pairvc.convert --checkpoint checkpoints/run/best.pt \
        --input me.wav --speaker 1 --output-dir converted/

Two modes, same result:
- offline (default): the whole file in one pass, output time-aligned with the input;
- --streaming: chunk by chunk (20 ms by default), exactly as a live app would
  run it; also reports the real-time factor (RTF < 1 = faster than real time).

Everything needed to rebuild the network is stored inside the checkpoint.
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
from torch import Tensor

from pairvc.config import DataConfig, ModelConfig
from pairvc.data import load_audio
from pairvc.model import PairVC
from pairvc.train import resolve_device

logger = logging.getLogger(__name__)


def load_model(checkpoint_path: Path, device: torch.device | str = "cpu") -> tuple[PairVC, int]:
    """Rebuild the generator from a checkpoint, in eval mode. Returns (model, sample_rate)."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = PairVC(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["generator"])
    model.to(device).eval()
    sample_rate = DataConfig(**checkpoint["data_config"]).sample_rate
    return model, sample_rate


def save_generator_only(checkpoint_path: Path, out_path: Path) -> Path:
    """Write a small file with only what inference needs (no discriminators, no optimizers).

    The result loads with `load_model` / `pairvc-convert` / `pairvc-export` like best.pt.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    keys = ("generator", "model_config", "data_config", "step")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: checkpoint[k] for k in keys if k in checkpoint}, out_path)
    return out_path


@dataclass(frozen=True)
class StreamStats:
    audio_seconds: float
    compute_seconds: float
    chunk_ms: float
    worst_chunk_ms: float
    latency_ms: float  # chunk + lookahead: how late the voice comes out

    @property
    def rtf(self) -> float:
        """Real-time factor: compute time / audio time (< 1 = keeps up with real time)."""
        return self.compute_seconds / self.audio_seconds


@torch.inference_mode()
def convert_streaming(
    model: PairVC, audio: Tensor, speaker: int, chunk_samples: int, sample_rate: int
) -> tuple[Tensor, StreamStats]:
    """Feed [T] audio chunk by chunk, as a live app would. Returns ([T] output, stats).

    The output is time-aligned with the input (the lookahead delay is removed).
    """
    device = model.enc_in.weight.device
    delay = model.config.lookahead_samples
    length = audio.shape[-1]
    total = length + delay
    audio = F.pad(audio, (0, -total % chunk_samples + delay)).to(device)[None]  # whole chunks
    spk = torch.tensor([speaker], device=device)
    states = model.init_state(1)
    outputs, times = [], []
    for start in range(0, audio.shape[-1], chunk_samples):
        t0 = time.perf_counter()
        out, states = model.stream_step(audio[:, start : start + chunk_samples], spk, states)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        outputs.append(out)
    converted = torch.cat(outputs, dim=-1)[0, delay : delay + length].cpu()
    stats = StreamStats(
        audio_seconds=length / sample_rate,
        compute_seconds=sum(times),
        chunk_ms=1000 * chunk_samples / sample_rate,
        worst_chunk_ms=1000 * max(times),
        latency_ms=1000 * (chunk_samples + delay) / sample_rate,
    )
    return converted, stats


def convert_file(model: PairVC, path: Path, speaker: int, sample_rate: int) -> Tensor:
    """Offline conversion of one file: [T] audio at `sample_rate`."""
    if not 0 <= speaker < model.config.n_speakers:
        raise ValueError(f"speaker must be in [0, {model.config.n_speakers - 1}]")
    device = model.enc_in.weight.device
    audio = load_audio(path, sample_rate).to(device)
    return model.convert(audio[None], torch.tensor([speaker], device=device))[0].cpu()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Convert speech to a target voice with PairVC.")
    p.add_argument("--checkpoint", type=Path, required=True, help="path to best.pt / last.pt")
    p.add_argument("--input", type=Path, required=True, nargs="+", help="one or more audio files")
    p.add_argument("--speaker", type=int, required=True, help="target voice id (0 .. n_speakers-1)")
    p.add_argument("--output-dir", type=Path, default=Path("converted"))
    p.add_argument("--streaming", action="store_true", help="chunk by chunk, report the RTF")
    p.add_argument("--chunk-ms", type=float, default=20.0, help="chunk size for --streaming")
    p.add_argument("--device", default="cpu", help="cpu (default: what real time runs on) | cuda | auto")
    return p


def run(argv: Sequence[str] | None = None) -> list[Path]:
    """Parse the flags, convert every input file, return the output paths."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    model, sample_rate = load_model(args.checkpoint, resolve_device(args.device))
    hop = model.config.hop_length
    chunk = max(hop, round(args.chunk_ms * sample_rate / 1000 / hop) * hop)  # whole frames
    args.output_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for path in args.input:
        if args.streaming:
            audio = load_audio(path, sample_rate)
            converted, stats = convert_streaming(model, audio, args.speaker, chunk, sample_rate)
            logger.info(
                "%s: RTF %.3f | chunk %.0f ms, worst %.1f ms | latency %.0f ms",
                path.name, stats.rtf, stats.chunk_ms, stats.worst_chunk_ms, stats.latency_ms,
            )  # fmt: skip
        else:
            converted = convert_file(model, path, args.speaker, sample_rate)
        out_path = args.output_dir / f"{path.stem}_to_{args.speaker}.wav"
        sf.write(out_path, converted.clamp(-1, 1).numpy(), sample_rate)
        logger.info("wrote %s", out_path)
        written.append(out_path)
    return written


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point (returns None: console scripts pass the return value to sys.exit)."""
    run(argv)


if __name__ == "__main__":
    main()
