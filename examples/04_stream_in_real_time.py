"""04 - Convert chunk by chunk, as a live app would, and measure the speed.

Feeds 20 ms chunks to `stream_step`, checks the result equals the offline
conversion, and prints the real-time factor (RTF < 1 = faster than real time).

    python examples/04_stream_in_real_time.py [--checkpoint examples/output/run/best.pt]
"""

import argparse
from pathlib import Path

import torch

from pairvc.convert import convert_streaming, load_model
from pairvc.data import load_audio

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, default=Path("examples/output/run/best.pt"))
args = parser.parse_args()

torch.set_num_threads(1)  # one CPU core, as in a live app
model, sample_rate = load_model(args.checkpoint)
paths = sorted(Path("examples/output/digits_data/val").glob("*_original.wav"))[:10]
audio = torch.cat([load_audio(p, sample_rate) for p in paths])  # ~6 s of speech
chunk = model.config.hop_length  # 320 samples = 20 ms

streamed, stats = convert_streaming(model, audio, speaker=0, chunk_samples=chunk, sample_rate=sample_rate)
with torch.inference_mode():
    offline = model.convert(audio[None], torch.tensor([0]))[0]
print(f"{stats.audio_seconds:.1f} s of audio in {stats.compute_seconds:.2f} s: RTF {stats.rtf:.3f}")
print(
    f"chunk {stats.chunk_ms:.0f} ms, slowest chunk {stats.worst_chunk_ms:.1f} ms, latency {stats.latency_ms:.0f} ms"
)
print(f"streaming == offline: max difference {(streamed - offline).abs().max():.2e}")
