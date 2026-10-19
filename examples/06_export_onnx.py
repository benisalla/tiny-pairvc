"""06 - Export the streaming model to ONNX and run it without PyTorch.

The ONNX graph converts one 20 ms chunk and carries the model's memory as
explicit state tensors; `run_onnx_streaming` is the loop a live app would run.

    python examples/06_export_onnx.py [--checkpoint examples/output/run/best.pt]
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from pairvc.convert import load_model
from pairvc.data import load_audio
from pairvc.export import export_onnx, run_onnx_streaming

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, default=Path("examples/output/run/best.pt"))
args = parser.parse_args()

model, sample_rate = load_model(args.checkpoint)
path = Path("examples/output/pairvc.onnx")
export_onnx(model, path, chunk_samples=model.config.hop_length)
print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")

source = min(Path("examples/output/digits_data/val").glob("*_original.wav"))
audio = load_audio(source, sample_rate).numpy()
onnx_out = run_onnx_streaming(path, audio, speaker=1)
with torch.inference_mode():
    torch_out = model(torch.from_numpy(audio)[None], torch.tensor([1]))[0].numpy()
n = min(len(onnx_out), len(torch_out))
print(f"ONNX == PyTorch: max difference {np.abs(onnx_out[:n] - torch_out[:n]).max():.2e}")
