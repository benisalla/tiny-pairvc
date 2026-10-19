"""03 - Convert one recording to each target voice (offline, whole file).

python examples/03_convert_a_file.py [--checkpoint examples/output/run/best.pt]
"""

import argparse
from pathlib import Path

import soundfile as sf
import torch

from pairvc.convert import convert_file, load_model
from pairvc.metrics import f0_metrics

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, default=Path("examples/output/run/best.pt"))
args = parser.parse_args()

model, sample_rate = load_model(args.checkpoint)
source_path = min(Path("examples/output/digits_data/val").glob("*_original.wav"))
out_dir = Path("examples/output/converted")
out_dir.mkdir(parents=True, exist_ok=True)

source, _ = sf.read(source_path)
print(f"{source_path.name}: pitch {f0_metrics(source, source, sample_rate)['median_f0_hz']:.0f} Hz")
for speaker in range(model.config.n_speakers):
    with torch.inference_mode():
        converted = convert_file(model, source_path, speaker, sample_rate).clamp(-1, 1).numpy()
    path = out_dir / f"{source_path.stem}_to_{speaker}.wav"
    sf.write(path, converted, sample_rate)
    pitch = f0_metrics(converted, converted, sample_rate)["median_f0_hz"]
    print(f"  voice {speaker}: pitch {pitch:.0f} Hz -> {path}")
