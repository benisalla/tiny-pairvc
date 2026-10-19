"""01 - Build a small *real speech* paired dataset in a minute.

Downloads the Free Spoken Digit Dataset once (16 MB) and plays the teacher with
a phase-vocoder pitch shift: voice 0 = +5 semitones, voice 1 = -5 semitones,
same timing. Prints what was written and the pitch of each voice.

    python examples/01_build_digits_dataset.py
"""

from pathlib import Path

import soundfile as sf

from pairvc.data import PairedVoiceDataset, find_pairs
from pairvc.digits import SHIFTS, VAL_SPEAKER, write_digits_dataset
from pairvc.metrics import f0_metrics

OUT = Path("examples/output/digits_data")

root = write_digits_dataset(OUT)
train = PairedVoiceDataset(root / "train", sample_rate=16_000, segment_samples=8_000)
val = PairedVoiceDataset(root / "val", sample_rate=16_000, segment_samples=8_000)
print(f"{len(train)} train pairs, {len(val)} val pairs (val speaker '{VAL_SPEAKER}' is never trained on)")
print(f"target voices {train.speakers}: shifts {SHIFTS} semitones")

for pair in find_pairs(root / "val")[:2]:
    source, sr = sf.read(pair.source)
    target, _ = sf.read(pair.target)
    f0_source = f0_metrics(source, source, sr)["median_f0_hz"]
    f0_target = f0_metrics(target, target, sr)["median_f0_hz"]
    print(f"{pair.target.name}: {len(source) / sr:.2f} s, pitch {f0_source:.0f} Hz -> {f0_target:.0f} Hz")
