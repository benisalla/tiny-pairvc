"""08 - A live voice changer with the ONNX model: 48 kHz, 20 ms frames, like a browser mic.

`LiveConverter` is what the Streamlit app (app/streamlit_app.py) runs on every
WebRTC audio frame: resample 48 -> 16 kHz, convert 20 ms with the ONNX graph
(keeping the model's memory), resample back, return a frame of the same size.
Here the "microphone" is a held-out recording cut into 960-sample frames.

    python examples/08_live_voice_changer.py [--onnx assets/pairvc_digits.onnx]
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from pairvc.realtime import LiveConverter

parser = argparse.ArgumentParser()
parser.add_argument(
    "--onnx", type=Path, default=Path("assets/pairvc_digits.onnx"), help="from example 06 or pairvc-export"
)
parser.add_argument("--speaker", type=int, default=1)
args = parser.parse_args()

rate, frame = 48_000, 960  # what WebRTC delivers: 20 ms frames at 48 kHz
clips = sorted(Path("assets/audio").glob("*_source.wav"))
mic = np.concatenate([resample_poly(sf.read(p, dtype="float32")[0], 3, 1) for p in clips]).astype(np.float32)

live = LiveConverter(args.onnx, speaker=args.speaker, rate=rate)
frames = [live.process(mic[i : i + frame]) for i in range(0, len(mic) - frame + 1, frame)]
out = np.concatenate(frames)

out_dir = Path("examples/output")
out_dir.mkdir(parents=True, exist_ok=True)
sf.write(out_dir / "live_input.wav", mic, rate)
sf.write(out_dir / f"live_voice{args.speaker}.wav", out, rate)
s = live.stats
print(
    f"{len(frames)} frames of {1000 * frame / rate:.0f} ms at {rate // 1000} kHz -> {len(s.times)} model chunks"
)
print(f"compute per 20 ms chunk: mean {s.mean_ms:.2f} ms, worst {s.worst_ms:.2f} ms, RTF {s.rtf:.3f}")
print(f"latency: {live.latency_ms:.0f} ms (20 ms chunk + 40 ms lookahead + resampling filters)")
print(f"wrote {out_dir}/live_input.wav and {out_dir}/live_voice{args.speaker}.wav")
