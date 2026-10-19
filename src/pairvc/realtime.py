"""Real-time conversion with the ONNX graph: what a live app (or app/streamlit_app.py) runs.

Like StreamVC, PairVC converts audio **20 ms at a time**: every 20 ms the
microphone delivers a chunk, the model converts it using its memory of the past
(the ONNX state tensors) and the converted chunk goes to the speakers. Nothing
waits for the end of the sentence, so the voice comes out 60 ms late (one
20 ms chunk + 40 ms lookahead), whatever the length of the speech.

- `OnnxStreamer`: one chunk in, one chunk out, the model's memory kept between calls.
- `StreamResampler`: sample-rate conversion that also works chunk by chunk
  (browsers and sound cards run at 48 kHz, the model at 16 kHz).
- `LiveConverter`: the two together for frames of any size at any rate, e.g.
  the 20 ms / 48 kHz frames of WebRTC: frame in, same-size frame out.

    from pairvc.realtime import LiveConverter
    live = LiveConverter("pairvc.onnx", speaker=0, rate=48_000)
    out = live.process(mic_frame)       # float32 mono, any length
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import firwin, lfilter

from pairvc.export import read_metadata


class OnnxStreamer:
    """The exported model plus its memory: `process(chunk)` converts the next `chunk` samples."""

    def __init__(self, path: Path | str, speaker: int = 0, threads: int = 1) -> None:
        import onnxruntime as ort  # optional dependency: pip install 'tiny-pairvc[onnx]'

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        self.chunk = int(inputs[0].shape[1])
        self._state_shapes = {i.name: i.shape for i in inputs[2:]}
        meta = read_metadata(self.session)
        self.sample_rate = meta.get("sample_rate", 16_000)
        self.n_speakers = meta.get("n_speakers", 2)
        self.lookahead = meta.get("lookahead_samples", 0)
        self.speaker = speaker
        self.reset()

    @property
    def latency_ms(self) -> float:
        """How late the converted voice comes out: one chunk + the lookahead."""
        return 1000 * (self.chunk + self.lookahead) / self.sample_rate

    def reset(self) -> None:
        """Forget the past (zero states), e.g. between two utterances."""
        self.states = {name: np.zeros(shape, dtype=np.float32) for name, shape in self._state_shapes.items()}

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """[chunk] float32 at `sample_rate` -> [chunk] converted (delayed by `lookahead`)."""
        feed = {
            "audio_in": np.asarray(chunk, dtype=np.float32).reshape(1, self.chunk),
            "speaker": np.array([self.speaker], dtype=np.int64),
            **self.states,
        }
        out, *new_states = self.session.run(None, feed)
        self.states = dict(zip(self.states, new_states, strict=True))
        return out[0]

    def convert(self, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """A whole clip, chunk by chunk as if it were live: ([T] output, per-chunk seconds).

        The output is time-aligned with the input (the lookahead delay is removed).
        """
        self.reset()
        length = len(audio)
        audio = np.pad(
            np.asarray(audio, dtype=np.float32), (0, -(length + self.lookahead) % self.chunk + self.lookahead)
        )
        outputs, times = [], []
        for start in range(0, len(audio), self.chunk):
            t0 = time.perf_counter()
            outputs.append(self.process(audio[start : start + self.chunk]))
            times.append(time.perf_counter() - t0)
        return np.concatenate(outputs)[self.lookahead : self.lookahead + length], np.array(times)


class StreamResampler:
    """Resample by up/down (integers, e.g. 1/3 for 48 -> 16 kHz) one block at a time.

    Zero-stuff by `up`, low-pass with an FIR filter whose state is kept between
    calls, keep one sample in `down`. Feeding a signal in blocks gives exactly
    the same output as feeding it at once.
    """

    def __init__(self, up: int, down: int, taps_per_phase: int = 16) -> None:
        g = math.gcd(up, down)
        self.up, self.down = up // g, down // g
        n_taps = taps_per_phase * max(self.up, self.down) + 1
        self.taps = firwin(n_taps, 0.95 / max(self.up, self.down)) * self.up
        self.reset()

    @property
    def delay(self) -> float:
        """Filter delay in output samples."""
        return (len(self.taps) - 1) / 2 / self.down

    def reset(self) -> None:
        self.zi = np.zeros(len(self.taps) - 1)
        self.phase = 0  # index of the next kept sample, relative to the next block

    def process(self, block: np.ndarray) -> np.ndarray:
        if self.up == self.down:
            return np.asarray(block, dtype=np.float32)
        stuffed = np.zeros(len(block) * self.up)
        stuffed[:: self.up] = block
        filtered, self.zi = lfilter(self.taps, 1.0, stuffed, zi=self.zi)
        out = filtered[self.phase :: self.down]
        self.phase = (self.phase - len(stuffed)) % self.down
        return out.astype(np.float32)


@dataclass
class LiveStats:
    """Timing of the last chunks (a rolling window), for a live display."""

    chunk_ms: float
    times: deque = field(default_factory=lambda: deque(maxlen=500))  # seconds per chunk

    @property
    def mean_ms(self) -> float:
        return 1000 * float(np.mean(self.times)) if self.times else 0.0

    @property
    def worst_ms(self) -> float:
        return 1000 * float(np.max(self.times)) if self.times else 0.0

    @property
    def rtf(self) -> float:
        """Compute time / audio time (< 1 = keeps up with real time)."""
        return self.mean_ms / self.chunk_ms


class LiveConverter:
    """Frames of any size at `rate` Hz in, converted frames of the same size out.

    Input is resampled to the model's rate and cut into model chunks; output is
    resampled back and queued. The queue starts with one chunk of silence, so
    every call can return a full frame: that is the price of chunking, and it
    is part of the reported latency.
    """

    def __init__(self, path: Path | str, speaker: int = 0, rate: int = 48_000, threads: int = 1) -> None:
        self.streamer = OnnxStreamer(path, speaker, threads)
        model_rate = self.streamer.sample_rate
        self.rate = rate
        self.to_model = StreamResampler(model_rate, rate)
        self.from_model = StreamResampler(rate, model_rate)
        self.stats = LiveStats(chunk_ms=1000 * self.streamer.chunk / model_rate)
        self.bypass = False  # True: return the input untouched (to compare with your own voice)
        self.reset()

    @property
    def speaker(self) -> int:
        return self.streamer.speaker

    @speaker.setter
    def speaker(self, value: int) -> None:
        self.streamer.speaker = value

    @property
    def latency_ms(self) -> float:
        """Total delay from microphone to output: chunk buffering + lookahead + resampling filters."""
        filters = 1000 * (self.to_model.delay / self.streamer.sample_rate + self.from_model.delay / self.rate)
        return self.streamer.latency_ms + filters

    def reset(self) -> None:
        self.streamer.reset()
        self.to_model.reset()
        self.from_model.reset()
        self.pending = np.zeros(0, dtype=np.float32)  # model-rate input not yet converted
        out_chunk = self.streamer.chunk * self.rate // self.streamer.sample_rate
        self.ready = np.zeros(out_chunk, dtype=np.float32)  # converted output not yet played

    def process(self, frame: np.ndarray) -> np.ndarray:
        """[n] float32 mono at `rate` -> [n] converted (or the input itself if `bypass`)."""
        frame = np.asarray(frame, dtype=np.float32)
        self.pending = np.concatenate([self.pending, self.to_model.process(frame)])
        chunk = self.streamer.chunk
        while len(self.pending) >= chunk:
            t0 = time.perf_counter()
            out = self.streamer.process(self.pending[:chunk])
            self.stats.times.append(time.perf_counter() - t0)
            self.pending = self.pending[chunk:]
            self.ready = np.concatenate([self.ready, self.from_model.process(np.clip(out, -1, 1))])
        n = len(frame)
        if len(self.ready) < n:  # only at start-up with odd frame sizes
            self.ready = np.concatenate([np.zeros(n - len(self.ready), dtype=np.float32), self.ready])
        out, self.ready = self.ready[:n], self.ready[n:]
        return frame if self.bypass else out


__all__ = ["LiveConverter", "LiveStats", "OnnxStreamer", "StreamResampler"]
