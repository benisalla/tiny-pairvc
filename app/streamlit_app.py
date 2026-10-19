"""PairVC playground: try the voice conversion in the browser, offline or live.

    pip install -e '.[app]'
    streamlit run app/streamlit_app.py

Everything runs the exported ONNX graph 20 ms at a time (pairvc.realtime), the
way a live voice changer does:

- "Record or upload": record a clip (or pick a file / a demo clip), convert it
  chunk by chunk, listen, and see how long each 20 ms chunk took.
- "Live microphone": your microphone through PairVC to your speakers, in real
  time, over WebRTC (48 kHz, 20 ms frames). Use headphones.

The default model is the small demo model of the README (assets/pairvc_digits.onnx,
trained on spoken digits). Export your own with
`pairvc-export --checkpoint best.pt --output my.onnx` and pick it in the sidebar.
"""

from __future__ import annotations

import io
import math
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import streamlit as st
from scipy.signal import resample_poly

from pairvc.realtime import LiveConverter, OnnxStreamer

ROOT = Path(__file__).resolve().parent.parent
DEMO_MODEL = ROOT / "assets" / "pairvc_digits.onnx"
DEMO_CLIPS = ROOT / "assets" / "audio"
DEMO_VOICES = {0: "voice 0: higher (+5 semitones)", 1: "voice 1: deeper (−5 semitones)"}

st.set_page_config(page_title="PairVC playground", page_icon="🎙️", layout="wide")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@st.cache_resource
def uploaded_model_path(data: bytes, name: str) -> Path:
    """Uploaded .onnx bytes -> a file onnxruntime can open (kept for the session)."""
    path = Path(tempfile.mkdtemp()) / name
    path.write_bytes(data)
    return path


with st.sidebar:
    st.header("Model")
    upload = st.file_uploader("ONNX model (default: the README's demo model)", type=["onnx"])
    model_path = uploaded_model_path(upload.getvalue(), upload.name) if upload else DEMO_MODEL
    if not model_path.exists():
        st.error(f"{model_path} not found. Export one: `pairvc-export --checkpoint best.pt --output my.onnx`")
        st.stop()
    threads = st.slider("CPU threads", 1, 4, 1, help="a live app usually gets one core")
    info = OnnxStreamer(model_path, threads=threads)
    voice_names = DEMO_VOICES if model_path == DEMO_MODEL else {}
    speaker = st.radio(
        "Target voice",
        range(info.n_speakers),
        format_func=lambda k: voice_names.get(k, f"voice {k}"),
    )
    st.caption(
        f"`{model_path.name}`: {info.sample_rate // 1000} kHz, {info.n_speakers} voices, "
        f"{1000 * info.chunk / info.sample_rate:.0f} ms chunks, "
        f"{1000 * info.lookahead / info.sample_rate:.0f} ms lookahead → "
        f"**{info.latency_ms:.0f} ms latency**"
    )
    if model_path == DEMO_MODEL:
        st.info(
            "Demo model: 0.67M parameters trained 1.5 h on a CPU on spoken digits. It has "
            "learned each voice's timbre, not yet clean pitch: expect a breathy voice. "
            "Train the full model on a GPU for the real thing."
        )

st.title("🎙️ PairVC playground")
st.write(
    "PairVC converts your voice **20 ms at a time**: every chunk is converted as soon as it "
    "arrives, using the model's memory of the past, so the new voice comes out "
    f"**{info.latency_ms:.0f} ms** after you speak (one chunk + the model's lookahead)."
)


def to_mono(audio: np.ndarray, rate: int, target: int) -> np.ndarray:
    """Any decoded file -> float32 mono at the model's rate."""
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != target:
        g = math.gcd(rate, target)
        audio = resample_poly(audio, target // g, rate // g)
    return audio.astype(np.float32)


def wav_bytes(audio: np.ndarray, rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, np.clip(audio, -1, 1), rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def spectrogram(audio: np.ndarray, rate: int, title: str):  # -> matplotlib Figure
    import matplotlib.pyplot as plt

    from pairvc.metrics import f0_metrics, log_mel

    fig, ax = plt.subplots(figsize=(5, 2.2))
    ax.imshow(log_mel(audio, rate).T, origin="lower", aspect="auto", cmap="magma", vmin=-9, vmax=2)
    pitch = f0_metrics(audio, audio, rate)["median_f0_hz"]
    ax.set_title(f"{title}: median pitch {pitch:.0f} Hz" if pitch else title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return fig


offline_tab, live_tab = st.tabs(["Record or upload", "Live microphone"])

# ---------------------------------------------------------------------------
# 1. Record or upload: convert a clip chunk by chunk, as if it were live
# ---------------------------------------------------------------------------

with offline_tab:
    source_kind = st.radio("Input", ["Record", "Upload a file", "Demo clip"], horizontal=True)
    clip = None
    if source_kind == "Record":
        recording = st.audio_input("Record yourself (a few words are enough)")
        if recording is not None:
            clip = sf.read(io.BytesIO(recording.getvalue()), dtype="float32")
    elif source_kind == "Upload a file":
        file = st.file_uploader("Audio file", type=["wav", "flac", "ogg", "mp3"])
        if file is not None:
            clip = sf.read(io.BytesIO(file.getvalue()), dtype="float32")
    else:
        demos = sorted(DEMO_CLIPS.glob("*_source.wav"))
        name = st.selectbox("Unseen speaker of the demo", [p.name for p in demos])
        clip = sf.read(DEMO_CLIPS / name, dtype="float32")

    if clip is not None:
        audio = to_mono(clip[0], clip[1], info.sample_rate)
        streamer = OnnxStreamer(model_path, speaker, threads)
        converted, times = streamer.convert(audio)
        chunk_ms = 1000 * streamer.chunk / streamer.sample_rate

        left, right = st.columns(2)
        with left:
            st.subheader("Input")
            st.audio(wav_bytes(audio, info.sample_rate), format="audio/wav")
            st.pyplot(spectrogram(audio, info.sample_rate, "input"))
        with right:
            st.subheader(voice_names.get(speaker, f"Voice {speaker}"))
            st.audio(wav_bytes(converted, info.sample_rate), format="audio/wav")
            st.pyplot(spectrogram(converted, info.sample_rate, "converted"))

        st.subheader("Real-time check")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("chunks", f"{len(times)} × {chunk_ms:.0f} ms")
        c2.metric("mean compute / chunk", f"{1000 * times.mean():.2f} ms")
        c3.metric("slowest chunk", f"{1000 * times.max():.2f} ms")
        c4.metric(
            "real-time factor", f"{1000 * times.mean() / chunk_ms:.3f}", help="< 1: faster than real time"
        )
        st.line_chart(
            {"compute time (ms)": 1000 * times, f"budget ({chunk_ms:.0f} ms)": np.full(len(times), chunk_ms)},
            x_label="chunk",
            y_label="ms",
        )
        st.caption(
            "Each 20 ms chunk must be converted in under 20 ms to keep up with a microphone. "
            "The output above is exactly what a live stream would produce (minus the delay)."
        )

# ---------------------------------------------------------------------------
# 2. Live: microphone -> PairVC -> speakers over WebRTC
# ---------------------------------------------------------------------------


class LiveSession:
    """One LiveConverter per browser stream, built on the first frame (when its rate is known)."""

    def __init__(self, path: Path, threads: int) -> None:
        self.path, self.threads = path, threads
        self.converter: LiveConverter | None = None
        self.speaker, self.bypass = 0, False
        self.lock = threading.Lock()

    def __call__(self, frame):  # av.AudioFrame -> av.AudioFrame
        import av

        pcm = frame.to_ndarray()  # packed s16: [1, samples * channels]
        channels = len(frame.layout.channels)
        mono = pcm.reshape(-1, channels).mean(axis=1).astype(np.float32) / 32768
        with self.lock:
            if self.converter is None or self.converter.rate != frame.sample_rate:
                self.converter = LiveConverter(self.path, rate=frame.sample_rate, threads=self.threads)
            self.converter.speaker, self.converter.bypass = self.speaker, self.bypass
            out = self.converter.process(mono)
        pcm_out = np.repeat((np.clip(out, -1, 1) * 32767).astype(np.int16)[:, None], channels, axis=1)
        new = av.AudioFrame.from_ndarray(pcm_out.reshape(1, -1), format="s16", layout=frame.layout.name)
        new.sample_rate, new.pts = frame.sample_rate, frame.pts
        if frame.time_base is not None:
            new.time_base = frame.time_base
        return new


with live_tab:
    try:
        from streamlit_webrtc import WebRtcMode, webrtc_streamer
    except ImportError:
        st.warning("Live mode needs streamlit-webrtc: `pip install -e '.[app]'`")
        st.stop()

    st.write(
        "Press **START**, allow the microphone and speak: you hear yourself in the target voice, "
        "converted 20 ms at a time. **Use headphones** (otherwise the speakers feed back into the "
        "microphone). Switch the voice in the sidebar while talking."
    )
    bypass = st.toggle("Bypass (hear your own voice, same delay path)", value=False)

    @st.cache_resource
    def live_session(path: Path, threads: int) -> LiveSession:
        return LiveSession(path, threads)

    session = live_session(model_path, threads)
    session.speaker, session.bypass = speaker, bypass
    ctx = webrtc_streamer(
        key="pairvc-live",
        mode=WebRtcMode.SENDRECV,
        audio_frame_callback=session,
        media_stream_constraints={"audio": {"echoCancellation": True, "noiseSuppression": False,
                                            "autoGainControl": False}, "video": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        async_processing=True,
    )  # fmt: skip

    stats_box = st.empty()
    while ctx.state.playing:
        converter = session.converter
        if converter is not None and converter.stats.times:
            s = converter.stats
            stats_box.markdown(
                f"**{converter.rate // 1000} kHz in/out** · {len(s.times)} recent chunks · "
                f"mean **{s.mean_ms:.2f} ms** / {s.chunk_ms:.0f} ms chunk · worst {s.worst_ms:.2f} ms · "
                f"RTF **{s.rtf:.3f}** · latency ≈ **{converter.latency_ms:.0f} ms** (+ browser/network)"
            )
        time.sleep(0.5)
