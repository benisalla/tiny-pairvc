# Using a trained model

You need a checkpoint written by `pairvc-train` (or the notebook):
`best.pt`, `last.pt`, or the small `pairvc_generator.pt`. Everything needed to
rebuild the network (its size, number of voices, lookahead, sample rate) is
stored inside, so no other flag has to match.

## 1. Command line

```bash
# offline: whole file at once, output aligned with the input
pairvc-convert --checkpoint checkpoints/run/best.pt --input me.wav --speaker 0
# -> converted/me_to_0.wav

# several files, another folder, target voice 1
pairvc-convert --checkpoint checkpoints/run/best.pt --input a.wav b.flac --speaker 1 --output-dir out

# streaming: chunk by chunk like a live app, and report the real-time factor
pairvc-convert --checkpoint checkpoints/run/best.pt --input me.wav --speaker 0 --streaming
# INFO me.wav: RTF 0.542 | chunk 20 ms, worst 17.2 ms | latency 60 ms
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--checkpoint` | required | `best.pt`, `last.pt` or `pairvc_generator.pt` |
| `--input` | required | one or more audio files (any format soundfile reads, any rate; stereo is averaged) |
| `--speaker` | required | target voice id, 0 … n_speakers − 1 |
| `--output-dir` | `converted` | writes `<name>_to_<speaker>.wav` |
| `--streaming` | off | convert chunk by chunk and log the real-time factor |
| `--chunk-ms` | 20 | chunk size for `--streaming` (rounded to whole 20 ms frames) |
| `--device` | cpu | `cpu`, `cuda`, `auto` |

Both modes give the same audio (tested). **RTF** = compute time / audio
duration: below 1 the model keeps up with real time.

## 2. Python

```python
import soundfile as sf
import torch
from pairvc.convert import load_model
from pairvc.data import load_audio

model, sample_rate = load_model("checkpoints/run/best.pt")  # eval mode, CPU
audio = load_audio("me.wav", sample_rate)  # [T] float32 mono
converted = model.convert(audio[None], torch.tensor([0]))[0]  # [T], aligned with the input
sf.write("me_to_0.wav", converted.clamp(-1, 1).numpy(), sample_rate)
```

`model.convert` removes the model's 40 ms lookahead delay. `model(audio, speaker)`
is the raw forward pass, whose output is delayed by that amount.

## 3. Streaming in Python

```python
states = model.init_state(batch_size=1)  # zeros: the model has heard nothing yet
speaker = torch.tensor([0])
with torch.inference_mode():
    for chunk in microphone_chunks():  # each [1, 320] (20 ms at 16 kHz), or n x 320
        out, states = model.stream_step(chunk, speaker, states)
        play(out)  # [1, 320]
```

The output is the input 40 ms + one chunk later (60 ms latency with 20 ms
chunks). Bigger chunks use the CPU more efficiently but add latency. Use
`torch.set_num_threads(1)` when you also run audio I/O on the same machine.

## 4. ONNX: real time outside PyTorch

```bash
pip install -e '.[onnx]'
pairvc-export --checkpoint checkpoints/run/best.pt --output pairvc.onnx --verify
# INFO wrote pairvc.onnx (chunk 320 samples, 60 state tensors)
# INFO verify: max |ONNX - PyTorch| = 1.68e-07
```

The graph converts **one chunk** (size fixed at export, `--chunk-ms`, default
20 ms). One self-contained file (~38 MB). Inputs and outputs:

| Name | Shape | |
|------|-------|-|
| `audio_in` | float32 [1, 320] | next chunk of input audio |
| `speaker` | int64 [1] | target voice id |
| `state_in_0` … `state_in_59` | float32 [1, C, L] | memory; zeros at start |
| `audio_out` | float32 [1, 320] | converted chunk |
| `state_out_0` … `state_out_59` | same as inputs | feed back as `state_in_*` next call |

```python
from pairvc.export import run_onnx_streaming

converted = run_onnx_streaming("pairvc.onnx", audio_numpy_float32, speaker=0)
```

`run_onnx_streaming` in `src/pairvc/export.py` is a 15-line reference loop to
port to C++, C#, Rust, etc. onnxruntime on one CPU thread converts a 20 ms
chunk in about 6 ms (RTF ≈ 0.3, measured on one thread of a desktop CPU).

## 5. Live: microphone in, converted voice out

[`pairvc.realtime`](../src/pairvc/realtime.py) wraps the ONNX graph for a live
app. The ONNX file stores its sample rate, number of voices and lookahead in
its metadata, so a player needs nothing else.

```python
from pairvc.realtime import LiveConverter

live = LiveConverter("pairvc.onnx", speaker=0, rate=48_000)  # the sound card's / browser's rate
out_frame = live.process(mic_frame)  # float32 mono, any length -> same length
live.speaker = 1  # switch voice on the fly
live.stats.mean_ms, live.stats.rtf, live.latency_ms
```

Each call resamples the frame to the model's rate with a streaming FIR
resampler (block-by-block == whole signal, tested), converts every complete
20 ms chunk with `OnnxStreamer`, resamples back and returns a frame of the same
size. The output queue starts with one chunk of silence, so latency is
**chunk + lookahead + ~1 ms of filters ≈ 61 ms**, plus the audio device's own
buffers.

The playground app uses it on every WebRTC audio frame (48 kHz, 20 ms):

```bash
pip install -e '.[app]'
streamlit run app/streamlit_app.py
```

- *Record or upload*: converts a clip with `OnnxStreamer.convert` (chunk by
  chunk, exactly the live output) and plots the compute time of every chunk
  against its 20 ms budget.
- *Live microphone*: `streamlit-webrtc` sends the microphone to the server,
  `LiveConverter` converts it, and the result plays back in the browser. Use
  headphones. On a remote server the browser needs HTTPS (or localhost) for the
  microphone, and WebRTC may need a TURN server behind strict NATs.
