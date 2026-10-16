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
