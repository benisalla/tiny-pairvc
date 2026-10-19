# Examples

Short scripts that use the `pairvc` package from Python, in the order you would
use it. Run them from the repository root, one after the other: each one takes
seconds to minutes on a CPU, and everything goes to `examples/output/`
(ignored by git).

They use the **spoken digits dataset** of `pairvc.digits`: real speech (the
Free Spoken Digit Dataset, 16 MB download) converted to two target voices by a
pitch-shift "teacher" (+5 and −5 semitones). It is the simplest real-speech
setting that still shows every part of PairVC working, and it is what the
README results come from.

| # | Script | What it shows | Time (CPU) |
|---|--------|---------------|------------|
| 01 | [01_build_digits_dataset.py](01_build_digits_dataset.py) | build a paired dataset: source speech + the same speech in 2 target voices | ~1 min |
| 02 | [02_train_on_digits.py](02_train_on_digits.py) | train a 0.67M-parameter PairVC: mel-only warm-up, then GAN (`--steps 3500` for the README numbers) | 15–25 min (700 steps) |
| 03 | [03_convert_a_file.py](03_convert_a_file.py) | convert a held-out recording to each target voice, print the pitch | seconds |
| 04 | [04_stream_in_real_time.py](04_stream_in_real_time.py) | 20 ms chunks as a live app does: real-time factor, latency, streaming == offline | seconds |
| 05 | [05_evaluate.py](05_evaluate.py) | metrics table: before vs after training, no conversion, teacher (MCD, F0, DNSMOS) | ~1 min |
| 06 | [06_export_onnx.py](06_export_onnx.py) | streaming ONNX export, run with onnxruntime, compare with PyTorch | seconds |
| 07 | [07_before_after_demo.py](07_before_after_demo.py) | the README visuals: spectrograms before/after, training GIF, curves, audio clips | ~1 min |

```bash
pip install -e '.[dev,examples]'   # examples = matplotlib + pillow, only for example 07
python examples/01_build_digits_dataset.py
python examples/02_train_on_digits.py --steps 700
python examples/03_convert_a_file.py
python examples/04_stream_in_real_time.py
python examples/05_evaluate.py
python examples/06_export_onnx.py
python examples/07_before_after_demo.py
```

Examples 03–07 take `--checkpoint` / `--run` to point at another run, e.g. a
model trained on the Kaggle/Colab notebook.
