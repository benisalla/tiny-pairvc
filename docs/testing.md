# Testing guide

Three levels of checks, from seconds to minutes:

1. **Unit tests** (`pytest`): every module, CPU only, no network.
2. **Overfit sanity check**: your real data, can the model memorise 8 pairs?
3. **Launch checks**: the real CLIs (`pairvc-train`, `torchrun`, `pairvc-convert`, `pairvc-export`) and the notebook.

## 1. Unit tests

```bash
pytest                       # everything (~2 min on a laptop CPU)
pytest tests/test_model.py   # one file
pytest -k streaming -v       # tests whose name contains "streaming"
ruff check src tests         # lint
ruff format --check src tests
```

No GPU and no internet needed: tests use tiny models and a synthetic paired
dataset written to a temporary folder (`pairvc.data.write_toy_dataset`).

| File | What it proves |
|------|----------------|
| `test_config.py` | invalid settings are rejected early (unknown discriminator, fp16, bad strides…); configs survive the round trip through a checkpoint |
| `test_data.py` | sources are matched with every `_converted_<k>` file; int16 / stereo / other sample rates load as mono float at 16 kHz; source and target are cropped at the same place (alignment kept); validation crops are deterministic |
| `test_model.py` | output length = input length (any length); **causality** (future input cannot change past output); the speaker changes the output; **streaming == whole clip** for several chunk sizes, tiny and full-size model; `convert` removes exactly the lookahead delay; gradient checkpointing gives identical gradients; receptive field |
| `test_losses.py` | mel filterbank is well formed; mel loss is 0 for identical audio and grows with noise; least-squares GAN values; every discriminator kind runs and sends gradients back to the audio |
| `test_train.py` | lookahead alignment; precision/device choice; a training run logs every metric and writes `last.pt`, `best.pt` and audio samples; **resume** continues the step count with identical weights; the mel loss goes down; the mel-only warm-up (`--gan-start-step`) skips the discriminators; `history.jsonl` is written; CLI flags (`--overfit`, segment length check) and the CLI end to end |
| `test_distributed.py` | **DDP**: 2 processes (gloo backend on CPU) start from *different* random weights, train across an epoch boundary and end with **identical** generator and discriminator weights; only rank 0 writes checkpoints |
| `test_metrics.py` | the pitch tracker finds the pitch of tones (within 30 cents); identical audio scores 0 / 100 %; MCD and F0 error grow with noise, timbre and pitch changes; the phase vocoder changes length not pitch; the digits teacher keeps the length and shifts the pitch by the asked semitones; `evaluate_split` and `pairvc-eval` report every system |
| `test_convert.py` | a checkpoint (or the generator-only file) reloads into the same model; streaming conversion == offline conversion; the CLI writes one file per input with the input's length; unknown speaker ids are rejected; **ONNX export == PyTorch** (skipped if onnxruntime is not installed) |

## 2. Overfit sanity check

See [training.md, section 2](training.md#2-first-the-overfit-sanity-check).
It catches what unit tests cannot: wrong data paths, misaligned
source/target files, wrong labels, a teacher that changed the timing.

## 3. Launch checks

```bash
# 2 processes on CPU (gloo backend): checks the torchrun path without GPUs
python -c "from pairvc.data import write_toy_dataset; write_toy_dataset('toy_data')"
torchrun --standalone --nproc_per_node=2 -m pairvc.train --device cpu \
    --data-dir toy_data --channels 8 --discriminators resolution \
    --max-steps 4 --eval-every 2 --log-every 1 --batch-size 2 --num-workers 0 \
    --output-dir /tmp/ddp-check

pairvc-convert --checkpoint /tmp/ddp-check/best.pt --input toy_data/val/clip0000_original.wav --speaker 1 --streaming
pairvc-export  --checkpoint /tmp/ddp-check/best.pt --output /tmp/check.onnx --verify
```

**The notebook** is generated from `src/` and must be rebuilt after any change
there:

```bash
python notebooks/build_notebook.py . notebooks/train_colab_kaggle.ipynb
```

To run it locally end to end (about 1 minute on CPU), set `SMOKE_TEST = True`
in its settings cell and execute it with Jupyter (`pip install jupyter`), or:

```bash
jupyter nbconvert --to notebook --execute notebooks/train_colab_kaggle.ipynb --output /tmp/run.ipynb
```

(after changing `SMOKE_TEST` to `True` in a copy).
