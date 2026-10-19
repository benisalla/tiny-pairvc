"""Build notebooks/train_colab_kaggle.ipynb: a self-contained training notebook.

The code of src/pairvc/{config,data,digits,model,losses,discriminators,train,metrics,evaluate}.py is
copied into the notebook at build time (imports between those modules are
removed, as everything lives in one namespace), so the notebook is identical to
the package and needs no GitHub access. Regenerate after changing src/pairvc:

    python notebooks/build_notebook.py . notebooks/train_colab_kaggle.ipynb
"""

import ast
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])


def module_source(module: str) -> str:
    """A module's code without `from __future__`, `from pairvc...` imports, __all__ and __main__."""
    text = (ROOT / "src" / "pairvc" / f"{module}.py").read_text()
    lines = text.splitlines()
    drop: set[int] = set()
    tree = ast.parse(text)
    for node in tree.body:
        is_local_import = isinstance(node, ast.ImportFrom) and (
            node.module == "__future__" or (node.module or "").startswith("pairvc")
        )
        is_all = isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "__all__"
        is_main = isinstance(node, ast.If) and "__main__" in ast.unparse(node.test)
        is_docstring = node is tree.body[0] and isinstance(node, ast.Expr)
        if is_local_import or is_all or is_main or is_docstring:
            drop.update(range(node.lineno - 1, node.end_lineno))
    kept = [line for i, line in enumerate(lines) if i not in drop]
    return "\n".join(kept).strip() + "\n"


def module_doc(module: str) -> str:
    """The module docstring, to show above its code."""
    text = (ROOT / "src" / "pairvc" / f"{module}.py").read_text()
    return ast.get_docstring(ast.parse(text)) or ""


cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


# ---------------------------------------------------------------------------
md("""
# tiny-pairvc: training notebook (Kaggle & Colab)

Train **PairVC**, a tiny real-time voice converter, on a **free Kaggle or Colab
GPU**. PairVC learns from **paired audio**: each source clip comes with the same
speech already converted to a target voice by a big (slow) teacher model, and
PairVC learns to reproduce it while staying small and causal enough to run
live, 20 ms at a time.

This notebook is **self-contained**: all the code (config, data, model, losses,
discriminators, training loop) is copied in from the `tiny-pairvc` repo, with
the **same default configuration**. The checkpoints it writes work with the
repo's `pairvc-convert`, `pairvc-export` and `pairvc-train --resume`.

| | Kaggle | Colab |
|---|---|---|
| GPU | *Settings → Accelerator → GPU T4 x2* (or P100) | *Runtime → Change runtime type → T4 GPU* |
| Your data | *Add Input → Datasets → your paired dataset* | Google Drive, or upload a zip |
| Session limit | 12 h per session, 30 h GPU / week | ~12 h, can disconnect when idle |
| Checkpoints kept in | `/kaggle/working` (saved with "Save Version") | Google Drive if `USE_GOOGLE_DRIVE = True` |

**No data yet?** Set `USE_DIGITS_DATA = True`: the notebook builds a small
*real speech* paired dataset (spoken digits, with a pitch-shift "teacher": the
repo's README results). Or `USE_TOY_DATA = True` / `SMOKE_TEST = True`: a tiny
synthetic dataset so every cell can be tried quickly.

**Sections:** 1. Install · 2. Config · 3. Data · 4. Model · 5. Losses &
discriminators · 6. Training code · 7. Live dashboard · 8. Train · 9. Results
(curves, listen, streaming speed, objective metrics: MCD, F0, DNSMOS) · 10. Download the trained model
""")

# ---------------------------------------------------------------------------
md("""
## 1. Install

Kaggle and Colab already have PyTorch, NumPy, SciPy and matplotlib; this makes
sure `soundfile` (reads/writes .wav) is there too.
""")
code("""
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "soundfile>=0.12"])

import torch

print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
""")

# ---------------------------------------------------------------------------
md(f"""
## 2. Config

### 2.1 Config classes (same as `src/pairvc/config.py`)

{module_doc("config")}
""")
code(module_source("config"))

md("""
### 2.2 Settings for this run

**Where is your data?** A folder with `train/` and `val/` inside, each holding
`<id>_original.wav` + `<id>_converted_<k>.wav` files (see the repo's
`docs/data.md`). On Kaggle, upload it as a Dataset and add it as input: the
cell below finds it under `/kaggle/input` by itself. On Colab, put it on Google
Drive (`USE_GOOGLE_DRIVE = True`) and set `DATA_DIR`, or upload a zip and unzip
it to `/content/data`.

Only two things adapt to the GPU, without changing what is learned:

- **batch size**: from the GPU memory (training needs ~0.8 GB of activations
  per second of audio in the batch; gradient checkpointing cuts the generator's
  share by ~10x for ~30% more time);
- **precision**: bf16 on GPUs with native support (A100, H100, L4, RTX 30xx/40xx),
  fp32 on older ones (T4, P100). fp16 is not used: GAN losses overflow it.
""")
code("""
import dataclasses
import os
from pathlib import Path

SMOKE_TEST = False          # True = tiny quick check of the whole notebook (toy data, tiny model)
USE_TOY_DATA = False        # True = synthetic paired data instead of yours
USE_DIGITS_DATA = False     # True = real spoken digits + a pitch-shift teacher (downloads 16 MB)
USE_GOOGLE_DRIVE = False    # Colab only: data + checkpoints on Drive (survive disconnects)
DATA_DIR = None             # None = find it automatically (Kaggle input, /content/data, ./data)
RESUME = True               # continue from last.pt in OUTPUT_DIR if it exists
TIME_BUDGET_HOURS = 11.0    # stop cleanly (after saving) before the 12 h session limit
N_SPEAKERS = None           # None = number of target voices found in the data

ON_KAGGLE = Path("/kaggle/working").exists()
ON_COLAB = "COLAB_RELEASE_TAG" in os.environ  # set by every Colab runtime
if ON_COLAB and USE_GOOGLE_DRIVE:
    from google.colab import drive

    drive.mount("/content/drive")
    OUTPUT_DIR = Path("/content/drive/MyDrive/tiny-pairvc/run")
elif ON_KAGGLE:
    OUTPUT_DIR = Path("/kaggle/working/checkpoints/run")
elif ON_COLAB:
    OUTPUT_DIR = Path("/content/checkpoints/run")
else:
    OUTPUT_DIR = Path("checkpoints/notebook-run")


def looks_like_paired_data(path: Path) -> bool:
    return (path / "train").is_dir() and any((path / "train").glob("*_original.wav"))


def find_data_dir() -> Path | None:
    \"\"\"First folder containing train/<id>_original.wav, in the usual places.\"\"\"
    roots = [Path("/kaggle/input"), Path("/content/data"), Path("/content/drive/MyDrive"), Path("data")]
    for root in roots:
        if not root.exists():
            continue
        candidates = [root, *sorted(p for p in root.rglob("*") if p.is_dir() and p.name != "train")]
        for path in candidates[:500]:
            if looks_like_paired_data(path):
                return path
    return None


if SMOKE_TEST or USE_TOY_DATA or USE_DIGITS_DATA:
    DATA_DIR = None  # written in section 3, once write_toy_dataset / write_digits_dataset are defined
elif DATA_DIR is None:
    DATA_DIR = find_data_dir()
    if DATA_DIR is None:
        raise FileNotFoundError(
            "No paired data found. Add your dataset as input (Kaggle) / set DATA_DIR, "
            "or set USE_TOY_DATA = True to try the notebook on synthetic data."
        )
print("data:", DATA_DIR or ("spoken digits" if USE_DIGITS_DATA and not SMOKE_TEST else "toy data (synthetic)"))

# --- device, precision, batch size ------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_config = DataConfig(data_dir=Path(DATA_DIR or "toy_data"))
model_config = ModelConfig()
disc_config = DiscConfig()

if device.type == "cuda":
    props = torch.cuda.get_device_properties(0)
    gpu_gb = props.total_memory / 1e9
    precision = "bf16" if props.major >= 8 else "fp32"  # native bf16 needs compute capability 8.x+
    # ~0.8 GB per second of audio in the batch; checkpointing below 40 GB. Lower it if "out of memory".
    batch_size = 32 if gpu_gb >= 70 else 16 if gpu_gb >= 14 else 8
    grad_checkpointing = gpu_gb < 40
    print(f"GPU: {props.name} ({gpu_gb:.0f} GB), compute capability {props.major}.{props.minor}")
else:
    precision, batch_size, grad_checkpointing = "fp32", 4, False
    print("No GPU: enable one in the notebook settings (CPU is fine only for SMOKE_TEST).")

train_config = TrainConfig(
    output_dir=OUTPUT_DIR,
    batch_size=batch_size,
    precision=precision,
    grad_checkpointing=grad_checkpointing,
    num_workers=min(4, os.cpu_count() or 1),
    device=device.type,
    log_every=50,
    eval_every=1000,
)
if USE_DIGITS_DATA and not SMOKE_TEST:
    data_config = dataclasses.replace(data_config, segment_samples=8000)  # a digit lasts ~0.5 s

if SMOKE_TEST:
    OUTPUT_DIR = OUTPUT_DIR.parent / "smoke-test"  # never mixed with a real run
    data_config = dataclasses.replace(data_config, segment_samples=3200)
    model_config = ModelConfig(channels=8, latent_dim=16, speaker_dim=16, residual_units=2)
    disc_config = DiscConfig(kinds=("resolution",))
    train_config = dataclasses.replace(
        train_config, output_dir=OUTPUT_DIR, max_steps=40, batch_size=4, log_every=2,
        eval_every=20, num_workers=0, g_lr=1e-3,
    )
    TIME_BUDGET_HOURS = 0.5
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for name, cfg in [("DataConfig", data_config), ("ModelConfig", model_config),
                  ("DiscConfig", disc_config), ("TrainConfig", train_config)]:
    print(f"\\n{name}:")
    for key, value in dataclasses.asdict(cfg).items():
        print(f"  {key:20} {value}")
""")

# ---------------------------------------------------------------------------
md(f"""
## 3. Data

### 3.1 Dataset code (same as `src/pairvc/data.py`)

{module_doc("data")}
""")
code(module_source("data"))
md(f"""
### 3.2 Real speech without a teacher model (same as `src/pairvc/digits.py`)

{module_doc("digits")}
""")
code(module_source("digits"))
code("""
if DATA_DIR is None and USE_DIGITS_DATA and not SMOKE_TEST:
    DATA_DIR = write_digits_dataset(Path("digits_data"))
    data_config = dataclasses.replace(data_config, data_dir=DATA_DIR)
elif DATA_DIR is None:  # toy data requested
    DATA_DIR = write_toy_dataset(Path("toy_data"), n_clips=64 if not SMOKE_TEST else 8, seconds=1.0)
    data_config = dataclasses.replace(data_config, data_dir=DATA_DIR)

root = Path(data_config.data_dir)
train_dataset = PairedVoiceDataset(
    root / data_config.train_split, data_config.sample_rate, data_config.segment_samples,
    train=True, limit=train_config.train_limit,
)
val_dataset = PairedVoiceDataset(
    root / data_config.val_split, data_config.sample_rate, data_config.segment_samples,
    limit=train_config.val_limit,
)
speakers = train_dataset.speakers
model_config = dataclasses.replace(model_config, n_speakers=N_SPEAKERS or max(speakers) + 1)
print(f"train: {len(train_dataset):,} pairs | validation: {len(val_dataset):,} pairs | "
      f"target voices: {speakers} -> n_speakers = {model_config.n_speakers}")
""")

md("""
### 3.3 Look at (and listen to) one pair
The source on top, the teacher's conversion below: the model learns to turn
the first into the second.
""")
code("""
import matplotlib.pyplot as plt
from IPython.display import Audio, Markdown, display


def show_spectrogram(audio: torch.Tensor, ax, title: str, sample_rate: int) -> None:
    spec = torch.stft(audio, 512, 128, window=torch.hann_window(512), return_complex=True).abs()
    ax.imshow((spec + 1e-5).log10().numpy(), origin="lower", aspect="auto",
              extent=[0, len(audio) / sample_rate, 0, sample_rate / 2000])
    ax.set_title(title)
    ax.set_ylabel("kHz")


pair = val_dataset.pairs[0]
source, target = load_audio(pair.source, data_config.sample_rate), load_audio(pair.target, data_config.sample_rate)
fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
show_spectrogram(source, axes[0], f"source: {pair.source.name}", data_config.sample_rate)
show_spectrogram(target, axes[1], f"teacher, voice {pair.speaker}: {pair.target.name}", data_config.sample_rate)
axes[1].set_xlabel("seconds")
plt.tight_layout()
plt.show()
display(Audio(source.numpy(), rate=data_config.sample_rate), Audio(target.numpy(), rate=data_config.sample_rate))
""")

# ---------------------------------------------------------------------------
md(f"""
## 4. Model

### 4.1 PairVC code (same as `src/pairvc/model.py`)

{module_doc("model")}
""")
code(module_source("model"))
md("""
### 4.2 Build it and check streaming
Converting a clip chunk by chunk (as a live app does) must give the same audio
as converting it in one go.
""")
code("""
torch.manual_seed(train_config.seed)
generator = PairVC(model_config)
print(f"generator: {sum(p.numel() for p in generator.parameters()) / 1e6:.2f}M parameters, "
      f"hop {model_config.hop_length} samples, lookahead {1000 * model_config.lookahead_samples / data_config.sample_rate:.0f} ms")

with torch.no_grad():
    x, spk = 0.1 * torch.randn(1, model_config.hop_length * 20), torch.tensor([0])
    full = generator.eval()(x, spk)
    states, chunks = generator.init_state(1), []
    for start in range(0, x.shape[-1], model_config.hop_length):
        out, states = generator.stream_step(x[:, start:start + model_config.hop_length], spk, states)
        chunks.append(out)
    error = (torch.cat(chunks, -1) - full).abs().max().item()
generator.train()
print(f"max |streaming - whole clip| = {error:.1e} (float rounding only)")
""")

# ---------------------------------------------------------------------------
md(f"""
## 5. Losses & discriminators

### 5.1 Losses (same as `src/pairvc/losses.py`)

{module_doc("losses")}
""")
code(module_source("losses"))
md(f"""
### 5.2 Discriminators (same as `src/pairvc/discriminators.py`)

{module_doc("discriminators")}
""")
code(module_source("discriminators"))
code("""
discriminator = Discriminators(disc_config)
print(f"discriminators {disc_config.kinds}: {sum(p.numel() for p in discriminator.parameters()) / 1e6:.2f}M parameters "
      "(training only, not needed to convert)")
""")

# ---------------------------------------------------------------------------
md(f"""
## 6. Training code (same as `src/pairvc/train.py`)

{module_doc("train")}

The command-line part is included but not used: section 8 calls `train()`
directly with the objects built above.
""")
code(module_source("train"))

# ---------------------------------------------------------------------------
md("""
## 7. Live dashboard

- **mel loss** (the one to watch): distance to the teacher's audio in log-mel
  space, train and validation. Lower is better.
- **adversarial / feature matching**: the GAN terms. They do not go steadily
  down: generator and discriminators compete. Worry only if the discriminator
  loss collapses to ~0 for a long time (it wins every time).
- **discriminator real / fake**: (1 - D(real))² and D(fake)²; both should stay
  away from 0.
- **gradient norms**, **learning rate**, **speed** (seconds of audio trained per second).

The dashboard redraws every `log_every` steps; metrics are saved to
`metrics.json` next to the checkpoints, so the curves survive a restart.
""")
code('''
import json
import time

from IPython.display import clear_output

METRICS_FILE = OUTPUT_DIR / "metrics.json"
history = {"train": [], "val": []}
if RESUME and METRICS_FILE.exists() and (OUTPUT_DIR / LAST_CHECKPOINT).exists():
    history = json.loads(METRICS_FILE.read_text())


class TimeBudgetReached(Exception):
    """Raised by the callback after a validation (= after saving) once the time is up."""


def draw_dashboard() -> None:
    train, val = history["train"], history["val"]
    clear_output(wait=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    (ax_mel, ax_gan, ax_d), (ax_grad, ax_lr, ax_speed) = axes

    def series(rows, key):
        return [r["step"] for r in rows], [r[key] for r in rows]

    ax_mel.plot(*series(train, "mel"), label="train", alpha=0.8)
    ax_mel.plot(*series(val, "mel"), "o-", label="validation")
    ax_mel.set_title("mel loss vs teacher (lower is better)")
    ax_gan.plot(*series(train, "adv"), label="adversarial")
    ax_gan.plot(*series(train, "fm"), label="feature matching")
    ax_gan.set_title("generator GAN terms")
    ax_d.plot(*series(train, "d_real"), label="real term")
    ax_d.plot(*series(train, "d_fake"), label="fake term")
    ax_d.set_title("discriminator")
    ax_grad.plot(*series(train, "g_grad"), label="generator")
    ax_grad.plot(*series(train, "d_grad"), label="discriminator")
    ax_grad.set_yscale("log")
    ax_grad.set_title("gradient norm")
    ax_lr.plot(*series(train, "lr_g"), label="generator")
    ax_lr.plot(*series(train, "lr_d"), label="discriminator")
    ax_lr.set_title("learning rate")
    ax_speed.plot(*series(train, "audio_sec_per_sec"), color="tab:green")
    ax_speed.set_title("speed (audio seconds / second)")
    for ax in axes.flat:
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
    fig.tight_layout()
    plt.show()
    if train:
        last = train[-1]
        print(f"step {last['step']:,}/{train_config.max_steps:,} (epoch {last['epoch']}) | "
              f"train mel {last['mel']:.4f} | adv {last['adv']:.3f} | fm {last['fm']:.3f}")
    if val:
        best = min(val, key=lambda r: r["mel"])
        print(f"last validation mel {val[-1]['mel']:.4f} | best {best['mel']:.4f} at step {best['step']:,}")
        print("latest audio sample (source | PairVC | teacher):")
        display(Audio(val[-1]["sample_path"]))


start_time = time.perf_counter()


def on_metrics(event: dict) -> None:
    history[event["kind"]].append(event)
    if event["kind"] == "val":
        METRICS_FILE.write_text(json.dumps(history))
        if (time.perf_counter() - start_time) / 3600 > TIME_BUDGET_HOURS:
            raise TimeBudgetReached
    draw_dashboard()
''')

# ---------------------------------------------------------------------------
md("""
## 8. Train

Same loop as the repo's `pairvc-train`. Every `eval_every` steps it validates,
writes an audio sample to `samples/` and saves `last.pt`, plus `best.pt` when
the validation mel loss improves.

**Interrupted, or hit the session limit?** Run the notebook again with
`RESUME = True`: it continues from `last.pt` (same step, same data order).
It also stops by itself after `TIME_BUDGET_HOURS`, right after a save.
On Kaggle, to continue in a *new* session, add the previous version's output
as input data and copy `checkpoints/run` back to `/kaggle/working` first.
""")
code("""
import logging

logging.basicConfig(level=logging.WARNING)  # the dashboard replaces the log lines

last_path = OUTPUT_DIR / LAST_CHECKPOINT
if RESUME and last_path.exists():
    train_config = dataclasses.replace(train_config, resume=last_path)
    print("resuming from", last_path)

train_loader, val_loader = build_dataloaders(train_dataset, val_dataset, train_config, device)
start_time = time.perf_counter()
try:
    state = train(
        generator, discriminator, train_loader, val_loader,
        data_config, model_config, disc_config, train_config, callback=on_metrics,
    )
    draw_dashboard()
    print(f"Training finished. Best validation mel {state.best_val_mel:.4f}")
except TimeBudgetReached:
    draw_dashboard()
    print(f"Time budget of {TIME_BUDGET_HOURS} h reached; {last_path} is saved. Run the notebook again to continue.")
""")

# ---------------------------------------------------------------------------
md("""
## 9. Results

### 9.1 Final curves
Saved as `training_curves.png` next to the checkpoints.
""")
code("""
curves_path = OUTPUT_DIR / "training_curves.png"
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot([r["step"] for r in history["train"]], [r["mel"] for r in history["train"]], label="train")
ax.plot([r["step"] for r in history["val"]], [r["mel"] for r in history["val"]], "o-", label="validation")
ax.set_title("mel loss vs teacher")
ax.set_xlabel("step")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(curves_path, dpi=120)
plt.show()
print("saved", curves_path)
""")

md("""
### 9.2 Listen: convert validation clips with the best checkpoint
Loads `best.pt` exactly as the repo's `pairvc-convert` does, converts a few
validation sources to every target voice, and plays source, PairVC and teacher.
""")
code("""
best = torch.load(OUTPUT_DIR / BEST_CHECKPOINT, map_location="cpu", weights_only=True)
best_model = PairVC(ModelConfig(**best["model_config"]))
best_model.load_state_dict(best["generator"])
best_model.to(device).eval()
print(f"best.pt from step {best['step']:,} (validation mel {best['best_val_mel']:.4f})")

seen = set()
for pair in val_dataset.pairs:
    if pair.source in seen or len(seen) >= 3:
        continue
    seen.add(pair.source)
    source = load_audio(pair.source, data_config.sample_rate)
    print(f"\\n=== {pair.source.name} ===\\nsource:")
    display(Audio(source.numpy(), rate=data_config.sample_rate))
    for other in val_dataset.pairs:
        if other.source != pair.source:
            continue
        converted = best_model.convert(source[None].to(device), torch.tensor([other.speaker], device=device))[0]
        teacher = load_audio(other.target, data_config.sample_rate)
        print(f"voice {other.speaker}: PairVC, then teacher")
        display(Audio(converted.clamp(-1, 1).cpu().numpy(), rate=data_config.sample_rate),
                Audio(teacher.numpy(), rate=data_config.sample_rate))
""")

md("""
### 9.3 Real-time check (CPU, 20 ms chunks)
Live conversion runs on a CPU, one 20 ms chunk at a time. **RTF** = compute
time / audio time: below 1 it keeps up with real time. (Colab/Kaggle CPUs are
slow and shared: a laptop is usually faster.)
""")
code("""
cpu_model = PairVC(best_model.config)
cpu_model.load_state_dict(best["generator"])
cpu_model.eval()
torch.set_num_threads(1)  # a live app usually gets one core
audio = load_audio(val_dataset.pairs[0].source, data_config.sample_rate)[: 5 * data_config.sample_rate]
chunk = model_config.hop_length  # 320 samples = 20 ms
states, spk, times = cpu_model.init_state(1), torch.tensor([0]), []
with torch.inference_mode():
    for start in range(0, len(audio) - chunk + 1, chunk):
        t0 = time.perf_counter()
        _, states = cpu_model.stream_step(audio[None, start:start + chunk], spk, states)
        times.append(time.perf_counter() - t0)
torch.set_num_threads(os.cpu_count() or 1)
rtf = sum(times) / (len(times) * chunk / data_config.sample_rate)
print(f"RTF {rtf:.3f} on 1 CPU thread | mean {1000 * sum(times) / len(times):.1f} ms, "
      f"worst {1000 * max(times):.1f} ms per 20 ms chunk | latency "
      f"{1000 * (chunk + model_config.lookahead_samples) / data_config.sample_rate:.0f} ms (chunk + lookahead)")
""")

md(f"""
### 9.4 Objective metrics (same as `src/pairvc/metrics.py` and `src/pairvc/evaluate.py`)

{module_doc("metrics")}

{module_doc("evaluate")}
""")
code(module_source("metrics"))
code(module_source("evaluate"))
code("""
subprocess.call([sys.executable, "-m", "pip", "install", "-q", "onnxruntime"])  # for DNSMOS
try:
    dnsmos = DNSMOS()
except Exception as error:  # no internet / no onnxruntime: the other metrics still work
    print("DNSMOS unavailable:", error)
    dnsmos = None

val_dir = Path(data_config.data_dir) / data_config.val_split
torch.manual_seed(train_config.seed)
untrained_model = PairVC(best_model.config).to(device)
before = evaluate_split(untrained_model, val_dir, data_config.sample_rate, limit=100, dnsmos=dnsmos)
after = evaluate_split(best_model, val_dir, data_config.sample_rate, limit=100, dnsmos=dnsmos, baselines=False)
results = {"no conversion": before["no conversion"], "PairVC, untrained": before["PairVC"],
           "PairVC, trained": after["PairVC"], "teacher": before["teacher"]}
(OUTPUT_DIR / "eval_results.json").write_text(json.dumps(results, indent=2))
display(Markdown(results_table(results)))
""")

# ---------------------------------------------------------------------------
md("""
## 10. Download the trained model

Writes `pairvc_generator.pt` (only what inference needs, ~38 MB instead of
the full `best.pt` with discriminators and optimizer states) and zips it with
the metrics and curves. Use it on your computer with the repo:

```bash
unzip tiny-pairvc-run.zip -d checkpoints/run
pairvc-convert --checkpoint checkpoints/run/pairvc_generator.pt --input me.wav --speaker 0
pairvc-export  --checkpoint checkpoints/run/pairvc_generator.pt --output pairvc.onnx --verify
```
""")
code("""
import shutil
import tempfile

generator_path = OUTPUT_DIR / "pairvc_generator.pt"
torch.save({k: best[k] for k in ("generator", "model_config", "data_config", "step")}, generator_path)
with tempfile.TemporaryDirectory() as tmp:
    bundle = Path(tmp) / "bundle"
    bundle.mkdir()
    for name in ["pairvc_generator.pt", "metrics.json", "eval_results.json", "training_curves.png"]:
        if (OUTPUT_DIR / name).exists():
            shutil.copy(OUTPUT_DIR / name, bundle / name)
    archive = shutil.make_archive(str(OUTPUT_DIR.parent / "tiny-pairvc-run"), "zip", bundle)
print("created", archive)

if ON_COLAB:
    from google.colab import files

    files.download(archive)
elif ON_KAGGLE:
    print("Kaggle: click 'Save Version', then download it from the notebook's Output tab.")
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "accelerator": "GPU",
    "colab": {"provenance": [], "gpuType": "T4"},
    "kaggle": {"accelerator": "gpu", "isInternetEnabled": True},
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
nbf.validate(nb)
nbf.write(nb, OUT)
print("wrote", OUT, len(cells), "cells")
