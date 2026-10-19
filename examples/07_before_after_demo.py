"""07 - Make the README's before/after visuals and audio from a training run.

From the run folder of example 02 (or any `pairvc-train` run) this writes:

- before_after.png    spectrograms: source | untrained | trained | teacher, for both voices
- training.gif        the validation sample improving over training (one frame per eval)
- training_curves.png train / validation mel loss and GAN losses over the steps
- audio/*.wav         a few held-out clips: source, untrained, trained, teacher

The untrained model is rebuilt with the training seed, so "before" is exactly
the network that training started from.

    python examples/07_before_after_demo.py [--run examples/output/run] [--out examples/output/demo]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from PIL import Image

from pairvc.convert import load_model
from pairvc.data import crop_pair, find_pairs, load_audio
from pairvc.metrics import f0_metrics, log_mel
from pairvc.model import PairVC
from pairvc.train import HISTORY_FILE, seed_everything

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=Path, default=Path("examples/output/run"))
parser.add_argument("--data-dir", type=Path, default=Path("examples/output/digits_data"))
parser.add_argument("--out", type=Path, default=Path("examples/output/demo"))
parser.add_argument("--seed", type=int, default=42, help="the training seed (pairvc-train --seed)")
parser.add_argument("--clips", type=int, default=3, help="held-out clips to save as audio")
args = parser.parse_args()

torch.set_num_threads(2)
args.out.mkdir(parents=True, exist_ok=True)
trained, sr = load_model(args.run / "best.pt")
seed_everything(args.seed)
untrained = PairVC(trained.config).eval()
pairs = find_pairs(args.data_dir / "val")


def mel_image(audio: np.ndarray) -> np.ndarray:
    return log_mel(audio, sr).T  # [mels, frames], low frequencies first


def show(ax: plt.Axes, audio: np.ndarray, title: str) -> None:
    ax.imshow(mel_image(audio), origin="lower", aspect="auto", cmap="magma", vmin=-9, vmax=2)
    pitch = f0_metrics(audio, audio, sr)["median_f0_hz"]
    ax.set_title(f"{title}\npitch {pitch:.0f} Hz", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


@torch.inference_mode()
def convert(model: PairVC, audio: torch.Tensor, speaker: int) -> np.ndarray:
    return model.convert(audio[None], torch.tensor([speaker]))[0].clamp(-1, 1).numpy()


# 1. Spectrograms before / after, both target voices, one held-out clip.
clip_id = pairs[0].source.name.removesuffix("_original.wav")
source = load_audio(pairs[0].source, sr)
fig, axes = plt.subplots(2, 4, figsize=(11, 4.6))
for k, row in enumerate(axes):
    target = load_audio(pairs[0].source.with_name(f"{clip_id}_converted_{k}.wav"), sr).numpy()
    show(row[0], source.numpy(), "source (unseen speaker)")
    show(row[1], convert(untrained, source, k), f"PairVC before training -> voice {k}")
    show(row[2], convert(trained, source, k), f"PairVC after training -> voice {k}")
    show(row[3], target, f"teacher, voice {k}")
fig.suptitle(f"'{clip_id}': log-mel spectrograms (time ->, frequency ^)", fontsize=10)
fig.tight_layout()
fig.savefig(args.out / "before_after.png", dpi=110)
plt.close(fig)

# 2. Training curves from history.jsonl.
events = [json.loads(line) for line in (args.run / HISTORY_FILE).read_text().splitlines()]
train_events = [e for e in events if e["kind"] == "train"]
val_events = [e for e in events if e["kind"] == "val"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.4))
ax1.plot([e["step"] for e in train_events], [e["mel"] for e in train_events], label="train")
ax1.plot(
    [e["step"] for e in val_events],
    [e["mel"] for e in val_events],
    "o-",
    ms=3,
    label="validation (unseen speaker)",
)
ax1.set(xlabel="step", ylabel="multi-scale mel loss", title="distance to the teacher")
ax1.legend()
for key, label in (("adv", "generator adversarial"), ("fm", "feature matching"), ("d_loss", "discriminator")):
    ax2.plot([e["step"] for e in train_events], [e[key] for e in train_events], label=label)
ax2.set(xlabel="step", title="GAN losses")
ax2.legend()
fig.tight_layout()
fig.savefig(args.out / "training_curves.png", dpi=110)
plt.close(fig)

# 3. GIF: the validation sample of every eval. train.py writes the first `segment_samples`
#    of the first validation pair as source | 0.5 s gap | PairVC | 0.5 s gap | teacher.
segment = torch.load(args.run / "best.pt", map_location="cpu", weights_only=True)["data_config"][
    "segment_samples"
]
source, teacher = crop_pair(load_audio(pairs[0].source, sr), load_audio(pairs[0].target, sr), segment, False)
frames = [(0, convert(untrained, source, pairs[0].speaker), None)]
for event in val_events:
    audio, _ = sf.read(args.run / "samples" / Path(event["sample_path"]).name)
    start = segment + sr // 2
    prediction = audio[start : start + segment - trained.config.lookahead_samples]
    frames.append((event["step"], np.pad(prediction, (0, segment - len(prediction))), event["mel"]))
teacher = teacher.numpy()
images = []
for step, prediction, val_mel in frames:
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.8), gridspec_kw={"width_ratios": [1, 1, 1, 1.2]})
    show(axes[0], source.numpy(), "source")
    show(axes[1], prediction, f"PairVC, step {step}")
    show(axes[2], teacher, f"teacher, voice {pairs[0].speaker}")
    axes[3].plot([e["step"] for e in val_events], [e["mel"] for e in val_events], color="0.6")
    if val_mel is not None:
        axes[3].plot([step], [val_mel], "o", color="crimson")
    axes[3].set(xlabel="step", title="validation mel loss")
    axes[3].tick_params(labelsize=7)
    fig.tight_layout()
    fig.canvas.draw()
    images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
    plt.close(fig)
durations = [1200] + [350] * (len(images) - 2) + [2500]  # linger on the first and last frames
images[0].save(args.out / "training.gif", save_all=True, append_images=images[1:], duration=durations, loop=0,
               optimize=True)  # fmt: skip

# 4. Audio of a few held-out clips (different digits), every system.
audio_dir = args.out / "audio"
audio_dir.mkdir(exist_ok=True)
seen = set()
for pair in pairs:
    clip_id = pair.source.name.removesuffix("_original.wav")
    if clip_id in seen or len(seen) == args.clips or clip_id[0] in {s[0] for s in seen}:
        continue  # one clip per digit
    seen.add(clip_id)
    source = load_audio(pair.source, sr)
    sf.write(audio_dir / f"{clip_id}_source.wav", source.numpy(), sr, subtype="PCM_16")
    for k in range(trained.config.n_speakers):
        teacher = load_audio(pair.source.with_name(f"{clip_id}_converted_{k}.wav"), sr).numpy()
        sf.write(
            audio_dir / f"{clip_id}_voice{k}_untrained.wav",
            convert(untrained, source, k),
            sr,
            subtype="PCM_16",
        )
        sf.write(
            audio_dir / f"{clip_id}_voice{k}_trained.wav", convert(trained, source, k), sr, subtype="PCM_16"
        )
        sf.write(audio_dir / f"{clip_id}_voice{k}_teacher.wav", teacher, sr, subtype="PCM_16")

for path in sorted(args.out.rglob("*")):
    if path.is_file():
        print(f"{path}  ({path.stat().st_size / 1e3:.0f} kB)")
