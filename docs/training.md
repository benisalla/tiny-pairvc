# Training guide

All training goes through one command, `pairvc-train` (same as
`python -m pairvc.train`). It runs on CPU, one GPU, several GPUs, or several
machines: only the way you **launch** it changes.

- [1. Install](#1-install)
- [2. First: the overfit sanity check](#2-first-the-overfit-sanity-check)
- [3. One GPU](#3-one-gpu)
- [4. Several GPUs on one machine](#4-several-gpus-on-one-machine)
- [5. Several machines (multi-node)](#5-several-machines-multi-node)
- [6. Resume a run](#6-resume-a-run)
- [7. What gets saved](#7-what-gets-saved)
- [8. Reading the logs](#8-reading-the-logs)
- [9. GPU memory](#9-gpu-memory)
- [10. All flags](#10-all-flags)
- [11. Tips](#11-tips)

---

> **No GPU?** Use the self-contained notebook
> [`notebooks/train_colab_kaggle.ipynb`](../notebooks/train_colab_kaggle.ipynb):
> upload it to Kaggle or Colab, add your paired dataset, pick a GPU, *Run all*.
> Same config and metrics as `pairvc-train`, plus live graphs and audio players.

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Install PyTorch **first**, for your hardware (pick the command for your CUDA
version on <https://pytorch.org/get-started/locally/>):

```bash
# NVIDIA GPU (example: CUDA 12.8)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# CPU only (small download, enough for tests, the overfit check and inference)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then the project itself:

```bash
pip install -e '.[dev]'
pytest            # should end with "N passed"
```

Your data goes in a folder with `train/` and `val/` inside; see
[data.md](data.md) for the layout.

## 2. First: the overfit sanity check

Before spending GPU hours, check the model can **memorise a few pairs**. If it
cannot, something is broken (data paths, alignment, labels) and a long run
would be wasted.

```bash
pairvc-train --data-dir data --overfit 8 --max-steps 2000 \
             --eval-every 200 --log-every 50 --output-dir checkpoints/overfit
```

`--overfit 8` trains *and validates* on the same 8 training pairs, always on
the start of each clip (no random crop). The validation `mel` should fall
steadily, and the audio samples in `checkpoints/overfit/samples/` should get
closer and closer to the teacher (the third part of each file). On a GPU this
takes a few minutes.

## 3. One GPU

```bash
pairvc-train --data-dir data --output-dir checkpoints/run
```

Defaults: 1 s clips, batch 16, both learning rates 1e-4, AdamW betas
(0.8, 0.99), LR × 0.9999996 per step, loss weights mel 15 / feature
matching 2 / adversarial 1, the `time` discriminator, bf16 on GPUs that
support it natively (else fp32). It runs for 500k steps; stop it whenever the
samples sound good (the best checkpoint is kept).

Long runs on a remote machine: start them in `tmux` (or `screen`) so they
survive a closed SSH session:

```bash
tmux new -s pairvc
pairvc-train --data-dir /data/paired --output-dir checkpoints/run 2>&1 | tee train.log
# detach: Ctrl+b then d        re-attach: tmux attach -t pairvc
```

## 4. Several GPUs on one machine

```bash
torchrun --standalone --nproc_per_node=4 -m pairvc.train \
    --data-dir data --output-dir checkpoints/run
```

torchrun starts one process per GPU; each one reads a different slice of the
data (`DistributedSampler`) and both networks are wrapped in
`DistributedDataParallel`, which averages gradients after every backward.
`--batch-size` is **per GPU**: 4 GPUs × 16 = 64 clips per step. Only rank 0
logs and writes checkpoints. If you raise the total batch a lot, you may raise
both learning rates a little (e.g. ×1.5–2 for ×4 batch).

## 5. Several machines (multi-node)

Same command on every machine, with the address of machine 0:

```bash
# on each of 2 machines with 8 GPUs; NODE_RANK = 0 on the first, 1 on the second
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=$NODE_RANK \
    --master_addr=10.0.0.1 --master_port=29500 \
    -m pairvc.train --data-dir /shared/data --output-dir /shared/checkpoints/run
```

The data and the output directory must be visible from every machine (shared
filesystem). With SLURM:

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
srun torchrun --nnodes=$SLURM_NNODES --nproc_per_node=8 \
    --rdzv_backend=c10d --rdzv_endpoint=$MASTER:29500 \
    -m pairvc.train --data-dir /shared/data --output-dir /shared/checkpoints/run
```

## 6. Resume a run

```bash
pairvc-train --data-dir data --output-dir checkpoints/run --resume checkpoints/run/last.pt
```

Weights of both networks, both optimizers, both LR schedules, the step counter
and every random generator are restored; the run continues with the same data
order. To train *longer* than planned, resume with a larger `--max-steps`.
Keep the other flags the same (the model shape must match the checkpoint).

## 7. What gets saved

```
checkpoints/run/
  last.pt          everything, written at every validation (to resume)
  best.pt          same, at the lowest validation mel loss so far
  history.jsonl    every logged train / val metric, one JSON object per line (for plots)
  samples/
    step_0001000.wav   source | PairVC | teacher, 0.5 s silence between them
```

A checkpoint holds the generator, the discriminators, both optimizer states
and the four configs (~230 MB with the default discriminator; ~690 MB with
all four). Files are written atomically: a crash never leaves a half-written
one. For inference only the generator is needed:

```python
from pairvc.convert import save_generator_only

save_generator_only("checkpoints/run/best.pt", "pairvc_generator.pt")  # ~38 MB
```

## 8. Reading the logs

```
train | epoch 3 | step 4200/500000 | mel 6.8123 | adv 0.412 | fm 0.231 | d 0.398 (real 0.201 fake 0.197) | grad g 92.1 d 3.4 | lr 9.98e-05 | 210.3 audio s/s
val | step 5000 | mel 6.9541 | l1 0.0412
```

| Field | Meaning |
|-------|---------|
| `mel` | multi-scale log-mel L1 to the teacher (sum of 7 scales). **The number to watch**: lower = closer to the teacher |
| `adv` | generator's adversarial loss, `(1 − D(fake))²`; ~0.3–0.6 is a healthy balance |
| `fm` | feature-matching loss |
| `d (real, fake)` | discriminator loss and its two halves. If `fake` → 0 and `adv` → 1 for thousands of steps, D wins too easily: lower `--d-lr` or try other discriminators |
| `grad g / d` | gradient norms before clipping (`--grad-clip` to cap spikes) |
| `audio s/s` | seconds of training audio processed per second (all GPUs) |
| `val mel / l1` | same mel loss on validation clips, and the mean absolute sample error |

GAN losses do not go down steadily; that is normal. Judge progress by the
validation mel and **by listening to `samples/`**.

## 9. GPU memory

Activation memory grows with **batch size × clip length**. Measured: about
0.2 GB per second of audio for the generator and 0.3 GB per discriminator pass
with `time`, so roughly **0.8 GB per second of audio in the batch** in total.

| GPU memory | Suggested settings |
|------------|--------------------|
| 16 GB (T4, V100-16) | `--batch-size 16 --grad-checkpointing` (default 1 s clips) |
| 24 GB (L4, RTX 3090/4090) | `--batch-size 16` |
| 40–80 GB (A100, H100) | `--batch-size 32` to `64` |

`--grad-checkpointing` recomputes the generator's activations in the backward
pass instead of storing them: ~10× less generator memory, ~30% slower steps.
Longer clips (`--segment-samples 32000`) give the discriminators more context
but cost memory linearly; keep it a multiple of 320.

## 10. All flags

| Flag | Default | Meaning |
|------|---------|---------|
| **data** | | |
| `--data-dir` | `data` | folder containing the splits |
| `--train-split` / `--val-split` | `train` / `val` | sub-folders |
| `--sample-rate` | 16000 | audio is resampled to this |
| `--segment-samples` | 16000 | training clip length (multiple of 320) |
| **model** | | |
| `--n-speakers` | 2 | number of target voices (> highest k in the data) |
| `--channels` | 32 | first-layer width (doubles at each stride) |
| `--latent-dim` | 64 | size of one latent frame |
| `--residual-units` | 3 | per block (dilations 1, 3, 9, …) |
| `--lookahead-frames` | 2 | future frames (20 ms each) the model may hear |
| `--discriminators` | `time` | any of `time period resolution band` |
| **training** | | |
| `--max-steps` | 500000 | total GAN steps |
| `--batch-size` | 16 | clips per step, **per GPU** |
| `--g-lr` / `--d-lr` | 1e-4 / 1e-4 | learning rates |
| `--betas` | 0.8 0.99 | AdamW betas |
| `--weight-decay` | 0 | AdamW weight decay |
| `--lr-decay` | 0.9999996 | LR multiplier per step |
| `--grad-clip` | 0 (off) | max gradient norm |
| `--lambda-mel` / `--lambda-fm` / `--lambda-adv` | 15 / 2 / 1 | loss weights |
| `--gan-start-step` | 0 | mel loss only before this step (discriminators skipped, ~3-5x faster steps): a warm-up that helps short / CPU runs. The examples use 2500 |
| `--precision` | auto | `auto` (bf16 if native, else fp32), `fp32`, `bf16` |
| `--grad-checkpointing` | off | less memory, slower |
| `--num-workers` | 4 | DataLoader processes (audio decoding) |
| `--train-limit` / `--val-limit` | all | use only the first N pairs |
| `--log-every` | 50 | steps between log lines |
| `--eval-every` | 1000 | steps between validation + checkpoint + audio sample |
| `--device` | auto | `auto`, `cpu`, `cuda`, `mps` |
| `--output-dir` | `checkpoints/run` | where checkpoints and samples go |
| `--resume` | none | checkpoint to continue from |
| `--seed` | 42 | random seed |
| `--overfit N` | off | sanity check: train and validate on the same N pairs |

## 11. Tips

- **Listen.** The mel loss tells you if the model gets closer to the teacher,
  not whether it sounds natural; the samples do.
- **Metallic / buzzy sound** after a long time: add discriminators
  (`--discriminators time period resolution`).
- **Words get mumbled**: more varied source data; check that teacher outputs
  are well aligned with the sources (wrong alignment = blurred targets).
- **Latency matters more than quality?** `--lookahead-frames 0` or `1`
  (must be the same at training and inference: it is stored in the checkpoint).
- **Data loading is the bottleneck** (GPU usage < 90%): raise `--num-workers`,
  store the data as 16 kHz mono wav on a local SSD.
