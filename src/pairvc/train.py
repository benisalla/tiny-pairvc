"""Train PairVC on paired (source, teacher-converted) audio.

The file is split into small pieces you can read top to bottom:

1. setup helpers   - seeding, device, mixed precision, multi-GPU (DDP) setup
2. data            - DataLoaders
3. optimisation    - two AdamW optimizers (generator, discriminators) + LR decay
4. one GAN step    - discriminator update, then generator update
5. evaluation      - validation mel loss, one audio sample to listen to
6. checkpoints     - save / load everything needed to resume
7. train()         - the training loop (takes ready-made objects, easy to test)
8. CLI             - main() builds everything from command-line flags

Run it with:  python -m pairvc.train --help

Multi-GPU / multi-node: launch the same command with torchrun, e.g.
    torchrun --nproc_per_node=4 -m pairvc.train ...
torchrun starts one process per GPU and sets RANK / WORLD_SIZE / LOCAL_RANK;
the code below picks those up and wraps both networks in DistributedDataParallel.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import os
import random
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from itertools import count, islice
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from pairvc.config import (
    DISCRIMINATOR_KINDS,
    DataConfig,
    DiscConfig,
    ModelConfig,
    TrainConfig,
    config_to_dict,
)
from pairvc.losses import (
    MultiScaleMelLoss,
    discriminator_loss,
    feature_matching_loss,
    generator_adversarial_loss,
)

logger = logging.getLogger(__name__)

LAST_CHECKPOINT = "last.pt"
BEST_CHECKPOINT = "best.pt"


# ---------------------------------------------------------------------------
# 1. Setup helpers
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    """Seed every random number generator we use."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds all CUDA devices


def resolve_device(name: str = "auto") -> torch.device:
    """Turn "auto" | "cpu" | "cuda" | "mps" into a torch.device."""
    # Under torchrun each process owns one GPU: cuda:LOCAL_RANK.
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if name == "cuda" or (name == "auto" and torch.cuda.is_available()):
        return torch.device("cuda", local_rank)
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_precision(precision: str, device: torch.device) -> torch.dtype | None:
    """Autocast dtype, or None for plain fp32.

    Only bf16 is offered: it has the range of fp32, so the GAN losses cannot
    overflow and no GradScaler is needed. GPUs without native bf16 (T4, P100,
    V100) train in fp32.
    """
    if precision == "fp32":
        return None
    if precision == "bf16":
        return torch.bfloat16
    if precision == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False):
            return torch.bfloat16
        return None
    raise ValueError(f"unknown precision: {precision!r}")


def autocast(device: torch.device, dtype: torch.dtype | None) -> AbstractContextManager:
    """Mixed-precision context; does nothing when dtype is None."""
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None)


@dataclass(frozen=True)
class DistInfo:
    """Who am I in a multi-process run? Defaults describe a single process."""

    rank: int = 0  # global id of this process (0 .. world_size-1)
    world_size: int = 1  # total number of processes (= total GPUs)
    local_rank: int = 0  # id of this process on its own machine

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        """Only the main process logs, writes audio samples and checkpoints."""
        return self.rank == 0


def setup_distributed(device_name: str = "auto") -> DistInfo:
    """Join the process group if we were launched by torchrun; no-op otherwise."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return DistInfo()
    local_rank = int(os.environ["LOCAL_RANK"])
    use_cuda = device_name in ("auto", "cuda") and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(local_rank)  # this process only talks to its own GPU
    # NCCL is the fast backend for GPUs; gloo works on CPU (handy for testing).
    dist.init_process_group(backend="nccl" if use_cuda else "gloo")
    return DistInfo(dist.get_rank(), dist.get_world_size(), local_rank)


def get_dist_info() -> DistInfo:
    """Current process-group state (single process if none was started)."""
    if not (dist.is_available() and dist.is_initialized()):
        return DistInfo()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return DistInfo(dist.get_rank(), dist.get_world_size(), local_rank)


# ---------------------------------------------------------------------------
# 2. Data
# ---------------------------------------------------------------------------


def seed_worker(worker_id: int) -> None:
    """Give each DataLoader worker its own python/numpy seed (random crops differ)."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_dataloaders(
    train_dataset: Dataset, val_dataset: Dataset, config: TrainConfig, device: torch.device
) -> tuple[DataLoader, DataLoader]:
    """Create the train and validation DataLoaders.

    With several processes, a DistributedSampler gives each one a different
    slice of the data, so batch_size is *per process* (per GPU).
    """
    info = get_dist_info()
    workers = config.num_workers
    common = {
        "batch_size": config.batch_size,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",  # faster CPU -> GPU copies
        "persistent_workers": workers > 0,  # do not restart workers every epoch
        "worker_init_fn": seed_worker if workers > 0 else None,
    }
    generator = torch.Generator()
    generator.manual_seed(config.seed + info.rank)  # + rank: different crops per process

    if info.enabled:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=config.seed, drop_last=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)
        train_loader = DataLoader(
            train_dataset, sampler=train_sampler, drop_last=True, generator=generator, **common
        )
        val_loader = DataLoader(val_dataset, sampler=val_sampler, drop_last=False, **common)
        return train_loader, val_loader

    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, generator=generator, **common)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader


def move_batch(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    """Send every tensor of the batch to the device."""
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


# ---------------------------------------------------------------------------
# 3. Optimisation
# ---------------------------------------------------------------------------


def build_optimizers(
    generator: nn.Module, discriminator: nn.Module, config: TrainConfig
) -> tuple[AdamW, AdamW]:
    """One AdamW per network: they are updated in turn, never together."""
    kwargs = {"betas": config.betas, "weight_decay": config.weight_decay}
    return (
        AdamW(generator.parameters(), lr=config.g_lr, **kwargs),
        AdamW(discriminator.parameters(), lr=config.d_lr, **kwargs),
    )


def build_schedulers(
    optim_g: Optimizer, optim_d: Optimizer, config: TrainConfig
) -> tuple[ExponentialLR, ExponentialLR]:
    """LR *= lr_decay after every step (0.9999996 -> x0.82 after 500k steps)."""
    return ExponentialLR(optim_g, config.lr_decay), ExponentialLR(optim_d, config.lr_decay)


def clip_gradients(model: nn.Module, max_norm: float) -> Tensor:
    """Total gradient norm (before clipping); clips only if max_norm > 0."""
    return nn.utils.clip_grad_norm_(model.parameters(), max_norm if max_norm > 0 else math.inf)


# ---------------------------------------------------------------------------
# 4. One GAN step
# ---------------------------------------------------------------------------


def align_with_lookahead(prediction: Tensor, target: Tensor, lookahead: int) -> tuple[Tensor, Tensor]:
    """Compare output sample t + lookahead with target sample t.

    The causal model may hear `lookahead` samples of future source audio before
    it must produce target sample t: a little latency, much better quality.
    """
    if lookahead == 0:
        return prediction, target
    return prediction[..., lookahead:], target[..., : target.shape[-1] - lookahead]


def gan_step(
    generator: nn.Module,
    discriminator: nn.Module,
    raw_discriminator: nn.Module,
    batch: dict[str, Tensor],
    mel_loss: MultiScaleMelLoss,
    optim_g: Optimizer,
    optim_d: Optimizer,
    config: TrainConfig,
    lookahead: int,
    amp_dtype: torch.dtype | None,
) -> dict[str, Tensor]:
    """One discriminator update then one generator update. Returns detached metrics.

    `discriminator` may be DDP-wrapped (its update is all-reduced);
    `raw_discriminator` is the same network unwrapped, used in the generator
    update with its weights frozen: there we only need gradients w.r.t. the
    *generated audio*, and going through DDP would wait for gradients that
    never come.
    """
    device = batch["source"].device
    with autocast(device, amp_dtype):
        prediction = generator(batch["source"], batch["speaker"])
    prediction, target = align_with_lookahead(prediction.float(), batch["target"], lookahead)

    # --- discriminators: real (teacher) -> 1, fake (PairVC) -> 0 ---
    with autocast(device, amp_dtype):
        real_scores, fake_scores, _, _ = discriminator(target, prediction.detach())
    d_loss, d_real, d_fake = discriminator_loss(real_scores, fake_scores)
    optim_d.zero_grad(set_to_none=True)
    d_loss.backward()
    d_grad = clip_gradients(raw_discriminator, config.grad_clip)
    optim_d.step()

    # --- generator: fool the discriminators + sound like the teacher ---
    raw_discriminator.requires_grad_(False)
    try:
        with autocast(device, amp_dtype):
            _, fake_scores, real_fmaps, fake_fmaps = raw_discriminator(target, prediction)
        adv = generator_adversarial_loss(fake_scores)
        fm = feature_matching_loss(real_fmaps, fake_fmaps)
        mel = mel_loss(prediction, target)
        g_loss = config.lambda_adv * adv + config.lambda_fm * fm + config.lambda_mel * mel
        optim_g.zero_grad(set_to_none=True)
        g_loss.backward()
    finally:
        raw_discriminator.requires_grad_(True)
    g_grad = clip_gradients(generator, config.grad_clip)
    optim_g.step()

    return {
        "d_loss": d_loss.detach(), "d_real": d_real, "d_fake": d_fake,
        "g_loss": g_loss.detach(), "adv": adv.detach(), "fm": fm.detach(), "mel": mel.detach(),
        "g_grad": g_grad.detach(), "d_grad": d_grad.detach(),
    }  # fmt: skip


# ---------------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------------


@torch.inference_mode()
def evaluate(
    generator: nn.Module,
    loader: DataLoader,
    mel_loss: MultiScaleMelLoss,
    lookahead: int,
    device: torch.device,
    amp_dtype: torch.dtype | None = None,
) -> dict[str, float]:
    """Mean mel loss and waveform L1 against the teacher, over a whole loader.

    The mel loss is the number to watch: lower = closer to the teacher.
    (Adversarial losses go up and down by design and say little on their own.)
    """
    was_training = generator.training
    generator.eval()
    sums = torch.zeros(3, device=device, dtype=torch.float64)  # mel, l1, n_clips
    for batch in loader:
        batch = move_batch(batch, device)
        with autocast(device, amp_dtype):
            prediction = generator(batch["source"], batch["speaker"])
        prediction, target = align_with_lookahead(prediction.float(), batch["target"], lookahead)
        n = target.shape[0]
        sums[0] += mel_loss(prediction, target).double() * n
        sums[1] += (prediction - target).abs().mean().double() * n
        sums[2] += n
    generator.train(was_training)
    if dist.is_initialized():
        dist.all_reduce(sums)  # every process saw a different slice: add them up
    n = sums[2].clamp(min=1)
    return {"mel": (sums[0] / n).item(), "l1": (sums[1] / n).item()}


@torch.inference_mode()
def save_audio_sample(
    generator: nn.Module, dataset: Dataset, path: Path, sample_rate: int, lookahead: int
) -> None:
    """Write source | PairVC | teacher for the first validation clip, one after the other."""
    was_training = generator.training
    generator.eval()
    item = dataset[0]
    device = next(generator.parameters()).device
    prediction = generator(item["source"][None].to(device), item["speaker"][None].to(device))[0]
    generator.train(was_training)
    prediction = prediction[lookahead:].float().cpu()  # time-align with the source
    gap = torch.zeros(sample_rate // 2)
    audio = torch.cat([item["source"], gap, prediction, gap, item["target"]]).clamp(-1, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.numpy(), sample_rate)


# ---------------------------------------------------------------------------
# 6. Checkpoints
# ---------------------------------------------------------------------------


@dataclass
class TrainState:
    """Counters that must survive a restart."""

    step: int = 0  # GAN steps done
    epoch: int = 0
    best_val_mel: float = math.inf


def rng_state() -> dict[str, Any]:
    """Snapshot every RNG so a resumed run continues the same random stream."""
    name, keys, pos, has_gauss, cached = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        # plain lists so torch.load(weights_only=True) can read them back
        "numpy": (name, keys.tolist(), pos, has_gauss, cached),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    """Restore what rng_state() saved."""
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    name, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.array(keys, dtype=np.uint32), pos, has_gauss, cached))
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: Path,
    generator: nn.Module,
    discriminator: nn.Module,
    optimizers: tuple[Optimizer, Optimizer],
    schedulers: tuple[ExponentialLR, ExponentialLR],
    state: TrainState,
    configs: dict[str, Any],
) -> None:
    """Save everything needed to resume (and to rebuild the model). Written atomically."""
    checkpoint = {
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "optim_g": optimizers[0].state_dict(),
        "optim_d": optimizers[1].state_dict(),
        "sched_g": schedulers[0].state_dict(),
        "sched_d": schedulers[1].state_dict(),
        "step": state.step,
        "epoch": state.epoch,
        "best_val_mel": state.best_val_mel,
        **{f"{name}_config": config_to_dict(cfg) for name, cfg in configs.items()},
        "rng_state": rng_state(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)  # atomic: a crash mid-save never leaves a half-written file


def load_checkpoint(
    path: Path,
    generator: nn.Module,
    discriminator: nn.Module | None = None,
    optimizers: tuple[Optimizer, Optimizer] | None = None,
    schedulers: tuple[ExponentialLR, ExponentialLR] | None = None,
) -> TrainState:
    """Load a checkpoint into the given objects and return the saved counters."""
    # weights_only=True refuses to run arbitrary pickled code (safe loading).
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    generator.load_state_dict(checkpoint["generator"])
    if discriminator is not None:
        discriminator.load_state_dict(checkpoint["discriminator"])
    if optimizers is not None:
        optimizers[0].load_state_dict(checkpoint["optim_g"])
        optimizers[1].load_state_dict(checkpoint["optim_d"])
    if schedulers is not None:
        schedulers[0].load_state_dict(checkpoint["sched_g"])
        schedulers[1].load_state_dict(checkpoint["sched_d"])
    set_rng_state(checkpoint["rng_state"])
    return TrainState(checkpoint["step"], checkpoint["epoch"], checkpoint["best_val_mel"])


# ---------------------------------------------------------------------------
# 7. Training loop
# ---------------------------------------------------------------------------


def epoch_batches(loader: DataLoader, epoch: int, seed: int, skip: int) -> Iterator[dict]:
    """Batches of one epoch, in a shuffle order that depends only on (seed, epoch).

    Re-seeding per epoch means a resumed run sees the same order, so we can
    skip the batches that were already used before the restart.
    """
    if isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(epoch)  # new shuffle each epoch, same on every process
    elif loader.generator is not None:
        loader.generator.manual_seed(seed + epoch)
    return islice(iter(loader), skip, None)


def train(
    generator: nn.Module,
    discriminator: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    data_config: DataConfig,
    model_config: ModelConfig,
    disc_config: DiscConfig,
    train_config: TrainConfig,
    callback: Callable[[dict[str, Any]], None] | None = None,
) -> TrainState:
    """Run the GAN training loop for `max_steps` steps and return the final counters.

    Everything is passed in already built, so tests can use tiny datasets.
    `callback` (optional) receives every logged metric as a dict, on the main
    process only; the notebook uses it to draw live training curves, and to stop
    at the session time limit by raising from a "val" event (sent after saving):
        {"kind": "train", "step", "epoch", "mel", "adv", "fm", "d_loss", "d_real",
         "d_fake", "g_grad", "d_grad", "lr_g", "lr_d", "audio_sec_per_sec"}
        {"kind": "val", "step", "epoch", "mel", "l1", "sample_path"}
    """
    cfg = train_config
    info = get_dist_info()
    device = resolve_device(cfg.device)
    amp_dtype = resolve_precision(cfg.precision, device)
    generator.to(device).train()
    discriminator.to(device).train()
    generator.grad_checkpointing = cfg.grad_checkpointing
    raw_g, raw_d = generator, discriminator  # unwrapped: checkpoints, eval, audio samples
    lookahead = model_config.lookahead_samples
    mel_loss = MultiScaleMelLoss(data_config.sample_rate).to(device)

    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
        raise ValueError("train set smaller than one batch: lower --batch-size")

    optimizers = build_optimizers(raw_g, raw_d, cfg)
    schedulers = build_schedulers(*optimizers, cfg)
    configs = {"data": data_config, "model": model_config, "disc": disc_config, "train": cfg}
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state = TrainState()
    if cfg.resume is not None:
        state = load_checkpoint(Path(cfg.resume), raw_g, raw_d, optimizers, schedulers)
        logger.info("resumed from %s at step %d", cfg.resume, state.step)

    if info.enabled:
        # DDP averages gradients over all processes after each backward pass and
        # copies rank 0's weights to everyone at start, so all begin equal.
        ids = [device.index] if device.type == "cuda" else None
        generator = DDP(generator, device_ids=ids)
        discriminator = DDP(discriminator, device_ids=ids)

    def checkpoint(path: Path) -> None:
        if info.is_main:  # one writer is enough: all processes hold the same weights
            save_checkpoint(path, raw_g, raw_d, optimizers, schedulers, state, configs)

    def emit(event: dict[str, Any]) -> None:
        if callback is not None and info.is_main:
            callback(event)

    def validate_and_save() -> None:
        metrics = evaluate(raw_g, val_loader, mel_loss, lookahead, device, amp_dtype)
        sample_path = output_dir / "samples" / f"step_{state.step:07d}.wav"
        if info.is_main:
            logger.info("val | step %d | mel %.4f | l1 %.4f", state.step, metrics["mel"], metrics["l1"])
            save_audio_sample(raw_g, val_loader.dataset, sample_path, data_config.sample_rate, lookahead)
        if metrics["mel"] < state.best_val_mel:
            state.best_val_mel = metrics["mel"]
            checkpoint(output_dir / BEST_CHECKPOINT)
        checkpoint(output_dir / LAST_CHECKPOINT)
        # Emitted after saving, so a callback may stop training here without losing work.
        emit({"kind": "val", "step": state.step, "epoch": state.epoch, **metrics,
              "sample_path": str(sample_path)})  # fmt: skip

    logger.info(
        "training: %d steps (%d per epoch), batch %d per process x %d processes, "
        "%.1f s clips, on %s, amp=%s",
        cfg.max_steps, steps_per_epoch, cfg.batch_size, info.world_size,
        data_config.segment_samples / data_config.sample_rate, device, amp_dtype,
    )  # fmt: skip

    window: dict[str, Tensor] = {}  # running sums on device: no GPU sync every step
    window_start = time.perf_counter()
    for epoch in count(state.step // steps_per_epoch):
        if state.step >= cfg.max_steps:
            break
        state.epoch = epoch
        done_in_epoch = state.step - epoch * steps_per_epoch  # > 0 only right after resume
        batches = epoch_batches(train_loader, epoch, cfg.seed, skip=done_in_epoch)
        for batch in islice(batches, cfg.max_steps - state.step):
            batch = move_batch(batch, device)
            metrics = gan_step(
                generator, discriminator, raw_d, batch, mel_loss, *optimizers, cfg,
                lookahead, amp_dtype,
            )  # fmt: skip
            for scheduler in schedulers:
                scheduler.step()
            state.step += 1
            for key, value in metrics.items():
                window[key] = window.get(key, 0.0) + value

            if state.step % cfg.log_every == 0:
                elapsed = time.perf_counter() - window_start
                means = {k: (v / cfg.log_every).item() for k, v in window.items()}
                audio_seconds = (
                    cfg.log_every * cfg.batch_size * info.world_size
                    * data_config.segment_samples / data_config.sample_rate
                )  # fmt: skip
                lr_g, lr_d = (s.get_last_lr()[0] for s in schedulers)
                logger.info(
                    "train | epoch %d | step %d/%d | mel %.4f | adv %.3f | fm %.3f | "
                    "d %.3f (real %.3f fake %.3f) | grad g %.1f d %.1f | lr %.2e | %.1f audio s/s",
                    epoch, state.step, cfg.max_steps, means["mel"], means["adv"], means["fm"],
                    means["d_loss"], means["d_real"], means["d_fake"], means["g_grad"],
                    means["d_grad"], lr_g, audio_seconds / elapsed,
                )  # fmt: skip
                emit({"kind": "train", "step": state.step, "epoch": epoch, **means,
                      "lr_g": lr_g, "lr_d": lr_d,
                      "audio_sec_per_sec": audio_seconds / elapsed})  # fmt: skip
                window.clear()
                window_start = time.perf_counter()

            if state.step % cfg.eval_every == 0 or state.step == cfg.max_steps:
                validate_and_save()
                window_start = time.perf_counter()  # don't count eval time in the speed

    logger.info("done | best val mel %.4f", state.best_val_mel)
    return state


# ---------------------------------------------------------------------------
# 8. Command line
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """All CLI flags. Defaults come from the config dataclasses."""
    d, m, dc, t = DataConfig(), ModelConfig(), DiscConfig(), TrainConfig()
    p = argparse.ArgumentParser(description="Train PairVC on paired voice-conversion data.")

    data = p.add_argument_group("data")
    data.add_argument("--data-dir", type=Path, default=d.data_dir, help="contains train/ and val/")
    data.add_argument("--train-split", default=d.train_split)
    data.add_argument("--val-split", default=d.val_split)
    data.add_argument("--sample-rate", type=int, default=d.sample_rate)
    data.add_argument("--segment-samples", type=int, default=d.segment_samples)

    model = p.add_argument_group("model")
    model.add_argument("--n-speakers", type=int, default=m.n_speakers)
    model.add_argument("--channels", type=int, default=m.channels)
    model.add_argument("--latent-dim", type=int, default=m.latent_dim)
    model.add_argument("--residual-units", type=int, default=m.residual_units)
    model.add_argument("--lookahead-frames", type=int, default=m.lookahead_frames)
    model.add_argument("--discriminators", nargs="+", choices=DISCRIMINATOR_KINDS, default=list(dc.kinds))

    opt = p.add_argument_group("training")
    opt.add_argument("--max-steps", type=int, default=t.max_steps)
    opt.add_argument("--batch-size", type=int, default=t.batch_size)
    opt.add_argument("--g-lr", type=float, default=t.g_lr)
    opt.add_argument("--d-lr", type=float, default=t.d_lr)
    opt.add_argument("--betas", type=float, nargs=2, default=list(t.betas))
    opt.add_argument("--weight-decay", type=float, default=t.weight_decay)
    opt.add_argument("--lr-decay", type=float, default=t.lr_decay)
    opt.add_argument("--grad-clip", type=float, default=t.grad_clip)
    opt.add_argument("--lambda-adv", type=float, default=t.lambda_adv)
    opt.add_argument("--lambda-fm", type=float, default=t.lambda_fm)
    opt.add_argument("--lambda-mel", type=float, default=t.lambda_mel)
    opt.add_argument("--precision", choices=["auto", "fp32", "bf16"], default=t.precision)
    opt.add_argument("--grad-checkpointing", action="store_true")
    opt.add_argument("--num-workers", type=int, default=t.num_workers)
    opt.add_argument("--train-limit", type=int, default=t.train_limit)
    opt.add_argument("--val-limit", type=int, default=t.val_limit)
    opt.add_argument("--log-every", type=int, default=t.log_every)
    opt.add_argument("--eval-every", type=int, default=t.eval_every)
    opt.add_argument("--device", default=t.device)
    opt.add_argument("--output-dir", type=Path, default=t.output_dir)
    opt.add_argument("--resume", type=Path, default=t.resume)
    opt.add_argument("--seed", type=int, default=t.seed)
    opt.add_argument(
        "--overfit",
        type=int,
        metavar="N",
        default=None,
        help="sanity check: train and validate on the same N training pairs",
    )
    return p


def configs_from_args(
    args: argparse.Namespace,
) -> tuple[DataConfig, ModelConfig, DiscConfig, TrainConfig]:
    """Turn parsed flags into the four configs."""
    data_config = DataConfig(
        data_dir=args.data_dir,
        train_split=args.train_split,
        val_split=args.val_split,
        sample_rate=args.sample_rate,
        segment_samples=args.segment_samples,
    )
    model_config = ModelConfig(
        n_speakers=args.n_speakers,
        channels=args.channels,
        latent_dim=args.latent_dim,
        residual_units=args.residual_units,
        lookahead_frames=args.lookahead_frames,
    )
    disc_config = DiscConfig(kinds=tuple(args.discriminators))
    train_config = TrainConfig(
        output_dir=args.output_dir,
        seed=args.seed,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        g_lr=args.g_lr,
        d_lr=args.d_lr,
        betas=tuple(args.betas),
        weight_decay=args.weight_decay,
        lr_decay=args.lr_decay,
        grad_clip=args.grad_clip,
        lambda_adv=args.lambda_adv,
        lambda_fm=args.lambda_fm,
        lambda_mel=args.lambda_mel,
        precision=args.precision,
        grad_checkpointing=args.grad_checkpointing,
        num_workers=args.num_workers,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        log_every=args.log_every,
        eval_every=args.eval_every,
        device=args.device,
        resume=args.resume,
    )
    if args.overfit is not None:
        # "Can the model memorise a tiny set?" If not, something is broken.
        n = args.overfit
        data_config = dataclasses.replace(data_config, val_split=data_config.train_split)
        train_config = dataclasses.replace(
            train_config,
            train_limit=n,
            val_limit=n,
            batch_size=min(train_config.batch_size, n),  # drop_last would leave 0 batches
        )
    if data_config.segment_samples % model_config.hop_length:
        raise ValueError(f"--segment-samples must be a multiple of {model_config.hop_length}")
    return data_config, model_config, disc_config, train_config


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point: set up (optional) multi-process training, then train.

    Returns None on purpose: console scripts pass the return value to sys.exit,
    and anything else would be printed and turned into exit code 1.
    """
    args = build_parser().parse_args(argv)
    configs = configs_from_args(args)
    info = setup_distributed(configs[-1].device)  # no-op unless launched by torchrun
    logging.basicConfig(
        level=logging.INFO if info.is_main else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        _build_and_train(args, *configs)
    finally:
        if info.enabled:
            dist.destroy_process_group()


def _build_and_train(
    args: argparse.Namespace,
    data_config: DataConfig,
    model_config: ModelConfig,
    disc_config: DiscConfig,
    train_config: TrainConfig,
) -> TrainState:
    """Build data and networks from the configs, then train."""
    from pairvc.data import PairedVoiceDataset
    from pairvc.discriminators import Discriminators
    from pairvc.model import PairVC

    seed_everything(train_config.seed)  # before building the networks: same init weights
    root = Path(data_config.data_dir)
    train_dataset = PairedVoiceDataset(
        root / data_config.train_split, data_config.sample_rate, data_config.segment_samples,
        train=args.overfit is None, limit=train_config.train_limit,  # no random crop when memorising
    )  # fmt: skip
    val_dataset = PairedVoiceDataset(
        root / data_config.val_split, data_config.sample_rate, data_config.segment_samples,
        limit=train_config.val_limit,
    )  # fmt: skip
    too_big = [s for s in train_dataset.speakers if s >= model_config.n_speakers]
    if too_big:
        raise ValueError(f"data has target voices {too_big}: set --n-speakers {max(too_big) + 1}")
    logger.info(
        "data: %d train / %d val pairs, target voices %s",
        len(train_dataset), len(val_dataset), train_dataset.speakers,
    )  # fmt: skip

    generator = PairVC(model_config)
    discriminator = Discriminators(disc_config)
    logger.info(
        "generator %.2fM parameters | discriminators %.2fM",
        sum(p.numel() for p in generator.parameters()) / 1e6,
        sum(p.numel() for p in discriminator.parameters()) / 1e6,
    )
    device = resolve_device(train_config.device)
    train_loader, val_loader = build_dataloaders(train_dataset, val_dataset, train_config, device)
    return train(
        generator, discriminator, train_loader, val_loader,
        data_config, model_config, disc_config, train_config,
    )  # fmt: skip


if __name__ == "__main__":
    main()
