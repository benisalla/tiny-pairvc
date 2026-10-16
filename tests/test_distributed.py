"""Multi-process (DDP) training test on CPU.

Starts 2 processes with the gloo backend, exactly like torchrun would on 2 GPUs,
and checks they stay in sync: same step count and identical final weights for
both the generator and the discriminators.
"""

import os
import socket
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from conftest import SEGMENT, tiny_disc_config, tiny_model_config, tiny_train_config

from pairvc.config import DataConfig
from pairvc.data import PairedVoiceDataset, write_toy_dataset
from pairvc.discriminators import Discriminators
from pairvc.model import PairVC
from pairvc.train import LAST_CHECKPOINT, build_dataloaders, get_dist_info, train

WORLD_SIZE = 2


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def worker(rank: int, port: int, tmp_path: Path) -> None:
    """What torchrun would run in each process."""
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
        RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(WORLD_SIZE),
    )  # fmt: skip
    dist.init_process_group("gloo", rank=rank, world_size=WORLD_SIZE)
    try:
        assert get_dist_info().world_size == WORLD_SIZE
        torch.manual_seed(rank)  # different init per rank: DDP must broadcast rank 0's
        generator, discriminator = PairVC(tiny_model_config()), Discriminators(tiny_disc_config())
        config = tiny_train_config(tmp_path / "run", max_steps=6, eval_every=3)
        root = tmp_path / "toy"
        train_set = PairedVoiceDataset(root / "train", 16_000, SEGMENT, train=True)
        val_set = PairedVoiceDataset(root / "val", 16_000, SEGMENT)
        train_loader, val_loader = build_dataloaders(train_set, val_set, config, torch.device("cpu"))
        # 16 pairs / 2 processes / batch 2 = 4 batches per process.
        assert len(train_loader) == 4
        state = train(
            generator, discriminator, train_loader, val_loader,
            DataConfig(data_dir=root, segment_samples=SEGMENT),
            tiny_model_config(), tiny_disc_config(), config,
        )  # fmt: skip
        torch.save(
            {"step": state.step, "g": generator.state_dict(), "d": discriminator.state_dict()},
            tmp_path / f"rank{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


def test_two_process_training_stays_in_sync(tmp_path: Path) -> None:
    write_toy_dataset(tmp_path / "toy", n_clips=8, seconds=0.25)
    mp.spawn(worker, args=(free_port(), tmp_path), nprocs=WORLD_SIZE, join=True)

    rank0 = torch.load(tmp_path / "rank0.pt", weights_only=True)
    rank1 = torch.load(tmp_path / "rank1.pt", weights_only=True)
    assert rank0["step"] == rank1["step"] == 6  # crosses an epoch boundary (4 per epoch)
    for net in ("g", "d"):
        for name, weight in rank0[net].items():
            torch.testing.assert_close(weight, rank1[net][name], msg=f"{net}.{name} differs")
    assert (tmp_path / "run" / LAST_CHECKPOINT).exists()  # written by rank 0 only
