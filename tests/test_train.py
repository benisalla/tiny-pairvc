"""Tests for pairvc.train. CPU only, offline: tiny model, tiny synthetic data."""

import dataclasses
from pathlib import Path

import pytest
import torch
from conftest import SEGMENT, tiny_disc_config, tiny_model_config, tiny_train_config

from pairvc.config import DataConfig, TrainConfig
from pairvc.data import PairedVoiceDataset
from pairvc.discriminators import Discriminators
from pairvc.model import PairVC
from pairvc.train import (
    BEST_CHECKPOINT,
    LAST_CHECKPOINT,
    align_with_lookahead,
    build_dataloaders,
    build_parser,
    configs_from_args,
    load_checkpoint,
    resolve_device,
    resolve_precision,
    train,
)


def run_training(data_config: DataConfig, config: TrainConfig, callback=None, limit=None):
    torch.manual_seed(0)
    root = Path(data_config.data_dir)
    train_set = PairedVoiceDataset(root / "train", 16_000, SEGMENT, train=True, limit=limit)
    val_set = PairedVoiceDataset(root / "val", 16_000, SEGMENT, limit=limit)
    generator, discriminator = PairVC(tiny_model_config()), Discriminators(tiny_disc_config())
    train_loader, val_loader = build_dataloaders(train_set, val_set, config, torch.device("cpu"))
    state = train(
        generator, discriminator, train_loader, val_loader,
        data_config, tiny_model_config(), tiny_disc_config(), config, callback=callback,
    )  # fmt: skip
    return state, generator


def test_align_with_lookahead() -> None:
    pred, target = torch.arange(10.0)[None], torch.arange(10.0)[None]
    p, t = align_with_lookahead(pred, target, 3)
    assert p.tolist() == [[3, 4, 5, 6, 7, 8, 9]] and t.tolist() == [[0, 1, 2, 3, 4, 5, 6]]
    assert align_with_lookahead(pred, target, 0)[0] is pred


def test_resolve_precision_and_device() -> None:
    cpu = torch.device("cpu")
    assert resolve_precision("auto", cpu) is None
    assert resolve_precision("fp32", cpu) is None
    assert resolve_precision("bf16", cpu) is torch.bfloat16
    assert resolve_device("cpu") == cpu


def test_training_run_logs_and_saves(data_config: DataConfig, tmp_path: Path) -> None:
    events = []
    config = tiny_train_config(tmp_path / "run", max_steps=4)
    state, _ = run_training(data_config, config, callback=events.append)
    assert state.step == 4
    kinds = [e["kind"] for e in events]
    assert kinds.count("train") == 4 and kinds.count("val") == 2  # val at steps 2 and 4
    for key in ("mel", "adv", "fm", "d_loss", "g_grad", "lr_g"):
        assert key in events[0]
    assert (tmp_path / "run" / LAST_CHECKPOINT).exists()
    assert (tmp_path / "run" / BEST_CHECKPOINT).exists()
    assert len(list((tmp_path / "run" / "samples").glob("*.wav"))) == 2


def test_resume_continues_the_step_count(data_config: DataConfig, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_training(data_config, tiny_train_config(run_dir, max_steps=2))
    state, generator = run_training(
        data_config, tiny_train_config(run_dir, max_steps=5, resume=run_dir / LAST_CHECKPOINT)
    )
    assert state.step == 5  # 2 before + 3 after the restart
    reloaded = PairVC(tiny_model_config())
    assert load_checkpoint(run_dir / LAST_CHECKPOINT, reloaded).step == 5
    for name, weight in generator.state_dict().items():
        torch.testing.assert_close(weight, reloaded.state_dict()[name])


def test_mel_loss_goes_down(data_config: DataConfig, tmp_path: Path) -> None:
    """With the GAN terms off, a few steps on 2 clips must reduce the mel loss."""
    events = []
    config = tiny_train_config(
        tmp_path / "run", max_steps=30, eval_every=30, g_lr=3e-3, lambda_adv=0.0, lambda_fm=0.0,
    )  # fmt: skip
    run_training(data_config, config, callback=events.append, limit=2)
    mel = [e["mel"] for e in events if e["kind"] == "train"]
    assert sum(mel[-5:]) / 5 < 0.8 * sum(mel[:5]) / 5


def test_cli_overfit_flags(toy_data: Path, tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "--data-dir",
            str(toy_data),
            "--overfit",
            "4",
            "--batch-size",
            "16",
            "--segment-samples",
            "3200",
            "--discriminators",
            "time",
            "band",
        ]
    )
    data, model, disc, train_config = configs_from_args(args)
    assert data.val_split == data.train_split == "train"  # validate on what we memorise
    assert train_config.train_limit == train_config.val_limit == 4
    assert train_config.batch_size == 4
    assert disc.kinds == ("time", "band")
    assert model.n_speakers == 2


def test_cli_rejects_segment_not_multiple_of_hop() -> None:
    args = build_parser().parse_args(["--segment-samples", "1000"])
    with pytest.raises(ValueError):
        configs_from_args(args)


def test_cli_end_to_end(toy_data: Path, tmp_path: Path) -> None:
    from pairvc.train import main

    main([
        "--data-dir", str(toy_data), "--segment-samples", "3200", "--channels", "4",
        "--latent-dim", "8", "--residual-units", "1", "--discriminators", "resolution",
        "--max-steps", "2", "--eval-every", "2", "--log-every", "1", "--batch-size", "2",
        "--num-workers", "0", "--device", "cpu", "--output-dir", str(tmp_path / "run"),
    ])  # fmt: skip
    checkpoint = torch.load(tmp_path / "run" / LAST_CHECKPOINT, weights_only=True)
    assert checkpoint["step"] == 2
    assert checkpoint["model_config"]["channels"] == 4
    assert dataclasses.asdict(tiny_disc_config())["kinds"] == checkpoint["disc_config"]["kinds"]
