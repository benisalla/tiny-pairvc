"""Score a trained model on a whole split: PairVC vs the teacher, with baselines.

    pairvc-eval --checkpoint checkpoints/run/best.pt --data-dir data --dnsmos

Every validation pair (whole clip, no cropping) is converted offline and
compared with the teacher's clip using pairvc.metrics. Three systems are
reported, so the numbers can be read without knowing the metrics:

- "no conversion": the source clip itself (what you hear without PairVC);
- "PairVC": the model;
- "teacher": the reference (0 distance by definition; its DNSMOS is the ceiling).

Pass `--untrained` to score a freshly initialised model with the same
architecture: the "before training" row of the README.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from pairvc.data import find_pairs, load_audio
from pairvc.metrics import DNSMOS, compare
from pairvc.model import PairVC

logger = logging.getLogger(__name__)

SYSTEMS = ("no conversion", "PairVC", "teacher")


@torch.inference_mode()
def evaluate_split(
    model: PairVC,
    split_dir: Path,
    sample_rate: int,
    limit: int | None = None,
    dnsmos: DNSMOS | None = None,
    baselines: bool = True,
) -> dict[str, dict[str, float]]:
    """Mean metrics per system over the pairs of `split_dir`: {system: {metric: value}}.

    `baselines=False` scores only PairVC (the baselines do not depend on the model).
    """
    model.eval()
    device = next(model.parameters()).device
    pairs = find_pairs(Path(split_dir))[:limit]
    if not pairs:
        raise ValueError(f"no pairs in {split_dir}")
    systems = SYSTEMS if baselines else ("PairVC",)
    rows: dict[str, list[dict[str, float]]] = {name: [] for name in systems}
    for pair in pairs:
        source = load_audio(pair.source, sample_rate)
        target = load_audio(pair.target, sample_rate).numpy()
        speaker = torch.tensor([pair.speaker], device=device)
        converted = model.convert(source[None].to(device), speaker)[0].clamp(-1, 1).cpu().numpy()
        audios = dict(zip(SYSTEMS, (source.numpy(), converted, target), strict=True))
        for name in systems:
            audio = audios[name]
            scores = compare(audio, target, sample_rate)
            if dnsmos is not None:
                scores.update(dnsmos(audio, sample_rate))
            rows[name].append(scores)
    return {name: {k: float(np.nanmean([r[k] for r in rs])) for k in rs[0]} for name, rs in rows.items()}


def results_table(results: dict[str, dict[str, float]]) -> str:
    """A Markdown table, one row per system (the format used in the README)."""
    columns = [
        ("mcd_db", "MCD dB ↓", "{:.2f}"),
        ("f0_rmse_cents", "F0 RMSE cents ↓", "{:.0f}"),
        ("voicing_agreement", "voicing ↑", "{:.1%}"),
        ("mel_l1", "mel L1 ↓", "{:.3f}"),
        ("dnsmos_ovrl", "DNSMOS OVRL ↑", "{:.2f}"),
        ("dnsmos_sig", "DNSMOS SIG ↑", "{:.2f}"),
    ]
    columns = [c for c in columns if any(c[0] in r for r in results.values())]
    lines = [
        "| system | " + " | ".join(title for _, title, _ in columns) + " |",
        "|---|" + "---:|" * len(columns),
    ]
    for name, scores in results.items():
        cells = [
            "" if key not in scores else "–" if np.isnan(scores[key]) else fmt.format(scores[key])
            for key, _, fmt in columns
        ]  # "–": not measurable (e.g. no voiced frame, so no pitch)
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Score PairVC against the teacher on a data split.")
    p.add_argument("--checkpoint", type=Path, required=True, help="best.pt / last.pt / generator file")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--split", default="val")
    p.add_argument("--limit", type=int, default=None, help="only the first N pairs")
    p.add_argument("--untrained", action="store_true", help="same architecture, random weights")
    p.add_argument("--dnsmos", action="store_true", help="add DNSMOS (needs onnxruntime, downloads 1 MB)")
    p.add_argument("--output", type=Path, default=None, help="write the results as JSON here")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0, help="init seed for --untrained")
    return p


def run(argv: Sequence[str] | None = None) -> dict[str, dict[str, float]]:
    from pairvc.convert import load_model

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    model, sample_rate = load_model(args.checkpoint, args.device)
    if args.untrained:
        torch.manual_seed(args.seed)
        model = PairVC(model.config).to(args.device)
    results = evaluate_split(
        model, args.data_dir / args.split, sample_rate, args.limit, DNSMOS() if args.dnsmos else None
    )
    print(results_table(results))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))
        logger.info("wrote %s", args.output)
    return results


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point (returns None: console scripts pass the return value to sys.exit)."""
    run(argv)


if __name__ == "__main__":
    main()
