"""02 - Train a small PairVC on the digits dataset (CPU).

Same training loop as `pairvc-train`, called from Python. The first 5/7 of the
steps use the mel loss only (fast, no discriminators: `--gan-start-step`), the
rest add the GAN losses. The default 700 steps take ~15-25 minutes on a laptop
CPU; the README results use `--steps 3500` (~1.5 h). Writes
examples/output/run/{best.pt, last.pt, history.jsonl, samples/}.

    python examples/02_train_on_digits.py [--steps 700]
"""

import argparse

from pairvc.train import main as train_main

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=700)
args = parser.parse_args()

train_main([
    "--data-dir", "examples/output/digits_data",
    "--output-dir", "examples/output/run",
    "--device", "cpu",
    "--channels", "8",            # 0.67M parameters (the default 32 channels = 9.6M)
    "--batch-size", "8",
    "--segment-samples", "8000",  # 0.5 s windows: a spoken digit
    "--g-lr", "1e-3", "--d-lr", "3e-4", "--lr-decay", "0.9995",
    "--gan-start-step", str(args.steps * 5 // 7),
    "--max-steps", str(args.steps),
    "--eval-every", str(max(1, args.steps // 20)),
    "--log-every", "25",
    "--num-workers", "0",
])  # fmt: skip
