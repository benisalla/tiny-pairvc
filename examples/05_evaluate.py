"""05 - Score the model: before training, after training, and the baselines.

Prints the Markdown tables of the README: MCD, F0 error, voicing, mel L1 and
DNSMOS (a neural MOS predictor; downloads 1.1 MB once, needs onnxruntime) on
the held-out speaker. Saves examples/output/results.json.

    python examples/05_evaluate.py [--checkpoint examples/output/run/best.pt] [--no-dnsmos]
"""

import argparse
import json
from pathlib import Path

import torch

from pairvc.convert import load_model
from pairvc.evaluate import evaluate_split, results_table
from pairvc.metrics import DNSMOS
from pairvc.model import PairVC
from pairvc.train import seed_everything

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, default=Path("examples/output/run/best.pt"))
parser.add_argument("--no-dnsmos", action="store_true")
args = parser.parse_args()

torch.set_num_threads(2)
trained, sample_rate = load_model(args.checkpoint)
seed_everything(42)  # the training seed: these are exactly the weights training started from
untrained = PairVC(trained.config).eval()
dnsmos = None if args.no_dnsmos else DNSMOS()
val_dir = Path("examples/output/digits_data/val")

before = evaluate_split(untrained, val_dir, sample_rate, dnsmos=dnsmos)
after = evaluate_split(trained, val_dir, sample_rate, dnsmos=dnsmos, baselines=False)
results = {
    "no conversion": before["no conversion"],
    "PairVC, untrained": before["PairVC"],
    "PairVC, trained": after["PairVC"],
    "teacher": before["teacher"],
}
print(results_table(results))
Path("examples/output/results.json").write_text(json.dumps(results, indent=2))
