# Evaluation guide

The training loss tells you the model is getting closer to the teacher. To know
*how* close, and whether the result sounds natural, use `pairvc-eval`:

```bash
pairvc-eval --checkpoint checkpoints/run/best.pt --data-dir data            # reference metrics
pairvc-eval --checkpoint checkpoints/run/best.pt --data-dir data --dnsmos   # + predicted MOS
pairvc-eval --checkpoint checkpoints/run/best.pt --data-dir data --untrained  # "before training" row
pairvc-eval ... --output results.json                                        # save the numbers
```

Every pair of the split (`val` by default) is converted offline, **whole clip,
no cropping**, and compared with the teacher's clip. The table has three rows,
so it can be read without knowing the metrics by heart:

| row | what it is | why it is there |
|-----|-----------|-----------------|
| no conversion | the source clip itself | the floor: what you get without PairVC |
| PairVC | the model | |
| teacher | the reference | 0 distance by definition; its DNSMOS is the ceiling PairVC can reach |

## The metrics

| metric | measures | better | notes |
|--------|----------|--------|-------|
| **MCD** (mel cepstral distortion, dB) | timbre / spectral envelope distance to the teacher | lower | 13 mel-cepstral coefficients without c0 (loudness), on the teacher's speech frames |
| **F0 RMSE** (cents) | pitch error, on frames voiced in both clips | lower | 100 cents = 1 semitone. "No conversion" scores ≈ 100 × the teacher's pitch shift |
| **voicing** (%) | frames where both agree on voiced vs unvoiced/silent | higher | catches mumbling and buzz in silences |
| **mel L1** | mean \|log10 mel − log10 mel\| at one scale | lower | the training loss, at one resolution |
| **DNSMOS OVRL / SIG** (1–5) | predicted listening-test score (overall / speech quality) | higher | Microsoft's DNSMOS P.835, no reference needed. `--dnsmos`, needs `onnxruntime`, downloads 1.1 MB once |

All of it lives in [`src/pairvc/metrics.py`](../src/pairvc/metrics.py): plain
NumPy, a YIN pitch tracker, no network except the DNSMOS model, so the tests
cover it.

### Limits, honestly

- MCD and F0 compare with the **teacher**, not with a real recording of the
  target speaker: PairVC can at best match the teacher.
- DNSMOS was trained on 16 kHz noisy speech. Band-limited audio (like the 8 kHz
  spoken digits) scores lower whatever the system, so compare rows of the same
  table, not numbers across datasets. Clips shorter than 9 s are repeated to
  fill the model's 9 s input, as the official script does.
- A real listening test (MOS / ABX with people) is still the final word for a
  production model. Speaker similarity with a speaker-verification model and
  CER with an ASR model are the next metrics to add for real paired data.

## In Python

```python
from pairvc.convert import load_model
from pairvc.evaluate import evaluate_split, results_table
from pairvc.metrics import DNSMOS, compare

model, sr = load_model("checkpoints/run/best.pt")
results = evaluate_split(model, "data/val", sr, dnsmos=DNSMOS())
print(results_table(results))

compare(converted_audio, teacher_audio, sr)   # one clip: {"mcd_db", "f0_rmse_cents", ...}
```

[`examples/05_evaluate.py`](../examples/05_evaluate.py) builds the README's
before/after table this way (the untrained model is rebuilt with the training
seed, so "before" is exactly where training started).
