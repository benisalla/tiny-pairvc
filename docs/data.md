# Paired data

PairVC is trained on **pairs**: a source clip and the *same speech* in a
target voice, aligned sample for sample. Real recordings of two people saying
the same words never line up like that, so the targets are made by a model:

1. Collect source speech: many speakers, many recording conditions (the more
   varied, the better PairVC handles voices it has never heard).
2. Run a big, high-quality but slow **teacher** voice-conversion model on every
   clip, once per target voice. This happens offline, once.
3. Train PairVC to map source → teacher output. At inference the teacher is no
   longer needed.

This is *knowledge distillation on synthetic parallel data*. PairVC can only
be as good as the teacher's conversions; whatever the teacher does well
(timbre, keeping the words and the intonation) PairVC learns, and the teacher's
artifacts are learned too.

**The teacher must keep the timing.** Target sample t has to correspond to
source sample t (the loss compares them directly). Most VC models that work
frame by frame (retrieval / kNN / SSL-feature + vocoder types) do; models that
re-generate speech with their own durations (TTS-style) do not.

## Folder layout

```
data/
  train/
    0001_original.wav        source speech (any speaker)
    0001_converted_0.wav     the same clip, converted to target voice 0
    0001_converted_1.wav     ... target voice 1
    0002_original.wav
    0002_converted_0.wav     (a source may have only some of the voices)
    ...
  val/
    ... same naming, different clips (ideally different source speakers)
```

- Every `<id>_original.wav` + `<id>_converted_<k>.wav` couple is **one training
  example** with label k: a source with 2 conversions gives 2 examples.
- Files without a partner are ignored.
- Any format `soundfile` reads (wav, flac, ogg), any bit depth, mono or
  stereo (averaged to mono), any sample rate (resampled to `--sample-rate`,
  16 kHz by default). Converting everything to 16 kHz mono wav beforehand makes
  loading faster.
- If source and target differ by a few samples in length, both are cut to the
  shorter one.
- Clips of any length work: training takes a random 1 s window
  (`--segment-samples 16000`) of each clip, a new one every epoch; validation
  always takes the first second, so its numbers are comparable over time.
  Shorter clips are zero-padded. Clips of 2–10 s are a good size.
- `--n-speakers` must be larger than the highest k in the data (training
  stops with a clear message otherwise).

## How much data?

There is no fixed number. As a rule of thumb for a distillation set: a few
hours of varied source speech per target voice is a reasonable start, and more
*different source speakers* matter more than more hours of the same ones (the
model must work on voices it never heard). The overfit check in
[training.md](training.md) tells you the pipeline works before you scale up.

## Check a dataset

```python
from pairvc.data import PairedVoiceDataset

ds = PairedVoiceDataset("data/train", sample_rate=16_000, segment_samples=16_000)
print(len(ds), "pairs, voices", ds.speakers)
item = ds[0]  # {"source": [16000], "target": [16000], "speaker": tensor(k)}
```

## No data yet? A toy dataset

```python
from pairvc.data import write_toy_dataset

write_toy_dataset("toy_data", n_clips=64, seconds=1.0)
```

Writes synthetic "voices" (harmonic tones; voice k = same melody, pitch × (k+2)/2,
different timbre). Good for checking the pipeline end to end; a model trained
on it does not convert real speech. The tests and the notebook's
`SMOKE_TEST` use it.
