# Training the "Cedar" wake word on Google Colab

The fastest, officially-supported way to train a custom openWakeWord model is the
`automatic_model_training.ipynb` notebook on a free Colab GPU (~1–2 h, no local setup).
This guide adapts that notebook to train **"Cedar" / "hey Cedar"** and drop the result
into the Reachy app.

## 1. Open the notebook in Colab

<https://colab.research.google.com/github/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb>

Then set a GPU runtime: **Runtime → Change runtime type → T4 GPU**.

## 2. Run the setup + data cells unchanged

Run every cell from the top **through the data-download cells** (environment install,
MIT RIRs, AudioSet background, and the two feature `.npy` files) exactly as written.
These take ~10–15 min and need no edits.

## 3. Replace the config cell with this one

Find the cell titled *"Modify values in the config and save a new version"* (it trains
"hey sebastian" by default). **Replace its entire contents** with:

```python
# Modify values in the config and save a new version — CEDAR
config["target_phrase"] = ["cedar", "hey cedar"]   # one binary model, fires on either
config["model_name"] = "cedar"                      # → produces my_custom_model/cedar.onnx
config["n_samples"] = 20000        # recommended minimum for a solid model
config["n_samples_val"] = 2000
config["steps"] = 50000

# Phrases that must NOT wake it — seed the model against Cedar's near-homophones.
config["custom_negative_phrases"] = [
    "seeder", "ceder", "see there", "see her", "seed her", "cedar wood", "cedar tree",
    "leader", "reader", "heater", "either", "theater",
    "hey seeder", "hey see there", "hey reader", "hey leader",
]

# The notebook only downloads AudioSet (there is no ./fma), so list just that:
config["background_paths"] = ['./audioset_16k']
config["false_positive_validation_data_path"] = "validation_set_features.npy"
config["feature_data_files"] = {"ACAV100M_sample": "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"}
# False-positive control is target_false_positives_per_hour (0.2) + max_negative_weight
# (1500) from the base config — leave them unless you tune later.

import yaml
with open('my_model.yaml', 'w') as f:
    yaml.dump(config, f)
print("wrote my_model.yaml for:", config["target_phrase"], "→", config["model_name"] + ".onnx")
```

**Sample-count knob** (`n_samples`): `20000` is the recommended minimum and trains a
strong model. Drop to `10000` if you're racing Colab's session limits; raise to `50000`
for best quality (longer generation). Generation is **resumable** — if a cell dies, just
re-run it and it continues to the target count.

**Single-word vs phrase:** if `cedar` alone false-triggers too much in practice, retrain
with `config["target_phrase"] = ["hey cedar"]` only.

## 4. Run the three training steps

Run the generate / augment / train cells in order:

```
train.py --generate_clips     # ~30–90 min at n_samples=20000 (the long one; resumable)
train.py --augment_clips       # a few minutes
train.py --train_model         # ~15–40 min on a T4
```

Output: `my_custom_model/cedar.onnx` (and a `.tflite` we don't need).

## 5. Download the model and drop it into Reachy

In a new Colab cell:

```python
from google.colab import files
files.download('my_custom_model/cedar.onnx')
```

Then on your Mac:

```bash
# quick standalone test with the live tester first:
cd "$HOME/Development Projects/openWakeWord"
cp ~/Downloads/cedar.onnx .
.venv-test/bin/python test_mic_live.py --model cedar.onnx --threshold 0.5 --device 1 --gain 4

# once it detects "Cedar" well, install it into the app:
cp ~/Downloads/cedar.onnx "$HOME/Development Projects/reachy/src/reachy/audio/models/cedar.onnx"
```

Then enable it in the Reachy app (cascade mode): `uv sync --extra wake_word` and
`REACHY_WAKE_WORD=on ./start.sh`. See
`reachy/docs/wake-word/wake-word.md` for the runtime knobs
(`REACHY_WAKE_WORD_THRESHOLD`, etc.).

## Notes

- Colab free GPU sessions can be reclaimed; if disconnected, re-run the generate cell
  (resumable) and continue. Keep the tab active.
- The `.tflite` conversion cell sometimes fails on Colab — ignore it, we only use ONNX.
- Record a few real "Cedar" WAVs and hard-negatives ("seeder", "cedar tree") and check
  the model against them with `test_mic_live.py` before trusting it on the robot.
