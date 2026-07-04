# Cedar wake-word training on the DGX Spark — handoff for Claude

You are picking up a task on a **DGX Spark (GB10)**. This file is a complete, self-contained
brief — you have none of the prior conversation. Read it fully before acting.

## Mission
Train a custom **"Cedar" / "hey Cedar"** wake-word model with openWakeWord, producing
`cedar.onnx`, to power a hands-free wake word for a **Reachy Mini** robot voice app. The app
integration (Mac side, separate repo) is already built and waiting for the model file; your
job is only to **produce `cedar.onnx` on this Spark** and hand it back.

## Why we're on the Spark (context, not optional)
This was first attempted on **Google Colab (free)** and failed repeatedly — not on the model,
but because the free runtime **disconnects when idle** and wiped hours of work twice, plus a
long chain of dependency breakages. The Spark is an always-on GPU box, and we run training in
a **detached Docker container** so it survives SSH/laptop disconnects. That resilience is the
whole point — don't reintroduce anything that depends on an interactive session staying alive.

## This machine (verified)
- **NVIDIA GB10** (Grace-Blackwell superchip), **aarch64/arm64**, DGX OS, **CUDA 13.0**,
  driver 580, GPU **compute capability sm_121 (12,1)**. ~128 GB unified LPDDR5X memory.
- Hostname `bigsur-spark`. Disk: `/` is 3.7 TB with ~2.2 TB free (plenty; need ~50 GB).
- **Already running (leave them alone):** Ollama serving `gemma4:26b` (~27 GB), two NVIDIA
  Riva/Speech NIM containers — `spark-riva-asr-nim-1` (parakeet) + `spark-riva-tts-nim-1`
  (magpie), which show up as `tritonserver` processes (~18 GB), and a `signal-api` container.
  Total GPU/unified use ~45 GB, so **~80 GB free — training coexists fine; do NOT stop the
  NIMs or Ollama.** These NIMs are the robot's live speech stack (a separate voice-server
  talks to them); killing them would break production.
- Docker + NVIDIA Container Toolkit work; **NGC auth is configured** (`docker login nvcr.io`,
  `NGC_API_KEY` set) so `nvcr.io/...` pulls succeed.

## The bundle: `~/cedar-train/`
Sourced from the `spark/` folder of the fork `https://github.com/mdwoodruff3/openWakeWord`
(cloned at `~/openWakeWord`; `git pull` there for updates, then re-copy changed files into
`~/cedar-train`). Files:
- **`Dockerfile`** — builds `cedar-train:1` from `nvcr.io/nvidia/pytorch:25.10-py3` (arm64:
  torch 2.9.0a0, CUDA 13.0.2, py3.12, sm_121-clean). Adds espeak-ng + the audio-augmentation
  stack. Deliberately installs **no tensorflow/tflite/transformers**.
- **`sitecustomize.py`** — soundfile-backed shim restoring `torchaudio.load/info/set_audio_backend`
  (removed in torchaudio 2.x). Baked into the image so it loads at every Python startup.
- **`setup.sh`** — clones the two training repos, downloads the piper voice model, applies two
  source patches (see pitfalls). Run once inside `~/cedar-train` before `docker build`.
- **`cedar.yml`** — training config (phrases, negatives, 20k samples, 50k steps).
- **`download_data.py`** — fetches RIRs / AudioSet / FMA / precomputed features (idempotent).
- **`run_all.sh`** — the pipeline: generate (GPU) → augment (CPU) → train+export (GPU).
- **`README.md`** — the runbook with exact commands + risk table.

## Current status at handoff
The image builds and the container runs. We just fixed the last dependency wall
(`datasets` 2.14.6 → 2.21.0 for modern pyarrow). **Next action: rebuild the image, re-run the
smoke test, and if it's green, launch the detached training run.** Nothing has been trained yet.

## Happy path (do this)
```bash
cd ~/openWakeWord && git pull                                   # get latest bundle
cp ~/openWakeWord/spark/*.py ~/openWakeWord/spark/*.sh ~/openWakeWord/spark/Dockerfile \
   ~/openWakeWord/spark/cedar.yml ~/cedar-train/                # refresh changed files
cd ~/cedar-train
bash setup.sh                                                   # clones + patches (idempotent)
docker build -t cedar-train:1 .                                 # base image is cached; fast

# SMOKE TEST — note the -i (heredoc needs stdin attached to the container)
docker run --rm -i --gpus all -v "$HOME/cedar-train:/work" -w /work cedar-train:1 python - <<'EOF'
import torch, torchaudio, onnxruntime, soundfile, datasets, speechbrain, torch_audiomentations, audiomentations, acoustics
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(), "| cap:", torch.cuda.get_device_capability())
print("datasets", datasets.__version__, "| torchaudio", torchaudio.__version__, "| shim:", torchaudio.get_audio_backend())
from espeak_phonemizer import Phonemizer; print("phonemes:", Phonemizer("en-us").phonemize("hey cedar"))
x = torch.rand(64,64,device="cuda") @ torch.rand(64,64,device="cuda"); print("GPU matmul OK:", x.sum().item() > 0)
EOF
# Expect (no traceback): cap (12, 1) | datasets 2.21.0 | torchaudio ... shim: soundfile | phonemes | GPU matmul OK: True

# LAUNCH — detached, disconnect-proof
docker run -d --name cedar-train --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$HOME/cedar-train:/work" -w /work -e HF_HOME=/work/hf_cache \
  cedar-train:1 bash /work/run_all.sh

docker logs -f cedar-train      # Ctrl-C detaches from logs only; training continues
```
Total ~2.5–4.5 h. Output: `~/cedar-train/my_custom_model/cedar.onnx`. If a run dies,
`docker start cedar-train` resumes (downloads cached, clips skipped; if it died mid-augment,
add `--overwrite` to the `--augment_clips` line in `run_all.sh` once, run, then remove it).

## Pitfalls already solved — DO NOT rediscover or "simplify" these away
Every one of these cost real time on Colab and is deliberately fixed in the bundle:
1. **piper repo:** use the **dscripka fork** of `piper-sample-generator` — the rhasspy repo
   **removed `generate_samples.py`**. Voice model: `en-us-libritts-high.pt` from rhasspy's
   **v1.0.0** release, placed at `piper-sample-generator/models/` (the fork ships the matching
   `.pt.json`; `generate_samples()` defaults to exactly this path — don't rename).
2. **`espeak_phonemizer`** (pip, pure-Python) + **`espeak-ng`** (apt) — the fork's generator
   imports `espeak_phonemizer`, not piper-phonemize.
3. **No TensorFlow / onnx_tf / tflite.** Only ONNX is needed; the TF imports in `train.py` are
   lazy (inside the tflite converter, never called). Installing TF wastes time and breaks.
4. **No `transformers`.** Nothing in the ONNX path needs it. Installing it triggers a
   `torchmetrics → transformers → huggingface_hub` skew (`is_offline_mode` import error).
   Leaving it out sidesteps that whole class — do not `pip install transformers`.
5. **torchaudio 2.x removed `set_audio_backend`/`get_audio_backend`/`info`/`load`** that
   `torch_audiomentations 0.11`, `speechbrain`, and `openwakeword/data.py` still call. Fixed by
   `sitecustomize.py` (soundfile-backed). Also: **NGC arm64 images ship no torchaudio**, so the
   Dockerfile pip-installs `torchaudio==2.9.0` (matching torch minor) with `--no-deps` so it
   can't clobber NVIDIA's torch.
6. **PyTorch ≥2.6 defaults `torch.load(weights_only=True)`** and the piper checkpoint is a full
   pickle → `setup.sh` patches `generate_samples.py` to `weights_only=False`.
7. **openWakeWord `ncpu` bug:** `compute_features_from_generator` accepts `ncpu` but never
   passes it to `AudioFeatures`, leaving ONNX feature extraction single-threaded → `setup.sh`
   patches `utils.py` to `AudioFeatures(device=device, ncpu=ncpu)` (huge speedup on the 20 Grace
   cores). If the patch's grep check says "not found," it's slower but not fatal.
8. **No aarch64 `onnxruntime-gpu` wheel.** So the **augment step must run with
   `CUDA_VISIBLE_DEVICES=""`** (already in `run_all.sh`) — otherwise `train.py` picks the GPU
   device and `AudioFeatures` demands `CUDAExecutionProvider`, which doesn't exist on arm64.
   Generation and training still use the GPU; only augment/feature-extraction is CPU.
9. **`onnxscript`** is required for `torch.onnx.export` (torch 2.9's dynamo exporter imports it).
   Installed. If export still errors, add `dynamo=False` to the `torch.onnx.export(...)` call at
   the end of `train.py`.
10. **`datasets==2.14.6` used `pyarrow.PyExtensionType`**, removed in modern pyarrow (no py3.12
    wheel exists for an old-enough pyarrow to pin back to). Fixed by `datasets==2.21.0` +
    `trust_remote_code=True` on the FMA/RIR `load_dataset` calls (FMA is a script dataset).
11. **`docker run` needs `-i`** for the heredoc smoke test (else `python -` gets empty stdin and
    silently does nothing).
12. **Memory:** you do NOT need to stop the NIMs/Ollama; the GB10 has plenty of unified-memory
    headroom. Only if `nvidia-smi` shows <~20 GB free, drop `tts_batch_size` to 50 in `cedar.yml`.

## Config decisions (in `cedar.yml`) — already set, change only if asked
- `target_phrase: ["cedar", "hey cedar"]` (one binary model, fires on either).
- `n_samples: 20000`, `n_samples_val: 2000`, `steps: 50000`, `tts_batch_size: 100`.
- `custom_negative_phrases`: Cedar near-homophones (seeder, cedar tree, leader, reader, …).

## After training — get the model to the robot (this is done from the Mac, not the Spark)
The user runs, on their Mac (Spark reachable via Tailscale as `bigsur-spark`):
```
scp bigsur-spark:~/cedar-train/my_custom_model/cedar.onnx \
    "~/Development Projects/reachy/src/reachy/audio/models/cedar.onnx"
```
Your job on the Spark ends at producing `cedar.onnx`. Just confirm it exists and report its path.

## The Reachy app side (context only — a different repo, on the Mac)
The robot app lives at `~/Development Projects/reachy` (Mac) with the wake-word integration on
branch **`feature/cedar-wake-word`** (built, not merged). It has a `CedarWakeDetector` that
loads `WAKE_WORD_MODEL` (default `src/reachy/audio/models/cedar.onnx`) and gates the mic behind
the wake word. Runtime env knobs: `REACHY_WAKE_WORD` (on/off), `REACHY_WAKE_WORD_THRESHOLD`
(0.6), `_ARMED_SECONDS` (8), `_DEBOUNCE_SECONDS` (2). To enable: `uv sync --extra wake_word` +
`REACHY_WAKE_WORD=on ./start.sh`. There's also a local tester `openWakeWord/test_mic_live.py`
(`--model cedar.onnx --device N --gain 4`) — note the **Reachy mic runs ~10–15 dB quieter than a
MacBook**, so real-world testing uses input gain.

## Quality expectation (important)
openWakeWord's **synthetic validation metrics are pessimistic** (a Colab run reported acc ~0.69
/ recall ~0.39 and the model may still work well in practice). **Judge the real model with
`test_mic_live.py` at a low threshold (0.3–0.5) before concluding it needs retraining.** Don't
auto-retrain on the synthetic numbers alone.

## Quick reference
- Fork (has this bundle): `https://github.com/mdwoodruff3/openWakeWord` → `spark/`
- Training repos (cloned by `setup.sh`): `dscripka/openwakeword`, `dscripka/piper-sample-generator`
- Container base: `nvcr.io/nvidia/pytorch:25.10-py3` (arm64)
- Work dir / bind mount: `~/cedar-train` ↔ `/work`
- Output: `~/cedar-train/my_custom_model/cedar.onnx`
- Full runbook + risk table: `~/cedar-train/README.md`
