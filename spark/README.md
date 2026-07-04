# Train the "Cedar" wake word on a DGX Spark (GB10)

Containerized, **disconnect-proof** openWakeWord training for the DGX Spark
(GB10 / Blackwell / sm_121 / aarch64). Runs in an NGC PyTorch container so the CUDA/torch
stack Just Works and never touches the box's system Python or the voice-server. All the
pipeline gotchas (piper fork, `weights_only`, torchaudio 2.x shim, `onnxscript`, no-TF path,
onnxruntime-CPU-only) are pre-solved. Output: `my_custom_model/cedar.onnx`.

Total ~2.5–4.5 h, fully unattended after `docker run -d`. It survives SSH/laptop drops
because the container is parented to dockerd, not your shell.

## Prereqs (already true on this Spark)
Docker + NVIDIA Container Toolkit, NGC auth (`docker login nvcr.io`), ~50 GB free disk.

## Steps (over Tailscale SSH to the Spark)

```bash
# 0. Preflight
nvidia-smi                 # GPU visible; note memory already used by the Riva NIMs
df -h ~                    # need ~50 GB free
docker ps                  # what NIMs are running (GPU-memory contention -> see Risks)

# 1. Work dir + these files. Either scp this spark/ folder, or git clone the fork:
mkdir -p ~/cedar-train && cd ~/cedar-train
#   e.g.  git clone https://github.com/mdwoodruff3/openWakeWord && cp openWakeWord/spark/* .
#   (then remove the nested clone: rm -rf openWakeWord)

# 2. Clone training repos + fetch model + apply source patches
bash setup.sh

# 3. Build the image (15–30 min, one-time)
docker build -t cedar-train:1 .

# 4. SMOKE TEST — do NOT skip (2 min; catches the one real unknown, torchaudio ABI)
docker run --rm --gpus all -v "$HOME/cedar-train:/work" -w /work cedar-train:1 python - <<'EOF'
import torch, torchaudio, onnxruntime, soundfile, datasets, speechbrain, torch_audiomentations, audiomentations, acoustics
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "| capability:", torch.cuda.get_device_capability())      # expect (12, 1)
print("torchaudio", torchaudio.__version__, "| shim backend:", torchaudio.get_audio_backend())
from espeak_phonemizer import Phonemizer
print("phonemes:", Phonemizer("en-us").phonemize("hey cedar"))
x = torch.rand(64, 64, device="cuda") @ torch.rand(64, 64, device="cuda")
print("GPU matmul OK:", x.sum().item() > 0)
EOF
# If `import torchaudio` fails with an undefined-symbol error, see Risk R1 below.

# 5. Launch training, DETACHED (survives your SSH session dying)
docker run -d --name cedar-train --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$HOME/cedar-train:/work" -w /work \
  -e HF_HOME=/work/hf_cache \
  cedar-train:1 bash /work/run_all.sh

# 6. Monitor (Ctrl-C detaches from logs only; training keeps running)
docker logs -f cedar-train
#   clip progress:  ls ~/cedar-train/my_custom_model/cedar/positive_train | wc -l   # target 20000
```

Output lands at `~/cedar-train/my_custom_model/cedar.onnx`.

## If it dies (power / OOM / crash) — resume, don't restart from scratch
`docker start cedar-train` re-runs the pipeline: HF downloads resume (cache on the mount),
`download_data.py` skips completed dirs, and `--generate_clips` skips any clip class already
≥95% complete. If it died *mid-augmentation*, add `--overwrite` to the `--augment_clips` line
in `run_all.sh` once (regenerates the partial feature `.npy`), run, then remove it.

## Get the model to the robot (from the Mac, over Tailscale)
```bash
mkdir -p "$HOME/Development Projects/reachy/src/reachy/audio/models"
scp <spark-host>:~/cedar-train/my_custom_model/cedar.onnx \
    "$HOME/Development Projects/reachy/src/reachy/audio/models/cedar.onnx"
# sanity: graph loads
cd "$HOME/Development Projects/reachy" && uv run --with onnxruntime python -c "
import onnxruntime as ort
s = ort.InferenceSession('src/reachy/audio/models/cedar.onnx', providers=['CPUExecutionProvider'])
print('inputs:', [(i.name, i.shape) for i in s.get_inputs()])"
```
Then test with `openWakeWord/test_mic_live.py --model cedar.onnx --device 1 --gain 4`, and
enable in the app: `uv sync --extra wake_word` + `REACHY_WAKE_WORD=on ./start.sh`.

## Risks & mitigations
- **R1 — torchaudio 2.9.0 stable wheel vs container's 2.9.0a0 torch** (undefined symbol at import).
  Caught by the smoke test. Fallback: `pip uninstall torchaudio` and ship a tiny pure-Python
  `torchaudio` stub (the shim funcs + a `Resample` wrapping `scipy.signal.resample_poly`), or
  use the community image `scitrera/dgx-spark-pytorch-dev` (torch+torchaudio co-built for Spark).
- **R2 — sm_121 kernel coverage**: 25.10+ is the GB10-clean NGC line; if a kernel error appears,
  bump the tag (25.12 is community-confirmed on Spark).
- **R3 — Riva NIMs competing for GPU memory** during TTS gen: if `nvidia-smi` free memory is low,
  `docker stop` the NIMs for the run (voice-server fails fast + recovers) or drop `tts_batch_size` to 50.
- **R5 (pre-solved)** — no aarch64 `onnxruntime-gpu`, so `--augment_clips` runs with
  `CUDA_VISIBLE_DEVICES=""` (CPU path on the 20 Grace cores; the `ncpu` patch multi-threads it).
- **R6 — datasets vs pyarrow / HF skew**: `datasets==2.21.0` (new enough for the container's
  modern pyarrow — 2.14.6 used the removed `pyarrow.PyExtensionType` — old enough to load the
  script-based FMA dataset via `trust_remote_code=True`). transformers isn't installed, so the
  older `huggingface_hub is_offline_mode` import chain can't occur regardless of hub version.
- **R8 — ONNX export**: `onnxscript` is installed; if the dynamo exporter still errors, add
  `dynamo=False` to the `torch.onnx.export(...)` call at the end of `train.py`.

Research + full rationale: this bundle was produced from a deep-research plan (sources include
NVIDIA NGC 25.10 release notes, the DGX Spark GB10/sm_121 writeups, the NVIDIA forum note that
NGC arm64 images omit torchaudio, and the openWakeWord/piper-fork sources).
