#!/usr/bin/env bash
# Run this INSIDE your work dir on the Spark (e.g. ~/cedar-train) BEFORE `docker build`.
# Clones the two repos, fetches the piper voice model, and applies the two source patches.
set -euo pipefail

# dscripka's fork HAS generate_samples.py (rhasspy's repo removed it)
[ -d openwakeword ] || git clone https://github.com/dscripka/openwakeword
[ -d piper-sample-generator ] || git clone https://github.com/dscripka/piper-sample-generator

# TTS checkpoint (255 MB) from rhasspy's v1.0.0 release; the fork ships the matching
# en-us-libritts-high.pt.json, and generate_samples() defaults to exactly this path.
mkdir -p piper-sample-generator/models
[ -f piper-sample-generator/models/en-us-libritts-high.pt ] || \
  wget -O piper-sample-generator/models/en-us-libritts-high.pt \
  https://github.com/rhasspy/piper-sample-generator/releases/download/v1.0.0/en-us-libritts-high.pt

# PATCH 1: PyTorch >=2.6 defaults torch.load(weights_only=True); piper checkpoint is a full pickle.
sed -i 's/torch.load(model_path)/torch.load(model_path, weights_only=False)/' \
  piper-sample-generator/generate_samples.py

# PATCH 2: openwakeword passes ncpu to compute_features_from_generator but never to
# AudioFeatures -> single-threaded ONNX feature extraction. Multi-threads the 20 Grace cores.
sed -i 's/AudioFeatures(device=device)/AudioFeatures(device=device, ncpu=ncpu)/' \
  openwakeword/openwakeword/utils.py

echo "--- patch check ---"
grep -q "weights_only=False" piper-sample-generator/generate_samples.py \
  && echo "PATCH1 (weights_only) applied" \
  || echo "PATCH1 FAILED — inspect piper-sample-generator/generate_samples.py torch.load line"
grep -q "ncpu=ncpu" openwakeword/openwakeword/utils.py \
  && echo "PATCH2 (ncpu) applied" \
  || echo "PATCH2 NOTE: pattern not found — feature extraction may be single-threaded (slower, not fatal)"
echo "setup done"
