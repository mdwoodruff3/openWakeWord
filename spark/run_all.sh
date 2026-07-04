#!/usr/bin/env bash
# Full Cedar training pipeline, run inside the container (bind-mounted /work).
set -euo pipefail
cd /work

# openwakeword editable, --no-deps (deps are baked into the image; stock deps would
# pull ai-edge-litert/speexdsp-ns we don't need for the ONNX path).
pip install --no-deps -e ./openwakeword

# Feature backbone models (melspectrogram.onnx + embedding_model.onnx) into the
# editable checkout on the bind mount -> persists across container restarts.
python -c "import openwakeword.utils as u; u.download_models()"

python download_data.py

echo '=== [1/3] TTS clip generation (GPU) ==='
python openwakeword/openwakeword/train.py --training_config cedar.yml --generate_clips

echo '=== [2/3] Augmentation + feature extraction (CPU, 20 Grace cores) ==='
# CUDA hidden on purpose: with CUDA visible, train.py picks device=gpu and AudioFeatures
# demands onnxruntime CUDAExecutionProvider (no aarch64 wheel). Hiding CUDA -> CPU path.
CUDA_VISIBLE_DEVICES="" python openwakeword/openwakeword/train.py --training_config cedar.yml --augment_clips

echo '=== [3/3] Model training + ONNX export (GPU) ==='
python openwakeword/openwakeword/train.py --training_config cedar.yml --train_model

echo '=== DONE ==='
ls -la my_custom_model/
