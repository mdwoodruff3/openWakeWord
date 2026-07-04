# Cedar wake-word training image for the DGX Spark (GB10 / sm_121 / aarch64).
# Base: NVIDIA NGC PyTorch 25.10 arm64 — PyTorch 2.9.0a0 + CUDA 13.0.2 + py3.12,
# built for Blackwell/GB10 (so we never fight torch/CUDA wheels).
FROM nvcr.io/nvidia/pytorch:25.10-py3

# espeak-ng -> libespeak-ng.so.1 for espeak_phonemizer; libsndfile1 backs soundfile.
RUN apt-get update && apt-get install -y --no-install-recommends \
      espeak-ng libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Torch-adjacent packages WITH --no-deps so pip can never replace NVIDIA's sm_121
# torch. torchaudio 2.9.0 matches the container's torch 2.9.0a0 minor version
# (NGC arm64 images omit torchaudio); PyPI ships a cp312 manylinux aarch64 wheel.
RUN pip install --no-cache-dir --no-deps \
      torchaudio==2.9.0 \
      speechbrain==0.5.14 \
      torch-audiomentations==0.11.0 \
      torchmetrics==1.2.0 \
      torchinfo==1.8.0

# Everything else, pinned to the known-good combo.
# Deliberate absences: NO tensorflow / onnx_tf / tflite (ONNX-only path);
# NO transformers (nothing here needs it -> the huggingface_hub 'is_offline_mode'
# skew can't occur); NO onnxruntime-gpu (no aarch64 wheels; CPU is fine).
RUN pip install --no-cache-dir \
      numpy==1.26.4 \
      soundfile librosa==0.10.2.post1 \
      onnx onnxscript onnxruntime \
      scipy scikit-learn requests tqdm pyyaml \
      hyperpyyaml sentencepiece joblib \
      julius torch-pitch-shift \
      audiomentations==0.33.0 acoustics==0.2.6 \
      mutagen==1.47.0 pronouncing==0.2.0 \
      espeak-phonemizer webrtcvad \
      "datasets==2.21.0"
# datasets 2.21 = new enough for the container's modern pyarrow (2.14.6 used the removed
# pyarrow.PyExtensionType), still old enough to load the script-based FMA dataset with
# trust_remote_code. hub/fsspec float to what datasets wants; transformers isn't installed,
# so the huggingface_hub 'is_offline_mode' skew still can't occur.

COPY sitecustomize.py /usr/local/lib/python3.12/dist-packages/sitecustomize.py
