"""Download the openWakeWord training data (idempotent — skips what already exists)."""
import os
import tarfile
from pathlib import Path

import numpy as np
import scipy.io.wavfile
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from datasets import load_dataset, Dataset, Audio


def write16k(path, arr):
    scipy.io.wavfile.write(path, 16000, (np.asarray(arr) * 32767).astype(np.int16))


# 1) Precomputed openWakeWord features (17.3 GB + 185 MB) — negatives + FP validation
for fn in ["openwakeword_features_ACAV100M_2000_hrs_16bit.npy", "validation_set_features.npy"]:
    if not os.path.exists(fn):
        hf_hub_download("davidscripka/openwakeword_features", fn,
                        repo_type="dataset", local_dir=".", local_dir_use_symlinks=False)

# 2) MIT environmental RIRs (271 clips)
os.makedirs("mit_rirs", exist_ok=True)
if len(os.listdir("mit_rirs")) < 270:
    rir = load_dataset("davidscripka/MIT_environmental_impulse_responses", split="train", trust_remote_code=True)
    rir = rir.cast_column("audio", Audio(sampling_rate=16000))
    for row in tqdm(rir, desc="MIT RIRs"):
        write16k(f"mit_rirs/{Path(row['audio']['path']).name}", row["audio"]["array"])

# 3) AudioSet background audio (~2.4 GB tar -> ~2000 clips at 16 kHz)
os.makedirs("audioset_16k", exist_ok=True)
if len(os.listdir("audioset_16k")) < 1500:
    tar_path = hf_hub_download("agkphysics/AudioSet", "data/bal_train09.tar",
                               repo_type="dataset", local_dir="audioset_dl",
                               local_dir_use_symlinks=False)
    tarfile.open(tar_path).extractall("audioset_raw")
    flacs = [str(p) for p in Path("audioset_raw").rglob("*.flac")]
    ds = Dataset.from_dict({"audio": flacs}).cast_column("audio", Audio(sampling_rate=16000))
    for row in tqdm(ds, desc="AudioSet -> 16k"):
        write16k(f"audioset_16k/{Path(row['audio']['path']).stem}.wav", row["audio"]["array"])

# 4) FMA music background (~1 hour, streamed)
os.makedirs("fma", exist_ok=True)
if len(os.listdir("fma")) < 100:
    fma = load_dataset("rudraml/fma", name="small", split="train", streaming=True, trust_remote_code=True)
    fma = iter(fma.cast_column("audio", Audio(sampling_rate=16000)))
    for i in tqdm(range(120), desc="FMA (1 hr of 30s clips)"):  # 120 * 30s = 1 hour
        row = next(fma)
        write16k(f"fma/fma_{i}.wav", row["audio"]["array"])

print("data download complete")
