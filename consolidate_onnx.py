#!/usr/bin/env python3
"""Merge an external-data ONNX (model.onnx + model.onnx.data) into one self-contained file.

openWakeWord's trainer exports the model as a tiny graph (.onnx) plus a separate
weights file (.onnx.data). The Reachy app ships only `*.onnx`, so the model must be
a single file. This loads both and rewrites the .onnx with weights embedded.

    .venv-test/bin/python consolidate_onnx.py cedar.onnx

Requires cedar.onnx.data to sit next to cedar.onnx. Verifies the result loads and
scores standalone afterward.
"""

import sys
from pathlib import Path

import onnx


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "cedar.onnx")
    data = path.with_suffix(path.suffix + ".data")
    if not path.is_file():
        print(f"ERROR: {path} not found")
        return 1
    if not data.is_file():
        print(f"ERROR: sidecar weights {data} not found — download it from Colab "
              f"(files.download('my_custom_model/{data.name}')) next to {path.name}")
        return 1

    print(f"Loading {path} + {data.name} ...")
    model = onnx.load(str(path))  # pulls in the external .data from the same dir

    # Re-save with all tensors embedded in the single .onnx file.
    onnx.save_model(model, str(path), save_as_external_data=False)
    onnx.checker.check_model(str(path))
    size_kb = path.stat().st_size / 1024
    print(f"Consolidated -> {path} ({size_kb:.0f} KB, self-contained)")

    # Prove it loads and scores standalone (no sidecar needed).
    import numpy as np
    from openwakeword.model import Model
    m = Model(wakeword_models=[str(path)], inference_framework="onnx")
    out = m.predict(np.zeros(1280, dtype=np.int16))
    print("Standalone load OK. prediction keys:", list(out.keys()))
    if path.stem not in out:
        print(f"WARNING: expected key {path.stem!r} (the app reads by file stem); got {list(out.keys())}")
    else:
        print(f"OK: key {path.stem!r} matches the app's WAKE_WORD_MODEL stem.")
    print(f"\nYou can delete the sidecar now: rm {data}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
