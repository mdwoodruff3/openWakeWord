#!/usr/bin/env python3
"""Live wake-word tester: score your mic in real time, log everything, capture detections.

Run from the openWakeWord repo with the inference-only venv:

    .venv-test/bin/python test_mic_live.py                        # pretrained 'hey_jarvis'
    .venv-test/bin/python test_mic_live.py --device 1 --gain 4    # Reachy mic, boosted
    .venv-test/bin/python test_mic_live.py --model trained_cedar_wakeword/cedar.onnx
    .venv-test/bin/python test_mic_live.py --list-devices
    .venv-test/bin/python test_mic_live.py --analyze wake_test_log.jsonl   # summarize a run

What gets recorded:
- wake_test_log.jsonl — one row per 80 ms frame (timestamp, score, input dBFS), plus
  detection/start/stop events. Everything the model heard, scored, on a timeline.
- wake_clips/detect_*.wav — ~3 s of the actual audio around every detection
  (2 s before + 1 s after), so you can replay exactly what triggered it. These double
  as training data later (real positives / mined false positives for Cedar).

macOS: grant the terminal Microphone access (System Settings > Privacy & Security).
"""

from __future__ import annotations

import json
import time
import wave
import queue
import argparse
from pathlib import Path
from datetime import datetime
from collections import deque

import numpy as np

RATE = 16000
CHUNK = 1280  # 80 ms — openWakeWord's native frame
PRE_S, POST_S = 2.0, 1.0  # audio saved around a detection


def analyze(log_path: Path) -> int:
    """Summarize a JSONL log: detections with the score ramp around each."""
    frames: list[dict] = []
    detections: list[dict] = []
    meta: dict = {}
    with log_path.open() as f:
        for line in f:
            rec = json.loads(line)
            kind = rec.get("kind")
            if kind == "frame":
                frames.append(rec)
            elif kind == "detection":
                detections.append(rec)
            elif kind == "start":
                meta = rec

    print(f"Log: {log_path}  model={meta.get('model')}  threshold={meta.get('threshold')}  gain={meta.get('gain')}")
    print(f"Frames: {len(frames)} (~{len(frames) * 0.08:.0f}s of audio)   Detections: {len(detections)}\n")

    if frames:
        scores = np.array([f["score"] for f in frames])
        dbfs = np.array([f["dbfs"] for f in frames])
        print("Score distribution:  "
              f">=0.9: {(scores >= 0.9).sum()}   0.5-0.9: {((scores >= 0.5) & (scores < 0.9)).sum()}   "
              f"0.1-0.5: {((scores >= 0.1) & (scores < 0.5)).sum()}   <0.1: {(scores < 0.1).sum()}")
        print(f"Input level while speaking (frames > -55 dBFS): "
              f"median {np.median(dbfs[dbfs > -55]):.1f} dBFS\n" if (dbfs > -55).any() else "\n")

    for i, d in enumerate(detections, 1):
        t = d["t"]
        idx = next((j for j, f in enumerate(frames) if f["t"] >= t), len(frames))
        before = [f["score"] for f in frames[max(0, idx - 25):idx]]          # ~2 s prior
        after = [f["score"] for f in frames[idx:idx + 13]]                   # ~1 s after
        ramp = " ".join(f"{s:.2f}" for s in before[-8:])                     # last ~0.6 s
        print(f"#{i}  {t}  score={d['score']:.3f}  in={d.get('dbfs', '?')} dBFS"
              + (f"  clip={d['clip']}" if d.get("clip") else ""))
        print(f"    ramp-in: {ramp or '(no frames)'}   "
              f"peak after: {max(after):.2f}" if after else "")
    if not detections:
        print("No detections in this run.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="hey_jarvis",
                   help="pretrained name (hey_jarvis, alexa, ...) or a path to a .onnx (e.g. cedar.onnx)")
    p.add_argument("--threshold", type=float, default=0.5, help="detection threshold (default 0.5)")
    p.add_argument("--debounce", type=float, default=2.0, help="seconds between logged detections")
    p.add_argument("--log", type=Path, default=Path("wake_test_log.jsonl"), help="JSONL log file")
    p.add_argument("--clips-dir", type=Path, default=Path("wake_clips"), help="where detection WAVs go")
    p.add_argument("--no-clips", action="store_true", help="don't save detection audio")
    p.add_argument("--device", type=int, default=None, help="input device index (see --list-devices)")
    p.add_argument("--gain", type=float, default=1.0,
                   help="software input gain multiplier (try 3-8 for a quiet device like Reachy)")
    p.add_argument("--verbose", action="store_true", help="print every frame's score")
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--analyze", type=Path, metavar="LOG", help="summarize an existing JSONL log and exit")
    args = p.parse_args()

    if args.analyze:
        return analyze(args.analyze)

    import sounddevice as sd
    from openwakeword.model import Model

    if args.list_devices:
        print(sd.query_devices())
        return 0

    model_arg = str(args.model)
    model = Model(wakeword_models=[model_arg], inference_framework="onnx")
    name = Path(model_arg).stem if model_arg.endswith(".onnx") else model_arg

    audio_q: "queue.Queue[np.ndarray]" = queue.Queue()

    def on_audio(indata, frames, t, status) -> None:  # noqa: ANN001
        if status:
            print(f"[audio] {status}")
        audio_q.put(indata[:, 0].copy())

    log = args.log.open("a")

    def emit(kind: str, **kw: object) -> None:
        rec = {"t": datetime.now().isoformat(timespec="milliseconds"), "kind": kind, **kw}
        log.write(json.dumps(rec) + "\n")

    def write_clip(frames: list[np.ndarray], score: float) -> str:
        args.clips_dir.mkdir(exist_ok=True)
        path = args.clips_dir / f"detect_{datetime.now():%Y%m%d_%H%M%S}_score{score:.2f}.wav"
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(np.concatenate(frames).tobytes())
        return str(path)

    print(f"Listening for {name!r}  threshold={args.threshold}  gain={args.gain}x")
    print(f"Log: {args.log}   Clips: {'off' if args.no_clips else args.clips_dir}/")
    print("Say the wake word. Ctrl+C to stop.  (in = input level dBFS; -20..-10 speaking is healthy)\n")
    emit("start", model=name, threshold=args.threshold, gain=args.gain, device=args.device)

    ring: deque[np.ndarray] = deque(maxlen=int(PRE_S * RATE / CHUNK))
    pending: list[dict] = []  # detections still collecting their POST_S of audio
    last_fire = 0.0
    peak_since_print, peak_dbfs = 0.0, -120.0
    last_print = time.monotonic()
    detections = 0

    with sd.InputStream(samplerate=RATE, blocksize=CHUNK, channels=1, dtype="int16",
                        device=args.device, callback=on_audio):
        try:
            while True:
                frame = audio_q.get()
                if args.gain != 1.0:
                    frame = np.clip(frame.astype(np.int32) * args.gain, -32768, 32767).astype(np.int16)
                ring.append(frame)
                rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
                dbfs = 20 * np.log10(max(rms, 1e-6) / 32768.0)
                peak_dbfs = max(peak_dbfs, dbfs)
                score = float(model.predict(frame).get(name, 0.0))
                now = time.monotonic()
                peak_since_print = max(peak_since_print, score)

                emit("frame", score=round(score, 4), dbfs=round(dbfs, 1))  # every 80 ms

                # finish any clips still collecting post-detection audio
                for pend in pending[:]:
                    pend["frames"].append(frame)
                    pend["remaining"] -= 1
                    if pend["remaining"] <= 0:
                        clip = write_clip(pend["frames"], pend["score"])
                        emit("clip", n=pend["n"], clip=clip)
                        print(f"   saved {clip}")
                        pending.remove(pend)

                if args.verbose and score > 0.01:
                    print(f"  score={score:.3f}  in={dbfs:5.1f} dBFS")
                if score >= args.threshold and (now - last_fire) >= args.debounce:
                    last_fire = now
                    detections += 1
                    stamp = datetime.now().strftime("%H:%M:%S")
                    print(f"\033[92m✔ DETECTED #{detections}  score={score:.3f}  in={dbfs:5.1f} dBFS  {stamp}\033[0m")
                    emit("detection", n=detections, score=round(score, 4), dbfs=round(dbfs, 1))
                    if not args.no_clips:
                        pending.append({"n": detections, "score": score,
                                        "frames": list(ring),
                                        "remaining": int(POST_S * RATE / CHUNK)})
                elif now - last_print >= 1.0:  # 1 Hz console meter (log already has every frame)
                    bar = "#" * int(peak_since_print * 30)
                    print(f"  peak {peak_since_print:.3f}  in {peak_dbfs:6.1f} dBFS |{bar:<30}|", end="\r")
                    peak_since_print, peak_dbfs, last_print = 0.0, -120.0, now
                log.flush()
        except KeyboardInterrupt:
            for pend in pending:  # flush any clip still collecting
                clip = write_clip(pend["frames"], pend["score"])
                emit("clip", n=pend["n"], clip=clip)
            print(f"\n\nStopped. {detections} detection(s). Log: {args.log}")
            print(f"Review: .venv-test/bin/python test_mic_live.py --analyze {args.log}")
            emit("stop", detections=detections)
            log.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
