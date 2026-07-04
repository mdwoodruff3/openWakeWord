# sitecustomize.py — restore torchaudio APIs removed in 2.x, backed by soundfile.
# Baked into the container's dist-packages so it runs at every Python startup and
# fixes openwakeword/data.py, torch_audiomentations 0.11, and speechbrain at once.
try:
    import types
    import torch
    import torchaudio
    import soundfile as sf

    def _load(filepath, frame_offset=0, num_frames=-1, normalize=True,
              channels_first=True, format=None, buffer_size=4096, backend=None):
        frames = -1 if num_frames in (-1, None) else int(num_frames)
        data, sr = sf.read(str(filepath), start=int(frame_offset), frames=frames,
                           dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.T.copy() if channels_first else data)
        return wav, sr

    def _info(filepath, format=None, buffer_size=4096, backend=None):
        i = sf.info(str(filepath))
        return types.SimpleNamespace(sample_rate=i.samplerate, num_frames=i.frames,
                                     num_channels=i.channels,
                                     bits_per_sample=16, encoding="PCM_S")

    torchaudio.load = _load
    torchaudio.info = _info
    torchaudio.set_audio_backend = lambda *a, **k: None
    torchaudio.get_audio_backend = lambda *a, **k: "soundfile"
    torchaudio.list_audio_backends = lambda *a, **k: ["soundfile"]
except Exception as e:  # never brick the interpreter
    import sys
    print(f"[torchaudio shim] not applied: {e}", file=sys.stderr)
