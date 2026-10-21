"""Audio → cached features (Phase 2).

Pipeline per song:
    audio file → Demucs htdemucs_6s stems → beat_this beats/downbeats (full mix)
    → per-stem CQT (log magnitude, 36 bins/octave folded to 12) saved to cache/<name>/features.npz

Only derived features are cached (no audio), so the audio and stems can be
deleted right after analysis (CLAUDE.md: 유튜브 오디오는 분석 후 삭제) and the
recognizer can still be re-tuned from the cache.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SR = 22050          # feature sample rate
HOP = 512           # ~23 ms
CQT_FMIN_MIDI = 28  # E1
CQT_OCTAVES = 6     # E1 .. ~E7
CQT_BPO = 12
STEMS = ("bass", "guitar", "piano", "other", "vocals", "drums")
FEATURE_VERSION = 1


def load_mix(path, sr=44100):
    import librosa
    y, _ = librosa.load(str(path), sr=sr, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    return y.astype(np.float32)


def separate(mix44: np.ndarray, device="cpu") -> dict:
    """Demucs htdemucs_6s. Returns {stem: mono float32 @44.1k}."""
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    model = get_model("htdemucs_6s")
    model.eval()
    wav = torch.from_numpy(mix44)
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    with torch.no_grad():
        out = apply_model(model, ((wav - mean) / std)[None], device=device,
                          split=True, overlap=0.25, progress=False)[0]
    out = out * std + mean
    return {name: out[i].mean(0).numpy() for i, name in enumerate(model.sources)}


def track_beats(mix44: np.ndarray, sr=44100):
    """beat_this on the full mix. Returns (beats_sec, downbeats_sec)."""
    from beat_this.inference import Audio2Beats
    a2b = Audio2Beats(checkpoint_path="final0", device="cpu", dbn=False)
    beats, downbeats = a2b(mix44.mean(0), sr)
    return np.asarray(beats, float), np.asarray(downbeats, float)


def cqt_logmag(y: np.ndarray, sr_in=44100) -> np.ndarray:
    """(n_bins, n_frames) log-compressed CQT magnitude, semitone bins from E1."""
    import librosa
    y = librosa.resample(y, orig_sr=sr_in, target_sr=SR)
    C = np.abs(librosa.cqt(y, sr=SR, hop_length=HOP,
                           fmin=librosa.midi_to_hz(CQT_FMIN_MIDI),
                           n_bins=CQT_OCTAVES * 36, bins_per_octave=36))
    # 3 bins per semitone → take the max so slight detuning doesn't split energy
    C = C.reshape(CQT_OCTAVES * 12, 3, -1).max(1)
    return np.log1p(100.0 * C).astype(np.float32)


def analyze(audio_path, cache_dir, device="cpu", log=print) -> Path:
    """Run separation + beats + CQT once and cache the result."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / "features.npz"
    if out.exists():
        with np.load(out) as z:
            if int(z.get("version", 0)) == FEATURE_VERSION:
                log(f"  캐시 사용: {out}")
                return out
    log(f"  오디오 읽기: {audio_path}")
    mix = load_mix(audio_path)
    log("  박자 추적 (beat_this)")
    beats, downbeats = track_beats(mix)
    log("  음원 분리 (htdemucs_6s, 몇 분 걸림)")
    stems = separate(mix, device=device)
    feats = {f"cqt_{k}": cqt_logmag(v) for k, v in stems.items()}
    feats["cqt_mix"] = cqt_logmag(mix.mean(0))
    np.savez_compressed(out, version=FEATURE_VERSION, sr=SR, hop=HOP,
                        fmin_midi=CQT_FMIN_MIDI, beats=beats, downbeats=downbeats,
                        duration=mix.shape[1] / 44100, **{k: v.astype(np.float16) for k, v in feats.items()})
    log(f"  저장: {out}")
    return out


def load_features(path) -> dict:
    with np.load(path) as z:
        return {k: (z[k].astype(np.float32) if k.startswith("cqt_") else z[k]) for k in z.files}
