"""Beat grid + root tracking (Phase 2). Pure numpy, no audio libraries.

Steps
  1. make_grid: beats/downbeats → bars → 8th-note frames (start/end seconds,
     position inside the bar). Tempo is never assumed constant.
  2. pool_frames: average CQT columns inside each 8th-note frame.
  3. chroma features: harmonic stems (guitar+piano+other) → 12-pc chroma;
     bass stem → low-register pitch-class salience.
  4. decode: Viterbi over 24 major/minor triads + N.C. per 8th-note frame.
     Changing chord costs a penalty that depends on where in the bar the change
     falls (cheap on beat 1/3, expensive on off-beats) — this is the
     "minimum harmonic rhythm" rule of PLAN.md 1-1, so passing notes and
     fills don't create fake changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

N_CHORDS = 24           # 0-11 major on root, 12-23 minor on root
NC = 24                 # no chord


def chord_root(i: int) -> int | None:
    return None if i == NC else i % 12


def chord_is_minor(i: int) -> bool:
    return 12 <= i < 24


# ------------------------------------------------------------------ grid

@dataclass
class Grid:
    bar_start: list            # seconds, one per bar
    bar_beats: list            # beats in each bar (time signature numerator)
    frame_t0: np.ndarray       # 8th-note frame start (s)
    frame_t1: np.ndarray       # 8th-note frame end (s)
    frame_bar: np.ndarray      # bar index of each frame
    frame_pos: np.ndarray      # 8th-note position inside its bar (0 = downbeat)
    bar_first_frame: list = field(default_factory=list)


def make_grid(beats, downbeats, duration=None, max_beats=7) -> Grid:
    beats = np.asarray(sorted(beats), float)
    downbeats = np.asarray(sorted(downbeats), float)
    if len(beats) < 2:
        raise ValueError("박이 2개 미만이라 박자 축을 만들 수 없습니다")
    if len(downbeats) == 0:
        downbeats = beats[::4]
    # snap downbeats to the nearest beat
    db_idx = sorted({int(np.argmin(np.abs(beats - d))) for d in downbeats})
    # final beat end: extrapolate one beat
    beat_end = np.append(beats[1:], beats[-1] + (beats[-1] - beats[-2]))
    if duration is not None:
        beat_end[-1] = min(beat_end[-1], duration)

    bar_start, bar_beats, t0, t1, fbar, fpos, first = [], [], [], [], [], [], []
    bounds = db_idx + [len(beats)]
    for bi in range(len(db_idx)):
        a, b = bounds[bi], bounds[bi + 1]
        n = b - a
        if n < 1:
            continue
        # beat tracker missed a downbeat: split very long bars into 4s
        chunks = [(a + k, min(a + k + 4, b)) for k in range(0, n, 4)] if n > max_beats else [(a, b)]
        for ca, cb in chunks:
            bar = len(bar_start)
            bar_start.append(float(beats[ca]))
            bar_beats.append(cb - ca)
            first.append(len(t0))
            for k, j in enumerate(range(ca, cb)):
                s, e = beats[j], beat_end[j]
                m = 0.5 * (s + e)
                t0 += [s, m]
                t1 += [m, e]
                fbar += [bar, bar]
                fpos += [2 * k, 2 * k + 1]
    return Grid(bar_start, bar_beats, np.array(t0), np.array(t1),
                np.array(fbar, int), np.array(fpos, int), first)


def pool_frames(feat: np.ndarray, grid: Grid, sr: int, hop: int) -> np.ndarray:
    """feat (D, T) at hop/sr seconds per column → (n_frames, D) mean per 8th note."""
    T = feat.shape[1]
    times = np.arange(T) * hop / sr
    out = np.zeros((len(grid.frame_t0), feat.shape[0]), np.float32)
    lo = np.searchsorted(times, grid.frame_t0)
    hi = np.searchsorted(times, grid.frame_t1)
    for i, (a, b) in enumerate(zip(lo, hi)):
        b = max(b, a + 1)
        if a < T:
            out[i] = feat[:, a:min(b, T)].mean(1)
    return out


# ------------------------------------------------------------------ chroma

def fold_chroma(semitones: np.ndarray, fmin_midi: int) -> np.ndarray:
    """(n, n_semitones) → (n, 12) pitch-class sum, column 0 = C."""
    n, S = semitones.shape
    out = np.zeros((n, 12), np.float32)
    for s in range(S):
        out[:, (fmin_midi + s) % 12] += semitones[:, s]
    return out


def harmonic_chroma(pooled: dict, fmin_midi: int, stems=("guitar", "piano", "other")) -> np.ndarray:
    """Chroma of the harmony instruments, roughly C2..C6 (vocals excluded)."""
    acc = None
    for s in stems:
        x = pooled.get(s)
        if x is None:
            continue
        acc = x.copy() if acc is None else acc + x
    if acc is None:
        acc = pooled["mix"].copy()
    lo = max(0, 36 - fmin_midi)          # C2
    hi = min(acc.shape[1], 84 - fmin_midi)  # C6
    acc[:, :lo] = 0
    acc[:, hi:] = 0
    # remove per-frame floor so broadband noise doesn't flatten the chroma
    acc = np.maximum(acc - np.median(acc, axis=1, keepdims=True), 0)
    return fold_chroma(acc, fmin_midi)


def bass_salience(pooled: dict, fmin_midi: int, fallback=("piano", "guitar")) -> tuple:
    """Pitch-class salience of the lowest sounding note.

    Uses the bass stem's E1..B2 fundamentals. Where the bass is silent, falls
    back to the lowest register of the keys/guitar. Returns (sal (n,12), energy (n,)).
    """
    def low_pc(x, lo_midi, hi_midi):
        a, b = max(0, lo_midi - fmin_midi), max(0, hi_midi - fmin_midi)
        seg = x[:, a:b]
        seg = np.maximum(seg - np.median(seg, axis=1, keepdims=True), 0)
        # weight lower notes more: the lowest strong partial is the bass note,
        # the octave/fifth above it are its harmonics
        w = np.linspace(1.0, 0.5, seg.shape[1])[None]
        pc = np.zeros((x.shape[0], 12), np.float32)
        for s in range(seg.shape[1]):
            pc[:, (lo_midi + s) % 12] += seg[:, s] * w[0, s]
        return pc, seg.max(1) if seg.shape[1] else np.zeros(x.shape[0])

    bass = pooled.get("bass")
    if bass is None:
        bass = pooled["mix"]
    sal, energy = low_pc(bass, 28, 52)              # E1..E3
    quiet = energy < 0.25 * (np.percentile(energy, 90) + 1e-6)
    if quiet.any():
        fb = None
        for s in fallback:
            if s in pooled:
                fb = pooled[s] if fb is None else fb + pooled[s]
        if fb is not None:
            fsal, _ = low_pc(fb, 36, 60)             # C2..C4
            sal[quiet] = fsal[quiet]
    return sal, energy


def normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-6)


# ------------------------------------------------------------------ templates

def chord_templates(third_w=0.8, fifth_w=0.7) -> np.ndarray:
    T = np.zeros((N_CHORDS, 12), np.float32)
    for r in range(12):
        T[r, [r, (r + 4) % 12, (r + 7) % 12]] = [1.0, third_w, fifth_w]
        T[12 + r, [r, (r + 3) % 12, (r + 7) % 12]] = [1.0, third_w, fifth_w]
    return normalize_rows(T)


# ------------------------------------------------------------------ decoding

@dataclass
class Params:
    w_harm: float = 1.0          # harmonic-stem chroma match
    w_bass: float = 0.8          # bass pitch class == root
    nc_level: float = 0.15       # score of N.C. (beats chords only when everything is quiet)
    silence_rel: float = 0.08    # frame energy below this * median → treated as silence
    # change penalties by 8th-note position in a 4/4 bar
    pen_bar: float = 0.6         # beat 1
    pen_half: float = 1.0        # beat 3
    pen_beat: float = 1.8        # beats 2, 4
    pen_push_bar: float = 1.0    # "and" of 4 → push into next bar (임시값, 실제 오디오로 튜닝 필요)
    pen_push_half: float = 1.6   # "and" of 2 → push into beat 3
    pen_offbeat: float = 4.0     # any other off-beat
    smooth: int = 0              # emission smoothing radius (frames); >0 blurs pushes


def position_penalty(pos: int, beats_in_bar: int, p: Params) -> float:
    """Cost of a chord change starting at 8th-note position `pos` of a bar."""
    n = 2 * beats_in_bar
    if pos == 0:
        return p.pen_bar
    if pos % 2 == 0:
        if beats_in_bar == 4 and pos == 4:
            return p.pen_half
        return p.pen_beat
    if pos == n - 1:
        return p.pen_push_bar
    if beats_in_bar == 4 and pos == 3:
        return p.pen_push_half
    return p.pen_offbeat


def emissions(harm: np.ndarray, bass_sal: np.ndarray, energy: np.ndarray, p: Params) -> np.ndarray:
    """(n_frames, 25) score of each chord per 8th-note frame (higher is better)."""
    H = normalize_rows(harm)
    E = p.w_harm * (H @ chord_templates().T)                       # cosine
    B = bass_sal / np.maximum(bass_sal.sum(1, keepdims=True), 1e-6)
    E += p.w_bass * np.concatenate([B, B], axis=1)
    if p.smooth > 0:
        k = 2 * p.smooth + 1
        pad = np.pad(E, ((p.smooth, p.smooth), (0, 0)), mode="edge")
        E = np.stack([pad[i:i + len(E)] for i in range(k)]).mean(0)
    nc = np.full((len(E), 1), p.nc_level, np.float32)
    loud = energy > p.silence_rel * (np.median(energy[energy > 0]) if (energy > 0).any() else 1)
    out = np.concatenate([E, nc], axis=1)
    out[~loud, :N_CHORDS] = -1.0        # silence → N.C.
    return out


def viterbi(E: np.ndarray, change_pen: np.ndarray) -> np.ndarray:
    """E (T, K) scores; change_pen (T,) cost of changing state at frame t."""
    T, K = E.shape
    score = E[0].copy()
    back = np.zeros((T, K), np.int32)
    for t in range(1, T):
        best_prev = int(np.argmax(score))
        switch = score[best_prev] - change_pen[t]
        stay = score
        use_switch = switch > stay
        back[t] = np.where(use_switch, best_prev, np.arange(K))
        score = np.where(use_switch, switch, stay) + E[t]
    path = np.zeros(T, np.int32)
    path[-1] = int(np.argmax(score))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(E: np.ndarray, grid: Grid, p: Params) -> np.ndarray:
    pen = np.array([position_penalty(int(pos), grid.bar_beats[int(b)], p)
                    for pos, b in zip(grid.frame_pos, grid.frame_bar)], np.float32)
    return viterbi(E, pen)


def segment_confidence(E: np.ndarray, path: np.ndarray) -> np.ndarray:
    """Per-frame confidence: margin of the chosen root over the best other root,
    averaged over the segment, squashed to 0..1."""
    T = len(path)
    conf = np.ones(T, np.float32)
    s = 0
    while s < T:
        e = s
        while e < T and path[e] == path[s]:
            e += 1
        c = int(path[s])
        if c != NC:
            seg = E[s:e, :N_CHORDS].mean(0)
            root = c % 12
            others = [seg[i] for i in range(N_CHORDS) if i % 12 != root]
            margin = seg[c] - max(others)
            conf[s:e] = float(1 / (1 + np.exp(-margin / 0.08)))
        s = e
    return conf


# ------------------------------------------------------------------ key

_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def estimate_key(chroma_sum: np.ndarray) -> tuple:
    """Krumhansl–Schmuckler. Returns (tonic_pc, minor)."""
    best, arg = -9, (0, False)
    x = chroma_sum - chroma_sum.mean()
    for minor, prof in ((False, _KS_MAJOR), (True, _KS_MINOR)):
        prof = prof - prof.mean()
        for t in range(12):
            r = float(np.dot(x, np.roll(prof, t)) / (np.linalg.norm(x) * np.linalg.norm(prof) + 1e-9))
            if r > best:
                best, arg = r, (t, minor)
    return arg


# ------------------------------------------------------------------ end to end

def recognize(feats: dict, p: Params | None = None):
    """Cached features → (path, conf, grid, key)."""
    p = p or Params()
    sr, hop, fmin = int(feats["sr"]), int(feats["hop"]), int(feats["fmin_midi"])
    grid = make_grid(feats["beats"], feats["downbeats"], float(feats["duration"]))
    pooled = {k[4:]: pool_frames(v, grid, sr, hop) for k, v in feats.items() if k.startswith("cqt_")}
    harm = harmonic_chroma(pooled, fmin)
    bass, _ = bass_salience(pooled, fmin)
    energy = harm.sum(1) + bass.sum(1)
    E = emissions(harm, bass, energy, p)
    path = decode(E, grid, p)
    conf = segment_confidence(E, path)
    key = estimate_key(harm.sum(0))
    return path, conf, grid, key
