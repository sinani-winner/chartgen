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
# Bass options per chord (Phase 3). State = chord * N_BASS + bass type; NC is last.
BASS_TYPES = ("root", "third", "fifth", "whole", "b7")   # C, C/E, C/G, C/D (D/E 꼴), C/Bb (A/G 꼴)
N_BASS = len(BASS_TYPES)
NC = N_CHORDS * N_BASS  # no chord


def state_chord(s: int) -> int | None:
    """State → chord index 0..23 (None for N.C.)."""
    return None if s == NC else s // N_BASS


def bass_offset(chord: int, btype: int) -> int:
    third = 3 if chord >= 12 else 4
    return (0, third, 7, 2, 10)[btype]


def state_bass(s: int) -> int | None:
    """State → bass pitch class."""
    c = state_chord(s)
    if c is None:
        return None
    return (c % 12 + bass_offset(c, s % N_BASS)) % 12


def chord_root(s: int) -> int | None:
    c = state_chord(s)
    return None if c is None else c % 12


def chord_is_minor(s: int) -> bool:
    c = state_chord(s)
    return c is not None and c >= 12


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


def regularize_downbeats(n_beats: int, db_idx, meter=4, switch_cost=4.0, miss_cost=0.5):
    """Pick one bar phase per beat so bars are `meter` beats long, changing phase
    only where the tracker's downbeats clearly moved (e.g. a real 2/4 bar).

    Fixes trackers that mark every 2nd beat as a downbeat in some sections.
    Viterbi over beat position 0..meter-1: +1 for a detected downbeat at
    position 0, -miss_cost for position 0 without one, -switch_cost to jump."""
    det = np.zeros(n_beats, bool)
    det[list(db_idx)] = True
    emit = np.where(det, 1.0, -miss_cost)
    score = np.where(np.arange(meter) == 0, emit[0], 0.0)
    back = np.zeros((n_beats, meter), np.int32)
    nxt = (np.arange(meter) - 1) % meter            # natural predecessor of each position
    for i in range(1, n_beats):
        stay = score[nxt]
        best = int(np.argmax(score))
        jump = score[best] - switch_cost
        use_jump = jump > stay
        back[i] = np.where(use_jump, best, nxt)
        score = np.where(use_jump, jump, stay) + np.where(np.arange(meter) == 0, emit[i], 0.0)
    pos = np.zeros(n_beats, np.int32)
    pos[-1] = int(np.argmax(score))
    for i in range(n_beats - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]
    out = [i for i in range(n_beats) if pos[i] == 0]
    return out or [0]


def make_grid(beats, downbeats, duration=None, max_beats=7, meter=4) -> Grid:
    beats = np.asarray(sorted(beats), float)
    downbeats = np.asarray(sorted(downbeats), float)
    if len(beats) < 2:
        raise ValueError("박이 2개 미만이라 박자 축을 만들 수 없습니다")
    if len(downbeats) == 0:
        downbeats = beats[::4]
    # snap downbeats to the nearest beat
    db_idx = sorted({int(np.argmin(np.abs(beats - d))) for d in downbeats})
    if meter:
        db_idx = regularize_downbeats(len(beats), db_idx, meter)
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

def peak_semitones(x: np.ndarray, compress: float = 0.5) -> np.ndarray:
    """Cached log-CQT (n, semitones) → linear magnitude kept only at local maxima
    across semitones, then mildly compressed.

    The cache stores log1p(100*|CQT|); log compression makes the leakage into
    the neighbouring semitone look almost as loud as the note itself (an E bass
    note showing up as D#). Peak picking in the linear domain removes that."""
    lin = np.expm1(np.maximum(x, 0)) / 100.0
    left = np.pad(lin, ((0, 0), (1, 0)))[:, :-1]
    right = np.pad(lin, ((0, 0), (0, 1)))[:, 1:]
    peak = (lin >= left) & (lin >= right)
    return np.where(peak, lin, 0.0) ** compress


def fold_chroma(semitones: np.ndarray, fmin_midi: int) -> np.ndarray:
    """(n, n_semitones) → (n, 12) pitch-class sum, column 0 = C."""
    n, S = semitones.shape
    out = np.zeros((n, 12), np.float32)
    for s in range(S):
        out[:, (fmin_midi + s) % 12] += semitones[:, s]
    return out


def harmonic_chroma(pooled: dict, fmin_midi: int, stems=("guitar", "piano", "other"),
                    peak: float | None = 0.5, lo_midi: int = 36) -> np.ndarray:
    """Chroma of the harmony instruments, roughly C2..C6 (vocals excluded).
    peak: compression exponent after semitone peak picking (None = old log path)."""
    acc = None
    for s in stems:
        x = pooled.get(s)
        if x is None:
            continue
        if peak is not None:
            x = np.expm1(np.maximum(x, 0)) / 100.0      # back to linear before summing stems
        acc = x.copy() if acc is None else acc + x
    if acc is None:
        acc = pooled["mix"].copy()
        if peak is not None:
            acc = np.expm1(np.maximum(acc, 0)) / 100.0
    lo = max(0, lo_midi - fmin_midi)     # below this the keys' left hand mostly doubles the bass
    hi = min(acc.shape[1], 84 - fmin_midi)  # C6
    acc[:, :lo] = 0
    acc[:, hi:] = 0
    if peak is not None:
        acc = peak_semitones(np.log1p(100.0 * acc), peak)
    else:
        # remove per-frame floor so broadband noise doesn't flatten the chroma
        acc = np.maximum(acc - np.median(acc, axis=1, keepdims=True), 0)
    return fold_chroma(acc, fmin_midi)


def bass_salience(pooled: dict, fmin_midi: int, fallback=("piano", "guitar"),
                  peak: float | None = 0.5) -> tuple:
    """Pitch-class salience of the lowest sounding note.

    Uses the bass stem's E1..B2 fundamentals. Where the bass is silent, falls
    back to the lowest register of the keys/guitar. Returns (sal (n,12), energy (n,)).
    """
    def low_pc(x, lo_midi, hi_midi):
        a, b = max(0, lo_midi - fmin_midi), max(0, hi_midi - fmin_midi)
        seg = x[:, a:b]
        if peak is not None:
            seg = peak_semitones(seg, peak)
        else:
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

def chord_templates(third_w=0.8, fifth_w=0.7, seventh_w=0.0) -> np.ndarray:
    """(n_variants, 24 labels) → returns (T (V,12), label (V,)).

    Output vocabulary is only major/minor triads, but a C chord played as Cmaj7
    or C7 (and Am as Am7) should still score as C / Am rather than as the triad
    hiding in its upper notes (Cmaj7 ⊃ Em). So each label gets extra 7th-shaped
    variants when seventh_w > 0, and the label's score is the best variant."""
    rows, labels = [], []
    for r in range(12):
        for minor in (False, True):
            third = 3 if minor else 4
            base = {r: 1.0, (r + third) % 12: third_w, (r + 7) % 12: fifth_w}
            variants = [base]
            if seventh_w > 0:
                for sev in ((10,) if minor else (11, 10)):
                    v = dict(base)
                    v[(r + sev) % 12] = seventh_w
                    variants.append(v)
            for v in variants:
                t = np.zeros(12, np.float32)
                for pc, w in v.items():
                    t[pc] = w
                rows.append(t)
                labels.append(r + (12 if minor else 0))
    return normalize_rows(np.array(rows)), np.array(labels)


# ------------------------------------------------------------------ decoding

@dataclass
class Params:
    w_harm: float = 1.0          # harmonic-stem chroma match
    w_bass: float = 0.6          # bass pitch class == root
    # bass-type priors (subtracted): how much evidence a slash chord needs over root position
    prior_third: float = 0.1
    prior_fifth: float = 0.5
    prior_whole: float = 0.4
    prior_b7: float = 0.2
    bass_change: float = 1.0     # bass-only change costs this × the position penalty
    third_w: float = 1.2
    fifth_w: float = 1.0
    seventh_w: float = 0.8       # >0: also match 7th-shaped voicings (label stays a triad)
    nc_level: float = 0.15       # score of N.C. (beats chords only when everything is quiet)
    silence_rel: float = 0.08    # frame energy below this * median → treated as silence
    # change penalties by 8th-note position in a 4/4 bar
    pen_bar: float = 0.1         # beat 1
    pen_half: float = 0.6        # beat 3
    pen_beat: float = 2.5        # beats 2, 4
    pen_push_bar: float = 1.2    # "and" of 4 → push into next bar
    pen_push_half: float = 2.5   # "and" of 2 → push into beat 3
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
    """(n_frames, N_CHORDS*N_BASS + 1) score of each (chord, bass) state per 8th note.

    Chord identity comes from the harmony stems; the bass term only asks
    "is this state's bass note what the bass is playing?". So D over an E bass
    is explained as D/E instead of turning the root into E."""
    H = normalize_rows(harm)
    T, lab = chord_templates(p.third_w, p.fifth_w, p.seventh_w)
    S = H @ T.T                                                    # cosine per variant
    C = np.full((len(H), N_CHORDS), -np.inf, np.float32)
    for v, l in enumerate(lab):
        C[:, l] = np.maximum(C[:, l], S[:, v])
    C *= p.w_harm
    B = bass_sal / np.maximum(bass_sal.sum(1, keepdims=True), 1e-6)
    prior = (0.0, p.prior_third, p.prior_fifth, p.prior_whole, p.prior_b7)
    E = np.zeros((len(H), NC + 1), np.float32)
    for c in range(N_CHORDS):
        for b in range(N_BASS):
            pc = (c % 12 + bass_offset(c, b)) % 12
            E[:, c * N_BASS + b] = C[:, c] + p.w_bass * B[:, pc] - prior[b]
    if p.smooth > 0:
        k = 2 * p.smooth + 1
        pad = np.pad(E, ((p.smooth, p.smooth), (0, 0)), mode="edge")
        E = np.stack([pad[i:i + len(E)] for i in range(k)]).mean(0)
    E[:, NC] = p.nc_level
    loud = energy > p.silence_rel * (np.median(energy[energy > 0]) if (energy > 0).any() else 1)
    E[~loud, :NC] = -1.0        # silence → N.C.
    return E


def viterbi(E: np.ndarray, change_pen: np.ndarray, bass_change: float = 1.0) -> np.ndarray:
    """E (T, K) scores with K = N_CHORDS*N_BASS + 1; change_pen (T,) cost of a
    chord change at frame t. Changing only the bass under the same chord costs
    bass_change × change_pen[t]."""
    T, K = E.shape
    group = np.array([state_chord(k) if k != NC else N_CHORDS for k in range(K)])
    score = E[0].copy()
    back = np.zeros((T, K), np.int32)
    idx = np.arange(K)
    for t in range(1, T):
        best_any = int(np.argmax(score))
        # best state inside each chord group (states of a chord are contiguous)
        blk = score[:NC].reshape(N_CHORDS, N_BASS)
        arg = blk.argmax(1)
        g_val = np.append(blk[np.arange(N_CHORDS), arg], score[NC])
        g_best = np.append(np.arange(N_CHORDS) * N_BASS + arg, NC)
        cand = score.copy()
        bp = idx.copy()
        sw = score[best_any] - change_pen[t]
        use = sw > cand
        cand = np.where(use, sw, cand)
        bp = np.where(use, best_any, bp)
        bsw = g_val[group] - bass_change * change_pen[t]
        use = bsw > cand
        cand = np.where(use, bsw, cand)
        bp = np.where(use, g_best[group], bp)
        back[t] = bp
        score = cand + E[t]
    path = np.zeros(T, np.int32)
    path[-1] = int(np.argmax(score))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(E: np.ndarray, grid: Grid, p: Params) -> np.ndarray:
    pen = np.array([position_penalty(int(pos), grid.bar_beats[int(b)], p)
                    for pos, b in zip(grid.frame_pos, grid.frame_bar)], np.float32)
    return viterbi(E, pen, p.bass_change)


def segment_confidence(E: np.ndarray, path: np.ndarray) -> np.ndarray:
    """Per-frame confidence: margin of the chosen root over the best other root
    (best state of each root), averaged over the segment, squashed to 0..1."""
    T = len(path)
    conf = np.ones(T, np.float32)
    roots = np.array([chord_root(k) if k != NC else -1 for k in range(E.shape[1])])
    s = 0
    while s < T:
        e = s
        while e < T and path[e] == path[s]:
            e += 1
        c = int(path[s])
        if c != NC:
            seg = E[s:e].mean(0)
            r = roots[c]
            other = seg[(roots != r) & (roots >= 0)].max()
            conf[s:e] = float(1 / (1 + np.exp(-(seg[c] - other) / 0.08)))
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

def prepare(feats: dict, peak: float | None = 1.0, harm_lo_midi: int = 36):
    """Cached features → (grid, harm chroma, bass salience, energy)."""
    sr, hop, fmin = int(feats["sr"]), int(feats["hop"]), int(feats["fmin_midi"])
    grid = make_grid(feats["beats"], feats["downbeats"], float(feats["duration"]))
    pooled = {k[4:]: pool_frames(v, grid, sr, hop) for k, v in feats.items() if k.startswith("cqt_")}
    harm = harmonic_chroma(pooled, fmin, peak=peak, lo_midi=harm_lo_midi)
    bass, _ = bass_salience(pooled, fmin, peak=peak)
    return grid, harm, bass, harm.sum(1) + bass.sum(1)


def recognize(feats: dict, p: Params | None = None, prepared=None):
    """Cached features → (path, conf, grid, key)."""
    p = p or Params()
    grid, harm, bass, energy = prepared or prepare(feats)
    E = emissions(harm, bass, energy, p)
    path = decode(E, grid, p)
    conf = segment_confidence(E, path)
    key = estimate_key(harm.sum(0))
    return path, conf, grid, key
