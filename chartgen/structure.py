"""Phase 4: local key / modulation and song form, from the decoded chord path.

Works on the recognizer output (one state per 8th note + the bar grid), so it
can be re-tuned from the cache in seconds.
"""
from __future__ import annotations

import numpy as np

from .harmony import NC, Grid, chord_is_minor, chord_root

# How typical each chord is in a major key, by scale degree of its root.
# (major-chord score, minor-chord score); anything else scores OUT.
DEGREE_FIT = {
    0: (2.0, -0.5),    # I
    2: (0.2, 1.0),     # II (V/V) / ii
    4: (0.2, 0.7),     # III (V/vi) / iii
    5: (1.5, -0.3),    # IV / iv (borrowed)
    7: (1.5, -0.5),    # V
    9: (-0.3, 1.2),    # VI / vi
    10: (0.3, -1.0),   # bVII
}
OUT = -1.0


def chord_key_fit(state: int, tonic: int) -> float:
    if state == NC:
        return 0.0
    deg = (chord_root(state) - tonic) % 12
    fit = DEGREE_FIT.get(deg)
    if fit is None:
        return OUT
    return fit[1] if chord_is_minor(state) else fit[0]


def local_keys(path, grid: Grid, switch_cost: float = 12.0, min_run: int = 8) -> list:
    """Major key (tonic pc) per bar. Viterbi over 12 keys; bar score = average
    fit of the chords sounding in it (per 8th note). A key change costs
    switch_cost, so a key only changes when several bars clearly prefer it.
    Runs shorter than min_run bars are folded into the neighbours."""
    n_bars = len(grid.bar_start)
    S = np.zeros((n_bars, 12))
    cnt = np.zeros(n_bars)
    for f, s in enumerate(path):
        b = int(grid.frame_bar[f])
        for t in range(12):
            S[b, t] += chord_key_fit(int(s), t)
        cnt[b] += 1
    S /= np.maximum(cnt, 1)[:, None]
    score = S[0].copy()
    back = np.zeros((n_bars, 12), int)
    for b in range(1, n_bars):
        best = int(np.argmax(score))
        stay = score
        sw = score[best] - switch_cost
        use = sw > stay
        back[b] = np.where(use, best, np.arange(12))
        score = np.where(use, sw, stay) + S[b]
    keys = [0] * n_bars
    keys[-1] = int(np.argmax(score))
    for b in range(n_bars - 1, 0, -1):
        keys[b - 1] = int(back[b, keys[b]])
    # fold short runs
    changed = True
    while changed:
        changed = False
        runs = []
        for b, k in enumerate(keys):
            if runs and runs[-1][2] == k:
                runs[-1][1] = b + 1
            else:
                runs.append([b, b + 1, k])
        for i, (a, e, k) in enumerate(runs):
            if e - a < min_run and len(runs) > 1:
                nb = runs[i - 1][2] if i > 0 else runs[i + 1][2]
                for b in range(a, e):
                    keys[b] = nb
                changed = True
                break
    return keys


# ------------------------------------------------------------------ form

def bar_chords(path, grid: Grid) -> list:
    """Per bar: tuple of (root, minor) per 8th note, (-1, False) for N.C."""
    out = [[] for _ in grid.bar_start]
    for f, s in enumerate(path):
        s = int(s)
        out[int(grid.frame_bar[f])].append((-1, False) if s == NC else (chord_root(s), chord_is_minor(s)))
    return out


def bar_codes(bars) -> np.ndarray:
    """(n_bars, 8) chord code per 8th note: root*2+minor, -1 for N.C. (bars resampled to 8)."""
    out = np.full((len(bars), 8), -1, int)
    for i, b in enumerate(bars):
        if not b:
            continue
        for t in range(8):
            r, m = b[t * len(b) // 8]
            out[i, t] = -1 if r < 0 else r * 2 + int(m)
    return out


def transpose_codes(c: np.ndarray, k: int) -> np.ndarray:
    return np.where(c < 0, c, ((c // 2 + k) % 12) * 2 + c % 2)


class BarSim:
    """S[k][i, j] = fraction of 8th notes where bar i transposed by k equals bar j."""

    SHIFTS = (0, 1, 2)          # same key, or a modulation up a half/whole step

    def __init__(self, bars, chroma=None, w_chroma=0.0):
        """chroma: optional (n_bars, 8, 12) harmony chroma per 8th note; blended
        in with weight w_chroma (cosine, transposed the same way)."""
        c = bar_codes(bars)
        self.n = len(bars)
        self.S, self.P = {}, {}
        n = self.n
        if chroma is not None and w_chroma > 0:
            X = chroma / np.maximum(np.linalg.norm(chroma, axis=2, keepdims=True), 1e-9)
        for k in range(12):
            t = transpose_codes(c, k)
            M = (t[:, None, :] == c[None, :, :]).mean(2)
            if chroma is not None and w_chroma > 0:
                Xk = np.roll(X, k, axis=2)          # pitch class p → p+k
                C = np.einsum("itp,jtp->ij", Xk, X) / 8
                M = (1 - w_chroma) * M + w_chroma * C
            self.S[k] = M
            # P[i+1, j+1] = sum of M along the diagonal ending at (i, j)
            P = np.zeros((n + 1, n + 1))
            for i in range(n):
                P[i + 1, 1:] = P[i, :-1] + M[i]
            self.P[k] = P

    def best_repeat(self, s, L, shifts=(0, 1, 2, 10, 11)) -> float:
        """Best mean similarity of block [s, s+L) with any non-overlapping block [c, c+L)."""
        n = self.n
        if s + L > n:
            return 0.0
        c = np.arange(0, n - L + 1)
        c = c[np.abs(c - s) >= L]
        if not len(c):
            return 0.0
        best = 0.0
        for k in shifts:
            P = self.P[k]
            v = (P[s + L, c + L] - P[s, c]) / L
            best = max(best, float(v.max()))
        return best

    def block(self, s1, s2, L, k=0) -> float:
        if s1 + L > self.n or s2 + L > self.n or L <= 0:
            return 0.0
        return float(self.S[k][np.arange(s1, s1 + L), np.arange(s2, s2 + L)].mean())


LEN_PRIOR = {8: 0.0, 16: -0.2, 4: -0.4, 12: -0.8}


def len_prior(n: int) -> float:
    if n in LEN_PRIOR:
        return LEN_PRIOR[n]
    near = min(abs(n - 8), abs(n - 16), abs(n - 4))
    return -1.0 - 0.4 * near


def stripe_starts(sim: BarSim, look=4, back=2) -> np.ndarray:
    """How strongly a repeated passage starts at each bar: the next `look` bars
    match some other place in the song while the `back` bars before don't."""
    n = sim.n
    out = np.zeros(n + 1)
    for b in range(back, n):
        L = min(look, n - b)
        c = np.arange(back, n - L + 1)
        c = c[np.abs(c - b) >= look]
        if not len(c):
            continue
        best = 0.0
        for k in (0, 1, 2):
            P = sim.P[k]
            ahead = (P[b + L, c + L] - P[b, c]) / L
            before = (P[b, c] - P[b - back, c - back]) / back
            best = max(best, float((ahead - before).max()))
        out[b] = best
    out[0] = out[n] = 1.0
    return out


def segment(sim: BarSim, vocal, w_rep=1.0, w_voc=0.5, w_nov=0.5, seg_cost=1.0, w_stripe=2.0,
            min_len=2, max_len=17):
    """DP segmentation into sections, driven by boundary cues: phrase-length
    prior (8 bars best), vocal on/off consistency inside a section, starting
    where the material changes (novelty) or where a repeated passage starts
    (stripe), minus a fixed cost per section.

    w_rep (reward for a block that repeats elsewhere) must stay small: a
    repeated verse+chorus pair scores as well as one 16-bar block, so it
    merges sections instead of finding them."""
    n = sim.n
    voc_on = vocal > 0.35
    S0 = sim.S[0]
    nov = np.zeros(n + 1)
    for b in range(1, n):
        prev = np.mean([S0[b - 1 - i, b + i] for i in range(min(2, b, n - b))])
        nov[b] = (1 - prev) + 0.5 * (voc_on[b] != voc_on[b - 1])
    nov[0] = nov[n] = 1.0
    stripe = stripe_starts(sim) if w_stripe else np.zeros(n + 1)
    best = np.full(n + 1, -np.inf)
    back = np.zeros(n + 1, int)
    best[0] = 0.0
    for e in range(1, n + 1):
        for s in range(max(0, e - max_len), e - min_len + 1):
            if best[s] == -np.inf:
                continue
            L = e - s
            v = voc_on[s:e]
            cons = max(v.mean(), 1 - v.mean())
            rep = sim.best_repeat(s, L) * L / 8 if w_rep else 0.0
            sc = (len_prior(L) + w_rep * rep + w_voc * (cons - 1) * 2
                  + w_nov * nov[s] + w_stripe * stripe[s] - seg_cost)
            if best[s] + sc > best[e]:
                best[e], back[e] = best[s] + sc, s
    cuts, e = [], n
    while e > 0:
        cuts.append((back[e], e))
        e = back[e]
    return cuts[::-1]


def cluster(sim: BarSim, segs, thr=0.55, mod_margin=0.15):
    """Group segments with the same chord sequence (allowing transposition).
    Returns (label per segment, shift per segment relative to its cluster's first instance)."""
    labels, shifts, reps = [], [], []
    for s, e in segs:
        L = e - s
        found = None
        for ci, (rs, re) in enumerate(reps):
            m = min(L, re - rs)
            if m < max(2, 0.5 * max(L, re - rs)):
                continue
            same = sim.block(rs, s, m, 0)
            cand = [(same, 0)]
            for k in BarSim.SHIFTS[1:]:
                v = sim.block(rs, s, m, k)
                if v >= same + mod_margin:       # only call it transposed if clearly better
                    cand.append((v, k))
            v, k = max(cand)
            if v >= thr and (found is None or v > found[2]):
                found = (ci, k, v)
        if found is None:
            reps.append((s, e))
            labels.append(len(reps) - 1)
            shifts.append(0)
        else:
            labels.append(found[0])
            shifts.append(found[1])
    return labels, shifts


def name_sections(segs, labels, vocal):
    """Section type per segment from vocals + repetition."""
    n = len(segs)
    voc = [float(np.mean(vocal[s:e] > 0.35)) >= 0.5 for s, e in segs]
    names = [None] * n
    vocal_idx = [i for i in range(n) if voc[i]]
    if not vocal_idx:
        return ["Verse"] * n
    first_v, last_v = vocal_idx[0], vocal_idx[-1]
    for i in range(n):
        if not voc[i]:
            names[i] = "Intro" if i < first_v else "Outro" if i > last_v else "Interlude"
    count = {}
    for i in vocal_idx:
        count[labels[i]] = count.get(labels[i], 0) + 1
    # chorus = the repeated vocal group sung loudest (worship choruses are the
    # big, full-band part); fall back to the most repeated group
    energy = {}
    for i in vocal_idx:
        s, e = segs[i]
        energy.setdefault(labels[i], []).append(float(np.mean(vocal[s:e])))
    rep = [l for l in count if count[l] >= 2]
    last = {labels[i]: i for i in vocal_idx}
    chorus = (max(rep, key=lambda l: (round(float(np.mean(energy[l])), 1), count[l], last[l]))
              if rep else max(count, key=count.get))
    verse = labels[first_v] if labels[first_v] != chorus else None
    chorus_pos = [i for i in vocal_idx if labels[i] == chorus]
    second_chorus = chorus_pos[1] if len(chorus_pos) > 1 else n
    for i in vocal_idx:
        lab = labels[i]
        if lab == chorus:
            names[i] = "Chorus"
        elif lab == verse:
            names[i] = "Verse"
        else:
            inst = [j for j in vocal_idx if labels[j] == lab]
            before_chorus = sum(1 for j in inst if j + 1 < n and labels[j + 1] == chorus)
            if before_chorus >= max(1, 0.6 * len(inst)):
                names[i] = "Pre-Chorus"
            elif inst[0] > second_chorus:
                names[i] = "Bridge"
            else:
                names[i] = "Verse"
    return names


def bar_chroma(harm: np.ndarray, grid: Grid) -> np.ndarray:
    """(n_bars, 8, 12) harmony chroma resampled to 8 slots per bar."""
    n = len(grid.bar_start)
    out = np.zeros((n, 8, 12))
    for b in range(n):
        fr = np.where(grid.frame_bar == b)[0]
        if len(fr):
            for t in range(8):
                out[b, t] = harm[fr[t * len(fr) // 8]]
    return out


def analyze_form(path, grid: Grid, vocal_per_bar, keys, harm=None, **kw):
    """→ (sections [(first_bar, end_bar, name)], keys per bar adjusted by transposed repeats)."""
    chroma = bar_chroma(harm, grid) if harm is not None else None
    bars = bar_chords(path, grid)
    sim = BarSim(bars, chroma, kw.get("w_chroma", 1.0))
    segs = segment(sim, vocal_per_bar, **{k: v for k, v in kw.items() if k in ("w_rep", "w_voc", "w_nov", "seg_cost", "w_stripe")})
    # grouping uses chord names: in one key every chord sounds alike to chroma
    csim = BarSim(bars, chroma, kw.get("cluster_chroma", 0.3))
    labels, shifts = cluster(csim, segs, thr=kw.get("cluster_thr", 0.45))
    names = name_sections(segs, labels, vocal_per_bar)
    # a repeat that matches its first instance only after transposing = a modulation:
    # the whole segment takes the first instance's key + shift
    keys = list(keys)
    first_of = {}
    for (s, e), lab, k in zip(segs, labels, shifts):
        if lab not in first_of:
            first_of[lab] = s
        elif k:
            base = keys[first_of[lab]]
            if keys[s] == ((base[0] + k) % 12, base[1]):
                continue
            for b in range(s, e):
                keys[b] = ((base[0] + k) % 12, base[1])
    return [(s, e, nm) for (s, e), nm in zip(segs, names)], keys
