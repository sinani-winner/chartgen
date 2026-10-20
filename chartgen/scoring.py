"""Score a predicted chart against a reference chart.

Both charts are expanded to the bar sequence actually played and laid out on
an 8th-note grid. Predicted bars are aligned to reference bars with a
dynamic-programming match on roots, so one missing or extra bar does not shift
every later bar into "wrong". Unmatched reference bars count as wrong;
changes inside unmatched predicted bars count as false changes.

Metrics follow PLAN.md section 4:
  1 root          root accuracy over reference 8th-note frames
  2 timing        change-point F1 (tolerance: one 8th note; whole bar in free bars)
  3 quality       chord quality accuracy on frames where the root is right
  4 bass          bass-note accuracy
  5 form          section boundary F1 (+-1 bar), section-sequence accuracy, bar count
  6 key           per-bar key accuracy and modulation-point F1 (+-1 bar, same target key)
  7 false         false changes and false tensions (counts, lower is better)
  +  free/stop    free-bar F1 and stop/break F1 (tempo-related markings)
"""
from __future__ import annotations

from dataclasses import dataclass

from .chart import Chart, Timeline, bar_frames, build_timeline, _same

MISSING = object()   # reference frame that has no aligned predicted bar
GAP_COST = 1.0


# ---------------------------------------------------------------- alignment

def _root_eq(a, b) -> bool:
    if a is MISSING or b is MISSING:
        return False
    if a is None or b is None:
        return a is None and b is None
    return a.root == b.root


def _bar_frames(tl: Timeline, i: int):
    s = tl.bar_start[i]
    return tl.frames[s:s + bar_frames(tl.bars[i].time)]


def _bar_sim(ref_tl, r, pred_tl, p) -> float:
    rf = _bar_frames(ref_tl, r)
    pf = _bar_frames(pred_tl, p)
    if not rf or not pf:
        return 0.0
    hits = sum(_root_eq(rf[t], pf[t * len(pf) // len(rf)]) for t in range(len(rf)))
    return hits / len(rf)


def align_bars(ref_tl: Timeline, pred_tl: Timeline):
    """Return list of (ref_bar | None, pred_bar | None) in order."""
    R, P = len(ref_tl.bars), len(pred_tl.bars)
    INF = float("inf")
    cost = [[INF] * (P + 1) for _ in range(R + 1)]
    back = [[None] * (P + 1) for _ in range(R + 1)]
    cost[0][0] = 0.0
    for i in range(R + 1):
        for j in range(P + 1):
            c = cost[i][j]
            if c == INF:
                continue
            if i < R and j < P:
                v = c + 1.0 - _bar_sim(ref_tl, i, pred_tl, j)
                if v < cost[i + 1][j + 1]:
                    cost[i + 1][j + 1], back[i + 1][j + 1] = v, (i, j)
            if i < R and c + GAP_COST < cost[i + 1][j]:
                cost[i + 1][j], back[i + 1][j] = c + GAP_COST, (i, j)
            if j < P and c + GAP_COST < cost[i][j + 1]:
                cost[i][j + 1], back[i][j + 1] = c + GAP_COST, (i, j)
    pairs = []
    i, j = R, P
    while (i, j) != (0, 0):
        pi, pj = back[i][j]
        pairs.append((pi if pi < i else None, pj if pj < j else None))
        i, j = pi, pj
    pairs.reverse()
    return pairs


# ---------------------------------------------------------------- helpers

def _match_points(ref_pts, pred_pts, tol_of):
    """Greedy nearest matching. ref_pts: list of positions; pred_pts: positions or None.
    tol_of(ref_index) -> tolerance. Returns number matched."""
    cands = []
    for i, r in enumerate(ref_pts):
        tol = tol_of(i)
        for j, p in enumerate(pred_pts):
            if p is not None and abs(p - r) <= tol:
                cands.append((abs(p - r), i, j))
    cands.sort()
    used_r, used_p = set(), set()
    for _, i, j in cands:
        if i not in used_r and j not in used_p:
            used_r.add(i)
            used_p.add(j)
    return len(used_r)


def _prf(n_match, n_ref, n_pred):
    if n_ref == 0 and n_pred == 0:
        return None, None, None
    p = n_match / n_pred if n_pred else 0.0
    r = n_match / n_ref if n_ref else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def _edit_distance(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(b)]


def bar_chord_text(tl: Timeline, i: int) -> str:
    """Compact text of the chords sounding in bar i, e.g. 'Em7 D/F#@3'."""
    b = tl.bars[i]
    s = tl.bar_start[i]
    n = bar_frames(b.time)
    fpb = 8 // b.time[1]
    out = []
    prev = MISSING
    for t in range(n):
        c = tl.frames[s + t]
        if prev is MISSING or not _same(prev, c):
            name = "N.C." if c is None else str(c)
            out.append(name if t == 0 else f"{name}@{1 + t / fpb:g}")
        prev = c
    return " ".join(out)


# ---------------------------------------------------------------- scoring

@dataclass
class SongScore:
    metrics: dict
    diffs: list           # human-readable wrong bars


def score(ref: Chart, pred: Chart) -> SongScore:
    rb, pb = ref.expand(), pred.expand()
    rt, pt = build_timeline(rb), build_timeline(pb)
    pairs = align_bars(rt, pt)
    r2p = {r: p for r, p in pairs if r is not None and p is not None}
    p2r = {p: r for r, p in pairs if r is not None and p is not None}

    # reference frame -> predicted label
    pred_at = [MISSING] * rt.n_frames
    for r, p in r2p.items():
        rs, ps = rt.bar_start[r], pt.bar_start[p]
        lr, lp = bar_frames(rb[r].time), bar_frames(pb[p].time)
        for t in range(lr):
            pred_at[rs + t] = pt.frames[ps + t * lp // lr]

    # predicted frame -> reference frame position (None if its bar is unaligned)
    def map_pred_frame(f):
        p = pt.bar_of_frame(f)
        if p not in p2r:
            return None
        r = p2r[p]
        lr, lp = bar_frames(rb[r].time), bar_frames(pb[p].time)
        return rt.bar_start[r] + (f - pt.bar_start[p]) * lr // lp

    # ---- 1 root / 3 quality / 4 bass / overall
    n = rt.n_frames
    root_ok = q_n = q_ok = tri_ok = bass_n = bass_ok = full_ok = 0
    for f in range(n):
        rc, pc = rt.frames[f], pred_at[f]
        if _root_eq(rc, pc):
            root_ok += 1
        if pc is not MISSING and _same(rc, pc):
            full_ok += 1
        if rc is None:
            continue
        bass_n += 1
        if pc is not MISSING and pc is not None:
            if pc.bass_pc == rc.bass_pc:
                bass_ok += 1
            if pc.root == rc.root:
                q_n += 1
                q_ok += pc.intervals == rc.intervals
                tri_ok += pc.triad == rc.triad

    # ---- 2 timing / 7 false changes
    ref_changes = rt.changes()
    pred_changes = [map_pred_frame(f) for f in pt.changes()]

    def tol_of(i, exact=False):
        b = rb[rt.bar_of_frame(ref_changes[i])]
        if b.free:
            return bar_frames(b.time)
        return 0 if exact else 1

    m_tol = _match_points(ref_changes, pred_changes, tol_of)
    m_exact = _match_points(ref_changes, pred_changes, lambda i: tol_of(i, exact=True))
    tp, tr, tf = _prf(m_tol, len(ref_changes), len(pred_changes))
    _, _, tf_exact = _prf(m_exact, len(ref_changes), len(pred_changes))
    false_changes = len(pred_changes) - m_tol

    # ---- 7 false tensions
    false_tension = 0
    tension_examples = []
    for s, e, c in pt.segments():
        if c is None:
            continue
        mapped = [map_pred_frame(f) for f in range(s, e)]
        mapped = [m for m in mapped if m is not None]
        if not mapped:
            continue
        counts = {}
        for m in mapped:
            k = rt.frames[m]
            key = None if k is None else k.key()
            counts.setdefault(key, [0, k])[0] += 1
        _, rc = max(counts.values(), key=lambda v: v[0])
        if rc is not None and rc.root == c.root and (c.tensions() - rc.tensions()):
            false_tension += 1
            if len(tension_examples) < 5:
                tension_examples.append(f"{c} (정답 {rc}, {rt.bar_of_frame(mapped[0]) + 1}마디)")

    # ---- 5 form
    def boundaries(bars):
        return [i for i in range(1, len(bars)) if bars[i].section_instance != bars[i - 1].section_instance]

    def map_pred_bar(p):
        if p in p2r:
            return p2r[p]
        for q in range(p + 1, len(pb)):       # next aligned bar
            if q in p2r:
                return p2r[q]
        return None

    rbnd = boundaries(rb)
    pbnd = [map_pred_bar(p) for p in boundaries(pb)]
    bp, br, bf = _prf(_match_points(rbnd, pbnd, lambda i: 1), len(rbnd), len(pbnd))
    sec_label_ok = sum(1 for r, p in r2p.items() if rb[r].section_type == pb[p].section_type)

    def seq(bars):
        out, last = [], None
        for b in bars:
            if b.section_instance != last:
                out.append(b.section_type)
                last = b.section_instance
        return out

    rseq, pseq = seq(rb), seq(pb)
    seq_acc = 1 - _edit_distance(rseq, pseq) / max(len(rseq), len(pseq), 1)

    # ---- 6 key
    key_n = key_ok = 0
    for r, b in enumerate(rb):
        if b.key is None:
            continue
        key_n += 1
        if r in r2p and pb[r2p[r]].key == b.key:
            key_ok += 1

    def mods(bars):
        return [(i, bars[i].key) for i in range(1, len(bars))
                if bars[i].key is not None and bars[i - 1].key is not None and bars[i].key != bars[i - 1].key]

    rmods = mods(rb)
    pmods = [(map_pred_bar(i), k) for i, k in mods(pb)]
    mod_match = 0
    used = set()
    for ri, rk in rmods:
        for j, (pi, pk) in enumerate(pmods):
            if j not in used and pi is not None and abs(pi - ri) <= 1 and pk == rk:
                used.add(j)
                mod_match += 1
                break
    _, _, mod_f = _prf(mod_match, len(rmods), len(pmods))

    # ---- free / stop markings
    def marked(bars, pred_side, flagset):
        idx = [i for i, b in enumerate(bars) if (b.free if flagset == "free" else any(f in flagset for f in b.flags))]
        return [map_pred_bar(i) for i in idx] if pred_side else idx

    rf_ = marked(rb, False, "free")
    pf_ = marked(pb, True, "free")
    _, _, free_f = _prf(_match_points(rf_, pf_, lambda i: 0), len(rf_), len(pf_))
    stops = {"stop", "break", "fermata"}
    rs_ = marked(rb, False, stops)
    ps_ = marked(pb, True, stops)
    _, _, stop_f = _prf(_match_points(rs_, ps_, lambda i: 1), len(rs_), len(ps_))

    def ratio(a, b):
        return a / b if b else None

    metrics = {
        "root": ratio(root_ok, n),
        "timing_f": tf, "timing_p": tp, "timing_r": tr, "timing_f_exact": tf_exact,
        "quality": ratio(q_ok, q_n), "quality_triad": ratio(tri_ok, q_n),
        "bass": ratio(bass_ok, bass_n),
        "chord_full": ratio(full_ok, n),
        "form_boundary_f": bf, "form_label": ratio(sec_label_ok, len(rb)), "form_seq": seq_acc,
        "bars_ref": len(rb), "bars_pred": len(pb),
        "key": ratio(key_ok, key_n), "mod_f": mod_f,
        "mods_ref": len(rmods), "mods_pred": len(pmods),
        "free_f": free_f, "stop_f": stop_f,
        "false_changes": false_changes,
        "false_tensions": false_tension,
        "false_changes_per100": 100 * false_changes / max(len(rb), 1),
        "false_tension_examples": tension_examples,
    }

    # ---- per-bar diffs for humans
    diffs = []
    for r, p in pairs:
        if r is None:
            diffs.append(f"  (정답에 없는 마디) 예측 {p + 1}: {bar_chord_text(pt, p)}")
            continue
        rtxt = bar_chord_text(rt, r)
        if p is None:
            diffs.append(f"  {r + 1:>3} [{rb[r].section_name}] 정답: {rtxt:<24} 예측: (빠짐)")
            continue
        s = rt.bar_start[r]
        frames_ok = all(pred_at[s + t] is not MISSING and _same(rt.frames[s + t], pred_at[s + t])
                        for t in range(bar_frames(rb[r].time)))
        if not frames_ok:
            diffs.append(f"  {r + 1:>3} [{rb[r].section_name}] 정답: {rtxt:<24} 예측: {bar_chord_text(pt, p)}")
    return SongScore(metrics=metrics, diffs=diffs)


# ---------------------------------------------------------------- aggregation

RATE_KEYS = ["root", "timing_f", "quality", "bass", "form_boundary_f", "form_seq",
             "key", "mod_f", "free_f", "stop_f", "chord_full", "timing_f_exact", "quality_triad", "form_label"]
COUNT_KEYS = ["false_changes", "false_tensions"]


def aggregate(per_song: dict) -> dict:
    out = {}
    for k in RATE_KEYS:
        vals = [m[k] for m in per_song.values() if m.get(k) is not None]
        out[k] = sum(vals) / len(vals) if vals else None
    for k in COUNT_KEYS:
        out[k] = sum(m[k] for m in per_song.values())
    bars = sum(m["bars_ref"] for m in per_song.values())
    out["false_changes_per100"] = 100 * out["false_changes"] / bars if bars else None
    out["false_tensions_per100"] = 100 * out["false_tensions"] / bars if bars else None
    out["songs"] = len(per_song)
    return out
