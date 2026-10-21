"""Phase 2 recognizer on synthetic signals (code correctness only, not a score)."""
import numpy as np
import pytest

from chartgen.assemble import build_chart
from chartgen.chart import chart_to_dict, chart_from_dict, expand, parse_chart
from chartgen.harmony import (NC, Params, make_grid, position_penalty, recognize,
                              chord_root, estimate_key)
from chartgen.scoring import score

librosa = pytest.importorskip("librosa")
from chartgen import audio  # noqa: E402

SR = 44100


def tone(midi, dur, sr=SR, amp=0.2):
    t = np.arange(int(dur * sr)) / sr
    f = 440 * 2 ** ((midi - 69) / 12)
    env = np.minimum(1, np.minimum(t / 0.01, (dur - t) / 0.02 + 1e-9))
    return amp * env * (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t))


def render(chords, bpm=100.0):
    """chords: list of (start_8th, root_pc, minor) on a 4/4 grid → (bass, harm, beats, downbeats)."""
    eighth = 30 / bpm
    n8 = chords[-1][0] + 8
    total = n8 * eighth + 0.5
    bass = np.zeros(int(total * SR))
    harm = np.zeros_like(bass)
    for i, (s, r, m) in enumerate(chords):
        e = chords[i + 1][0] if i + 1 < len(chords) else n8
        a, d = s * eighth, (e - s) * eighth
        ia = int(a * SR)
        b = tone(36 + r, d, amp=0.4)
        bass[ia:ia + len(b)] += b
        for iv in (0, 3 if m else 4, 7):
            h = tone(60 + (r + iv) % 12, d)
            harm[ia:ia + len(h)] += h
    beats = np.arange(0, n8, 2) * eighth
    return bass, harm, beats, beats[::4], total


def feats_from(bass, harm, beats, downbeats, total):
    return {"sr": audio.SR, "hop": audio.HOP, "fmin_midi": audio.CQT_FMIN_MIDI,
            "beats": beats, "downbeats": downbeats, "duration": total,
            "cqt_bass": audio.cqt_logmag(bass), "cqt_piano": audio.cqt_logmag(harm),
            "cqt_mix": audio.cqt_logmag(bass + harm)}


def test_grid_bars_and_positions():
    beats = np.arange(12) * 0.5
    g = make_grid(beats, beats[::4], duration=6.0)
    assert len(g.bar_start) == 3 and g.bar_beats == [4, 4, 4]
    assert list(g.frame_pos[:8]) == list(range(8))
    assert np.allclose(g.frame_t1[:2], [0.25, 0.5])


def test_penalties_prefer_strong_beats():
    p = Params()
    assert position_penalty(0, 4, p) < position_penalty(4, 4, p) < position_penalty(2, 4, p)
    assert position_penalty(7, 4, p) < position_penalty(5, 4, p)


def test_roots_and_push_end_to_end():
    # | C | Am | F | >G |   (push: G arrives on the "and" of beat 4 of bar 3)
    C, A, F, G = 0, 9, 5, 7
    seq = [(0, C, False), (8, A, True), (16, F, False), (23, G, False)]
    feats = feats_from(*render(seq))
    path, conf, grid, key = recognize(feats)
    chart = build_chart(path, conf, grid, key)
    ref = parse_chart("Key: C\n| C | Am | F | >G |")
    s = score(ref, chart).metrics
    assert s["root"] >= 0.95, s
    assert s["timing_f_exact"] == 1.0, chart.to_text()
    assert s["false_changes"] == 0
    # JSON round trip keeps the push
    again = chart_from_dict(chart_to_dict(chart))
    assert any(e.push for b in expand(again) for e in b.events)


def test_short_passing_note_is_not_a_change():
    # One 8th-note E-major blip in the middle of a C bar must not become a chord.
    seq = [(0, 0, False), (3, 4, False), (4, 0, False), (8, 5, False), (16, 7, False)]
    feats = feats_from(*render(seq))
    path, conf, grid, key = recognize(feats)
    roots = [chord_root(int(c)) for c in path[:8]]
    assert roots.count(4) == 0, roots


def test_key_estimate_major():
    prof = np.zeros(12)
    for pc, w in [(7, 3), (11, 1), (2, 2), (0, 1.5), (4, 1), (9, 1), (6, 0.7)]:
        prof[pc] = w
    assert estimate_key(prof) == (7, False)
