"""Decoded 8th-note chord path → Chart (same format as the reference charts)."""
from __future__ import annotations

from .chart import Bar, Chart, Directive, Event, Section
from .chords import PC_NAMES_FLAT, PC_NAMES_SHARP, parse_chord
from .harmony import NC, Grid, chord_is_minor, state_bass, state_chord

SHARP_KEYS_MAJOR = {7, 2, 9, 4, 11, 6}    # G D A E B F#
SHARP_KEYS_MINOR = {4, 11, 6, 1, 8}       # Em Bm F#m C#m G#m


def key_name(tonic: int, minor: bool) -> str:
    sharp = tonic in (SHARP_KEYS_MINOR if minor else SHARP_KEYS_MAJOR)
    return (PC_NAMES_SHARP if sharp else PC_NAMES_FLAT)[tonic] + ("m" if minor else "")


def chord_name(idx: int, key: tuple) -> str:
    """State index → chord symbol spelled for the key (slash bass when bass ≠ root)."""
    if idx == NC:
        return "N.C."
    tonic, minor = key
    sharp = tonic in (SHARP_KEYS_MINOR if minor else SHARP_KEYS_MAJOR)
    names = PC_NAMES_SHARP if sharp else PC_NAMES_FLAT
    c = state_chord(idx)
    name = names[c % 12] + ("m" if chord_is_minor(idx) else "")
    b = state_bass(idx)
    if b != c % 12:
        name += "/" + names[b]
    return name


def build_chart(path, conf, grid: Grid, key, meta: dict | None = None,
                low_conf=0.5, sections=None) -> Chart:
    """One event per chord change, placed on the 8th-note grid.

    A change on an off-beat is written as a push ('>') onto the next beat; a
    change on the last 8th of a bar is a push onto beat 1 of the next bar.
    Every bar also restates its opening chord on beat 1, like the references.

    key: (tonic, minor) for the whole song, or a list with one per bar
    (key changes become [Key: X] directives, chords are spelled per local key).
    sections: [(first_bar, end_bar, name), ...] covering all bars; default one "Song".
    """
    n_bars = len(grid.bar_start)
    bar_keys = list(key) if isinstance(key, list) else [key] * n_bars
    bar_keys = [k if isinstance(k, tuple) else (k, False) for k in bar_keys]
    bars_events = [dict() for _ in range(n_bars)]   # beat -> Event

    def make_event(beat, frame, push, bar=None):
        idx = int(path[frame])
        k = bar_keys[int(grid.frame_bar[frame]) if bar is None else bar]
        chord = None if idx == NC else parse_chord(chord_name(idx, k))
        c = round(float(conf[frame]), 2)
        return Event(beat=float(beat), chord=chord, push=push, conf=c if c < 1.0 else None)

    prev = None
    for f in range(len(path)):
        b = int(grid.frame_bar[f])
        pos = int(grid.frame_pos[f])
        nb = grid.bar_beats[b]
        if pos == 0 and 1.0 not in bars_events[b]:
            bars_events[b][1.0] = make_event(1, f, False)
        if prev is not None and path[f] != prev:
            if pos % 2 == 0:
                bars_events[b][pos / 2 + 1] = make_event(pos / 2 + 1, f, False)
            elif pos == 2 * nb - 1 and b + 1 < n_bars:
                bars_events[b + 1][1.0] = make_event(1, f, True, bar=b + 1)
            elif pos == 2 * nb - 1:
                pass   # push past the last bar: ignore
            else:
                beat = (pos + 1) / 2 + 1
                bars_events[b][beat] = make_event(beat, f, True)
        prev = path[f]

    if not sections:
        sections = [(0, n_bars, "Song")]
    out = []
    for a, e, name in sections:
        sec = Section(name=name)
        for b in range(a, e):
            if b > 0 and bar_keys[b] != bar_keys[b - 1]:
                sec.items.append(Directive("key", key_name(*bar_keys[b])))
            evs = [bars_events[b][k] for k in sorted(bars_events[b])]
            sec.items.append(Bar(events=evs, time=(grid.bar_beats[b], 4),
                                 start_sec=round(grid.bar_start[b], 3)))
        out.append(sec)
    m = {"time": (4, 4), "key": key_name(*bar_keys[0])}
    m.update(meta or {})
    return Chart(meta=m, sections=out)
