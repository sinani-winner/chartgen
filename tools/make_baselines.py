#!/usr/bin/env python3
"""Build reference baselines from the answer charts (no audio involved).

These are NOT recognizer output. They answer: "if a player walked in with a
typical simplified chord chart, how would it score against this live arrangement?"
That is the floor a recognizer has to beat.

  simple_chart    tensions, slash basses and pushes removed (chord-site level)
  simple_form     simple_chart + repeats collapsed to the "original song" form
                  (x2 -> x1, back-to-back identical sections merged)

  python tools/make_baselines.py data/refs runs
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chartgen.chart import Bar, Event, Section, load_chart  # noqa: E402
from chartgen.chords import Chord  # noqa: E402


def simplify_chord(c: Chord | None) -> Chord | None:
    if c is None:
        return None
    triad = "maj" if c.triad == "5" else c.triad
    return Chord(root=c.root, triad=triad, seventh=None, ext=frozenset(), bass=None)


def simplify_bar(b: Bar) -> Bar:
    events = [Event(e.beat, simplify_chord(e.chord), False, None) for e in b.events]
    return replace(b, events=events, flags=[f for f in b.flags if f == "free"])


def simple_chart(chart):
    for s in chart.sections:
        s.items = [simplify_bar(it) if isinstance(it, Bar) else it for it in s.items]
    return chart


def _bars_sig(sec: Section):
    return [tuple((e.beat, e.chord) for e in it.events) for it in sec.items if isinstance(it, Bar)]


def simple_form(chart):
    chart = simple_chart(chart)
    out = []
    for s in chart.sections:
        s.repeat = 1
        if out and out[-1].type == s.type and _bars_sig(out[-1]) == _bars_sig(s) \
                and not any(not isinstance(i, Bar) for i in s.items):
            continue
        out.append(s)
    chart.sections = out
    return chart


def main():
    refs, runs = Path(sys.argv[1]), Path(sys.argv[2])
    for name, fn in (("simple_chart", simple_chart), ("simple_form", simple_form)):
        d = runs / name
        d.mkdir(parents=True, exist_ok=True)
        for f in sorted(refs.glob("*.chart")):
            if f.name.startswith("_"):
                continue
            c = fn(load_chart(f))
            (d / (f.stem + ".json")).write_text(json.dumps(c.to_dict(), ensure_ascii=False, indent=1),
                                                encoding="utf-8")
        print(f"{d}: 생성 완료")


if __name__ == "__main__":
    main()
