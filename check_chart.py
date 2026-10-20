#!/usr/bin/env python3
"""정답 차트 입력 확인용: 파일을 읽어 실제로 연주되는 순서대로 마디를 펼쳐 보여준다.

  python check_chart.py data/refs/song1.chart
  python check_chart.py data/refs            # 폴더 안 전부 검사 (오류만 보고)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chartgen.chart import ChartError, build_timeline, load_chart  # noqa: E402
from chartgen.chords import ChordParseError  # noqa: E402
from chartgen.scoring import bar_chord_text  # noqa: E402


def show(path: Path):
    chart = load_chart(path)
    bars = chart.expand()
    tl = build_timeline(bars)
    m = chart.meta
    print(f"{m.get('title', path.stem)}  Key {m.get('key', '?')}  {m['time'][0]}/{m['time'][1]}"
          f"  BPM {m.get('bpm', '?')}  → 총 {len(bars)}마디, 코드 체인지 {len(tl.changes())}개")
    last_inst, last_key = None, None
    for i, b in enumerate(bars):
        if b.section_instance != last_inst:
            extra = " (free)" if b.free else ""
            print(f"\n[{b.section_name or '(이름 없음)'}] → 종류: {b.section_type}{extra}")
            last_inst = b.section_instance
        if b.key != last_key:
            print(f"   ── 키: {b.key}")
            last_key = b.key
        flags = " ".join(f"({f})" for f in b.flags if f != "free")
        ts = f" {{{b.time[0]}/{b.time[1]}}}" if b.time != tuple(m["time"]) else ""
        print(f"  {i + 1:>3}{ts}  {bar_chord_text(tl, i)}  {flags}".rstrip())


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    p = Path(sys.argv[1])
    if p.is_dir():
        bad = 0
        for f in sorted(p.glob("*.chart")):
            try:
                n = len(load_chart(f).expand())
                print(f"OK   {f.name} ({n}마디)")
            except (ChartError, ChordParseError) as e:
                bad += 1
                print(f"오류 {f.name}: {e}")
        sys.exit(1 if bad else 0)
    try:
        show(p)
    except (ChartError, ChordParseError) as e:
        sys.exit(f"오류: {e}")


if __name__ == "__main__":
    main()
