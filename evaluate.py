#!/usr/bin/env python3
"""채점기: 예측 차트를 정답 차트와 항목별로 비교한다.

사용법
  python evaluate.py data/refs runs/baseline            # 폴더끼리 (파일 이름으로 짝지음)
  python evaluate.py data/refs/song1.chart out.json     # 한 곡
  python evaluate.py data/refs runs/baseline --diff     # 틀린 마디 목록도 출력
  python evaluate.py data/refs runs/baseline --log "베이스 최저음 근음 추정"   # EXPERIMENTS.md 에 기록

정답 파일: .chart (텍스트, docs/CHART_FORMAT.md)
예측 파일: .chart 또는 .json (같은 이름)
'_'로 시작하는 파일은 건너뜀.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chartgen.chart import ChartError, load_chart  # noqa: E402
from chartgen.chords import ChordParseError  # noqa: E402
from chartgen.scoring import aggregate, score  # noqa: E402

COLUMNS = [  # (key, header, kind)
    ("root", "1근음", "pct"),
    ("timing_f", "2타이밍F", "pct"),
    ("timing_f_exact", "2정확", "pct"),
    ("quality", "3성격", "pct"),
    ("bass", "4베이스", "pct"),
    ("form_boundary_f", "5경계F", "pct"),
    ("form_seq", "5순서", "pct"),
    ("key", "6키", "pct"),
    ("mod_f", "6전조F", "pct"),
    ("false_changes", "7가짜체인지", "int"),
    ("false_tensions", "7가짜텐션", "int"),
    ("free_f", "프리F", "pct"),
    ("stop_f", "스탑F", "pct"),
    ("chord_full", "전체일치", "pct"),
]


def width(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def pad(s: str, w: int, right=True) -> str:
    sp = " " * max(w - width(s), 0)
    return sp + s if right else s + sp


def fmt(v, kind):
    if v is None:
        return "-"
    if kind == "pct":
        return f"{100 * v:.1f}"
    return str(v)


def find_pairs(ref: Path, pred: Path):
    if ref.is_file():
        if not pred.is_file():
            sys.exit(f"예측 파일이 없습니다: {pred}")
        return [(ref.stem, ref, pred)], []
    pairs, missing = [], []
    for r in sorted(ref.glob("*.chart")):
        if r.name.startswith("_"):
            continue
        cands = [pred / (r.stem + ext) for ext in (".chart", ".json")]
        hit = next((c for c in cands if c.exists()), None)
        if hit:
            pairs.append((r.stem, r, hit))
        else:
            missing.append(r.stem)
    return pairs, missing


def print_table(per_song: dict, agg: dict):
    names = list(per_song) + ["평균"]
    name_w = max(width(n) for n in names) + 1
    widths = [max(width(h), 6) for _, h, _ in COLUMNS]
    print(pad("곡", name_w, False) + " ".join(pad(h, w) for (_, h, _), w in zip(COLUMNS, widths)))
    rows = list(per_song.items()) + [("평균", agg)]
    for name, m in rows:
        if name == "평균":
            print("-" * (name_w + sum(widths) + len(widths)))
        print(pad(name, name_w, False) + " ".join(
            pad(fmt(m.get(k), kind), w) for (k, _, kind), w in zip(COLUMNS, widths)))
    print()
    print("점수는 %, 7번(가짜 체인지/텐션)은 개수(낮을수록 좋음), '-'는 해당 없음.")
    print("2타이밍F: 8분음표 하나 차이까지 허용 / 2정확: 푸시까지 정확히 맞아야 인정")
    if agg.get("false_changes_per100") is not None:
        print(f"가짜 체인지 {agg['false_changes_per100']:.1f}개 / 가짜 텐션 {agg['false_tensions_per100']:.1f}개 (100마디당)")


def log_experiment(path: Path, pred_dir: str, note: str, per_song: dict, agg: dict, missing):
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    hdr = "| 곡 | " + " | ".join(h for _, h, _ in COLUMNS) + " |"
    sep = "|---|" + "---:|" * len(COLUMNS)
    lines = [f"\n## {now} — `{pred_dir}`", "", f"메모: {note}", "", hdr, sep]
    for name, m in list(per_song.items()) + [("**평균**", agg)]:
        lines.append(f"| {name} | " + " | ".join(fmt(m.get(k), kind) for k, _, kind in COLUMNS) + " |")
    if missing:
        lines += ["", f"예측 없음(채점 제외): {', '.join(missing)}"]
    if not path.exists():
        path.write_text("# 실험 기록\n\n`evaluate.py --log` 로 자동 누적. 같은 정답 세트로만 비교할 것.\n", encoding="utf-8")
    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"→ {path} 에 기록했습니다")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref", type=Path, help="정답 .chart 파일 또는 폴더")
    ap.add_argument("pred", type=Path, help="예측 .chart/.json 파일 또는 폴더")
    ap.add_argument("--diff", action="store_true", help="틀린 마디 목록 출력")
    ap.add_argument("--log", metavar="메모", help="EXPERIMENTS.md 에 결과 추가")
    ap.add_argument("--json", type=Path, help="상세 결과를 JSON으로 저장")
    args = ap.parse_args(argv)

    pairs, missing = find_pairs(args.ref, args.pred)
    if not pairs:
        sys.exit("채점할 곡이 없습니다 (정답 폴더에 .chart 파일, 예측 폴더에 같은 이름의 파일 필요)")

    per_song, diffs = {}, {}
    for name, r, p in pairs:
        try:
            ref, pred = load_chart(r), load_chart(p)
        except (ChartError, ChordParseError) as e:
            sys.exit(f"{name}: 차트를 읽을 수 없습니다 → {e}")
        s = score(ref, pred)
        per_song[name] = s.metrics
        diffs[name] = s.diffs

    agg = aggregate(per_song)
    print_table(per_song, agg)
    if missing:
        print(f"\n예측 파일 없음 (채점 제외): {', '.join(missing)}")

    for name, m in per_song.items():
        ex = m.get("false_tension_examples")
        if ex:
            print(f"\n[{name}] 가짜 텐션 예: " + "; ".join(ex))
        if m["bars_ref"] != m["bars_pred"]:
            print(f"[{name}] 마디 수: 정답 {m['bars_ref']} / 예측 {m['bars_pred']}")
    if args.diff:
        for name, d in diffs.items():
            print(f"\n=== {name}: 틀린 마디 {len(d)}개")
            print("\n".join(d) if d else "  없음")

    if args.json:
        args.json.write_text(json.dumps({"per_song": per_song, "aggregate": agg, "missing": missing},
                                        ensure_ascii=False, indent=2), encoding="utf-8")
    if args.log is not None:
        log_experiment(Path(__file__).resolve().parent / "EXPERIMENTS.md", str(args.pred),
                       args.log, per_song, agg, missing)


if __name__ == "__main__":
    main()
