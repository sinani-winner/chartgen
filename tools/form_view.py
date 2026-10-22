"""정답 구간과 인식 구간 나란히 보기: python3 tools/form_view.py runs/<방법> 곡이름"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chartgen.chart import build_timeline, load_chart  # noqa: E402
from chartgen.scoring import align_bars  # noqa: E402


def runs(bars):
    out, last = [], None
    for i, b in enumerate(bars):
        if b.section_instance != last:
            out.append([i, i + 1, b.section_name or b.section_type, str(b.key)])
            last = b.section_instance
        else:
            out[-1][1] = i + 1
    return out


def main():
    run, song = sys.argv[1], sys.argv[2]
    ref, pred = load_chart(f"data/refs/{song}.chart"), load_chart(f"{run}/{song}.json")
    rb, pb = ref.expand(), pred.expand()
    p2r = {p: r for r, p in align_bars(build_timeline(rb), build_timeline(pb)) if r is not None and p is not None}
    rr, pr = runs(rb), runs(pb)
    print(f"{'정답 구간':34s} | 인식 구간 (정답 마디 위치)")
    for i in range(max(len(rr), len(pr))):
        a = f"{rr[i][0]+1:3d}-{rr[i][1]:3d} {rr[i][2]:14s} {rr[i][3]:3s}" if i < len(rr) else ""
        if i < len(pr):
            s, e = pr[i][0], pr[i][1]
            at = next((p2r[q] + 1 for q in range(s, e) if q in p2r), "?")
            b = f"{s+1:3d}-{e:3d} ({e-s:2d}) {pr[i][2]:11s} {pr[i][3]:3s} → 정답 {at}"
        else:
            b = ""
        print(f"{a:34s} | {b}")


if __name__ == "__main__":
    main()
