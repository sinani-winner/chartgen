"""캐시된 곡들로 Params 탐색 (오디오 불필요, 몇 초~몇 분).

    python3 tools/tune.py                     # 기본값 점수
    python3 tools/tune.py --sweep             # 좌표 탐색 (한 번에 값 하나씩 바꿔 가며)

목표 = 근음 + 0.5*타이밍F - 0.002*가짜체인지(100마디당 x100) — 근음이 최우선 (PLAN.md 1장 순위).
5곡뿐이라 과적합 위험: 곡별 점수도 같이 보고, 한 곡만 크게 오르는 변경은 의심할 것.
"""
import argparse
import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chartgen.assemble import build_chart  # noqa: E402
from chartgen.audio import load_features  # noqa: E402
from chartgen.chart import load_chart  # noqa: E402
from chartgen.harmony import Params, prepare, recognize  # noqa: E402
from chartgen.scoring import score  # noqa: E402

GRID = {
    "w_bass": [0.4, 0.6, 0.8, 1.0, 1.2],
    "w_bass_tone": [0.2, 0.4, 0.6, 0.8],
    "seventh_w": [0.6, 0.8, 1.0, 1.2],
    "third_w": [0.8, 1.0, 1.2, 1.4],
    "fifth_w": [0.6, 0.8, 1.0],
    "pen_bar": [0.1, 0.2, 0.3, 0.5],
    "pen_half": [0.3, 0.45, 0.6, 0.8],
    "pen_beat": [2.0, 2.5, 3.0, 4.0],
    "pen_push_bar": [0.5, 0.8, 1.2],
    "pen_push_half": [1.0, 1.6, 2.5],
}

# 튜닝 전 처음 값 (과적합 점검은 여기서 출발)
START = Params(w_bass=0.8, w_bass_tone=0.0, seventh_w=0.0, third_w=0.8, fifth_w=0.7, pen_bar=0.6,
               pen_half=1.0, pen_beat=1.8, pen_push_bar=1.0, pen_push_half=1.6)


def load_songs(cache, refs):
    songs = []
    for d in sorted(Path(cache).iterdir()):
        r = Path(refs) / f"{d.name}.chart"
        if (d / "features.npz").exists() and r.exists():
            f = load_features(d / "features.npz")
            songs.append((d.name, f, prepare(f), load_chart(r)))
    return songs


def evaluate(songs, p):
    per = {}
    for name, f, prep, ref in songs:
        path, conf, grid, key = recognize(f, p, prep)
        m = score(ref, build_chart(path, conf, grid, key)).metrics
        per[name] = m
    n = len(per)
    root = sum(m["root"] for m in per.values()) / n
    tim = sum(m["timing_f"] for m in per.values()) / n
    fc = sum(m["false_changes"] for m in per.values()) / max(1, sum(m["bars_ref"] for m in per.values())) * 100
    return root + 0.5 * tim - 0.002 * fc, root, tim, fc, per


def show(tag, res):
    obj, root, tim, fc, per = res
    songs = " ".join(f"{k[:10]}:{100*m['root']:.0f}" for k, m in per.items())
    print(f"{tag:38s} 목표 {obj:.4f} | 근음 {100*root:5.1f} 타이밍 {100*tim:5.1f} 가짜/100마디 {fc:4.1f} | {songs}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(ROOT / "cache"))
    ap.add_argument("--refs", default=str(ROOT / "data" / "refs"))
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--loo", action="store_true", help="한 곡씩 빼고 튜닝 → 뺀 곡 점수 (과적합 점검)")
    a = ap.parse_args()
    songs = load_songs(a.cache, a.refs)
    if a.loo:
        tot_base = tot_loo = 0.0
        for i, held in enumerate(songs):
            train = songs[:i] + songs[i + 1:]
            p = sweep(train, START, a.rounds, verbose=False)
            b = evaluate([held], Params())[4][held[0]]["root"]
            t = evaluate([held], p)[4][held[0]]["root"]
            tot_base += b
            tot_loo += t
            print(f"{held[0]:20s} 나머지 4곡으로 튜닝 → 이 곡 근음 {100*t:5.1f} (현재 기본값 {100*b:5.1f})", flush=True)
        n = len(songs)
        print(f"평균: 빼고 튜닝 {100*tot_loo/n:.1f} / 현재 기본값 {100*tot_base/n:.1f}")
        return
    p = Params()
    show("기본값", evaluate(songs, p))
    if a.sweep:
        p = sweep(songs, p, a.rounds)
        print("\n최종:", {k: getattr(p, k) for k in GRID})


def sweep(songs, p, rounds, verbose=True):
    best = evaluate(songs, p)
    for rnd in range(rounds):
        for k, vals in GRID.items():
            for v in vals:
                if getattr(p, k) == v:
                    continue
                q = dataclasses.replace(p, **{k: v})
                res = evaluate(songs, q)
                if res[0] > best[0] + 1e-4:
                    best, p = res, q
                    if verbose:
                        show(f"[{rnd+1}] {k}={v}", res)
    return p


if __name__ == "__main__":
    main()
