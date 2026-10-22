"""구성(폼) 설정 탐색: 코드 인식은 한 번만 하고 analyze_form 값만 바꿔 가며 채점.

    python3 tools/tune_form.py            # 현재 기본값
    python3 tools/tune_form.py --sweep    # 좌표 탐색
목표 = 구간경계F + 구간순서 + 구간이름(마디별) — 5곡뿐이라 곡별 점수도 같이 볼 것.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chartgen.assemble import build_chart  # noqa: E402
from chartgen.audio import load_features  # noqa: E402
from chartgen.chart import load_chart  # noqa: E402
from chartgen.harmony import prepare, recognize, vocal_per_bar  # noqa: E402
from chartgen.scoring import score  # noqa: E402
from chartgen.structure import analyze_form  # noqa: E402

GRID = {
    "w_rep": [0.5, 1.0, 1.5, 2.0],
    "w_voc": [0.5, 1.0, 2.0],
    "w_nov": [0.0, 0.5, 1.0],
    "seg_cost": [0.5, 1.0, 1.5, 2.0],
    "cluster_thr": [0.45, 0.55, 0.65],
    "w_chroma": [0.0, 0.7, 1.0],
}
DEFAULT = {"w_rep": 1.0, "w_voc": 8.0, "w_nov": 2.0, "seg_cost": 3.0, "cluster_thr": 0.55, "w_chroma": 1.0}


def load(cache, refs):
    out = []
    for d in sorted(Path(cache).iterdir()):
        r = Path(refs) / f"{d.name}.chart"
        if (d / "features.npz").exists() and r.exists():
            f = load_features(d / "features.npz")
            prep = prepare(f)
            path, conf, grid, key = recognize(f, prepared=prep)
            out.append((d.name, path, conf, grid, key, vocal_per_bar(f, grid), load_chart(r), prep[1]))
    return out


def evaluate(songs, kw):
    per = {}
    for name, path, conf, grid, key, voc, ref, harm in songs:
        secs, keys = analyze_form(path, grid, voc, key, harm=harm, **kw)
        per[name] = score(ref, build_chart(path, conf, grid, keys, sections=secs)).metrics
    avg = {k: sum(m[k] or 0 for m in per.values()) / len(per)
           for k in ("form_boundary_f", "form_seq", "form_label", "key")}
    return avg["form_boundary_f"] + avg["form_seq"] + avg["form_label"], avg, per


def show(tag, res):
    obj, a, per = res
    songs = " ".join(f"{k[:10]}:{100*m['form_boundary_f']:.0f}/{100*m['form_seq']:.0f}/{100*m['form_label']:.0f}"
                     for k, m in per.items())
    print(f"{tag:26s} 목표 {obj:.3f} | 경계 {100*a['form_boundary_f']:4.1f} 순서 {100*a['form_seq']:4.1f} "
          f"이름 {100*a['form_label']:4.1f} 키 {100*a['key']:4.1f} | {songs}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--rounds", type=int, default=2)
    a = ap.parse_args()
    songs = load(ROOT / "cache", ROOT / "data" / "refs")
    kw = dict(DEFAULT)
    best = evaluate(songs, kw)
    show("기본값", best)
    if not a.sweep:
        return
    for rnd in range(a.rounds):
        for k, vals in GRID.items():
            for v in vals:
                if kw[k] == v:
                    continue
                res = evaluate(songs, {**kw, k: v})
                if res[0] > best[0] + 1e-4:
                    best, kw = res, {**kw, k: v}
                    show(f"[{rnd+1}] {k}={v}", res)
    print("최종:", kw)


if __name__ == "__main__":
    main()
