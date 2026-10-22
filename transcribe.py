"""오디오 → 연주용 코드 차트 JSON (Phase 2: 박자 축 + 근음 추적).

    # 분석(음원 분리·박자·CQT)은 곡당 한 번, 결과는 cache/<곡>/features.npz 에 저장
    python3 transcribe.py audio/AWC2022_05.mp3 audio/AWC2023_14.mp3 --method phase2_root

    # 캐시만으로 다시 인식 (파라미터 튜닝용, 오디오 필요 없음)
    python3 transcribe.py --from-cache --method phase2_root

    # 분석 후 오디오 삭제 (CLAUDE.md 원칙)
    python3 transcribe.py audio/*.mp3 --method phase2_root --delete-audio

오디오 파일 이름(확장자 제외) = 정답 차트 이름이어야 채점됨 (예: AWC2022_05.mp3 ↔ data/refs/AWC2022_05.chart).
"""
import argparse
import json
import sys
from pathlib import Path

from chartgen.assemble import build_chart
from chartgen.chart import chart_to_dict, load_chart
from chartgen.harmony import Params, prepare, recognize, vocal_per_bar
from chartgen.structure import analyze_form

ROOT = Path(__file__).resolve().parent


def run_one(name, feats_path, out_dir, params, refs):
    from chartgen.audio import load_features
    feats = load_features(feats_path)
    prep = prepare(feats)
    path, conf, grid, key = recognize(feats, params, prep)
    meta = {}
    ref = refs / f"{name}.chart"
    if ref.exists():
        rm = load_chart(ref).meta
        meta = {k: rm[k] for k in ("title", "url") if k in rm}
    sections, key = analyze_form(path, grid, vocal_per_bar(feats, grid), key, harm=prep[1])
    chart = build_chart(path, conf, grid, key, meta, sections=sections)
    out = out_dir / f"{name}.json"
    out.write_text(json.dumps(chart_to_dict(chart), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{name}: {len(grid.bar_start)}마디, 키 {chart.meta['key']} → {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", nargs="*", help="오디오 파일들")
    ap.add_argument("--method", required=True, help="결과 폴더 이름 (runs/<method>)")
    ap.add_argument("--cache", default=str(ROOT / "cache"))
    ap.add_argument("--refs", default=str(ROOT / "data" / "refs"))
    ap.add_argument("--from-cache", action="store_true", help="캐시에 있는 곡 전부 다시 인식")
    ap.add_argument("--delete-audio", action="store_true", help="분석 끝난 오디오 파일 삭제")
    ap.add_argument("--device", default="cpu", help="cpu / mps / cuda (음원 분리)")
    ap.add_argument("--params", help="Params 덮어쓰기 JSON, 예: '{\"pen_beat\": 2.0}'")
    a = ap.parse_args(argv)

    cache, refs = Path(a.cache), Path(a.refs)
    out_dir = ROOT / "runs" / a.method
    out_dir.mkdir(parents=True, exist_ok=True)
    params = Params(**json.loads(a.params)) if a.params else Params()
    if a.params:
        (out_dir / "params.json").write_text(a.params, encoding="utf-8")

    jobs = []
    for f in map(Path, a.audio):
        if not f.exists():
            sys.exit(f"파일 없음: {f}")
        from chartgen.audio import analyze
        print(f"[{f.stem}] 분석")
        feats = analyze(f, cache / f.stem, device=a.device)
        if a.delete_audio:
            f.unlink()
            print(f"  오디오 삭제: {f}")
        jobs.append((f.stem, feats))
    if a.from_cache:
        jobs += [(d.name, d / "features.npz") for d in sorted(cache.iterdir())
                 if (d / "features.npz").exists() and d.name not in {j[0] for j in jobs}]
    if not jobs:
        sys.exit("처리할 곡이 없습니다 (오디오 파일을 주거나 --from-cache)")
    for name, feats in jobs:
        run_one(name, feats, out_dir, params, refs)


if __name__ == "__main__":
    main()
