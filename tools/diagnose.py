"""근음 오류 진단: 정렬된 마디 안에서 틀린 근음이 정답과 몇 도 차이인지, 정답 베이스와 같은지 등.

    python3 tools/diagnose.py data/refs runs/<방법>
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chartgen.chart import bar_frames, build_timeline, load_chart  # noqa: E402
from chartgen.scoring import align_bars  # noqa: E402

IV = ["1", "b2", "2", "b3", "3", "4", "b5", "5", "b6", "6", "b7", "7"]


def diagnose(ref_path, pred_path):
    ref, pred = load_chart(ref_path), load_chart(pred_path)
    rb, pb = ref.expand(), pred.expand()
    rt, pt = build_timeline(rb), build_timeline(pb)
    pairs = align_bars(rt, pt)
    miss_ref = sum(1 for r, p in pairs if p is None)
    extra_pred = sum(1 for r, p in pairs if r is None)
    n = ok = 0
    iv, kinds = Counter(), Counter()
    for r, p in pairs:
        if r is None or p is None:
            continue
        rs, ps = rt.bar_start[r], pt.bar_start[p]
        lr, lp = bar_frames(rb[r].time), bar_frames(pb[p].time)
        for t in range(lr):
            rc, pc = rt.frames[rs + t], pt.frames[ps + t * lp // lr]
            if rc is None:
                continue
            n += 1
            if pc is None:
                kinds["N.C.로 예측"] += 1
                continue
            if pc.root == rc.root:
                ok += 1
                continue
            iv[IV[(pc.root - rc.root) % 12]] += 1
            if rc.bass is not None and pc.root == rc.bass:
                kinds["분수코드 베이스를 근음으로"] += 1
            elif rc.seventh or rc.ext:
                kinds["정답이 7th/텐션 코드"] += 1
            else:
                kinds["기타"] += 1
    return dict(matched_acc=ok / max(n, 1), miss_ref=miss_ref, extra_pred=extra_pred,
                ref_bars=len(rb), pred_bars=len(pb), iv=iv, kinds=kinds, n=n)


def main():
    refs, preds = Path(sys.argv[1]), Path(sys.argv[2])
    tot_iv, tot_k = Counter(), Counter()
    for p in sorted(preds.glob("*.json")):
        r = refs / f"{p.stem}.chart"
        if not r.exists():
            continue
        d = diagnose(r, p)
        tot_iv += d["iv"]
        tot_k += d["kinds"]
        print(f"{p.stem:20s} 정렬된 마디 근음 {100*d['matched_acc']:5.1f}%  "
              f"정답 {d['ref_bars']}마디 중 짝 없음 {d['miss_ref']}  예측 {d['pred_bars']}마디 중 남음 {d['extra_pred']}")
        print("   오류 음정(예측-정답):", ", ".join(f"{k}:{v}" for k, v in d["iv"].most_common(6)))
    print("\n전체 오류 음정:", ", ".join(f"{k}:{v}" for k, v in tot_iv.most_common()))
    print("전체 오류 종류:", ", ".join(f"{k}:{v}" for k, v in tot_k.most_common()))


if __name__ == "__main__":
    main()
