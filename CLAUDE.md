# CLAUDE.md

라이브 합주 유튜브 영상 → 연주용 코드 차트 생성기. 전체 기획은 `PLAN.md`.

## 작업 원칙
- 이 프로젝트의 기준은 **"연주자가 이 차트로 바로 합주할 수 있는가"** 하나다.
- 각 Phase 끝마다 `python evaluate.py data/refs runs/<방법이름> --log "메모"` 로 채점하고, 항목별 점수로 보고한다.
- 부가 기능 제안 금지 (PDF, 셋리스트, 라이브러리 꾸미기 등). 정확도에 기여하지 않는 작업은 하지 않는다.
- 출력은 화성(코드 차트)까지만. 멜로디·리프·필인·음표 채보 기능은 만들지 않는다.
- 애드립 음이 가짜 체인지·가짜 텐션을 만들면 안 된다 (PLAN.md 1-1 규칙).
- 애매하면 텐션 없는 단순한 코드로 표기하고 신뢰도를 낮게 남긴다 (텍스트에선 `코드?`, JSON에선 `conf`).
- 화면 작업(Phase 5)은 Phase 4 점수가 만족스러울 때까지 시작하지 않는다.
- 착수 전 맥 환경(Apple Silicon 여부, 파이썬 버전) 확인: `uname -m; python3 --version; sw_vers`.
  라이브러리 설치가 안 되면 대안으로 바꾸고 이유를 설명한다.
- 유튜브 오디오는 개인 채보 용도로만 쓰고, 분석 후 삭제한다 (`audio/`, `stems/`는 git에 올리지 않음).

## 구조
```
chartgen/chords.py    코드 기호 파싱 (근음·성격·베이스, 이명동음 동일 취급)
chartgen/chart.py     차트 텍스트/JSON 형식, 반복 펼치기, 8분음표 타임라인
chartgen/scoring.py   항목별 채점 (마디 정렬 → 근음/타이밍/성격/베이스/구성/키/가짜)
evaluate.py           채점 CLI (+ EXPERIMENTS.md 기록)
check_chart.py        정답 입력 확인 CLI
data/refs/*.chart     정답 세트 (사람이 입력, 형식: docs/CHART_FORMAT.md)
runs/<방법>/*.json    방법별 예측 결과 (정답과 같은 파일 이름)
tests/                pytest
```

## 명령
- 테스트: `python3 -m pytest -q tests`
- 정답 확인: `python3 check_chart.py data/refs` (전체) / `python3 check_chart.py data/refs/곡.chart` (펼쳐 보기)
- 채점: `python3 evaluate.py data/refs runs/<방법> [--diff] [--log "메모"]`

## 채점 메모
- 해상도는 8분음표. 예측 마디는 정답 마디에 근음 기준 DP로 정렬된다(마디 하나 빠져도 뒤가 전부 틀리지 않음). 빠진 마디는 오답, 남는 마디의 체인지는 가짜 체인지.
- 타이밍: `timing_f`는 8분음표 1개 차이 허용, `timing_f_exact`는 푸시까지 정확해야 인정. free 구간은 마디 단위 허용.
- 성격(quality)은 근음이 맞은 구간에서만 계산. 가짜 텐션 = 근음은 맞는데 정답에 없는 7th/6th/9th 등을 붙인 코드 개수.
- 예측 JSON 형식은 `chart_to_dict()` 출력과 같다 (`"format": "chartgen/1"`). 마디별 `start_sec`(영상 싱크용) 선택 필드 있음.
