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

## 사용 방식
- 보통 세트 5곡 중 편곡 확인이 필요한 2~3곡만 돌린다 → 클라우드 CPU로 충분 (음원 분리만 곡당 몇 분, 이후 튜닝은 캐시로 몇 초). 속도보다 정확도 우선.
- 유튜브는 클라우드 서버에서 봇 차단/한국 지역 제한으로 못 받는 경우가 많다 → 사용자가 맥에서 받은 mp3를 올리는 것이 기본 경로.
- 새 클라우드 세션은 `.claude/hooks/session-start.sh`가 requirements 설치 + 모델 가중치를 받아 둔다.

## 구조
```
chartgen/chords.py    코드 기호 파싱 (근음·성격·베이스, 이명동음 동일 취급)
chartgen/chart.py     차트 텍스트/JSON 형식, 반복 펼치기, 8분음표 타임라인
chartgen/scoring.py   항목별 채점 (마디 정렬 → 근음/타이밍/성격/베이스/구성/키/가짜)
chartgen/audio.py     오디오 → 음원 분리(htdemucs_6s) + 박자(beat_this) + 스템별 CQT → cache/<곡>/features.npz
chartgen/harmony.py   8분음표 박자 축, 크로마/베이스, 근음 Viterbi (위치별 체인지 벌점), 키 추정
chartgen/assemble.py  인식 결과 → 차트 (오프비트 체인지 = 푸시 '>')
transcribe.py         인식 CLI (오디오 → runs/<방법>/곡.json)
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
- 인식: `python3 transcribe.py audio/곡.mp3 --method <방법> [--delete-audio]` (오디오 이름 = 정답 차트 이름)
- 재인식(캐시만, 튜닝용): `python3 transcribe.py --from-cache --method <방법> --params '{"pen_beat": 2.0}'`
- 설치: `pip install -r requirements.txt` (모델 가중치는 첫 실행 때 dl.fbaipublicfiles.com, cloud.cp.jku.at 에서 받음)

## 채점 메모
- 해상도는 8분음표. 예측 마디는 정답 마디에 근음 기준 DP로 정렬된다(마디 하나 빠져도 뒤가 전부 틀리지 않음). 빠진 마디는 오답, 남는 마디의 체인지는 가짜 체인지.
- 타이밍: `timing_f`는 8분음표 1개 차이 허용, `timing_f_exact`는 푸시까지 정확해야 인정. free 구간은 마디 단위 허용.
- 성격(quality)은 근음이 맞은 구간에서만 계산. 가짜 텐션 = 근음은 맞는데 정답에 없는 7th/6th/9th 등을 붙인 코드 개수.
- 예측 JSON 형식은 `chart_to_dict()` 출력과 같다 (`"format": "chartgen/1"`). 마디별 `start_sec`(영상 싱크용) 선택 필드 있음.

## Phase 2 메모 (2026-10-21 완료: 근음 76.0 / 타이밍F 90.0, 한 곡씩 빼고 튜닝 시 근음 73.9)
- 캐시(`cache/`)에는 오디오가 아니라 스템별 CQT·박 위치만 저장한다 → 오디오는 분석 직후 지워도 튜닝 가능.
- 튜닝: `python3 tools/tune.py [--sweep] [--loo]` (캐시로 몇 초), 오류 분석: `python3 tools/diagnose.py data/refs runs/<방법>`.
- 코드 어휘는 장/단 3화음 + N.C.뿐. 7th 모양은 판정에만 쓰고 출력은 3화음 (사용자: 7th보다 3화음 중심).
- 박자: beat_this 다운비트가 구간에 따라 2박마다 찍히는 경우가 있어 4박 위상으로 정리한다 (`regularize_downbeats`).
- 캐시 CQT는 log 압축이라 옆 반음 누출이 커 보임 → 선형으로 되돌려 반음 피크만 사용 (`peak_semitones`).
- 정답 세트는 리드시트 기준이라 라이브에서 더 부른 반복(예: AWC2022_05 후렴 +16마디, 엔딩 +10마디)은 정답에 없다.
  반복 기호 해석 문제는 아님 (리드시트 확인함). 남는 마디는 가짜 체인지 수에만 반영된다.
- 남은 큰 오류 (Phase 3에서): 분수코드 베이스를 근음으로 잡음 (D/E → E, C/E → C 등). 근음은 화성 스템, 베이스는 베이스 스템으로 따로 정해야 함.
