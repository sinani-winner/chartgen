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
chartgen/harmony.py   8분음표 박자 축, 크로마/베이스, (코드×베이스) Viterbi (위치별 체인지 벌점), 키 추정
chartgen/structure.py 마디별 키·전조, 구성(구간 나누기·묶기·이름)
chartgen/assemble.py  인식 결과 → 차트 (오프비트 체인지 = 푸시 '>', 키에 맞춘 음 이름 표기)
transcribe.py         인식 CLI (오디오 → runs/<방법>/곡.json)
evaluate.py           채점 CLI (+ EXPERIMENTS.md 기록)
check_chart.py        정답 입력 확인 CLI
data/refs/*.chart     정답 세트 (사람이 입력, 형식: docs/CHART_FORMAT.md)
runs/<방법>/*.json    방법별 예측 결과 (정답과 같은 파일 이름)
tools/tune.py         캐시로 Params 탐색 / 한 곡씩 빼고 튜닝(--loo)
tools/diagnose.py     근음 오류 유형 분석
tools/tune_form.py    구성 설정 탐색 (경계+순서+이름)
tools/form_view.py    정답 구간 ↔ 인식 구간 나란히 보기
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

## Phase 3 메모 (2026-10-21: 근음 77.6 / 베이스 76.4→87.9 / 3화음 96.8 / 타이밍F 91.5, 한 곡씩 빼고 튜닝 시 근음 75.5)
- 사용자 기준: 7th보다 3화음 중심. 출력 = 장/단 3화음 + 분수 베이스. 채점표에 `3화음`(7th 무시 성격) 칸 추가.
- 상태 = 24코드 × 베이스 종류(근음·3음·5음·온음 위·b7) + N.C. 근음은 화성 스템, 베이스 음은 베이스 스템이 정한다.
  베이스 종류별 사전 비용(prior_*)으로 근음 위치를 기본으로 두고, 베이스만 바뀌면 bass_change × 위치 벌점.
- 해 봤는데 효과 없어서 뺀 것: 3음 대비 항(근음 급락), D/E 상태에 7sus4 모양 허용(변화 없음), 화성 음역 하한 올리기(악화).
- 남은 오류: AWC2025_03의 D/E를 밴드가 E·A·B·D(sus)로 쳐서 E로 읽힘(271프레임), X/3음 ↔ 3음 단3화음 혼동(A/C#→C#m, G/B→Bm),
  AWC2023_14 Gmaj7→Bm (프리코러스·후렴에 몰림, 라이브 편곡 차이일 수도 — 귀로 확인 필요).

## Phase 4 메모 (2026-10-22: 키 62→98.4 / 전조F 50 / 구성 경계 53.5·순서 53.7·이름 62.9)
- 정답 세트에 free/stop 표시가 없어 프리·스탑은 채점 불가 → 아직 구현 안 함.
- 키: 인식된 코드가 장조 음계 어디에 맞는지로 마디별 키 Viterbi (`local_keys`, 장조만). 반복 구간이 반음/온음 올려
  반복되면 그 구간 키를 올림. 남은 오류: AWC2022_16 전조(G→A)를 8마디 늦게 잡음 (그 후렴 코드 인식 오류가 G에 맞음).
- 구성: 경계 신호 DP (코드 진행 변화, 반복 시작 지점, 보컬 on/off, 8마디 선호, 구간당 비용).
  "반복되는 덩어리" 보너스를 크게 주면 벌스+후렴이 16마디로 합쳐진다 → w_rep 작게 (`tests/test_structure.py`가 막아 줌).
  나누기는 크로마 유사도, 묶기는 코드 이름(+크로마 0.3) 유사도. 후렴 = 가장 크게 부른 반복 묶음.
- 해 봤는데 뺀 것: 같은 이름 연속 구간 합치기(경계 53→40), 음정만 크로마 대신 코드만 쓰기(이름 정확도 낮음).
- 5곡뿐이라 구성 점수는 설정에 따라 출렁인다. 리드시트 기준 정답이라 라이브에서 더 부른 반복은 정답에 없다.
- 오디오 원본은 2026-10-22 삭제 (분석 결과는 cache/에 있음).
