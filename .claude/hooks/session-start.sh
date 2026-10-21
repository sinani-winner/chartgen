#!/bin/bash
# 클라우드 세션 시작 시 의존성 설치 + 모델 가중치 미리 받기 (여러 번 실행해도 안전)
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"
pip install -q -r requirements.txt 2>&1 | grep -v "Running pip as the 'root' user" || true

# Demucs(htdemucs_6s), beat_this 가중치를 캐시에 받아 둠 (이미 있으면 바로 끝남)
python3 - <<'PY' 2>&1 | grep -vi "warning" || true
from demucs.pretrained import get_model
get_model("htdemucs_6s")
from beat_this.inference import load_model
load_model("final0", device="cpu")
print("model weights ready")
PY
