#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${API_BASE_URL:-http://localhost:5000}"

curl -fsS "${BASE_URL}/health"
echo
curl -fsS "${BASE_URL}/forecast?horizon=3"
echo
curl -fsS "${BASE_URL}/model-info"
echo
