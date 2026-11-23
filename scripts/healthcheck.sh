#!/usr/bin/env bash
# Print the API health report and exit non-zero unless every dependency is up.
set -euo pipefail
URL="${RAG_API_URL:-http://localhost:8000}"
body="$(curl -fsS --max-time 10 "$URL/health" || true)"
if [[ -z "$body" ]]; then
  echo "API not reachable at $URL" >&2
  exit 2
fi
echo "$body" | python3 -m json.tool
status="$(echo "$body" | python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])')"
[[ "$status" == "ok" ]]
