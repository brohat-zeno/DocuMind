#!/usr/bin/env bash
# Run from the repository root. Requires the configured Compose stack.
set -euo pipefail
BASE_URL="${BASE_URL:-http://localhost:8000}"
DEMO_SESSION="documind-demo-$(date +%s)"
BOOKING_DATE="${BOOKING_DATE:-$(uv run python -c 'from datetime import datetime,timedelta; from zoneinfo import ZoneInfo; print((datetime.now(ZoneInfo("Asia/Kathmandu"))+timedelta(days=30)).date())')}"

for strategy in fixed_size recursive; do
  curl --fail-with-body -sS "$BASE_URL/api/v1/documents" \
    -F 'file=@tests/fixtures/handbook.pdf;type=application/pdf' \
    -F "chunking_strategy=$strategy"
  printf '\n'
done

chat() {
  uv run python -c 'import json,sys; print(json.dumps({"session_id":sys.argv[1],"message":sys.argv[2]}))' \
    "$1" "$2" |
    curl --fail-with-body -sS "$BASE_URL/api/v1/chat" \
      -H 'Content-Type: application/json' --data-binary @-
  printf '\n'
  # Pace this manual demo for free-tier provider limits.
  sleep 12
}

chat "$DEMO_SESSION" 'Which two training plans does the DocuMind Learning Centre offer? List their names in handbook order.'
chat "$DEMO_SESSION" 'What is the monthly price of the second plan, and how many instructor-led workshops does it include?'
chat "$DEMO_SESSION-unknown" 'According to the documents, what launch date and engine thrust are specified for the Artemis lunar rocket?'
chat "$DEMO_SESSION-smalltalk" 'Hello! Hope you are doing well.'
chat "$DEMO_SESSION-booking" 'I would like to book an interview.'
chat "$DEMO_SESSION-booking" 'My name is Demo Example and my email is demo@example.com.'
chat "$DEMO_SESSION-booking" "Use $BOOKING_DATE at 11:30."
# Inspect the summary above. If the slot is occupied, choose another slot interactively.
chat "$DEMO_SESSION-booking" 'yes'
