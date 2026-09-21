#!/usr/bin/env bash
# One request to a running local-jev server: one prompt, three questions (a yes/no, a category and a score).
#
#   scripts/example.sh
#   scripts/example.sh "My order arrived broken and support is ignoring me."
#   LOCAL_JEV_URL=http://host:8765 scripts/example.sh
URL="${LOCAL_JEV_URL:-http://127.0.0.1:8765}"
if ! curl -s -o /dev/null --max-time 3 "$URL/healthz"; then
  echo "local-jev is not running at $URL." >&2
  echo "Start it in another terminal with:  .venv/bin/local-jev serve   (add --ui for the browser portal)" >&2
  exit 1
fi
PROMPT="${1:-You charged my card twice this month for the same subscription. Please refund the extra charge.}"
PROMPT_JSON=$(printf '%s' "$PROMPT" | python3 -c 'import json, sys; print(json.dumps(sys.stdin.read()))')

curl -s "$URL/v1/systemone" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${LOCAL_JEV_API_KEY:-local}" \
  -d @- <<JSON | python3 -m json.tool
{
  "model": "jev-latest",
  "state": $PROMPT_JSON,
  "questions": {
    "refund":   {"type": "noul",   "instructions": "The customer is asking for money back"},
    "team":     {"type": "choice", "instructions": "Which team should handle this",
                 "criteria": {"billing": "Charges, invoices and refunds", "technical": "Bugs and outages", "sales": "Pricing and new plans"}},
    "severity": {"type": "score",  "instructions": "How serious is the problem",
                 "criteria": ["Minor annoyance", "Real problem with a workaround", "Blocking"]}
  }
}
JSON
