#!/usr/bin/env bash
# Like example.sh, but first asks which model should answer: one prompt, three questions (a yes/no, a category and a score).
#
#   scripts/example-model.sh
#   scripts/example-model.sh "My order arrived broken and support is ignoring me."
#   LOCAL_JEV_URL=http://host:8765 scripts/example-model.sh
URL="${LOCAL_JEV_URL:-http://127.0.0.1:8765}"
if ! curl -s -o /dev/null --max-time 3 "$URL/healthz"; then
  echo "local-jev is not running at $URL." >&2
  echo "Start it in another terminal with:  .venv/bin/local-jev serve   (add --ui for the browser portal)" >&2
  exit 1
fi
PROMPT="${1:-You charged my card twice this month for the same subscription. Please refund the extra charge.}"
AUTH="Authorization: Bearer ${LOCAL_JEV_API_KEY:-local}"

# The models the server offers, without the aliases (jev-latest and friends all mean the default).
MODELS=$(curl -s -H "$AUTH" "$URL/v1/models" | python3 -c '
import json, sys
models = json.load(sys.stdin)["models"]
default = next((m["description"][len("Alias of "):].rstrip(".") for m in models if m["name"] == "jev-latest"), "")
for m in models:
    if not m["description"].startswith("Alias of"):
        about = m["description"].split(" read at ")[0].split(". ")[0].rstrip(".")
        print("|".join([m["name"], "default" if m["name"] == default else "", about]))
')
if [ -z "$MODELS" ]; then
  echo "Could not read the model list from $URL/v1/models." >&2
  exit 1
fi

echo "Which model should answer?"
i=0
DEFAULT_INDEX=1
while IFS='|' read -r name flag about; do
  i=$((i + 1))
  [ "$flag" = "default" ] && DEFAULT_INDEX=$i
  printf '  %d) %-19s %s%s\n' "$i" "$name" "$about" "$([ "$flag" = "default" ] && echo "   <- default")"
done <<< "$MODELS"
printf 'Model [1-%d, Enter for %d]: ' "$i" "$DEFAULT_INDEX"
read -r PICK
PICK="${PICK:-$DEFAULT_INDEX}"
case "$PICK" in
  ''|*[!0-9]*) echo "Not a number: $PICK" >&2; exit 1 ;;
esac
if [ "$PICK" -lt 1 ] || [ "$PICK" -gt "$i" ]; then
  echo "Pick a number from 1 to $i." >&2
  exit 1
fi
MODEL=$(sed -n "${PICK}p" <<< "$MODELS" | cut -d'|' -f1)
echo "Asking $MODEL (the first request to a model loads it, which can take a minute)..."
echo

PROMPT_JSON=$(printf '%s' "$PROMPT" | python3 -c 'import json, sys; print(json.dumps(sys.stdin.read()))')
curl -s "$URL/v1/systemone" \
  -H "Content-Type: application/json" \
  -H "$AUTH" \
  -d @- <<JSON | python3 -m json.tool
{
  "model": "$MODEL",
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
