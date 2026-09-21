#!/usr/bin/env bash
# Smoke-test a running local-jev server with curl.
#
#   scripts/test-api.sh                        # http://127.0.0.1:8765, the server's default model
#   scripts/test-api.sh http://host:9000       # another server
#   MODEL=nli-deberta-large scripts/test-api.sh
#   LONG=1 scripts/test-api.sh                 # also send a state longer than the model's window
#
# If the server was started with LOCAL_JEV_API_KEY set, export the same variable here.
# Needs curl and python3 (to check the JSON). Exits non-zero if any check fails.
set -uo pipefail

URL="${1:-${LOCAL_JEV_URL:-http://127.0.0.1:8765}}"
URL="${URL%/}"
MODEL="${MODEL:-jev-latest}"
AUTH=()
[ -n "${LOCAL_JEV_API_KEY:-}" ] && AUTH=(-H "Authorization: Bearer ${LOCAL_JEV_API_KEY}")

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
PASS=0
FAIL=0

if [ -t 1 ]; then GREEN=$'\033[32m' RED=$'\033[31m' RESET=$'\033[0m'; else GREEN= RED= RESET=; fi
ok()   { PASS=$((PASS + 1)); printf '  %sok%s    %s\n' "$GREEN" "$RESET" "$1"; }
fail() { FAIL=$((FAIL + 1)); printf '  %sFAIL%s  %s\n' "$RED" "$RESET" "$1"; [ -n "${2:-}" ] && printf '        %s\n' "$2"; }

# request METHOD PATH [JSON]: writes the body to $TMP/body and headers to $TMP/headers; sets STATUS and TIME
request() {
  local args=(-s -o "$TMP/body" -D "$TMP/headers" -w '%{http_code} %{time_total}' -X "$1" ${AUTH[@]+"${AUTH[@]}"})
  [ $# -ge 3 ] && args+=(-H 'Content-Type: application/json' --data-binary "$3")
  read -r STATUS TIME < <(curl "${args[@]}" "$URL$2" || echo "000 0")
}

# check NAME PYTHON_EXPR: evaluates the expression with `r` bound to the parsed response body
check() {
  local out
  if out=$(python3 - "$TMP/body" "$2" 2>&1 <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
if not eval(sys.argv[2]):
    sys.exit(json.dumps(r)[:300])
PY
  ); then ok "$1"; else fail "$1" "$out"; fi
}

expect_status() {  # expect_status NAME CODE
  if [ "$STATUS" = "$2" ]; then ok "$1 (HTTP $STATUS, ${TIME}s)"; else fail "$1" "expected HTTP $2, got $STATUS: $(head -c 300 "$TMP/body")"; fi
}

echo "local-jev API test against $URL (model: $MODEL)"

echo
echo "Server"
request GET /healthz
if [ "$STATUS" = "000" ]; then
  fail "server reachable" "nothing answered at $URL; start it with: local-jev serve"
  exit 1
fi
expect_status "GET /healthz" 200
request GET /v1/models
expect_status "GET /v1/models" 200
check "models are listed" "len(r['models']) > 0 and all('name' in m for m in r['models'])"
python3 -c "import json,sys; r=json.load(open(sys.argv[1])); print('        ' + ', '.join(m['name'] for m in r['models']))" "$TMP/body" 2>/dev/null
request GET /
expect_status "GET /" 200

echo
echo "System One: one request with all three question types"
request POST /v1/systemone "$(cat <<JSON
{
  "model": "$MODEL",
  "state": "You charged my card twice this month for the same subscription. Please refund the extra charge.",
  "questions": {
    "refund":   {"type": "noul",   "instructions": "The customer is asking for money back"},
    "team":     {"type": "choice", "instructions": "Which team should handle this",
                 "criteria": {"billing": "Charges, invoices and refunds", "technical": "Bugs and outages", "sales": "Pricing and new plans"}},
    "severity": {"type": "score",  "instructions": "How serious is the problem",
                 "criteria": ["Minor annoyance", "Real problem with a workaround", "Blocking"]}
  }
}
JSON
)"
expect_status "POST /v1/systemone" 200
check "response names the model and has usage" "r['model'] and r['usage']['input_tokens'] > 0"
check "noul is a probability" "0 <= r['answers']['refund']['noul'] <= 1"
check "choice is one of the keys" "r['answers']['team']['choice'] in ('billing', 'technical', 'sales')"
check "choice probabilities sum to 1" "abs(sum(r['answers']['team']['probabilities'].values()) - 1) < 0.01"
check "choice confidence is in [0, 1]" "0 <= r['answers']['team']['confidence'] <= 1"
check "score is within the legend" "0 <= r['answers']['severity']['score'] <= 2 and len(r['answers']['severity']['legend']) == 3"
check "score probabilities sum to 1" "abs(sum(r['answers']['severity']['probabilities'].values()) - 1) < 0.01"
check "sanity: a double charge goes to billing" "r['answers']['team']['choice'] == 'billing'"
check "sanity: a refund request is a refund request" "r['answers']['refund']['noul'] > 0.5"
python3 -c "
import json, sys
a = json.load(open(sys.argv[1]))['answers']
print('        refund %.2f | team %s (%.2f) | severity %.2f' % (a['refund']['noul'], a['team']['choice'], a['team']['confidence'], a['severity']['score']))
" "$TMP/body" 2>/dev/null

request POST /v1/systemone "{\"model\": \"$MODEL\", \"state\": \"Are we still on for lunch on Friday?\",
  \"questions\": {\"refund\": {\"type\": \"noul\", \"instructions\": \"The customer is asking for money back\"}}}"
expect_status "unrelated text" 200
check "sanity: a lunch invite is not a refund request" "r['answers']['refund']['noul'] < 0.5"

echo
echo "Errors"
request POST /v1/systemone '{"model": "no-such-model", "state": "hi", "questions": {"q": {"type": "noul", "instructions": "x"}}}'
expect_status "unknown model is 404" 404
request POST /v1/systemone "{\"model\": \"$MODEL\", \"state\": \"hi\", \"questions\": {\"q\": {\"type\": \"choice\", \"instructions\": \"x\"}}}"
expect_status "choice without criteria is 422" 422
check "the error points at the question" "'q' in json.dumps(r['detail'])"
request POST /v1/systemone "{\"model\": \"$MODEL\", \"state\": \"hi\", \"questions\": {\"q\": {\"type\": \"maybe\", \"instructions\": \"x\"}}}"
expect_status "unknown question type is 422" 422
request POST /v1/systemone '{not json'
expect_status "malformed JSON is 422" 422

if [ -n "${LONG:-}" ]; then
  echo
  echo "Long input (LONG=1): a state longer than the model's window is shortened, not refused"
  python3 - "$TMP/long.json" "$MODEL" <<'PY'
import json, sys
state = "Invoice 4471: the customer was charged twice for the same subscription. " * 6000   # about 90k tokens
json.dump({"model": sys.argv[2], "state": state,
           "questions": {"billing": {"type": "noul", "instructions": "This is about billing"}}}, open(sys.argv[1], "w"))
PY
  request POST /v1/systemone "$(cat "$TMP/long.json")"
  expect_status "very long state" 200
  if grep -qi '^x-local-jev-truncated: *true' "$TMP/headers"; then ok "x-local-jev-truncated header is set"; else fail "x-local-jev-truncated header is set"; fi
fi

echo
echo "$PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
