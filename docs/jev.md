# Jev — How It Works and What It Can Do

> Source: [docs.typesafe.ai](https://docs.typesafe.ai/introduction) and its `llms.txt` index. Jev is TypeSafe's flagship **System One** model; current version `jev-1.13.0`.

## 1. What Jev is

Jev is not a chatbot and not a small LLM. It is a **System One model**: a model trained to make **fast, structured decisions that software can use directly**.

The problem it exists to solve: ordinary LLMs are text-generation systems that you then *force* to emit structured decisions, which your code parses, validates and hopes is well-formed. That round trip is slow, expensive and unreliable. Jev removes it — it **returns typed values and probability distributions directly**. There is no parsing step, because there is no text.

The documentation is blunt about what it will not do: System One models **"do not write replies, produce code, or generate explanations of their reasoning."**

### System One vs. System Two

| | System One (Jev) | LLM (System Two) |
|---|---|---|
| Output | Constrained, typed values + calibrated probabilities | Freeform generated text |
| Trained for | Calibrated decisions — "probabilities are optimized against outcomes to reflect uncertainty" | Next-token prediction over language |
| Latency | ~100–150 ms | Seconds |
| Role in your system | A primitive your code calls, like `strcmp` with judgment | An agent that decides its own next step |
| Cost profile | ~$0.042 / M input tokens; output free | Orders of magnitude more |

The governing principle for building with it: **"keep code in control and give System One narrow, structured decisions."** Your code owns the control flow. Jev supplies what the docs call *programmable common sense*.

## 2. The mental model — State + Questions

Every request is one **state** evaluated against one or more **questions**.

Think of it as handing a case file to a panel of experts and asking each a single well-scoped question they could answer in seconds. Crucially, **every question sees the same state and is evaluated independently** — no question can see another's answer. That independence is the whole architecture: it prevents *context-rot*, and it means questions evaluate in parallel.

### State

The material to evaluate. Three accepted shapes:

```python
# String — a single message
"My card was charged twice."

# Object — most requests; related fields with descriptive names
{"message": "My card was charged twice.", "order_id": "A-104"}

# Array — sequences such as conversation threads or records
["Hi", "My customer number is TS1337.", "My card was charged twice."]
```

Rules: **text only** — images, audio and video are not supported (yet). English is primary; other languages including CJK scripts are accepted but currently have lower accuracy. Keep state and questions separate: **state holds content and facts, questions define judgments.**

## 3. The three primitives

### Choice — select one option from a known, unordered set

```python
Choice(
    instructions="Which team should handle this",
    criteria={
        "billing":   "Payment or subscription issues",
        "technical": "Bugs or integration problems",
        "sales":     "Pricing or account questions",
    },
)
```

Returns `{choice, probabilities, confidence}`. **Up to 255 options.** Provide a complete list — add `other` or `none of the above` when coverage is uncertain, because the distribution is over *your* options only; the model cannot invent one.

*Use for:* routing tickets to departments, detecting programming languages, classifying document types.

### Score — rate along an ordered spectrum

```python
Score(
    instructions="How frustrated the customer appears",
    criteria=["Calm, just stating facts", "Frustrated but civil", "Very angry, strong language"],
)
```

Returns `{score, legend, probabilities, confidence}`. **2–10 levels.** The score is an **expected value, not an argmax**: multiply each level number by its probability and sum. Three levels (0–2) with 0.57/0.43 on levels 1 and 2 yields **1.43** — so scores land *between* levels, which is the point.

*Writing levels:* **describe situations, not degrees.** "Broken or degraded feature, but workaround exists" beats "Moderately severe."

*Use for:* bug severity, customer frustration, candidate qualification, content formality.

### Noul — a yes/no proposition where the probability *is* the answer

```python
Noul(instructions="The message conveys urgency or time-sensitivity")
```

Returns `{noul}` — a single number in 0–1: the probability that the answer is yes. Near 1 is strong affirmation, near 0 negation, near 0.5 genuine uncertainty.

**Nouls carry no separate `confidence` field**, because the probability already conveys both the answer and the certainty. Optional `criteria` with `true`/`false` keys pins down subtle boundaries.

*Use for:* PII detection, refund requests, jailbreak flags, feature mentions in a résumé. For multi-part decisions, ask **several Nouls in parallel** rather than cramming conditions into one.

## 4. Confidence — the second decision axis

Choice and Score answers carry a **`confidence` in 0–1 derived from the shape of the probability distribution**. All the mass on one option gives 1.0; the more evenly it spreads, the lower the confidence. (For three options the demo uses `(3 × p_max − 1) / 2`.)

This is what makes Jev safe to automate against: **"I don't know" is a useful signal.** A flat distribution means ambiguity, competing factors or insufficient information — and your code can route on that instead of guessing.

Starting bands from the docs:

| Confidence | Suggested action |
|---|---|
| **> 0.9** | Act automatically, no human involved |
| **0.5 – 0.9** | Proceed cautiously — consider user confirmation or review |
| **< 0.5** | Do not act — route to a human or ask for clarification |

```python
if confidence < 0.5:
    route_to_human(user_message)
elif high_stakes_action and confidence <= 0.9:
    ask_user_to_confirm(account_id)
else:
    proceed_automatically()
```

Two caveats the docs stress: thresholds are **per domain and per use case** — start conservative, test on your own data, adjust. And **thresholds are not portable between primitives**: a Noul of 0.7 and a Choice confidence of 0.7 are not the same quantity.

## 5. The API

**`POST https://api.typesafe.ai/v1/systemone`**, `Authorization: Bearer <API_KEY>`, `Content-Type: application/json`.

**Request**

```json
{
  "state": "string | object | array",
  "model": "jev-latest",
  "questions": {
    "<question_id>": {
      "type": "noul | choice | score",
      "instructions": "string | object | array",
      "criteria": "map (choice, ≤255) | array (score, 2–10) | optional true/false object (noul)"
    }
  }
}
```

**Response**

```json
{
  "model": "jev-1.13.0",
  "answers": {
    "department":  {"type": "choice", "choice": "technical", "confidence": 0.78},
    "frustration": {"type": "score",  "score": 1.0},
    "is_urgent":   {"type": "noul",   "noul": 1.0}
  },
  "usage": {"input_tokens": 392, "output_tokens": 65}
}
```

Answers come back **keyed by your own question ids**. `GET /v1/models` lists availability for your account.

**Errors**

| Status | Meaning |
|---|---|
| 401 | Missing or invalid API key |
| 422 | Request body failed validation (missing field, malformed question) |
| 429 | Rate limit exceeded — back off and retry |
| 529 | TypeSafe temporarily overloaded — retry after a short delay |

Use exponential backoff on 429/529; the SDKs do this automatically.

## 6. Model, limits and pricing

| | `jev-1.13.0` |
|---|---|
| Aliases | `jev-latest` (SDK default), `jev-preview` — both currently point here |
| Price | **$0.042 per million input tokens** ($42/B). **Output is free.** |
| Rate limits | 250,000 tokens/sec; 1,200 requests/min |
| Context | **64k tokens total per request**; **32k for state plus the longest question** |
| Input | Text only — strings, JSON objects, or arrays of text |
| Latency | ~100–150 ms typical |

> Rate limits are noted as adjusting dynamically under demand and may change without notice. Enterprise limits via sales@typesafe.ai.

## 7. SDKs

```bash
pip install typesafe-sdk
# Claude Code agent skill:
claude plugin marketplace add typesafe-ai/skills
claude plugin install typesafe@typesafe-ai
```

```python
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

client = TypeSafeClient()

response = client.system_one(
    state="Hi, I've been trying to connect my Stripe account for 3 days and the "
          "integration keeps failing. I'm losing sales. Please help ASAP.",
    questions={
        "department": Choice(
            instructions="Which team should handle this",
            criteria={
                "billing":   "Payment or subscription issues",
                "technical": "Bugs or integration problems",
                "sales":     "Pricing or account questions",
            },
        ),
        "frustration": Score(
            instructions="How frustrated the customer appears",
            criteria=["Calm, just stating facts", "Frustrated but civil", "Very angry, strong language"],
        ),
        "is_urgent": Noul(
            instructions="The message conveys urgency or time-sensitivity",
        ),
    },
)
```

- **Clients:** `TypeSafeClient` (sync) and `AsyncTypeSafeClient` (async context manager). Options include `model` (default `jev-latest`) and a `RetryPolicy` with `max_retries`, `backoff_max`, `timeout`.
- **Env vars:** `TYPESAFE_API_KEY` (required), `TYPESAFE_BASE_URL`, `TYPESAFE_DEFAULT_MODEL`, `TYPESAFE_LOG_LEVEL`.
- **Reading results:** `result.choices[name].choice`, `result.scores[name].score`, `result.nouls[name].noul`, plus `result.request_id` and `result.raw_http_response`.
- **Typed responses:** pass `response_model=` a Pydantic model inheriting `SystemOneResponse` (or a plain `BaseModel`).
- **Errors:** catch `TypeSafeAPIError` — carries `status` and `request_id`.
- **Forward compatibility:** `extra_body=` for newer API fields; raw question dicts (`{"billing": {"type": "noul", ...}}`) always work; unrecognised answer types are reachable via `raw_http_response.json()`.

A **JavaScript/TypeScript SDK** exists with the same shape — `TypeSafeClient`, `choice()`/`score()`/`noul()` helpers, `ChoiceQuestion<T>`/`ScoreQuestion<T>`/`NoulQuestion` interfaces, `SystemOneRequest<Q>` / `SystemOneResult<Q>`, and a full typed error hierarchy (`APIError`, `RateLimitError`, `AuthenticationError`, …).

## 8. How to build with it — the method

Five steps, in order:

1. **Use code when possible.** Keep deterministic work, rules and control flow in code. Don't reach for an agent loop when a workflow will do.
2. **Decompose the state.** Include only context relevant to the current questions. Structure it as nested JSON and reference specific values with backticked paths like `` `support.tickets[0].message` ``.
3. **Decompose questions atomically.** This matters most: *"broad questions hide several judgments behind one answer. Atomic questions expose those judgments so you can inspect, tune, and combine them."* Not "Is this spam?" — instead: does it request credentials, does the sender display name conflict with the domain, does it promise an unexpected reward.
4. **Structure question definitions** as objects with named fields rather than long template strings, so code can swap context without rewriting the question.
5. **Ask many questions in parallel** over the same state in one request. Maximum intelligence per dollar, no serial round trips.

Then **combine in code** (deterministic rules, weighted sums) and **route on confidence**. The cardinal sin: letting one question's result become hidden context for another. That destroys composability and reintroduces context-rot.

**Anti-patterns:** one broad question instead of atomic ones · irrelevant context in state · relying on the model's world knowledge · treating outputs as hidden context · guessing instead of escalating · serial round trips.

## 9. Patterns

### Speculative fan-out
Send many questions in one call — **including ones you may not need** — and let your code decide what was relevant. Since questions evaluate in parallel, adding more barely moves response time. Ask for `bug_severity` even though it is meaningless for a feature request: you get the entire decision tree from one call.

```python
category    = response.answers["category"]
bug_severity= response.answers["bug_severity"]
bug_repro   = response.answers["has_reproducible_steps"]
refund      = response.answers["refund_requested"]
frustration = response.answers["frustration"]

if category.choice == "bug_report":
    if bug_severity.score > 1.5 and bug_repro.noul > 0.6:
        escalate_to_engineering(ticket_id, severity="high")
    else:
        add_to_bug_backlog(ticket_id)
elif category.choice == "billing":
    route_to_billing_with_flag(ticket_id, refund_likely=refund.noul > 0.7)
elif category.choice == "feature_request":
    log_feature_request(ticket_id)

if frustration.score > 1.5:
    flag_for_priority_response(ticket_id)
```

### Confidence-gated routing
Treat confidence as a second decision axis. Different actions in the same system get different thresholds based on **stakes and reversibility** — not one global number.

### Composite scoring
Ask several Score questions on separate dimensions, **normalise each by its maximum**, then combine with weights in code. Keeps each judgment inspectable and tunable independently.

### Intent routing
Classify a user's intent with a Choice, then dispatch to the specialised handler — cheap, fast semantic dispatch in front of expensive machinery.

## 10. Advanced structure

Instructions, Choice option descriptions, Score level descriptions and Noul `criteria.true/false` **all accept JSON structure** (string, object, array or null), not just strings.

```json
"instructions": {
  "field": {
    "name": "invoice_number",
    "type": "string",
    "description": "The identifier printed on the invoice."
  },
  "extracted_value": "4471",
  "question": "Does `extracted_value` match the `field`?"
}
```

- **Choice options as objects** with `what` / `not_for` / `examples` sharpen the boundary between confusable options.
- **Taxonomies**: nest subtrees as option values so the model weighs branches before committing to a path (hierarchical classification).
- **Score levels as objects** with `summary` and `signals` clarify each tier with concrete indicators.
- **Noul `criteria`** with `true`/`false` objects pins down subtle boundaries using definitions and contrasting examples.

Start with plain strings; reach for structure when options get confusable.

## 11. Known jagged edges (jev-1.13)

Jev is strong at common-sense semantic judgment and weak wherever a task needs multiple reasoning steps or numeric precision. The documented failure modes, with the prescribed workaround:

| Weakness | Workaround |
|---|---|
| **Literal interpretation** — answers the exact question, misses implied meaning | State conditions and boundary cases explicitly; split ambiguous questions and combine in code |
| **Math** — cannot reliably do arithmetic | All computation in code; model for semantic judgment only |
| **Counting** — unreliable over characters, terms, list items; degrades with size | Count in code (regex/parsers); or query per item and aggregate |
| **Numeric representations** — hex, RGB triples, binary; cannot compare proximity | Convert to semantic form (colour names) or bucket in code first |
| **Score interpolation** — cannot reconstruct exact numbers between levels | Use scores for threshold checks, not numeric reconstruction |
| **Dates/times** — treated as text, not ordered quantities | Extract with Choice over enumerated options; all comparison and arithmetic in code |
| **Indirection / multi-hop** — double negatives, logical chains | Simplify; point directly at the relevant state instead of requiring inference |
| **Irrelevant context** — large unrelated state acts as a distractor | Filter and retrieve in code first; use intermediate Nouls to find relevant passages |
| **Adversarial content** — does not treat input as hostile; injected instructions can steer it | Write explicit criteria; test edge cases before deployment |

Plus three structural limits: **results are not comparable across primitives** (don't carry a threshold from a Noul to a Choice); **contradictory instructions vs. criteria** cause confusion, so keep phrasing consistent; and **it is not a text generator** — use a generative model for that.

## 12. Where it fits

Five high-level categories from the docs:

1. **AI automation software** — "run it a million times in the background without a human co-pilot."
2. **Real-time applications** — frontier intelligence at ~150 ms, fast enough to sit inside a UI or a game loop.
3. **AI map-reduce over big data** — search, classify and extract predictive features across large corpora affordably.
4. **Universal verification** — check inputs, extractions, reasoning and tool calls from *other* AI systems at a fraction of their cost.
5. **Harness engineering** — semantic queries for model routing, context retrieval, error detection and classification at scale.

Concrete domains covered by the use-case map: search & reranking for RAG · scientific screening · model routing · LLM guardrails · semantic code linting · ML feature extraction · recruiting · lead scoring · customer support triage · insurance claims · financial crime · legal & compliance · marketplace listings · trust & safety · advertising · gaming · risk assessment · demand forecasting · knowledge-graph alignment.

Cookbooks worth reading before building: self-consistency (nouls / choices), parallel questions, re-ranking, line-by-line semantic search, structure recovery, function calling, hierarchical classification, classifying RAG passages, citation checking, LLM guardrails, SDE cascade, date extraction, classification using confidence.

---

### Reference links

- Introduction: <https://docs.typesafe.ai/introduction> · Full index: <https://docs.typesafe.ai/llms.txt>
- System One · State · Primitives (Choice/Score/Noul) · Advanced structure · Confidence · Patterns · Models · API · Jaggedness · Cookbooks — all under <https://docs.typesafe.ai/>
