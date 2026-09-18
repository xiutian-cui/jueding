# Jueding

[![CI](https://github.com/xiutian-cui/jueding/actions/workflows/ci.yml/badge.svg)](https://github.com/xiutian-cui/jueding/actions/workflows/ci.yml)
[![coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)](#development)
[![python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://github.com/xiutian-cui/jueding/blob/main/pyproject.toml)

**Jueding** (决定, pronounced *jué dìng*) means **to decide** or **a decision** in
Chinese.

Jueding is a small, typed Python wrapper for TypeSafe's [Jev](https://typesafe.ai/)
model. It uses the official [`typesafe-sdk`](https://pypi.org/project/typesafe-sdk/)
package. Jueding turns the probabilistic judgments of Jev into ordinary function calls.
The boundary is simple:

> Unstructured state goes in. A typed value, a confidence, and a complete probability
> distribution come out. Your Python code decides what happens next.

Jev answers three kinds of question. Jueding gives each kind a typed Python primitive:

- a choice among labels
- a yes/no judgment
- a score on an ordered rubric

Jueding is not an agent framework. It does not take control of your application.

> [!NOTE]
> Jev is in early access behind a waitlist. You must have a TypeSafe API key to reach
> the real model.

## Install

```bash
uv add "jueding @ git+https://github.com/xiutian-cui/jueding"
```

This installs `typesafe-sdk`, which Jueding requires. Jueding adds no other runtime
dependency.

## Quick start

```python
from enum import StrEnum

from typesafe_sdk import TypeSafeClient

from jueding import Decision
from jueding.typesafe import TypeSafeBackend


class Intent(StrEnum):
    SEARCH = "search"
    ANALYZE = "analyze"
    CHAT = "chat"


client = TypeSafeClient()          # reads TYPESAFE_API_KEY
backend = TypeSafeBackend(client=client)

decide_intent = Decision(
    instructions="Determine the user's primary intent.",
    choices=Intent,
    backend=backend,
    criteria={
        Intent.SEARCH: "Retrieve existing information.",
        Intent.ANALYZE: "Reason about or compare information.",
        Intent.CHAT: "Primarily conversational.",
    },
    threshold=0.8,
)

intent = decide_intent(
    user_input="Compare these two implementation approaches.",
    context={"surface": "developer_tool"},
)

if intent == Intent.ANALYZE:
    run_analysis()
```

The result compares directly against the enum. It also carries every value that the
model reported:

```python
intent.value          # Intent.ANALYZE
intent.confidence     # 0.91
intent.probabilities  # {Intent.SEARCH: 0.06, Intent.ANALYZE: 0.91, Intent.CHAT: 0.03}
intent.ranked         # ((Intent.ANALYZE, 0.91), (Intent.SEARCH, 0.06), (Intent.CHAT, 0.03))
intent.margin         # 0.85
intent.meets_threshold
intent.model, intent.usage, intent.request_id, intent.latency_seconds
```

## What goes where

A call has two inputs. The difference between them is important:

| | What it holds | Lifetime |
|---|---|---|
| **The question** — `instructions` and `criteria` | What to decide, and what each outcome means, with definitions and examples | Fixed when you build the decision |
| **The state** — the call arguments | The facts of this particular case | Per call |

Criteria accept text, mappings, or sequences. Put stable domain knowledge here:

```python
criteria={
    "returns": {"what": "Exchanges, refunds, wrong or damaged items",
                "examples": ["wrong size", "arrived broken"]},
    "billing": {"what": "Charges, invoices, payment problems",
                "not_for": ["refund requests, which are returns"]},
}
```

State is an ordinary JSON document. **The model reads the field names.** Therefore the
keyword names are part of the question, not plumbing:

```python
decide(message=text, order={"id": "A-104"}, customer_tier="gold")   # keywords become state
decide("a plain string is used as state directly")                  # or one positional value
```

A positional argument can be a string, a mapping, a sequence, or a dataclass instance.
If you mix positional state and keyword state, the call raises an error.

## Primitives

### `Decision[Enum]`

`Decision` selects one member of a string-valued enum. It returns a `DecisionResult`
with the complete distribution, `ranked`, and `margin`.

```python
if not intent.meets_threshold and intent.margin < 0.1:
    top_two = [member for member, _ in intent.ranked[:2]]
    ask_user_to_pick(top_two)
```

A narrow `margin` is different from diffuse uncertainty. It shows that two specific
candidates compete. That is the case to disambiguate.

### `Condition`

A yes/no judgment that does not collapse into a bare boolean.

```python
from jueding import Condition

is_urgent = Condition(
    instructions="Does this request explicitly communicate urgency?",
    backend=backend,
    true="The request includes time pressure or a deadline.",
    false="The request does not communicate time pressure.",
)

result = is_urgent("Please fix this before payroll closes today.")
if result:
    escalate()

result.probabilities   # {True: 0.98, False: 0.02}
```

Confidence here is `max(p, 1 - p)`. It never falls below 0.5. A threshold under 0.5 is
always met.

### `Score`

`Score` returns an expected score over an **ordered** rubric. The levels are ranked, so
a value between two levels is meaningful. `1.3` means mostly level 1, with some level 2.

```python
from jueding import Score

severity = Score(
    instructions="How severe is the reported issue?",
    rubric=[
        "Cosmetic; no impact to functionality",
        "Broken or degraded feature, but a workaround exists",
        "Blocking issue; no workaround exists",
    ],
    backend=backend,
)

result = severity(bug_report)
if result >= 1.5:
    page_oncall()

result.value          # 1.3 — the probability-weighted mean of the level numbers
result.probabilities  # {0: 0.0, 1: 0.7, 2: 0.3}
result.legend         # {0: "Cosmetic; ...", 1: ..., 2: ...}
```

Use `Score` when the outcomes lie on a spectrum. Use `Decision` when they do not. There
is no midpoint between `search` and `chat`. There is one between "degraded" and
"blocking".

## Thresholds

`threshold` is optional and has no default. Without a threshold, Jueding applies no
confidence policy. It reports what the model said.

With a threshold, a result below that threshold **refuses to act as a certain value**:

```python
intent = decide_intent(user_input=text)      # confidence 0.42, threshold 0.8

intent == Intent.ANALYZE     # raises LowConfidenceError
bool(is_urgent(ticket))      # raises LowConfidenceError
severity(bug) >= 1.5         # raises LowConfidenceError

intent.value                 # Intent.ANALYZE — reading never raises
intent.confidence            # 0.42
intent.meets_threshold       # False
```

The guard fires only against the type of the answer. A comparison between two results is
never guarded. A comparison against an unrelated type is never guarded. Therefore results
stay usable in collections. A comparison between two judgments uses
their **values only**, not their confidences.

Jueding has no retry and no fallback. `typesafe-sdk` owns the retries at the transport
layer. Your own code must decide what to do about a low-confidence answer.

## Design principles

- **Normal Python:** decisions are callable objects. Results work with `if`, `match`,
  dictionaries, and sets.
- **Typed values:** an enum decision returns an enum member, not a string to parse.
- **Visible uncertainty:** Jueding always keeps the confidence and the full distribution.
- **Explicit policy:** no default threshold, no retry, no fallback, no side effect.
- **Thin integration:** `typesafe-sdk` owns HTTP, authentication, response parsing, and
  transport retries. Jueding adds types. Jueding never wraps the exceptions of the SDK.

## Development

```bash
uv sync
uv run pdoc jueding jueding.typesafe -o site   # API docs from docstrings
```

## License

MIT — see [LICENSE](https://github.com/xiutian-cui/jueding/blob/main/LICENSE).
