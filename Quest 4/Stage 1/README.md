# GlobalCart Operations Resolver Agent — Quest #04, Stage 1

A single autonomous agent that resolves GlobalCart support tickets end to end:
it reads a free-text customer ticket, investigates using the starter kit's four
tools, makes one operational decision — automatic refund, rejection with a
cited policy, escalation to a human, or a request for identifying information —
and returns machine-parseable JSON containing an auditable reasoning chain and
a customer-ready reply in the customer's language.

Built on the raw Anthropic SDK with a hand-rolled agent loop (a deliberate
choice, argued below), a Pydantic-enforced output contract, and a regression
suite that runs all nine scenario families with ~100 machine-checked
assertions per pass, verified on two model tiers with zero code changes. The
starter kit is untouched.

```
ticket ──► [ LLM turn ◄──► tool dispatch ]  (loop, max 8 turns)
                     │
                     ▼
        validated JSON: reasoning_chain · action_taken · customer_response
```

---

## 1. Architecture and framework choice

### Why no framework

I built the agent loop by hand on the Anthropic SDK instead of adopting
LangChain/LangGraph, CrewAI, or PydanticAI. Three reasons:

1. **The loop is the assignment.** Stop conditions, error handling, tool
   dispatch, and agent/tool layer separation are the graded content of this
   quest. A framework does those for you, which means it does them *instead*
   of you, and you learn very little by letting it. For a single agent the
   abstraction is overkill; it becomes justified in Stage 2, when real
   orchestration complexity appears. No need to over-complicate an
   implementation that doesn't require it.
2. **Transparency.** With a hand-rolled loop, debugging is `print(messages)`:
   you see exactly what the model saw and said. Several findings in the
   incident log below were only diagnosable because the full transcript was a
   first-class object in my own code.
3. **The knowledge transfers.** Every abstraction in the modern frameworks is
   a named wrapper around something in this repo, which makes Part B's
   planned migration cheap:

| This repo (Part A) | LangGraph / LangChain 1.x equivalent (Part B) |
|---|---|
| the `messages` list | `MessagesState` |
| the `stop_reason == "tool_use"` branch | `tools_condition` conditional edge |
| `TOOL_REGISTRY[name](**args)` dispatch | `ToolNode` |
| the `MAX_STEPS` cap | `recursion_limit` |
| the ~120-line loop | `create_agent` |

The plan for Part B is to wrap this agent as one node in a LangGraph graph,
adopting the framework at the moment orchestration complexity actually
justifies it, not before.

### Module map

```
agent/
  __init__.py   puts starter-kit/ on sys.path before any submodule runs
  config.py     .env loading, model name, loop budgets, no logic
  prompts.py    the system prompt (the agent's job description)
  schemas.py    Pydantic output contract: enum, field rules, consistency
  loop.py       the agent loop: model turn -> tool dispatch -> repeat
  run.py        CLI: python -m agent.run "<ticket text>"
starter-kit/    the provided tools and fixtures, byte-for-byte untouched
tests/
  run_scenarios.py     11-run regression suite over the 9 scenario families
  last_run_report.json full outputs of the latest suite run (generated)
```

**Layer separation.** `loop.py` contains no business rules, it does not know
what a refund cap is. `mock_services.py` does not know an LLM exists. The
system prompt defines *conduct* (honesty, citation discipline, decision
semantics); the tools own *policy* (windows, caps, escalation triggers). If a
rule can be enforced in code, it is never merely requested in prose.

**Division of labor in prompting.** The tool docstrings answer "when do I call
this?"; the system prompt answers "who am I, what is my authority, how do I
behave when things go wrong, and what do I output." Policy numbers are never
duplicated into the prompt, the agent must *cite* the rulebook, not memorize
a copy that can drift.

### Configuration

All tunables live in `.env` (template committed as `.env.example`):

| Variable | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | required; the SDK reads it from the environment |
| `MODEL` | `claude-sonnet-5` | any tool-capable model id; verified on two tiers (§4) |
| `MAX_STEPS` | `8` | hard cap on model turns per ticket |
| `MAX_FORMAT_RETRIES` | `1` | validation-failure retries for the final JSON |

The config separation earned its keep on day one: swapping models via `.env`
surfaced that a newer model generation rejects a manual sampling parameter the
loop was passing, a one-line fix, and the API named the exact knob, precisely
because model choice was never hard-coded. Parameter surfaces are
model-generation-specific; keeping the model in config is what makes swaps a
configuration event instead of a refactor.

### The import bootstrap

The starter kit is not an installable package, so `starter-kit/` must be put
on `sys.path` before `import mock_services` runs. That bootstrap lives in
`agent/__init__.py`, not in a sibling module, because Python *guarantees* a
package initializer executes before any of its submodules. An earlier version
kept the bootstrap in `config.py` and relied on import order inside `loop.py`;
an auto-formatter re-sorted the imports and broke it (incident 1). Invariants
should be structural, not conventional.

---

## 2. The tools, and how to run it

### The four tools

The agent imports `mock_services` directly and receives the kit's
`TOOL_SCHEMAS` untranslated (they are already in Anthropic's tool format);
`TOOL_REGISTRY` maps each name to the callable that my loop dispatches. The
tools, their schemas and the fixtures are used exactly as provided, the
assignment grades how the agent *wires and drives* them, and editing the kit
would make submissions incomparable.

| Tool | Role in the chain |
|---|---|
| `get_order_details(order_id)` | first stop for any ticket naming an order; also yields `user_id` |
| `get_user_profile(user_id)` | tier, fraud score, flags, refund history |
| `check_return_policy(order_id, reason)` | **the rulebook**: verdict, eligibility, escalation flags, `applicable_policies`, explanation |
| `process_refund(order_id, amount, reason)` | **the only tool with a (simulated) side effect**; enforces the cap in code |

The brief's "at least 2 tools" is a capability requirement across the suite,
not a per-ticket quota: the hallucination trap is *correctly* resolved with a
single lookup, while the suite asserts all four tools are exercised overall
and that each scenario's required/forbidden calls match the policy below.

Conduct rules the prompt adds on top of the tools:

- `process_refund` is called only after `check_return_policy` has reported the
  claim eligible; ineligible claims are rejected and risk-flagged claims are
  escalated without a refund attempt.
- For an eligible claim the agent requests the **full merited amount** and
  lets `process_refund` arbitrate the cap. It never splits or reduces a
  request to sneak under its own authority, a behavior the tools would
  otherwise permit: `process_refund("ORD-1002", 50.0)` returns `APPROVED`
  even though the customer's claim is $150. The cap guardrail enforces the
  letter of the policy; the agent is responsible for its intent.
- If a ticket states no return reason, the agent runs the check with the most
  plausible reason, **states that assumption explicitly in its reasoning**,
  and lets the verdict decide, clarification is reserved for cases that
  cannot be established at all (see incident 4).
- Internal reasoning never reaches the customer message: policy ids, tool
  names, fraud scores and risk status, a risk-flagged customer is never told
  they are one, live only in the reasoning chain. The suite's hygiene
  assertions enforce this.

### Setup

Run from inside project directory (Stage 1)

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env           # then paste your ANTHROPIC_API_KEY
```

### Run

```bash
# sanity-check the kit's fixtures and rule engine (33 assertions, no API key needed)
python starter-kit/examples/verify_scenarios.py

# resolve one ticket (defaults to the scenario-1 ticket if no text given)
python -m agent.run "Order ORD-1002. The espresso machine is dented and leaking. I paid 150 dollars for this."

# full regression suite: 11 live runs over the 9 scenario families
python tests/run_scenarios.py
```

Notes: run everything from the repo root. On Windows PowerShell, avoid `$`
inside double-quoted tickets (`"$150"` is variable interpolation; write
`150 dollars`). A full suite run costs a few cents. The verbose tool trace
printed by `agent.run` is the agent's audit trail, every call, every result.

**Reproducibility.** All date arithmetic in the kit is measured against a
fixed reference date, so decisions, refund ids and cited policies are stable
over time. The date is defined in `starter-kit/data/policies.json`
(`reference_date: 2026-08-05`) and read on every call by
`mock_services.reference_date()`, which honors a `QUEST4_REFERENCE_DATE`
environment-variable override (`YYYY-MM-DD`), a kit feature, consumed by the
kit's code, not by `agent/config.py`. To time-travel, set it in the shell
(that also covers the kit's own verifier); an entry in `.env` reaches agent
runs only, because `load_dotenv` populates the same process environment the
kit later reads. Separately, the loop sets no sampling parameters, newer
model generations manage sampling internally and reject overrides, so
*wording* and occasionally tool-call grouping vary between runs while
tool-arbitrated *outcomes* do not; the regression suite asserts the outcomes.

---

## 3. Reasoning mechanism, guardrails, and edge cases

### The output contract, requested by prompt, enforced by code

The agent's final message must be a single JSON object with three fields:
`reasoning_chain` (auditable steps citing concrete data and policy ids),
`action_taken` (distinct tools in first-call order, a decision from a closed
enum, refund amount and id), and `customer_response`. A prompt can only
*request* structure; `schemas.py` *enforces* it with Pydantic:

- a `Literal` enum for `decision`, a typo like `"ESCALATED"` dies at the
  boundary instead of silently poisoning downstream checks;
- `extra="forbid"`, invented fields are contract drift, caught loudly;
- a cross-field validator making decision/payload inconsistency **structurally
  impossible**: the JSON cannot claim `AUTO_REFUND_APPROVED` without a refund
  id, or carry money on a rejection.

On validation failure the loop feeds the compacted Pydantic error back to the
model and allows exactly one retry (a normal turn, inside the same `MAX_STEPS`
budget); a second failure returns a loud `INVALID_OUTPUT` with the raw text
preserved. Errors are data all the way down, including the loop's own
failures.

### Decision semantics

`NEEDS_MORE_INFO` is reserved for cases that cannot be established at all
(order/customer not found, unintelligible ticket). If the order exists and
policy blocks this channel from paying, wrong status, outside the window,
non-returnable category, the decision is `REJECTED`, and the correct route
(e.g. billing) belongs in the customer reply. Escalation is never presented to
the customer as anything but "our team will review"; internal signals (fraud
scores, flags, claim counts, policy ids) never appear in customer text.

The reply is also matched to the customer's emotional register, frustration
is acknowledged before process, long-tenured customers are thanked, and every
concrete statement (amounts, dates, what happens next) is a verified fact from
a tool result. There is no sentiment tool: the model reads tone from the
ticket itself, and the prompt requires the reply to promise nothing the
decision did not deliver.

### Layered defenses

| Threat | Mechanism | Where |
|---|---|---|
| infinite loop | hard `MAX_STEPS` cap; exceeding it returns data, not an exception | `loop.py` |
| retrying identical calls | after 2 identical `(tool, args)` calls, further attempts get a synthetic `REPEATED_CALL` error result | `loop.py` |
| hallucinated tool names | unknown names dispatch to an `UNKNOWN_TOOL` error result, the loop cannot crash on a tool | `loop.py` |
| business errors (`ORDER_NOT_FOUND`…) | treated as answers: report honestly, never retry, never guess | prompt + kit design |
| malformed final output | Pydantic validation + one feedback retry | `loop.py` + `schemas.py` |
| lying about tools used | claimed vs. executed comparison (distinct, first-call order) | test suite |
| wasteful repeats | per-tool invocation budget (≤2) fails the suite | test suite |
| ungrounded citations | every `POL-*` in reasoning must appear verbatim in the run's full tool results | test suite |
| leaking internals to customers | reply hygiene assertions (`POL-`, "fraud", "flag") | test suite |

Read vs. write asymmetry: a duplicate *read* costs tokens; a duplicate *write*
is a double payment. The kit's `process_refund` is stateless and deterministic
(same request → same `refund_id`), which is the mock's stand-in for the real
answer, idempotency keys at the tool layer, not hope at the prompt layer.

### Edge-case behavior (all machine-verified, see#4)

- **Hallucination trap (`ORD-2222`)**: one lookup, an honest "I couldn't find
  this order, please double-check the number", `NEEDS_MORE_INFO`, no invented
  dates, no payment.
- **Authority breach ($150 / cap $50)**: the agent requests the full merited
  amount, receives `ESCALATION_REQUIRED`, reports escalation, and never
  claims a refund happened. The suite would fail an agent that said otherwise.
- **Greed test ($999 demanded on a $35 order)**: the brief's scenario 8 is a
  tool-level bad-input battery already covered by the kit's own verifier, so I
  reinterpreted it at the agent level: the merited amount is what the customer
  paid, not what they demand. The agent requested $35 directly and was
  approved for exactly that.
- **Risky customer (`ORD-1005`)**: eligibility and risk are independent axes,
  the claim is eligible *and* must escalate. Three escalation reasons are
  cited in the reasoning; none of them reach the customer.
- **Boundary pair ($48 / $52)**: approved and escalated respectively; off-by-
  one authority reasoning is tested directly.
- **Unshipped orders**: rejected citing `POL-REF-04`, routed to billing in
  plain language. A side finding: status `delayed` (ORD-1004) is refundable
  under no rule but named by no routing rule either, a genuine gap in the
  fixture's policy text, documented rather than papered over.

### Incident log

Every failure encountered during development, with root cause and fix. The
common thread: across all of them, the agent never paid out wrongly, never
invented data, and never crashed, the defects were in the harness, the
tooling, and the specifications around it.

1. **The import-order landmine.** An auto-formatter sorted
   `import mock_services` above the module that set `sys.path`, breaking every
   run. Fix: move the bootstrap to `agent/__init__.py`, which Python
   guarantees runs before any submodule. Lesson: invariants that live in
   conventions a formatter can't see will eventually be formatted away.
2. **`tools_called` ambiguity.** The model once re-read the order record after
   fraud flags appeared (a reasonable diligence pattern) and then reported the
   *distinct* tools it used; the suite compared against the raw invocation
   log and failed it. The contract never said which was right. Fix: pin the
   contract ("distinct tool names, in the order first called"), compare
   accordingly, and give runaway repetition its own assertion (per-tool budget
   ≤2) plus a runtime guard on identical calls. Lesson: one property per
   assertion, an honesty check moonlighting as a loop detector will
   eventually have to betray one job to do the other.
3. **The phantom `none` tool.** Twice, both times on the suite's most
   semantically awkward scenario, the model emitted a `tool_use` named
   `none`, a no-op convention bleeding in from agent traces in its training
   data. The speculative `UNKNOWN_TOOL` branch converted it to an error
   result; the model recovered and answered correctly, honestly listing only
   real tools. The suite now compares honesty against *real* executed tools
   and surfaces phantom attempts as visible notes (three would fail the
   budget). The behavior appears model-dependent: it has not been observed on
   the smaller tier. Lesson: defensive code you hope is dead code isn't; and
   "emitted only well-formed calls" is a different property from "reported
   honestly."
4. **The refusal to guess.** Given "I'd like a refund please" on a cancelled
   order, the only ticket with no inferable return reason, the agent
   declined to invent the `reason` argument and asked for clarification: the
   anti-hallucination rule ("never fill gaps with guesses"), written for order
   data, over-generalizing to tool inputs. Defensible service, wrong shape for
   a single-shot resolver. Fix: an explicit rule to act under a *disclosed*
   assumption when the ambiguity cannot change the action (here the status
   gate is reason-independent, so no assumption could flip the verdict), plus
   pinned `NEEDS_MORE_INFO` semantics. Verified by controlled rerun: same
   ticket, the assumption stated in the reasoning chain, `REJECTED` citing
   `POL-REF-04`. Lesson: single-shot agents need pinned decision semantics;
   honesty and paralysis are different things.

---

## 4. Testing and verification

Two layers, mirroring the kit's own philosophy:

- **`starter-kit/examples/verify_scenarios.py`** (provided): 33 assertions
  proving the fixtures and rule engine are sound. Run first; if it passes,
  every wrong answer later is the agent's.
- **`tests/run_scenarios.py`** (mine): 11 live agent runs covering the nine
  scenario families, ~100 assertions across four layers, **outcomes**
  (decision, amount, refund id, all derived from the rule engine rather than
  guessed), **honesty** (claimed vs. executed tools; the required call /
  forbidden call signature of the sequencing policy), **grounding** (every
  cited policy id and refund id must appear verbatim in that run's *full*
  serialized tool results, full on purpose: checking selected fields alone
  produced a false hallucination-accusation during development), and
  **hygiene** (no policy codes or risk signals in customer text). Full
  outputs land in `tests/last_run_report.json`.

The suite itself was validated before trusting it: an *oracle* fake agent
(ideal outputs over real tool results) must pass everything, proving the
expectations are consistent with the rule engine, and four *mutation* fakes
(wrong decision, fabricated citation, dishonest tool claim, leaked codes) must
each be caught by exactly the assertion built for them. A test suite is code;
untested tests are rumors.

**Portability.** The identical suite, zero code, prompt, or expectation
changes, passes on two model tiers (a frontier-tier and a small-tier model).
The contract behaviors transferred intact: on the smaller model the agent
likewise disclosed its assumed reason on the reason-less ticket and took the
direct merited-amount path on the greed test. Model choice is a `.env` line,
and the regression suite is the gate for every swap.

Current status: all scenarios green on both tiers, including the controlled
rerun of incident 4.

---

## 5. Known limitations and the Part B runway

- **Stochastic variance**: without sampling control, wording and tool-call
  grouping vary between runs. Outcomes are pinned by the tools and asserted by
  the suite; prose is not, by design.
- **Single-shot by design**: the agent produces one decision per ticket. The
  disclosed-assumption rule is the single-shot substitute for a clarifying
  dialogue; a multi-turn product would relax it.
- **Per-name repeat budget** assumes single-order tickets (true of this
  suite); the runtime guard already uses the principled key, `(tool, args)`.
- **Part B**: wrap this agent as a LangGraph node (the migration table in#1
  is the plan), and note that `ORD-1005`, eligible, risk-flagged, address
  changed two days before delivery, is explicitly the seed of Part B's fraud
  crew. The agent already sniffed at that record unprompted once (incident 2).

---

## 6. Demo video

https://www.loom.com/share/05b5ce7c2a7b449aa94107e7b9f2865a

Three runs, tickets prepared in advance, the printed audit trail as narration:
scenario 1 end-to-end with the reasoning chain; scenario 2 refusing to exceed
authority and escalating honestly; scenario 9 declining to invent an order.
