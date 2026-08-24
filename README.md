
**Contextor** (`agents/contextor.py`) rewrites the latest user turn into a standalone query using an LLM with structured output (`ContextorOutput`), carrying forward an active order ID from memory when the user doesn't repeat it. It also does a deliberate query-expansion step: if the message mentions a damaged/broken/wrong item, it appends terms like *"reporting window timeframe deadline requirements"* to the rewritten query so the retriever is more likely to surface time-limited policy sections, not just the general damage-policy chunk.

**Router** (`agents/router.py`) is a structured-output LLM classifier (`RouteDecision`) choosing `order` / `rag` / `direct`, with explicit few-shot edge cases in the prompt (e.g. "final-sale bag arrived broken" → `rag`, not `order`).

**OrderTool** (`tools/order_lookup.py`) loads `data/orders.json`, normalizes the ID (`.strip().upper()`), and on a hit passes the raw record through a Pydantic `SafeOrder` model (`schemas/safe_order.py`) whose fields are an explicit allowlist (`order_id`, `membership_tier`, `items`, timestamps, `status`, carrier/tracking, `estimated_delivery`, `customer_safe_message`) with `Config.extra = "ignore"` — any internal-only field on the raw record (email, address, notes, risk score, etc.) is silently dropped before it ever reaches the LLM. Cancelled/returned orders have `estimated_delivery` forced to `None` before serialization, so the model can't relay a stale ETA. A `tool_calls` record (`hit: True/False`) is emitted for observability regardless of outcome; a missing order ID short-circuits with no lookup attempted and no fabricated `tool_calls` entry.

**RAGTool** (`tools/rag_lookup.py`) queries the shared `VectorStore` singleton for the top-8 chunks.

Retrieval pipeline (`rag/`):
- `loader.py` reads every `.md` in `knowledge-base/`.
- `parser.py` splits YAML front matter from body and validates required fields (`document_id`, `title`, `status`, `policy_authority`), raising per-file rather than failing the whole ingest.
- `chunker.py` splits each document at H2 boundaries (H3+ stays folded into the parent H2 as content, not a separate chunk), producing a `raw_text` (shown to the LLM/user, source of truth for citation) and a markdown-stripped `embedding_text` (what's actually embedded) per chunk. It also extracts every `**bolded**` span as a `key_facts` list — the document authors bolded the numbers that matter ("30 calendar days", "$6.95 return shipping fee"), so this gives a cheap, deterministic string to check against later without depending on the LLM getting the paraphrase exactly right.
- `vectorstore.py` flattens list/dict metadata into Chroma-safe scalars, computes an `authoritative` flag at ingest time (`status == "active" AND policy_authority == "official"`), and **filters retrieval to `authoritative=True` by default** — superseded and internal-only documents never compete with current policy in a normal query unless a caller explicitly opts out (`authoritative_only=False`), which nothing in the current pipeline actually does.

**Synthesizer** (`agents/synthesizer.py`) is where grounding, privacy, injection-handling, and conflict rules live, almost entirely as a single system prompt against a structured `SynthesizerOutput` (`answer`, `sources_used`, `confidence`, `handoff`, `injection_detected`). Code-level enforcement on top of the LLM's own judgment:
- **Handoff precedence override**: if `injection_detected` is true, `handoff` is forced `False` regardless of what the model returned (never send a prompt-injection attempt to human support); otherwise, an order lookup that returned "not found"/"error"/"invalid" forces `handoff = True`, and `confidence in {"insufficient", "conflicting"}` forces `handoff = True`. This turns the "strict precedence order" described in the prompt into an actual boolean guarantee instead of trusting the model to apply it consistently.
- **Source validation**: any cited chunk `id` that wasn't actually in this turn's retrieved set is dropped before it's translated to a `source_file`.
- **Source backfill** (`_backfill_sources_from_key_facts`): a normalized substring match between the final answer and every retrieved chunk's `key_facts` list, adding a chunk's `source_file` to `sources_used` if a fact clearly appears in the answer text even when the model forgot to cite it — a deterministic safety net for citation completeness, not a replacement for it.

**Tool-call isolation**: the LLM never receives `orders.json` or the vector index directly — only whatever `order_context` / `rag_context` a graph node explicitly writes into state that turn. `[ORDER DATA]` and `[KNOWLEDGE BASE]` are appended to the system prompt as clearly delimited, explicitly-untrusted blocks per Rule 5, so retrieved text (which may itself contain injected instructions, per the corpus's `14-internal-content-migration-notes.md`) is never treated as an instruction source.

**Multi-turn state**: `GraphState` is a `TypedDict` with `messages` merged via LangGraph's `add_messages` reducer; `MemorySaver` checkpoints full state per `thread_id`, so `order_id` and conversation history persist turn-to-turn within a session without the Router or Synthesizer needing to re-derive them.

## 5. Command for running evaluations

```bash
python evaluator.py
```

`evaluator.py` loads `evaluation/visible-cases.json` (required) and `evaluation/custom-cases.json` (optional, additive — the harness runs it automatically if present, no separate flag needed) and drives each case through `graph.invoke`. Matching is intentionally not exact-substring:
- `_flexible_contains` normalizes hyphens/quotes/whitespace/markdown-bold and tolerates simple singular/plural and delivery/delivered-style variants before checking containment.
- `_flexible_date_match` checks month/day/year components independently rather than requiring an exact date string.
- `_invented_check` flags a forbidden term only when it appears in a sentence **without** a nearby negation cue (`"not"`, `"unavailable"`, `"can't"`, etc.), so "we cannot share the risk score" doesn't wrongly fail a `must_not_invent`/`must_refuse_to_disclose` check for "risk score".
- `_sources_match` compares by filename prefix.

Per-case assertions come from each case's `expect` block: `tool` (`order_lookup` / `not_called` / `not_called_without_id`), `handoff`, `must_include`, `must_not_include`, `must_not_invent`, `must_refuse_to_disclose`, `required_sources`, `forbidden_sources_as_authority`. Results are reported per-case and rolled up by `category` (whatever categories exist in the case files), plus a separate visible-vs-custom breakdown.

**Honesty note on this section**: the assignment asks for baseline vs. final evaluation numbers by category. I do not have `evaluation/visible-cases.json` or an actual execution trace in front of me, so I'm not going to invent pass/fail counts. Run `python evaluator.py` against a clean checkout and paste the printed `TEST EXECUTION SUMMARY` block here before submitting — the harness already prints category breakdown, visible/custom breakdown, and a final score in exactly the shape needed for this section.

## 6. Bug diary

**Bug 1 — Model cited a fact without citing its source chunk**
- *How reproduced*: reviewing `synthesizer.py`, the `_backfill_sources_from_key_facts` function exists specifically because, per its own docstring, "the LLM sometimes uses a fact from a retrieved chunk without citing that chunk's id (e.g. it states '30 calendar days' but only lists a different chunk in `sources_used`)."
- *Root cause*: Rule 11 in the system prompt asks the model to return the id of every chunk whose fact appears in the answer, but with several retrieved chunks in context the structured-output call would sometimes emit a correct, well-grounded answer while under-citing — an attention/compliance gap in the LLM call itself, not a retrieval problem (the right chunk was retrieved; it just wasn't listed).
- *Fix*: added a deterministic post-processing pass — normalize the final answer text, normalize every retrieved chunk's `key_facts` (the bolded numeric/deadline spans extracted at chunk time), and append any chunk's `source_file` whose fact clearly appears in the answer but was missing from the model's own `sources_used`.
- *Regression test*: `required_sources` assertions in `evaluator.py` check citation completeness against `sources_used` (via `_sources_match`) independently of whatever the model's raw structured output claimed, so under-citation now fails the case even if the answer text itself was accurate.

**Bug 2 — Handoff flag didn't reliably follow the stated precedence rules**
- *How reproduced*: `synthesizer_node`'s own inline comment states the reasoning directly: "The LLM prompt is great, but boolean logic is best enforced in code." The structured-output `handoff` field, produced purely by the LLM against a 13-rule strict-precedence prompt, would not consistently reflect that precedence — e.g. an injection case could still come back with `handoff=True`, or a failed order lookup could come back `False`.
- *Root cause*: precedence logic ("injection overrides everything → lookup-failed forces true → insufficient/conflicting forces true → default false") is a deterministic boolean cascade, and asking an LLM to hold a 4-level precedence order in its head across every response is not reliable even at `temperature=0`.
- *Fix*: moved the precedence cascade into plain Python after the structured-output call — `injection_detected` short-circuits `handoff=False`; otherwise a string-matched "not found"/"error"/"invalid" in `order_context` forces `handoff=True`; otherwise `confidence in {"insufficient","conflicting"}` forces `handoff=True`. The LLM's own `handoff` guess is only trusted for the remaining case (privacy/action requests, rule 13.4), which is closer to a single semantic judgment than a multi-branch cascade.
- *Regression test*: `evaluator.py`'s `handoff` assertion (`expect["handoff"]`) directly checks the final, code-adjusted `handoff` value returned in graph state, so this now fails deterministically on any regression in the override logic, independent of how the LLM phrases its own guess.

**Bug 3 — Damage-related queries failed to retrieve the reporting-window/deadline policy**
- *How reproduced*: `contextor_node`'s system prompt contains an explicit, separately-flagged rule ("Query Expansion (CRITICAL)") requiring that any damaged/broken/defective/incorrect-item mention have terms like "reporting window timeframe deadline requirements" appended to the rewritten query before it reaches the retriever.
- *Root cause*: a query like "my bag arrived broken" is semantically close to `04-damaged-or-wrong-items.md`'s main body but not close to the specific reporting-deadline sub-section, which uses different vocabulary — plain vector similarity on the literal user phrasing under-ranked the deadline chunk against the more topically-broad damage-policy chunk.
- *Fix*: query rewriting in Contextor deliberately over-generates retrieval terms for this specific query shape rather than passing the user's literal phrasing straight to the vector store.
- *Regression test*: this is exactly the kind of case `evaluator.py`'s `must_include` assertion is built to catch — a damage-related case whose `expect.must_include` names the deadline language (e.g. "reporting window") will fail if the deadline chunk drops out of the top-8 again.

## 7. Evaluation breakdown

The evaluation suite categorizes cases by whatever `category` string each case in `evaluation/visible-cases.json` / `evaluation/custom-cases.json` declares, and checks (per case, not just overall):

- **Retrieval / source precedence** — `required_sources` (must appear, matched by filename) and `forbidden_sources_as_authority` (must not appear as a cited source), which directly tests whether the vector store's `authoritative=True` filter is doing its job of keeping superseded/internal docs out of normal answers.
- **Groundedness / abstention** — `must_include` (flexibly matched, including date-aware matching) and `must_not_invent` (forbidden terms that must not appear un-negated), catching both missing required facts and fabricated ones.
- **Tool use / privacy** — `tool` (`order_lookup` called correctly, or explicitly *not* called for `not_called` / `not_called_without_id` cases) and `must_refuse_to_disclose` for internal-only order fields.
- **Multi-turn behavior** — handled by cases whose `messages` array has more than one `user` turn against the same `thread_id`; the harness invokes the graph once per user message on the same thread, so context loss across turns shows up as a failure on the final turn's assertions.
- **Handoff correctness** — `expect.handoff` is checked against the code-enforced value from Bug 2's fix, not the raw LLM guess.

Deterministic assertions (source ids, tool calls, forbidden disclosures, abstention text) are the default; nothing in `evaluator.py` calls out to a second LLM to grade responses — matching is entirely string/regex-based against the graph's own structured output fields.

## 8. Known limitations / what I'd improve before production

- **In-memory checkpointing.** `graph.py` uses `langgraph.checkpoint.memory.MemorySaver`, which is process-local and lost on restart. A real deployment needs a durable, atomic checkpointer (e.g. a Redis- or Postgres-backed LangGraph checkpointer) so conversation state survives restarts and works across multiple app instances.
- **No durable execution for multi-step mutations.** The agent is currently read-only (order lookup, retrieval) with no write actions, so this isn't exercised today — but the assignment explicitly asks for reasoning here: any future action like "process refund" or "cancel order" would need a Durable Execution pattern (e.g. Temporal, or LangGraph's own persistence combined with idempotency keys) so a crash mid-mutation can't leave an order in a half-updated state, and so retries don't double-apply a cancellation.
- **Single-process vector store.** Chroma's `PersistentClient` here is a local file store. At real scale this would move to a managed/distributed vector store (e.g. a hosted Chroma/pgvector/Pinecone deployment) so ingestion and multiple app replicas aren't fighting over one local directory.
- **Conflict detection is LLM-judged, not code-enforced.** Rule 2 (surfacing genuine active-source conflicts) and Rule 12 (citing all conflicting chunk ids) rely entirely on the structured-output call correctly identifying a conflict; unlike the handoff precedence (Bug 2), there's no deterministic backstop in `synthesizer.py` that independently checks the retrieved chunk set for two `authoritative=True` chunks disagreeing on the same fact. This is the most likely place for a silent regression.
- **Unbounded session history.** `messages` accumulates via `add_messages` for the life of a thread with no truncation or summarization — a long support conversation will eventually blow the context window and increase per-turn cost.
- **`app.py` doesn't write to `graph_execution.log`.** Structured logging (`logging.basicConfig(...)`) is only configured in `graph.py`'s own module scope; the Streamlit entrypoint imports `graph` (so the handler is technically registered) but never calls `logger.info`/`logger.exception` itself, and its own except-block only shows the error in the UI — so a Streamlit-driven session doesn't produce the same route/result trace that the `graph.py` CLI harness does. Production observability should log per-turn (route, retrieved chunk ids + scores, tool call args/result, final `confidence`/`handoff`/`injection_detected`) from a single place regardless of entrypoint.
- **Malformed vs. unknown order IDs aren't distinguished.** `order_tool_node` normalizes and does a direct equality lookup; a malformed ID (wrong format) and a syntactically valid but nonexistent ID both just fall through to the same "not found" path. That satisfies "handle safely" but doesn't give the user a more specific "that doesn't look like an order ID" message.
- **Single embedding/LLM provider.** Both `embedder.py` and `llm.py` hard-depend on OpenAI with no fallback; a provider outage stalls ingestion, retrieval, and generation simultaneously.

## 9. AI coding tools used

Used Claude throughout this project for: reviewing the graph/prompt design against the assignment's required capabilities, drafting the deterministic backfill/precedence logic described in the bug diary, and producing this README from the actual source.

**Example of an AI-generated suggestion that was wrong or incomplete**: an early suggestion for the conflict-detection rule was to resolve conflicting active sources by preferring whichever document had the more recent `effective_date`. That's factually reasonable-sounding but directly contradicts the assignment's own scenario (a newer date shouldn't silently win over genuine policy disagreement) — the system prompt now explicitly states "*A newer `effective_date` does NOT automatically resolve a conflict — only explicit supersession metadata does*" specifically to override that instinct.

## 10. Demo

*(Embed the 2–4 minute GIF/video here — showing one KB question with citations, one order lookup, one multi-turn conversation, one abstain/human-handoff case, and the evaluation suite running.)*

---

**Author**: Nilay — nilayshahane@gmail.com