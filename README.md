# Aster & Row Support Agent

A reliability-focused RAG support agent for Aster & Row, an ecommerce company selling bags, drinkware, and travel accessories. The system combines semantic retrieval, source-authority ranking, tool calling, conversation memory, and deterministic safeguards to deliver accurate, grounded, and traceable support responses.

It is designed to prioritize reliable knowledge retrieval, authoritative sources, accurate order information, contextual conversations, and safe handling of untrusted content, with an evaluation suite used to validate the agent's behavior across realistic support scenarios.

---

## 1. Setup and Run Instructions (clean clone)

```bash
git clone <your-repo-url>
cd ai-agent-intern-test

# create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt

# configure environment variables
cp .env.example .env
# edit .env and set GROQ_API_KEY

# build the vector index (run once, or after any knowledge-base/chunker change)
python -m app.ingest.build_index

# run the agent
python run.py
```

---

## 2. Environment Variables

`.env.example`:

```
GROQ_API_KEY=
GROQ_MODEL=openai/gpt-oss-120b
EMBEDDING_MODEL=all-MiniLM-L6-v2
TRACE_LOG_PATH=logs/trace.jsonl
```

---

## 3. Model, Embedding, Framework, Storage

| Choice | What | Why |
|---|---|---|
| LLM | Groq-hosted `openai/gpt-oss-120b`, temperature=0 | Low cost, tool-calling support; temp=0 minimizes run-to-run phrasing variance. |
| Embeddings | `sentence-transformers` / `all-MiniLM-L6-v2`, local | No external embedding API, fully offline after first model download, deterministic. |
| Framework | Plain Python, no agent framework | The workflow is one bounded retrieve → generate → tool-call → respond loop. |
| Vector storage | NumPy array + JSON metadata (`index/vectors.npy`, `index/metadata.json`) | No vector DB needed; cosine similarity computed directly at query time. |

---

## 4. Architecture

![Architecture diagram](assets/architecture.png)

### File Map

| File | Role |
|---|---|
| `app/ingest/loader.py` | Parses KB markdown + YAML front matter into `Document` objects. |
| `app/ingest/chunker.py` | Splits documents by `##` heading; prepends doc title + heading to chunk text before embedding. |
| `app/ingest/build_index.py` | Loads, chunks, embeds, and persists the full KB index. |
| `app/retrieval/embedder.py` | Wraps `sentence-transformers` for batch (index-build) and single-query embedding. |
| `app/retrieval/store.py` | NumPy-backed vector store; cosine similarity top-k query. |
| `app/retrieval/rank.py` | Authority re-ranking, citable-chunk filtering, candidate-conflict detection. |
| `app/tools/order_lookup.py` | Deterministic, side-effect-free order lookup; whitelist-only return schema — the sole module allowed to touch raw `orders.json`. |
| `app/agents/prompts.py` | System prompt: trust boundaries, citation rules, handoff triggers, worked examples. |
| `app/agents/session.py` | Per-session in-memory conversation history. |
| `app/agents/tool_schema.py` | OpenAI-style tool schema for `get_order_status`. |
| `app/agents/orchestrator.py` | Ties retrieval, LLM calls, tool execution, and deterministic handoff backstops together (`handle_turn`). |
| `app/observability/tracer.py` | Writes one structured JSONL trace event per turn. |
| `app/models/schemas.py` | Pydantic models: `Chunk`, `RetrievedChunk`, `OrderLookupResult`, `TraceEvent`, etc. |
| `evaluation/run_eval.py` | Deterministic + heuristic eval harness; category-level reporting. |
| `evaluation/cases_custom.json` | 6 original test cases beyond the supplied visible cases. |

---

## 5. Running Evaluations

```bash
python -m evaluation.run_eval               # all cases (15 visible + custom)
python -m evaluation.run_eval --visible-only # supplied visible-cases.json only
python -m evaluation.run_eval --verbose      # show answer text for passing cases too
```

---

## 6. Baseline and Final Evaluation Results

**Baseline** — early integration run, before the deterministic-handoff backstops, chunking fix, and prompt refinements described in the bug diary:

| Category | Passed |
|---|---|
| retrieval | 1/2 |
| groundedness | 0/2 |
| multi-source-grounding | 0/1 |
| conversation | 0/1 |
| tool-use | 0/3 |
| tool-reliability | 0/5 |
| privacy | 1/1 |
| prompt-security | 0/3 |
| abstention | 0/1 |
| source-conflict | 0/2 |
| **TOTAL** | **2/21** |

**Final** — after all fixes below, run against all 15 supplied visible cases + all 6 original custom cases:

| Category | Passed |
|---|---|
| retrieval | 2/2 |
| groundedness | 2/2 |
| multi-source-grounding | 1/1 |
| conversation | 1/1 |
| tool-use | 3/3 |
| tool-reliability | 5/5 |
| privacy | 1/1 |
| prompt-security | 3/3 |
| abstention | 1/1 |
| source-conflict | 2/2 |
| **TOTAL** | **21/21** |

---

## 7. Bug Diary

**Bug 1 — Supporting source was not cited when another source was sufficient**
- *Reproduced by:* the `final-sale-damaged-exception` evaluation case. The answer was correctly grounded, but the model cited only the source it directly relied on, even though both `03-final-sale-and-promotions.md` and `04-damaged-or-wrong-items.md` contained relevant supporting information.
- *Root cause:* the LLM treated citations as "sources used to formulate the answer" rather than "all relevant authoritative sources retrieved." Since one document was sufficient, it omitted the other.
- *Fix:* strengthened the system prompt to require citation of every relevant authoritative source retrieved for the answer, even when one source alone is sufficient.
- *Regression:* reran `final-sale-damaged-exception` and verified the response cites both the final-sale and damaged/wrong-item sections while still giving the correct 7-calendar-day requirement.

**Bug 2 — Retrieval missed relevant sections**
- *Reproduced:* `canada-multiturn` repeatedly missed the duties/taxes disclosure even though it existed in the KB.
- *Cause:* embeddings used only `chunk.text`; section headings were excluded, so short sections like "Duties and taxes" were hard to retrieve.
- *Fix:* added document title + section heading to the text used for embedding.
- *Regression:* rebuilt the index and compared trace scores — the duties/taxes chunk improved from 0.313 → 0.324 and consistently entered the top-8 context.

**Bug 3 — KB content overrode tool results**
- *Reproduced:* an order with no delivery estimate received a generic KB delivery-time range instead.
- *Cause:* the model had both the specific order-tool result and general shipping-policy content, with no explicit precedence rule.
- *Fix:* added a prompt rule — specific tool results are authoritative for the order; never substitute a general KB estimate when the tool value is unavailable.
- *Regression:* custom eval requires no invented arrival date and requires the "delivery estimate unavailable" response.

**Bug 4 — Abstention backstop caused false handoffs**
- *Reproduced:* a custom order/coupon case incorrectly returned `handoff=True` even though the agent correctly handled the tool result.
- *Cause:* the abstention heuristic interpreted missing fields in a valid tool result as insufficient KB information.
- *Fix:* scoped the KB-abstention backstop so it does not trigger when a tool was successfully called.
- *Regression:* custom regression case confirms `handoff=False`.

---

## 8. Known Limitations

- **Generation variance** — temperature=0 reduces but does not eliminate output variation on the shared Groq endpoint. *Improvement:* use dedicated/self-hosted inference for strict consistency.
- **Exact-phrase evaluation sensitivity** — a correct, well-grounded answer can still fail an eval case that checks for specific wording rather than meaning. *Improvement:* semantic/concept-level assertions alongside exact-phrase checks.
- **Transient API failures** — Groq requests can still fail after retries. *Improvement:* add a circuit breaker and/or fallback model.
- **Single-round tool calling** — one tool-calling round per turn. *Improvement:* multi-step tool orchestration if future tools require it.
- **Unbounded session history** — not currently summarized or truncated. *Improvement:* token-aware history truncation/summarization.
- **Malformed order IDs** — treated as formatting issues rather than automatic escalation. *Improvement:* confirm whether malformed IDs should trigger handoff.

---

## 9. AI Coding Tools Used

Claude and ChatGPT were used throughout for: reviewing modules against the assignment requirements, diagnosing evaluation failures using trace logs and test evidence, writing and revising implementation fixes, and refining the README.

**Example of a corrected AI suggestion:** during debugging, Claude initially suggested that increased retrieval results were caused by the conflict-detection threshold and recommended changing `min_score`. Instead of applying the change immediately, retrieval scores were checked against the trace logs — the evidence showed conflict detection wasn't the root cause; the actual issue was the heading-embedding gap described in Bug 2. This reinforced the project's debugging approach: **verify against trace-log and evaluation evidence before changing code.**

---

## 10. Demo

🎥 **Demo Video:**

<video src="assets/agent_demo.mp4" controls width="700"></video>

If your viewer doesn't render inline video, watch/download it directly: [`assets/agent_demo.mp4`](assets/agent_demo.mp4), or view on [Google Drive](https://drive.google.com/drive/folders/1LuIPnrR54TwwoV1HK9Kwocal8SwStqMi?usp=sharing).

The demo demonstrates:
- One knowledge-base question with citations.
- One order lookup.
- One multi-turn conversation.
- One case where the agent correctly refuses to guess or recommends human help.
- The evaluation suite running (21/21 cases passed).