# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## How to explain things to the user (important)

The user is learning. When explaining ANY concept, follow these rules — strictly:

- **Simple language.** Plain English, not jargon. If you must use a technical term, define it in one sentence.
- **Always include a concrete example.** Code snippet, before/after comparison, real-world analogy, or a worked-through case. Never explain a concept abstractly without showing it.
- **Crisp and short.** Aim for ~30-50 lines per explanation. Cut anything that isn't the core point.
- **No exhaustive surveys.** If there are 5 options, pick ONE recommendation and explain it. Don't list all 5 unless asked.
- **Use tables sparingly** — only when comparing 2-3 things side-by-side adds real value.
- **No fluff phrases** like "as we discussed," "great question," or filler intros. Start with the answer.

Bad: "Dependency injection is a design pattern where dependencies are provided to a class rather than created within it. There are several forms including constructor injection, setter injection, and interface injection. Each has its own tradeoffs..."

Good: "DI = pass the thing in instead of building it inside. Example: `Search(pinecone=client)` instead of `Search()` then `self.pinecone = PineconeClient(...)` inside. Win: you can swap `client` for a fake in tests."

## Working rules

- **The user writes new scripts themselves.** Guide first (skeleton + key calls). Only write the code when they say "write it", "fix", "complete", etc.
- **Ask before anything that costs money or writes to an external service:** OpenAI calls (an eval run is ~$0.77), Pinecone upserts, DagsHub/MLflow tag or run writes, Supabase writes. Reads from DagsHub/Supabase are fine.
- **Never read or print `.env`.** It holds real secrets (Supabase password, `DAGSHUB_USER_TOKEN`, API keys).
- **Parameters are changed in `src/config.py` defaults, not `.env`.** Don't suggest env-var overrides for tunables.

## What this project is

**PaperPilot** (GitHub `k-Rohit/PaperPilot`; local folder still named `Arxiv_Paper_Curator`) — an end-to-end RAG + agentic RAG system over arXiv papers, built as a learning/portfolio project. It started from the [jamwithai/production-agentic-rag-course](https://github.com/jamwithai/production-agentic-rag-course) (tags `week2.0` … `week7.0`; fetch files via `https://raw.githubusercontent.com/jamwithai/production-agentic-rag-course/<tag>/<path>`), then deviated: Pinecone instead of OpenSearch, GitHub Actions instead of Airflow, OpenAI instead of Ollama, LangSmith instead of Langfuse, Supabase instead of local Postgres.

The LLMOps part (experiment tracking + regression gate) follows [Himanshu-1703/llmops-rag-app](https://github.com/Himanshu-1703/llmops-rag-app).

`README.md` is the up-to-date public overview. `docs/` is gitignored and local-only (`AGENTS_GUIDE.md`, `INTERVIEW_PREP.md`, `AUDIT_LOG.md`); some of it predates the Pinecone migration.

## Common commands

Python tooling is **`uv`**, Python `>=3.12,<3.13`.

```
uv sync                                   # install deps
make dev                                  # start Redis only (Postgres = Supabase, Pinecone = hosted)
make api                                  # uvicorn with reload → http://localhost:8000 (chat UI at /, docs at /docs)
make start / make stop                    # Redis + API in Docker

uv run ruff check src evals scripts tests # lint (I, F, E, W, B, RET, SIM, UP)
uv run ruff format src evals scripts tests
uv run python -m py_compile <file>        # quick syntax check after edits

uv run python -m scripts.ingest_papers           # daily ingestion (same as the GH Actions cron)
uv run python -m scripts.reindex_from_postgres   # rebuild Pinecone from Postgres
```

`make test/lint/format/setup/clean` are in `.PHONY` but not implemented.

### Evals, tracking, regression gate

```
uv run python -m evals.data.generate_goldens      # once; goldens are committed
uv run python -m evals.evaluate_rag_app           # PAID (~$0.77): regenerates answers, grades, logs an MLflow run on DagsHub
uv run python -m scripts.create_thresholds        # recompute evals/thresholds.json from baseline + noise runs
uv run pytest tests/                              # the gate: regression + promotion tests (reads DagsHub)
uv run python -m scripts.promote promote|reject   # move stage tags based on the gate result (writes DagsHub)
```

## Architecture

Dependencies flow downward.

```
ENTRYPOINTS     src/main.py (FastAPI; lifespan builds services onto app.state)
                src/routers/{ping,ask,agentic_ask}.py   (/api/v1/health, /ask, /agentic_ask)
                scripts/ingest_papers.py  (.github/workflows/ingest.yml, weekdays 06:00 UTC)
                evals/ , scripts/{create_thresholds,promote}.py, tests/
       ↓
AGENT           src/services/agents/  (LangGraph: condense → guardrail → router → retrieve → grade →
                rewrite / live fetch → generate; nodes/ one file per node; AsyncPostgresSaver checkpoints)
       ↓
SERVICES        metadata_fetcher.py (ingestion orchestrator), arxiv/, pdf_parser/ (Docling for ingestion,
                pypdfium2 for live fetch), indexing/ (TextChunker + HybridIndexingService),
                embeddings/ (OpenAI), pinecone_/ (hybrid search), openai_/ (LLM + rag_system.txt), cache/ (Redis)
       ↓
DATA ACCESS     src/repositories/paper.py (PaperRepository), src/db/, src/models/paper.py
       ↓
FOUNDATION      src/config.py, src/exceptions.py, src/schemas/
```

### Key patterns

- **Service factories.** Each service has `factory.py` with `@lru_cache(maxsize=1) def make_<service>()`. Copy this for new services. The Docling parser is the expensive one (~1GB of models) — never bypass the cache.
- **FastAPI DI.** Services are built once in `main.py`'s lifespan and stored on `app.state`; routes take the `*Dep` aliases from `src/dependencies.py`.
- **Repository owns DB queries.** No `session.query(...)` or raw SQL outside `PaperRepository`. The session is injected by the caller.
- **`config.py` is the single settings source.** `Settings` holds per-domain sub-configs as `Field(default_factory=...)`: `arxiv`, `pdf_parser`, `openai_embeddings`, `openai_client`, `redis`, `deepeval`, `retrieval`, `chunking`, `pinecone_`. Every tunable that can move eval scores lives here (top_k, alpha, chunking sizes, LLM model/temperature, judge model, threshold).
- **Pinecone hybrid.** Dense OpenAI (1024 dims) + `pinecone-sparse-english-v0`, dotproduct index, α-weighted. Sparse vectors are L2-normalized on upsert *and* query — without it sparse scores (~1000) drown out dense (≤1).

### Eval + LLMOps pipeline

- `evals/evaluate_rag_app.py::track_experiment(run_type)` → generate fresh answers → `collect_params()` → DeepEval (5 metrics) → `metrics_from_result()` → MLflow on DagsHub (experiment `paperpilot-rag`; artifacts = eval dataset JSON + Markdown report).
- **Two tags per run.** `run_type` (baseline / noise / experiment) is permanent. `stage` moves: `challenger` → `champion` / `rejected` / `demoted`; old champion → `archived`. Invariant: exactly one champion, at most one challenger. Only `run_type=experiment` runs get `stage=challenger`, set *after* everything is logged.
- `evals/mlflow_utils.py` holds `EXPERIMENT_NAME`, `find_runs_by_stage`, `get_run_by_stage` (exactly-one or raise), `demote_stale_challengers`. Import from here — don't copy them.
- `evals/thresholds.json` = std-dev of each metric across same-config runs (baseline + noise). `pass_rate` is excluded (too noisy).
- **Gate rule:** `tests/test_regression.py` — no metric below `champion − 2×noise` (parametrized per metric). `tests/test_promotion.py` — at least `MIN_IMPROVED_METRICS` (1) metric above `champion + 2×noise`. pytest's exit code decides; `scripts/promote.py` only moves tags.

## Gotchas

1. **ORM registration order.** `Base` lives in `src/db/interfaces/postgresql.py`; `Paper` only registers when `src/models/paper.py` is imported. Import `Paper` before `make_database()` (hence the `# ruff: noqa: I001` in `evals/eval_utils.py`).
2. **Indentation bugs disguised as logic bugs** — `return` nested inside `if`, methods detached from their class, loops returning on the first iteration. Run `py_compile` and re-read indentation after non-trivial edits.
3. **The eval grades the newest saved dataset.** `evaluate_rag_app.py` regenerates it first; running grading alone just re-grades old answers.
4. **Don't import from `tests/` in app code.** Test modules run DagsHub queries at import time and raise when there's no challenger.
5. **DagsHub's MLflow server rejects `IN` in `filter_string`** — query one tag value at a time. Use `mlflow.get_experiment_by_name` in read-only code (`set_experiment` creates one if missing).
6. **`mlflow.set_tag` tags the *active* run only;** to retag an existing run use `MlflowClient().set_tag(run_id, ...)`.
7. **Logging:** scripts use `basicConfig(level=WARNING)` + `logger.setLevel(INFO)` for their own logger; root INFO floods the output with httpx/DeepEval logs.
8. **arXiv rate limit (HTTP 429)** — ~1 req / 3s; repeated runs in a short window get you banned for minutes. Wait, don't retry harder.
9. **Supabase pooler:** use the Session pooler (port 5432); the pool is kept small (`postgres_pool_size=5`, no overflow).

## Where things live

- New tunables / env settings: `src/config.py` (+ add them to `collect_params()` in `evals/eval_utils.py` if they affect eval scores).
- DB schema: `src/models/paper.py` (ORM) + `src/schemas/arxiv/paper.py` (Pydantic) — keep in sync.
- Prompts: `src/services/openai_/prompts/rag_system.txt` (plain RAG), `src/services/agents/prompts.py` (agent).
- Agent nodes: `src/services/agents/nodes/`; graph wiring in `agentic_rag.py`. Redraw `assets/agent_graph.png` after changing edges (command in README).
- CI: `.github/workflows/ingest.yml` (daily ingestion). Eval CI workflow not written yet.
