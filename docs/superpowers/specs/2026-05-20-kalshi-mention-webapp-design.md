# Kalshi Mention Webapp Design

## Goal

Build a production-grade web app that wraps the existing `kalorie` mention-engine pipeline so a user can:

1. Browse all open Kalshi earnings mention markets.
2. Open a market detail page.
3. Upload market-specific `EX-99.*` files via drag-and-drop.
4. Submit background training/rerun jobs that use existing historical ingestion and model workflows.
5. See job progress without blocking the UI.
6. View phrase-contract-level market data plus model prediction outputs.

This design prioritizes shipping speed and low risk by orchestrating existing CLI/model code rather than rebuilding modeling logic.

## Confirmed Product Decisions

- Architecture option: thin web orchestrator (FastAPI + React).
- Frontend stack: React + Tailwind + shadcn/ui.
- Job policy: run all independent jobs in parallel.
- Market scope: open earnings mention markets only (`KXEARNINGSMENTION*`).
- Detail table granularity: one row per phrase contract (not per transcript token).
- History window default: all available quarters.
- Data mode default: mixed best effort (local cache first, fetch missing data when possible).
- Run selection behavior:
  - On `/markets/:ticker`, if a run exists, show latest completed run by default.
  - If multiple runs exist, show a top-right dropdown to switch runs.
- Rerun optimization:
  - Persist event/company-level historical cache in `runs/web/events/<company_symbol>/<event_key>/data/`.
  - Store execution-specific artifacts in `runs/web/events/<company_symbol>/<event_key>/runs/<run_id>/`.
  - Reuse cached historical data for reruns unless explicitly refreshed.
- Leakage control:
  - Every job has an explicit effective decision cutoff timestamp.
  - Evidence newer than cutoff is excluded from dataset/training inputs.
- API reliability:
  - Job creation uses idempotency keys and duplicate-submit deduplication.
- Price normalization:
  - Standardize on dollar fixed-point fields when available and preserve decimal precision end-to-end.

## Existing Backend Context (From Repo)

The model/evidence pipeline already exists and should be reused. Relevant capabilities already implemented:

- Transcript/news/SEC ingestion commands.
- Dataset builders (synthetic + real paths).
- `model1` optimized training and company-specific retraining.
- Evaluation and prediction artifact generation.
- Kalshi market discovery and market comparison helpers.

Primary orchestration entrypoint today: `src/kalorie/app/cli.py`.

## Architecture

### Backend

Add a FastAPI service layer that orchestrates existing pipeline commands and exposes stable API contracts for the frontend.

Core responsibilities:

- Open market discovery for `KXEARNINGSMENTION*`.
- Market detail retrieval and normalized top-of-book stats.
- Async job submission and tracking.
- Per-market run listing and retrieval.
- Run artifact serving for completed jobs.
- Job stream broadcasting via WebSocket.

### Frontend

Create a React app (Vite) with Tailwind and shadcn that includes:

- Home market list page.
- Market detail page with upload + submit.
- Global right-side background jobs rail.
- Completion toast deep-linking to the finished run view.
- Run selector dropdown on market page.

## Data and Run Layout

### Event/Company Cache (Persistent Across Runs)

`runs/web/events/<company_symbol>/<event_key>/data/`

Where `event_key` uses Kalshi `event_ticker` when available, otherwise deterministic fallback:

`<company_symbol>-FY<year>Q<quarter>-<event_date>`

Contains reusable historical inputs shared across phrase contracts in the same event, such as:

- historical transcript slices
- normalized SEC manifest snapshots
- normalized news manifest snapshots
- phrase catalog expansions and merged manifest snapshots
- cached intermediate files used by dataset build steps
- `cache_manifest.json` with version signatures and validity metadata

### Execution Run Directory

`runs/web/events/<company_symbol>/<event_key>/runs/<run_id>/`

Contains run-specific files:

- `uploads/` (user-provided `EX-99.*` originals)
- `inputs.json` (job options and run config)
- `job_status.json` (state/progress/errors)
- `job_log.txt` (command output logs)
- `artifacts/` (run outputs and links/copies)
- `result.json` (frontend-ready phrase-contract table)
- `summary.json` (timings, warnings, data coverage, key metrics)

Optional market index pointers for fast lookup from market routes:

- `runs/web/market-index/<market_ticker>.json` -> latest run metadata + event cache path

### Run ID

Use sortable timestamp + short unique suffix:

`YYYYMMDD-HHMMSS-<short_id>`

## API Contract

### Markets

- `GET /api/markets/open`
  - returns open `KXEARNINGSMENTION*` markets with pagination under the hood
- `GET /api/markets/{market_ticker}`
  - returns market metadata and latest snapshot fields used by UI

### Runs

- `GET /api/markets/{market_ticker}/runs`
  - returns runs for dropdown (newest first)
- `GET /api/markets/{market_ticker}/runs/latest`
  - convenience endpoint for default-run behavior
- `GET /api/markets/{market_ticker}/runs/{run_id}`
  - returns selected run details + result payload

### Jobs

- `POST /api/markets/{market_ticker}/jobs`
  - multipart file upload (`EX-99.*`) + options payload
  - supports `Idempotency-Key` header (required in clients)
  - request includes `decision_cutoff_ts` (optional from client)
  - server always resolves and stores `effective_decision_cutoff_ts`
  - returns `job_id` immediately
- `GET /api/jobs`
  - list active and recent jobs
- `GET /api/jobs/{job_id}`
  - detailed status/log pointers
- `WS /api/jobs/stream`
  - push job state and progress events

## Job State Machine

`queued -> preparing_data -> hydrating_cache -> fetching_missing_data -> training -> predicting -> finalizing -> completed`

Alternative terminal states:

- `failed`
- `canceled`

Each transition emits a WebSocket event and writes to `job_status.json`.

## Parallelism and Locking

Default behavior: run submitted jobs in parallel.

Serialize only when required by dependency conflicts, such as:

- same `run_id` path collision (should be avoided by unique IDs)
- shared temp artifact path collisions
- explicit resource lock requirements in reused pipeline steps

Keep lock scope minimal to preserve throughput.

## Resource Budgets and Concurrency Caps

Parallel-by-default still uses explicit scheduler budgets to prevent thrash:

- `max_active_jobs` (global parallel job cap)
- `max_cpu_training_jobs` (CPU-heavy stage cap)
- `max_openai_requests_in_flight` (provider cap)
- `max_news_provider_requests_in_flight` (source cap)
- per-provider retry/backoff limits

Recommended initial behavior:

- Jobs can all be submitted immediately.
- Scheduler admits work as resources are available.
- UI shows `queued (waiting for resources)` clearly when budget-limited.

## Leakage Control Contract

Each run must carry a strict decision timestamp contract:

- `effective_decision_cutoff_ts` is written to `inputs.json` and `summary.json`.
- Evidence inclusion rule: `published_at <= effective_decision_cutoff_ts`.
- Cache entries are cutoff-aware; future-dated artifacts are not reused for earlier-cutoff reruns.
- Run outputs include cutoff provenance so Brier evaluations remain auditable.

If client omits cutoff, backend sets it to request-received timestamp and returns it explicitly.

## Cache Versioning and Invalidation

Event/company cache reuse requires signature validation. `cache_manifest.json` includes:

- `pipeline_version`
- `feature_schema_version`
- `model_recipe_version`
- `phrase_catalog_hash`
- `source_manifest_hash`
- `cutoff_policy_version`

Invalidation behavior:

- Exact signature match -> reuse cache.
- Partial mismatch -> selective rebuild of affected cache segments.
- Major mismatch -> full cache refresh for event scope.

Cache decisions are logged in `job_log.txt` and summarized in `summary.json`.

## Job Idempotency and Retry Semantics

`POST /api/markets/{market_ticker}/jobs` deduplicates by:

- `Idempotency-Key`
- authenticated user/session identity
- normalized request payload hash

Behavior:

- Same key + same payload within dedupe window -> return existing `job_id`.
- Same key + different payload -> `409` conflict with explanation.
- Retry after network drop is safe with same idempotency key.

This prevents accidental duplicate expensive runs from reconnect/retry flows.

## Market Price Normalization Contract

To avoid subpenny/format drift:

- Prefer Kalshi dollar fixed-point fields when present (`*_dollars`).
- Parse into `Decimal` internally (no float for persisted prices).
- Preserve canonical string precision in API responses where applicable.
- Only derive reciprocal ask fields from normalized Decimal bids.

Result payload exposes consistent price fields across runs and markets.

## Frontend UX

### Home (`/`)

- list open mention markets
- quick search/filter
- navigate to `/markets/:ticker`

### Market Detail (`/markets/:ticker`)

- header with market metadata and live snapshot
- run selector dropdown in top-right if multiple runs
- if no run query param:
  - load latest completed run by default when available
- drag/drop upload zone for `EX-99.*` files
- submit button triggers async job
- result table (phrase contract rows):
  - phrase/contract
  - yes bid
  - yes ask (derived where needed)
  - spread
  - volume
  - model prediction

### Right Rail

- always-visible background jobs panel
- shows job stage, elapsed time, status, and actions (retry/cancel when valid)

### Toast

- shadcn toast on completion
- click opens `/markets/:ticker?run=<run_id>`

## Pipeline Reuse Strategy

The web layer should call existing CLI/module workflows rather than rewriting model logic.

Execution pattern:

1. Build or update event/company cache in `runs/web/events/<company_symbol>/<event_key>/data/`.
2. Materialize run-specific input set in `runs/web/events/<company_symbol>/<event_key>/runs/<run_id>/`.
3. Execute existing dataset/build/train/predict pipeline stages.
4. Normalize run outputs into `result.json` for frontend rendering.

## Error Handling

- Mixed best-effort mode:
  - continue when optional sources fail
  - collect warnings in `summary.json`
- hard failures:
  - mark run `failed`, preserve all logs and partial artifacts
- run retrieval:
  - if requested run missing, return typed 404 with valid run options
- upload validation:
  - reject unsupported file types and empty uploads with clear messages

## Testing and Verification Plan

### Backend

- unit tests for run store/cache path behavior
- unit tests for default latest-run selection
- API tests for runs endpoints and job state transitions
- orchestration tests ensuring cache reuse on rerun

### Frontend

- route tests for default run loading
- run dropdown behavior tests
- toast click navigation tests
- job rail rendering for concurrent jobs

### End-to-End Smoke

1. Submit run for one market with upload.
2. Verify `runs/web/events/<company_symbol>/<event_key>/data/` populated.
3. Submit rerun for another phrase market in same event and verify shared cache reuse.
4. Verify latest run auto-load on `/markets/:ticker`.
5. Verify switching runs via dropdown.

## Non-Goals (Initial Slice)

- Trade execution or order placement.
- Full observability platform rollout.
- Replacing existing model internals.
- Token-level transcript prediction UI.

## Open Questions Resolved in This Spec

- Parallel job execution: yes, for independent jobs.
- Run selection behavior: latest by default + dropdown for multiple runs.
- Rerun data persistence: required event/company cache in `runs/web/events/<company_symbol>/<event_key>/data/`.
- Leakage control: explicit decision cutoff contract is mandatory on every run.
- Job API reliability: idempotency-key dedupe is required.

