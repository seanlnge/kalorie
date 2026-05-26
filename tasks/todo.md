# Kalorie Advantage Engine MVP Implementation Plan

> **For agentic workers:** Do not implement this plan until the user approves the plan check-in. Use checkbox (`- [ ]`) syntax for tracking. Production code tasks must follow the cycle: write failing tests, run them to confirm failure, implement the smallest useful behavior, then verify.

**Goal:** Build the first realistic MVP slice of the earnings-call mention-market advantage engine: parse the local CAVA earnings release, normalize market/evidence data, create deterministic labels and simple features, and compare a transparent probability baseline against paper market prices.

**Architecture:** Start with a Python package and CLI because the first slice is data, parsing, labeling, and modeling heavy. Keep exchange/vendor integrations behind small client interfaces with mocked tests so the local CAVA workflow can run without live credentials. Defer the Next.js dashboard, databases, websockets, and automated trading until the core data path is testable.

**Tech Stack:** Python 3.11+, `pytest`, `pydantic`, `pydantic-settings`, `httpx`, `typer`, `rich`, `pdfplumber`, `pandas`, `scikit-learn`, `ruff`. No secrets are read or printed; env validation checks only presence and file path shape.

**Context Inspected:** `kalorie/advantage-engine-design.md` defines a larger architecture for market ingestion, document ingestion, transcript processing, labeling, feature generation, calibration, and paper trading. Current local files include `advantage-engine-design.md`, `Earnings-Release-2026-Q1.pdf`, `.env`, and `kalshi.rsa`. The secret files exist but must not be opened. `Earnings-Release-2026-Q1.pdf` was confirmed as the CAVA Q1 2026 earnings press release.

---

## MVP Boundary

- [x] Build only a local, reproducible research workflow for one CAVA earnings release and mocked market/vendor data.
- [x] Support live credentials only through explicit config validation and client abstractions; tests must not require real API calls.
- [x] Produce inspectable JSON artifacts for documents, labels, features, predictions, and paper comparisons.
- [x] Use deterministic settlement-style exact matching for labels; semantic or lexical similarity is predictive evidence, not settlement truth.
- [x] Include one CLI entry point because it provides a repeatable workflow and verification target. Do not build a dashboard in this slice.
- [x] Exclude real order placement, live websocket ingestion, persistent databases, vector databases, hosted embeddings, LLM calls, and browser/search backfills from this slice.

## Planned File Structure

- [x] Create `pyproject.toml` for the Python package, test runner, formatter, linter, and dependencies.
- [x] Create `.env.example` with empty example variable names only:
  - `API_NINJAS_API_KEY=`
  - `KALSHI_API_KEY_ID=`
  - `KALSHI_PRIVATE_KEY_PATH=`
  - `KALSHI_BASE_URL=https://api.elections.kalshi.com/trade-api/v2`
- [x] Create `src/kalorie/__init__.py` with package metadata.
- [x] Create `src/kalorie/config.py` for redacted settings and mode validation.
- [x] Create `src/kalorie/models.py` for normalized data models.
- [x] Create `src/kalorie/documents.py` for local PDF/text ingestion and chunking.
- [x] Create `src/kalorie/api_ninjas.py` for earnings calendar and transcript client abstractions.
- [x] Create `src/kalorie/kalshi.py` for public and authorized market-data client abstractions.
- [x] Create `src/kalorie/markets.py` for mention-market title parsing and normalized market snapshots.
- [x] Create `src/kalorie/labeling.py` for deterministic exact and lexical mention labeling.
- [x] Create `src/kalorie/features.py` for lexical and TF-IDF similarity features.
- [x] Create `src/kalorie/modeling.py` for transparent baseline probabilities.
- [x] Create `src/kalorie/paper.py` for implied probability, spread-aware edge, and paper-trade comparison.
- [x] Create `src/kalorie/cli.py` for the local CAVA workflow.
- [x] Create `tests/unit/` for fast unit tests with no network and no secrets.
- [x] Create `tests/integration/` for local-file workflow tests using `Earnings-Release-2026-Q1.pdf`.
- [x] Create `data/.gitkeep` and `runs/.gitkeep` if the implementation needs output directories; generated run artifacts should be ignored by git.

---

## Task 0: Plan Approval Gate

**Files:**
- Existing: `tasks/todo.md`

- [x] **Step 1: Get user approval before code changes**

Ask the user to approve, narrow, or revise this plan. Do not create `pyproject.toml`, package files, tests, or generated data until approval is explicit.

- [x] **Step 2: Confirm the initial market phrase set**

Use a small phrase list for the first run so the CAVA document path is concrete:

```text
traffic
same restaurant sales
digital revenue
geopolitical uncertainty
value proposition
margin
```

Expected result: the first implementation has enough phrases to exercise exact matches, multi-word matches, absent phrases, and paper-market comparison without pretending to cover all possible markets.

---

## Task 1: Scaffold Python Package and Test Harness

**Files:**
- Create: `pyproject.toml`
- Create: `src/kalorie/__init__.py`
- Create: `tests/conftest.py`
- Create: `.env.example`

- [x] **Step 1: Write the failing scaffold smoke test**

Create `tests/unit/test_scaffold.py` with assertions that `kalorie.__version__` exists and the package imports without reading environment files.

Expected test intent:

```python
def test_package_import_has_version():
    import kalorie

    assert isinstance(kalorie.__version__, str)
    assert kalorie.__version__
```

- [x] **Step 2: Run the scaffold test to verify failure**

Run: `python -m pytest tests/unit/test_scaffold.py -q`

Expected: fail because `pyproject.toml` and `src/kalorie` do not exist yet.

- [x] **Step 3: Add the minimal package scaffold**

Create `pyproject.toml` with package metadata, dependencies, `pytest` config, and `ruff` config. Use `src` layout. Keep dependencies limited to the MVP libraries listed in the header.

Create `src/kalorie/__init__.py` with `__version__ = "0.1.0"`.

Create `.env.example` with empty variable names only. Do not copy values from `.env`.

- [x] **Step 4: Verify the scaffold passes**

Run:

```powershell
python -m pip install -e ".[dev]"
python -m pytest tests/unit/test_scaffold.py -q
python -m ruff check src tests
```

Expected: pytest passes; ruff reports no lint failures.

---

## Task 2: Config and Secret-Safe Env Validation

**Files:**
- Create: `src/kalorie/config.py`
- Create: `tests/unit/test_config.py`

- [x] **Step 1: Write failing tests for redaction and mode validation**

Test cases:

- `Settings.model_dump()` must not include secret values when rendered for logs.
- `Settings.redacted_dict()` returns booleans like `api_ninjas_configured=True` instead of key material.
- Missing API Ninjas key is allowed for local-only runs.
- Missing Kalshi auth is allowed for public market-data mode.
- Authorized Kalshi mode requires `KALSHI_API_KEY_ID` and `KALSHI_PRIVATE_KEY_PATH`.
- `KALSHI_PRIVATE_KEY_PATH` validation checks path presence only; it never reads or prints file contents.

- [x] **Step 2: Run config tests to verify failure**

Run: `python -m pytest tests/unit/test_config.py -q`

Expected: fail because `kalorie.config` does not exist.

- [x] **Step 3: Implement minimal settings**

Use `pydantic-settings` with explicit fields:

- `api_ninjas_api_key: SecretStr | None`
- `kalshi_api_key_id: SecretStr | None`
- `kalshi_private_key_path: Path | None`
- `kalshi_base_url: AnyHttpUrl`
- `mode: Literal["local", "api", "kalshi_public", "kalshi_authorized"]`

Expose a `validate_for_mode(mode)` method and a `redacted_dict()` method. Avoid custom `__repr__` that could accidentally leak fields.

- [x] **Step 4: Verify config behavior**

Run:

```powershell
python -m pytest tests/unit/test_config.py -q
python -m ruff check src/kalorie/config.py tests/unit/test_config.py
```

Expected: tests pass; no secret values appear in assertion output.

---

## Task 3: Normalized Data Models

**Files:**
- Create: `src/kalorie/models.py`
- Create: `tests/unit/test_models.py`

- [x] **Step 1: Write failing model serialization tests**

Cover these models:

- `Company(symbol="CAVA", name="CAVA Group, Inc.")`
- `EarningsEvent(company_symbol="CAVA", fiscal_year=2026, fiscal_quarter=1, event_date=date(2026, 5, 19))`
- `SourceDocument(source_id, company_symbol, document_type, source_path, published_at, content_hash)`
- `DocumentChunk(document_id, chunk_index, text, section, token_start, token_end)`
- `TargetPhrase(phrase, normalized_phrase, aliases)`
- `MarketSnapshot(venue, market_id, title, yes_bid, yes_ask, observed_at)`
- `MentionLabel(target_phrase, exact_mentioned, lexical_mentioned, match_spans)`
- `FeatureVector(target_phrase, features)`
- `Prediction(target_phrase, model_version, probability, reasons)`
- `PaperTradeComparison(target_phrase, model_probability, market_probability, edge, side)`

- [x] **Step 2: Run model tests to verify failure**

Run: `python -m pytest tests/unit/test_models.py -q`

Expected: fail because `kalorie.models` does not exist.

- [x] **Step 3: Implement the models**

Use `pydantic.BaseModel` for validation and JSON serialization. Store all timestamps as timezone-aware `datetime`. Use `Decimal` for market prices when precision matters and float for ML features.

- [x] **Step 4: Verify model tests**

Run:

```powershell
python -m pytest tests/unit/test_models.py -q
python -m ruff check src/kalorie/models.py tests/unit/test_models.py
```

Expected: model serialization round trips and validation errors are explicit.

---

## Task 4: Local CAVA Press Release Ingestion

**Files:**
- Create: `src/kalorie/documents.py`
- Create: `tests/unit/test_documents.py`
- Create: `tests/integration/test_cava_press_release.py`
- Use local input: `Earnings-Release-2026-Q1.pdf`

- [x] **Step 1: Write failing unit tests for text normalization and chunking**

Use small in-test strings to verify:

- repeated whitespace is normalized,
- document hashes are stable,
- chunking preserves chunk order,
- chunk metadata includes section labels when headings are detected,
- extraction never reads `.env`, `.rsa`, `.pem`, or files matching `*key*`.

- [x] **Step 2: Write failing integration test for the CAVA PDF**

Test that ingesting `Earnings-Release-2026-Q1.pdf` returns:

- `company_symbol == "CAVA"`,
- `document_type == "earnings_press_release"`,
- extracted text containing `CAVA GROUP REPORTS FIRST QUARTER 2026 RESULTS`,
- extracted text containing `same restaurant sales`,
- at least one chunk with non-empty text,
- a non-empty content hash.

- [x] **Step 3: Run ingestion tests to verify failure**

Run:

```powershell
python -m pytest tests/unit/test_documents.py tests/integration/test_cava_press_release.py -q
```

Expected: fail because the ingestion module does not exist.

- [x] **Step 4: Implement local document ingestion**

Implement:

- `extract_text_from_pdf(path: Path) -> str` using `pdfplumber`.
- `normalize_text(text: str) -> str`.
- `content_hash(bytes_or_text) -> str` using SHA-256.
- `chunk_text(text: str, max_chars: int = 1800, overlap_chars: int = 200) -> list[DocumentChunk]`.
- `ingest_local_press_release(path, company_symbol, fiscal_year, fiscal_quarter, published_at) -> tuple[SourceDocument, list[DocumentChunk]]`.

Reject secret-looking paths before opening files.

- [x] **Step 5: Verify CAVA ingestion**

Run:

```powershell
python -m pytest tests/unit/test_documents.py tests/integration/test_cava_press_release.py -q
python -m ruff check src/kalorie/documents.py tests/unit/test_documents.py tests/integration/test_cava_press_release.py
```

Expected: CAVA PDF is parsed locally; no secrets are opened.

---

## Task 5: API Ninjas Client Abstraction

**Files:**
- Create: `src/kalorie/api_ninjas.py`
- Create: `tests/unit/test_api_ninjas.py`

- [x] **Step 1: Write failing mocked-client tests**

Use `httpx.MockTransport` to verify:

- API key is sent only in the request header required by API Ninjas.
- Earnings calendar responses map into `EarningsEvent`.
- Transcript responses map into `SourceDocument` and transcript chunks.
- HTTP 401 raises `VendorAuthError`.
- HTTP 429 raises `VendorRateLimitError`.
- Unit tests do not load `.env`.

- [x] **Step 2: Run API Ninjas tests to verify failure**

Run: `python -m pytest tests/unit/test_api_ninjas.py -q`

Expected: fail because `kalorie.api_ninjas` does not exist.

- [x] **Step 3: Implement the client interface**

Implement `ApiNinjasClient` with injected `httpx.Client`, base URL, and API key. Add methods:

- `get_earnings_calendar(ticker: str, start_date: date, end_date: date) -> list[EarningsEvent]`
- `get_transcript(ticker: str, fiscal_year: int, fiscal_quarter: int) -> SourceDocument`

Keep response parsing narrow and explicit. Do not add news ingestion in this slice unless the earnings/transcript path is already green.

- [x] **Step 4: Verify mocked API Ninjas behavior**

Run:

```powershell
python -m pytest tests/unit/test_api_ninjas.py -q
python -m ruff check src/kalorie/api_ninjas.py tests/unit/test_api_ninjas.py
```

Expected: mocked responses pass; no real network is required.

---

## Task 6: Kalshi Market Data Client Abstraction

**Files:**
- Create: `src/kalorie/kalshi.py`
- Create: `tests/unit/test_kalshi.py`

- [x] **Step 1: Write failing public-mode tests**

Use `httpx.MockTransport` to verify:

- public market metadata can be fetched without credentials,
- market snapshots map into `MarketSnapshot`,
- prices are normalized to decimal probabilities between `0` and `1`,
- stale or missing bid/ask fields raise a parse error with the market ID.

- [x] **Step 2: Write failing authorized-mode tests**

Use a fake signer object to verify:

- authorized mode requires configured key ID and private key path,
- request signing is delegated to a signer interface,
- tests never open `kalshi.rsa`,
- auth headers are attached only in authorized mode.

- [x] **Step 3: Run Kalshi tests to verify failure**

Run: `python -m pytest tests/unit/test_kalshi.py -q`

Expected: fail because `kalorie.kalshi` does not exist.

- [x] **Step 4: Implement Kalshi client boundaries**

Implement:

- `KalshiPublicClient` for public event/market discovery.
- `KalshiAuthorizedClient` for authenticated market-data reads only.
- `KalshiSigner` protocol with a later concrete RSA implementation.

Do not implement order placement, cancellation, or execution endpoints in this slice.

- [x] **Step 5: Verify Kalshi boundaries**

Run:

```powershell
python -m pytest tests/unit/test_kalshi.py -q
python -m ruff check src/kalorie/kalshi.py tests/unit/test_kalshi.py
```

Expected: public and authorized modes are separated; secret material is never loaded in tests.

---

## Task 7: Mention Market Normalization

**Files:**
- Create: `src/kalorie/markets.py`
- Create: `tests/unit/test_markets.py`

- [x] **Step 1: Write failing parser tests**

Use titles such as:

- `Will CAVA mention traffic during earnings?`
- `Will CAVA mention "same restaurant sales" on its earnings call?`
- `Will CAVA mention geopolitical uncertainty?`

Expected parsed fields:

- `company_symbol == "CAVA"`,
- `target_phrase` extracted without surrounding quotes,
- normalized phrase is lowercase,
- unsupported titles return a structured parse error.

- [x] **Step 2: Run parser tests to verify failure**

Run: `python -m pytest tests/unit/test_markets.py -q`

Expected: fail because `kalorie.markets` does not exist.

- [x] **Step 3: Implement simple market title parsing**

Implement `parse_mention_market_title(title: str) -> TargetPhrase`. Keep the parser narrow and auditable; expand only for observed Kalshi/Polymarket title patterns.

- [x] **Step 4: Verify market normalization**

Run:

```powershell
python -m pytest tests/unit/test_markets.py -q
python -m ruff check src/kalorie/markets.py tests/unit/test_markets.py
```

Expected: target phrases normalize consistently for the CAVA MVP phrase set.

---

## Task 8: Deterministic Mention Labeling

**Files:**
- Create: `src/kalorie/labeling.py`
- Create: `tests/unit/test_labeling.py`

- [x] **Step 1: Write failing exact-label tests**

Test that:

- `traffic` matches `Guest Traffic growth of 6.8%`,
- `same restaurant sales` matches regardless of case,
- `digital revenue` matches `Digital Revenue Mix`,
- `robotaxi` does not match the CAVA release text,
- exact matching uses word boundaries so `AI` does not match inside `said`.

- [x] **Step 2: Write failing lexical-label tests**

Test aliases:

- target `margin` with alias `restaurant-level profit margin`,
- target `value proposition` with alias `compelling value proposition`,
- target `geopolitical uncertainty` with no alias.

Expected: exact and lexical labels are stored separately.

- [x] **Step 3: Run labeling tests to verify failure**

Run: `python -m pytest tests/unit/test_labeling.py -q`

Expected: fail because `kalorie.labeling` does not exist.

- [x] **Step 4: Implement deterministic labelers**

Implement:

- `normalize_phrase(text: str) -> str`,
- `find_exact_mentions(text: str, phrase: str) -> list[MatchSpan]`,
- `find_lexical_mentions(text: str, target: TargetPhrase) -> list[MatchSpan]`,
- `label_document_chunks(chunks, targets) -> list[MentionLabel]`.

Use case-insensitive matching, Unicode quote normalization, and word boundaries. Keep settlement exact labels separate from alias-based lexical labels.

- [x] **Step 5: Verify deterministic labeling**

Run:

```powershell
python -m pytest tests/unit/test_labeling.py -q
python -m ruff check src/kalorie/labeling.py tests/unit/test_labeling.py
```

Expected: exact labels are auditable and do not depend on embeddings or LLMs.

---

## Task 9: Baseline Lexical and Semantic Feature Extraction

**Files:**
- Create: `src/kalorie/features.py`
- Create: `tests/unit/test_features.py`

- [x] **Step 1: Write failing feature tests**

For each target phrase, verify feature keys:

- `exact_match_count`,
- `lexical_match_count`,
- `chunk_count`,
- `target_token_count`,
- `max_tfidf_similarity`,
- `mean_top3_tfidf_similarity`,
- `chunks_above_0_20_similarity`,
- `appears_in_headline_or_first_chunk`.

- [x] **Step 2: Run feature tests to verify failure**

Run: `python -m pytest tests/unit/test_features.py -q`

Expected: fail because `kalorie.features` does not exist.

- [x] **Step 3: Implement baseline feature extraction**

Use deterministic text features first:

- count exact and lexical labels from Task 8,
- compute TF-IDF cosine similarity between target phrases and chunks using `scikit-learn`,
- aggregate top-k similarity features,
- include simple document-position signals.

Call these "baseline semantic/lexical" features. Do not add hosted embeddings in this slice; TF-IDF gives a reproducible first semantic-ish retrieval baseline that can be tested locally.

- [x] **Step 4: Verify feature extraction**

Run:

```powershell
python -m pytest tests/unit/test_features.py -q
python -m ruff check src/kalorie/features.py tests/unit/test_features.py
```

Expected: feature vectors are stable across runs and serializable to JSON.

---

## Task 10: Simple Probability Baseline

**Files:**
- Create: `src/kalorie/modeling.py`
- Create: `tests/unit/test_modeling.py`

- [x] **Step 1: Write failing probability tests**

Test two model paths:

- `RuleBasedBaseline` maps feature vectors to probabilities using transparent weights and clamps output to `[0.01, 0.99]`.
- `LogisticBaseline` trains on a small in-test dataframe and returns probabilities for held-out rows.

Expected behavior:

- exact current-material mention increases probability,
- zero evidence stays near the configured base rate,
- higher TF-IDF similarity increases probability more than unrelated chunks,
- model output includes reason codes.

- [x] **Step 2: Run modeling tests to verify failure**

Run: `python -m pytest tests/unit/test_modeling.py -q`

Expected: fail because `kalorie.modeling` does not exist.

- [x] **Step 3: Implement transparent baselines**

Implement:

- `RuleBasedBaseline(base_rate: float = 0.25)`,
- `LogisticBaseline(min_training_rows: int = 20)`,
- `predict_proba(feature_vector) -> Prediction`.

Use the rule-based baseline for the local CAVA demo because there is not yet a historical dataset. Keep logistic regression ready for the first labeled historical backfill.

- [x] **Step 4: Verify probability output**

Run:

```powershell
python -m pytest tests/unit/test_modeling.py -q
python -m ruff check src/kalorie/modeling.py tests/unit/test_modeling.py
```

Expected: probabilities are deterministic, bounded, and explainable.

---

## Task 11: Paper-Trade Comparison

**Files:**
- Create: `src/kalorie/paper.py`
- Create: `tests/unit/test_paper.py`

- [x] **Step 1: Write failing paper-comparison tests**

Test:

- `market_probability` for buying yes uses `yes_ask`,
- `market_probability` for buying no uses `1 - yes_bid`,
- spread is present in the comparison,
- side is `yes` when model probability exceeds yes ask by the configured threshold,
- side is `no` when no probability exceeds no implied probability by the configured threshold,
- side is `skip` when edge is below threshold.

- [x] **Step 2: Run paper tests to verify failure**

Run: `python -m pytest tests/unit/test_paper.py -q`

Expected: fail because `kalorie.paper` does not exist.

- [x] **Step 3: Implement paper comparison**

Implement:

- `implied_yes_probability(snapshot: MarketSnapshot) -> Decimal`,
- `implied_no_probability(snapshot: MarketSnapshot) -> Decimal`,
- `compare_prediction_to_market(prediction, snapshot, min_edge=Decimal("0.05")) -> PaperTradeComparison`.

Include reason codes such as `edge_below_threshold`, `wide_spread`, `yes_edge`, and `no_edge`.

- [x] **Step 4: Verify paper comparison**

Run:

```powershell
python -m pytest tests/unit/test_paper.py -q
python -m ruff check src/kalorie/paper.py tests/unit/test_paper.py
```

Expected: paper comparisons are deterministic and never place orders.

---

## Task 12: Minimal CLI for the Local CAVA Workflow

**Files:**
- Create: `src/kalorie/cli.py`
- Create: `tests/integration/test_cli_cava.py`
- Modify: `pyproject.toml` to expose console script `kalorie=kalorie.cli:app`.

- [x] **Step 1: Write failing CLI integration test**

Use Typer's `CliRunner` to run:

```powershell
python -m kalorie.cli run-local-cava `
  --pdf Earnings-Release-2026-Q1.pdf `
  --market-title "Will CAVA mention traffic during earnings?" `
  --yes-bid 0.38 `
  --yes-ask 0.45 `
  --out runs/cava-q1-2026
```

Expected files:

- `runs/cava-q1-2026/document.json`,
- `runs/cava-q1-2026/chunks.jsonl`,
- `runs/cava-q1-2026/labels.json`,
- `runs/cava-q1-2026/features.json`,
- `runs/cava-q1-2026/prediction.json`,
- `runs/cava-q1-2026/paper_comparison.json`.

- [x] **Step 2: Run CLI test to verify failure**

Run: `python -m pytest tests/integration/test_cli_cava.py -q`

Expected: fail because `kalorie.cli` does not exist.

- [x] **Step 3: Implement the CLI workflow**

Wire together:

1. ingest local CAVA PDF,
2. parse mention-market title,
3. label chunks,
4. extract features,
5. score with `RuleBasedBaseline`,
6. compare to supplied bid/ask,
7. write JSON/JSONL artifacts.

Return a non-zero exit code for missing PDF, invalid bid/ask, or unsupported market title.

- [x] **Step 4: Verify the local CAVA run**

Run:

```powershell
python -m pytest tests/integration/test_cli_cava.py -q
python -m kalorie.cli run-local-cava --pdf Earnings-Release-2026-Q1.pdf --market-title "Will CAVA mention traffic during earnings?" --yes-bid 0.38 --yes-ask 0.45 --out runs/cava-q1-2026
python -m ruff check src tests
```

Expected: CLI writes all run artifacts; no network or secrets are required.

---

## Task 13: Full Verification Before Plan Completion

**Files:**
- Existing: all files created by Tasks 1-12

- [x] **Step 1: Run the full local test suite**

Run:

```powershell
python -m pytest -q
```

Expected: all unit and integration tests pass.

- [x] **Step 2: Run linting**

Run:

```powershell
python -m ruff check src tests
```

Expected: no lint failures.

- [x] **Step 3: Run the local CAVA smoke workflow**

Run:

```powershell
python -m kalorie.cli run-local-cava --pdf Earnings-Release-2026-Q1.pdf --market-title "Will CAVA mention traffic during earnings?" --yes-bid 0.38 --yes-ask 0.45 --out runs/cava-q1-2026
```

Expected:

- the CLI exits with status `0`,
- generated JSON files contain no API keys, private key material, or env values,
- `labels.json` reports an exact or lexical hit for `traffic`,
- `paper_comparison.json` reports a model probability, market probability, edge, side, and reason codes.

- [x] **Step 4: Inspect generated artifacts**

Run a small verification script or pytest assertion that loads every generated JSON/JSONL file and confirms:

- required fields are present,
- probabilities are between `0` and `1`,
- market bid is less than or equal to market ask,
- source document hash is present,
- no serialized field name contains `secret`, `private_key`, or raw API key values.

Expected: artifacts are machine-readable and safe to share for review.

---

## Review: Risks, Assumptions, and Real-Trading Gates

**Implementation Review**

- Completed the MVP local workflow: CAVA PDF ingestion, normalized models, mocked vendor/exchange clients, deterministic labels, TF-IDF features, transparent probability scoring, paper comparison, and CLI artifact generation.
- Verification completed with `python -m pytest -q`, `python -m ruff check src tests`, the required `run-local-cava` command, and a generated-artifact safety/shape inspection.
- Minor implementation note: `parse_mention_market_title` returns a small `MentionMarket` wrapper containing both `company_symbol` and `TargetPhrase`, because the approved parser expectations include both fields.

**Risks**

- [ ] The first probability baseline is a research baseline, not a calibrated trading model.
- [ ] The local CAVA press release proves document parsing and feature generation, but it does not prove historical predictive edge.
- [ ] TF-IDF similarity is deterministic and testable, but it may miss true semantic relationships that embeddings could capture later.
- [ ] API response shapes may differ from mocked fixtures; the first live API run needs careful contract verification.
- [ ] Market title parsing can be brittle; expand only from observed real market titles.
- [ ] Exact phrase labeling is suitable for settlement-style checks, but market resolution rules may include venue-specific nuance.
- [ ] Paper EV can overstate opportunity if liquidity, fees, and fill probability are not modeled.

**Assumptions**

- [ ] `Earnings-Release-2026-Q1.pdf` is the intended local CAVA earnings press release.
- [ ] Python is the right first scaffold because this slice is modeling and document-processing centered.
- [ ] A CLI is justified as the smallest useful interface for reproducible local runs.
- [ ] Local-only tests should pass with no `.env` file and no private key access.
- [ ] API Ninjas and Kalshi live calls will be introduced only after mocked client contracts pass.

**Must Confirm Before Real Trading**

- [ ] Settlement criteria for each market are understood and encoded before a prediction is actionable.
- [ ] Historical labels are built with strict evidence cutoff times to avoid leakage.
- [ ] Probability outputs are calibrated on time-based holdouts, not random splits.
- [ ] Paper trading survives realistic spreads, fees, depth, and stale-market checks.
- [ ] Authenticated Kalshi code has been reviewed for key handling and never logs request signatures or private material.
- [ ] The system has explicit risk controls, position limits, and manual approval before any live order placement.

## Self-Review of This Plan

- [ ] The scope is MVP-first: one local CAVA document, mocked API clients, deterministic labeling, baseline features, transparent probability scoring, and paper comparison.
- [ ] Production implementation is gated on user approval.
- [ ] The plan avoids dashboard/database/websocket/LLM/vector-store overreach.
- [ ] Tests precede implementation in every production-code task.
- [ ] Verification commands are listed only after the task scaffolds the relevant tooling.
- [ ] Secret files are acknowledged as present but are not read or printed.

---

## Addendum: CAVA Q1 2026 Transcript Verification

**Goal:** Verify the MVP against the CAVA Q1 2026 earnings call transcript using a repeatable local artifact path, without requiring live API credentials or reading secret files.

**Source plan:** Prefer the existing `ApiNinjasClient.get_transcript()` abstraction for future live transcript ingestion, but use a public local transcript artifact for repeatable verification in this pass.

- [x] Save transcript source artifact to `data/raw/cava-q1-2026-transcript.txt` with provenance metadata.
- [x] Write a failing integration test for a local transcript CLI workflow that emits document/chunk/label/feature/prediction/paper comparison artifacts.
- [x] Implement the smallest local transcript ingestion helper and CLI command needed to pass the test.
- [x] Run transcript-specific pytest.
- [x] Run full `python -m pytest -q`.
- [x] Run `python -m ruff check src tests`.
- [x] Run the transcript CLI smoke command against `data/raw/cava-q1-2026-transcript.txt`.
- [x] Record final source, commands, output paths, phrase labels, and prediction summary here.

**Verification Results**

- Source: Alphastreet, `CAVA Group Inc (CAVA) Q1 2026 Earnings Call Transcript`, `https://news.alphastreet.com/cava-group-inc-cava-q1-2026-earnings-call-transcript/`, fetched 2026-05-19.
- Local transcript artifact: `data/raw/cava-q1-2026-transcript.txt`.
- Generated run artifacts: `runs/cava-q1-2026-transcript/document.json`, `chunks.jsonl`, `labels.json`, `features.json`, `prediction.json`, and `paper_comparison.json`.
- Red test confirmed before implementation: `python -m pytest tests/integration/test_cli_transcript.py -q` failed with `No such command 'run-local-transcript'`.
- Focused verification passed: `python -m pytest tests/integration/test_cli_transcript.py -q` -> `1 passed`.
- Smoke verification passed: `python -m kalorie.cli run-local-transcript --transcript data/raw/cava-q1-2026-transcript.txt --market-title "Will CAVA mention traffic during earnings?" --yes-bid 0.38 --yes-ask 0.45 --out runs/cava-q1-2026-transcript`.
- Full verification passed: `python -m pytest -q` -> `45 passed`.
- Lint verification passed after splitting one long transcript `source_id` line: `python -m ruff check src tests` -> `All checks passed!`.
- Phrase labels from `labels.json`: `traffic=true`, `same restaurant sales=true`, `digital revenue=false`, `geopolitical uncertainty=true`, `value proposition=true`, `margin=true`.
- `traffic` prediction from `prediction.json`: model `rule-based-v0`, probability `0.615266`, reasons `base_rate`, `exact_match`, `tfidf_similarity`.
- `traffic` paper comparison from `paper_comparison.json`: side `yes`, model probability `0.62`, market probability `0.45`, edge `0.17`, reason `yes_edge`.

## Addendum: OpenAI Config and Metrics Verification

**Goal:** Accept `OPENAI_API_KEY` safely for future optional embedding work, keep the local MVP pipeline independent of live OpenAI calls, and output Brier score for the transcript run.

- [x] Add `OPENAI_API_KEY` support to `Settings` without exposing the value in `model_dump()` or `redacted_dict()`.
- [x] Add `.env.example` placeholder `OPENAI_API_KEY=` without copying any real value.
- [x] Add optional embedding provider boundaries: `OpenAIEmbeddingProvider` for future live/manual use and `FakeEmbeddingProvider` for deterministic tests.
- [x] Keep transcript tests and local pipeline free of required live OpenAI/network calls.
- [x] Add binary evaluation reporting with explicit `brier_score`.
- [x] Add `evaluate-run` CLI command for run-artifact smoke metrics.
- [x] Run focused OpenAI/config/metrics tests.
- [x] Run transcript smoke command.
- [x] Run metrics smoke command.
- [x] Run full `python -m pytest -q`.
- [x] Run `python -m ruff check src tests`.

**Metrics Results**

- Metrics command: `python -m kalorie.cli evaluate-run --run runs/cava-q1-2026-transcript --out runs/cava-q1-2026-transcript/metrics.json`.
- Metrics artifact: `runs/cava-q1-2026-transcript/metrics.json`.
- Brier score: `0.131603`.
- Sample count: `6` phrase outcomes from the CAVA transcript phrase set.
- Training status: `trained_model=false`; this is smoke-only evaluation over one transcript run, not statistically meaningful training performance.
- Note: for binary probability labels, Brier score is already the squared probability-error metric, so MSE is not reported separately.

**Verification Results**

- Focused tests passed: `python -m pytest tests/unit/test_config.py tests/unit/test_embeddings.py tests/unit/test_evaluation.py tests/integration/test_cli_metrics.py -q` -> `8 passed`.
- Transcript smoke passed: `python -m kalorie.cli run-local-transcript --transcript data/raw/cava-q1-2026-transcript.txt --market-title "Will CAVA mention traffic during earnings?" --yes-bid 0.38 --yes-ask 0.45 --out runs/cava-q1-2026-transcript`.
- Metrics smoke passed and printed Brier score `0.131603`.
- Full tests passed: `python -m pytest -q` -> `49 passed`.
- Ruff passed: `python -m ruff check src tests` -> `All checks passed!`.

## Addendum: Browser and Kalshi Historical Seed Data

**Goal:** Add real seed data beyond the original local CAVA PDF by using Browser-discovered public earnings release URLs and Kalshi public historical market metadata.

- [x] Use Browser search to discover official/public earnings release sources.
- [x] Collect NVIDIA Fiscal Q1 2026 earnings release into `data/raw/` and `data/manifests/`.
- [x] Collect NVIDIA Fiscal Q2 2026 earnings release into `data/raw/` and `data/manifests/`.
- [x] Collect NVIDIA Fiscal Q3 2026 earnings release into `data/raw/` and `data/manifests/`.
- [x] Probe CAVA and Tesla official IR sources; record that they block direct HTTP collection with `403`.
- [x] Add Kalshi historical market metadata collection from the public `/markets` endpoint with `status=closed`.
- [x] Save CAVA closed-market raw data and parsed mention contracts under `data/kalshi/`.
- [x] Save NVIDIA closed-market raw data and parsed mention contracts under `data/kalshi/`.
- [x] Remove duplicate MSE reporting and keep Brier score as the primary probability metric.

### Artifacts (This Run)

- `data/manifests/nvda-2026-q1-press-release.json`
- `data/manifests/nvda-2026-q2-press-release.json`
- `data/manifests/nvda-2026-q3-press-release.json`
- `data/kalshi/cava-closed-markets-raw.json`
- `data/kalshi/cava-closed-mention-markets.json`
- `data/kalshi/nvda-closed-markets-raw.json`
- `data/kalshi/nvda-closed-mention-markets.json`

## Addendum: 2,500-Sample Real Training Data Expansion

**Goal:** Build a larger real-data training artifact with at least 2,500 full examples, where each example is backed by a local transcript, a SEC EX-99.1 press release, and a Kalshi mention-market target/price snapshot.

- [x] Add SEC mapping API support for `https://api.sec-api.io/mapping/{resolveBy}/{value}` so transcript company folder names can resolve to CIKs.
- [x] Cache resolved company CIKs in `data/sec/company_to_cik.json` without storing API keys or request secrets.
- [x] Collect SEC EX-99.1 press releases across mapped transcript companies, using compliant SEC user-agent headers and provenance manifests.
- [x] Merge available Kalshi mention-market contracts from CAVA and NVIDIA historical files into the target set.
- [x] Build at least 2,500 examples by pairing matched company-quarter transcripts and press releases with all available mention-market contracts.
- [x] Train the base historical model on the 2,500-example artifact and report Brier score only as the primary metric.
- [x] Verify with focused tests, full pytest, Ruff, and artifact sanity checks.

### Results (This Run)

- SEC collection artifact: `data/manifests/sec-corpus-ex99-1.json` with 350 press release manifests across 18 companies.
- Kalshi target artifact: `data/kalshi/combined-closed-mention-markets.json` with 13 unique mention-market contracts.
- Quarantined prototype training artifact: `data/datasets/quarantine/prototype-invalid/training/historical/sec-corpus-2500-examples.json` with 2,500 examples across 10 companies and 193 company-quarter periods.
- Quarantined prototype evaluation artifact: `data/datasets/quarantine/prototype-invalid/testing/historical/sec-corpus-2500-eval.json`.
- Global Brier score: `0.169838`.
- Company-adapted Brier score: `0.171226`.
- Verification: `python -m ruff check src tests` and `python -m pytest -q` passed with 71 tests.

## Addendum: Filing-Plus-Five Market Snapshot Correction

**Goal:** Correct market probability timing from near market close to five minutes after SEC supplemental material filing, and collect all `EX-99.*` supplemental exhibits instead of only `EX-99.1`.

- [x] Update SEC parsing to retain every `EX-99.*` exhibit attached to a filing.
- [x] Update corpus SEC collection to store each supplemental exhibit as its own manifest/raw artifact.
- [x] Add Kalshi candlestick client support for filing-time market snapshots.
- [x] Collect CAVA Q1 2026 SEC supplemental material and use its filing time as the anchor.
- [x] Write Kalshi filing-plus-five-minute snapshot artifact for the current mention contracts.
- [x] Rebuild the 2,500-row dataset with expanded `EX-99.*` supplemental material.
- [x] Retrain and write an updated forward-pass file with filing-plus-five-minute Kalshi probability fields.
- [x] Verify with focused tests, full pytest, Ruff, and artifact sanity checks.

**Results**

- Supplemental corpus manifests: `data/manifests/sec-corpus-ex99-supplemental.json` with 456 manifests across `sec_ex_99_1_supplemental`, `sec_ex_99_2_supplemental`, and `sec_ex_99_3_supplemental`; each regenerated manifest preserves `raw_original_path` and, where text extraction applies, `extracted_text_path`.
- CAVA supplemental manifest: `data/manifests/cava-sec-ex99-supplemental.json`.
- Kalshi filing-plus-five snapshot artifact: `data/kalshi/combined-closed-mention-market-filing-plus-5m-snapshots.json`, anchored at `2026-05-19T16:22:46-04:00`.
- Quarantined updated training artifact: `data/datasets/quarantine/prototype-invalid/training/historical/sec-corpus-ex99-supplemental-2500-examples.json`.
- Quarantined updated evaluation artifact: `data/datasets/quarantine/prototype-invalid/testing/historical/sec-corpus-ex99-supplemental-2500-eval.json`.
- Quarantined updated forward-pass artifact: `data/datasets/quarantine/prototype-invalid/testing/historical/sec-corpus-ex99-supplemental-2500-test-forward-passes-with-kalshi-filing-plus-5m.json`.
- Global Brier score after regenerating with original/extracted artifact preservation: `0.166589`.
- Verification: `python -m ruff check src tests` and `python -m pytest -q` passed with 74 tests.

## Addendum: Dataset Layout and Event-Aligned Repair Plan

**Goal:** Separate generated training/testing artifacts from raw SEC/Kalshi/transcript sources, then replace the current prototype corpus assembly with an event-aligned dataset that can support real market comparison.

### Verified Current State

- [x] Current tests pass: `python -m pytest -q` -> `74 passed`.
- [x] Current lint passes: `python -m ruff check src tests` -> `All checks passed!`.
- [x] SEC supplemental manifest has 456 rows across 18 companies.
- [x] Stored SEC originals are preserved: 449 `.htm` originals and 7 `.txt` originals; every manifest path exists.
- [x] The latest example artifact has 2,500 rows across 10 companies, 13 target phrases, and labels `618` positive / `1,882` negative.
- [x] The latest reported global Brier score is `0.166589`, but it is only a pipeline smoke metric because the examples are not event-market aligned.
- [x] All 2,500 current examples use CAVA Kalshi market IDs, including non-CAVA company rows.
- [x] ACN fiscal period assignment is broken in the manifest: for example, ACN 2024 Q2 points to a 2015 SEC exhibit, confirming the zip-by-recency matching issue.

### Dataset Layout Completed

- [x] Move training example artifacts to `data/datasets/training/historical/`.
- [x] Move eval, forward-pass, and readable result artifacts to `data/datasets/testing/historical/`.
- [x] Update embedded artifact references from `data/training/...` to the new structured paths.
- [x] Remove the old `data/training/` folder after moving all files.
- [x] Quarantine the known-invalid prototype datasets under `data/datasets/quarantine/prototype-invalid/` so they are audit artifacts, not active training/testing inputs.

### Clean Benchmark Repair Plan

- [ ] **Task 1: Add a canonical event table**
  - Create a model with `company_symbol`, `company_name`, `cik`, `fiscal_year`, `fiscal_quarter`, `call_date`, and `transcript_path`.
  - Build it from the transcript corpus and cached SEC company map.
  - Persist it at `data/datasets/events/earnings-events.json`.
  - Test that every event has one transcript path, a resolved CIK when available, and a timezone-aware call date.

- [ ] **Task 2: Replace zip-by-recency SEC matching**
  - Stop assigning SEC filings to transcript periods by sorted order.
  - Match candidate `8-K` / `EX-99.*` filings to events by CIK and date proximity to `call_date`.
  - Enforce a maximum date window and record unmatched events instead of fabricating pairs.
  - Test the ACN regression so ACN 2024 Q2 can no longer match a 2015 exhibit.

- [ ] **Task 3: Preserve raw SEC originals and parse into derived text**
  - Treat `.htm`, `.html`, `.pdf`, and raw `.txt` as immutable source artifacts under `data/sec/`.
  - Store extracted text separately and keep `raw_original_path`, `raw_original_content_hash`, `extracted_text_path`, and `extraction_method`.
  - Add PDF support with a parser boundary so `docling` can be used for PDF text extraction without changing the dataset builder.
  - Detect image-heavy PDFs and mark extraction quality explicitly; do not silently train on empty or low-quality parsed text.

- [ ] **Task 4: Build real company/event market mapping**
  - Parse Kalshi event/market metadata into `company_symbol`, event date, target phrase, market ID, and settlement rules.
  - Only create examples where the Kalshi market company/event matches the earnings event company/date.
  - Keep CAVA markets scoped to CAVA events; do not reuse CAVA target contracts across AMZN, BAC, ACN, or other companies.
  - Test that a non-CAVA event cannot receive a `KXEARNINGSMENTIONCAVA-*` market ID.

- [ ] **Task 5: Use event-specific filing-plus-one-minute odds**
  - Anchor each market snapshot to that event's matched SEC filing timestamp.
  - Use filing `+1m` as the default comparison snapshot and store missing-stale-wide-spread reasons.
  - Persist market comparison test artifacts under `data/datasets/testing/historical/`.
  - Test that odds from one company/event cannot be joined onto another company/event.

- [ ] **Task 6: Rebuild labels and features from the repaired event dataset**
  - Keep strict phrase-presence transcript labels as settlement-style labels.
  - Build features only from evidence documents available before the cutoff.
  - Drop rows with missing transcript, missing matched SEC evidence, missing matched market, or low-quality parsed source text.
  - Persist clean examples under `data/datasets/training/historical/`.

- [ ] **Task 7: Retrain and report trustworthy smoke metrics**
  - Train only on the repaired event-aligned examples.
  - Use a time split with no cross-event leakage.
  - Report sample count, company count, market count, label balance, Brier score, and explicit warnings if the sample is too small for edge claims.
  - Keep the current prototype Brier score as historical context only, not model evidence.

### Review Notes

- [ ] The current raw ingredients are useful, but the 2,500-row assembled dataset should not be used for trading decisions.
- [x] The first code change added guardrails against CAVA market reuse and far-year SEC exhibit pairing, including a failing-then-passing ACN date-alignment regression test.
- [ ] The next code change should create a persisted event table with externally sourced or otherwise trustworthy call dates.
- [ ] PDF/docling work should focus on preserving raw binaries and extraction-quality metadata before increasing corpus size.

### Implementation Progress

- [x] Added a regression test proving non-CAVA transcript rows are dropped when only CAVA Kalshi contracts are available.
- [x] Added a regression test for short-symbol substring leakage, so ticker `ALL` cannot match text like `What will...`.
- [x] Added a regression test proving ACN 2024 Q2 collection skips a 2015 filing when a 2024 filing is available.
- [x] Updated historical example generation to filter mention-market contracts to the transcript company before labels/features are built.
- [x] Verified that rebuilding with the current CAVA-only Kalshi contract artifact now yields `0` rows instead of fabricated cross-company examples.
- [x] Replaced SEC record/filing `zip()` pairing with conservative filing selection that rejects filings more than one calendar year from the transcript fiscal year.
- [x] Added a training validation gate that rejects obvious company/market mismatches before metrics are computed.
- [x] Quarantined the existing generated example/eval/forward-pass artifacts because they predate the guardrails and remain known-invalid.
- [x] Added a synthetic phrase-presence dataset builder so transcript/material training no longer depends on real Kalshi market availability.
- [x] Rebuilt `data/datasets/training/historical/synthetic-phrase-presence-examples.json` with 507 clean examples across 7 companies and 13 target phrases.
- [x] Persisted `market_venue="synthetic"` on synthetic examples so they cannot be mistaken for real Kalshi market rows.
- [x] Updated time splitting to keep all phrase rows from the same company/fiscal-period event on the same side of the train/test split.
- [x] Wrote `data/datasets/testing/historical/synthetic-phrase-presence-eval.json`; global Brier score is `0.175335` on a 130-row event-grouped time holdout.
- [x] Verified rebuilt rows have no company/market ID mismatches and no evidence cutoff years more than one year from the labeled fiscal year.
- [x] Enforced local-only evidence cutoff `filing_time <= call_time` during example generation using a transcript-mtime/fiscal-window proxy (no extra API calls).
- [x] Added Model 1 implementation as a persisted base-plus-company-finetune predictor (`mention-base-company-v1`) with CLI training/inference commands.
- [ ] Keep `build-real-training-dataset` scoped to market-comparison/prototype work until event-level real Kalshi joins exist; use `build-synthetic-phrase-dataset` for phrase-presence training.
- [ ] Rebuild a larger 2,500+ row historical dataset after collecting/rematching more event-aligned materials; the old 2,500-row dataset remains quarantined as prototype-only.
- [x] Added LLM template-phrase catalog generation and template embedding similarity features, wired as optional inputs to synthetic/real training dataset builders.
- [x] Organized `src/kalorie/` with helper subpackages for `data_grepping`, `data_cleaning`, `model_eval`, `model_test`, and `kalshi_pull` workflows.

## Addendum: Model1 Brier Optimization Loop (2026-05-20)

**Goal:** Optimize for lower out-of-sample Brier score and better company-specific calibration (especially CAVA-style obscure phrase markets), while reducing end-to-end training runtime.

### Plan (Kalshi Phrase Expansion)

- [x] **Track A: Data/feature pipeline speed**
  - Add transcript/evidence chunk caching during dataset builds so files are not re-read for each phrase/record path.
  - Remove redundant per-target TF-IDF re-fit overhead in feature extraction by sharing chunk vectorizer work.
  - Verify runtime reduction by re-running synthetic dataset build and recording elapsed time against current baseline (~288s with templates).

- [x] **Track B: Company/market phrase coverage**
  - Add optional market-contract phrase ingestion to synthetic dataset building so company-specific terms (e.g., `auv`, `sweet potato`, `glazed salmon`) can be included automatically.
  - Add optional company-scoped material snippet loading for template generation so phrase variants are grounded in target-company docs.
  - Verify generated dataset includes CAVA-style market targets and no duplicate phrase normalization regressions.

- [x] **Track C: Model1 probability quality (Brier-first)**
  - Add optimized model1 training with hyperparameter search and probability calibration (time-aware, no leakage).
  - Add model1 time-split evaluation command/report that measures both global and company-adapted Brier/log-loss on held-out events.
  - Tune on synthetic+template dataset and report deltas versus current benchmark (`global_brier_score ~0.171`).
  - Evaluate tuned model on company-specific holdout slices (including available CAVA clean examples) and record where calibration still fails.

### Review Notes (to fill after implementation)

- [x] Final baseline vs optimized Brier comparison
  - Historical logistic evaluator baseline (`historical-eval-with-templates.json`): global Brier `0.172079`.
  - Baseline model1-with-templates eval (`model1-baseline-with-templates-eval.json`): Brier `0.169467`.
  - Optimized model1 search after adding target-indicator features (`model1-optimization-report.json`): best holdout Brier `0.109462` with isotonic calibration (`C=10.0`, `min_company_rows=8`, `blend_weight=0.7`, `class_weight_balanced=True`, `include_target_indicator=True`).
  - Final tuned model eval on synthetic template dataset (`model1-final-synthetic-eval.json`): Brier `0.077105`.

- [x] Runtime benchmark before/after
  - Prior synthetic+template build runtime (recorded run): ~`288s`.
  - Current synthetic+template build runtime after TF-IDF sharing/caching updates: ~`249s`.
  - Net improvement: ~`39s` faster (`~13.5%` reduction) on the same 2,340-row build.

- [x] Data sufficiency verdict for real deployment
  - Synthetic multi-company training set is sufficient for base phrase-presence calibration improvements.
  - Real company-specific fine-tuning data is insufficient: only `13` clean CAVA rows are available, so CAVA behavior is still high-variance.
  - Out-of-domain base optimized model on CAVA slice is poor (`0.549659` Brier); direct in-sample CAVA fine-tune can look extremely strong (`0.039590`), but honest leave-one-out CAVA evaluation is `0.236630` (`model1-final-cava-loo-eval.json`), confirming company-specific data is still the limiting factor.

- [x] Recommended next data acquisitions (transcripts, docs, alternative context feeds)
  - Add historical transcripts for CAVA and other target names (at least 8-12 prior calls per company) in corpus layout.
  - Expand per-company pre-call materials (SEC exhibits + IR supplemental decks + shareholder letters) with strict pre-call cutoff timestamps.
  - Build a broader real mention-market set beyond a single CAVA event and link each market to event-level filing+time snapshots.
  - Add external context only after transcript/material coverage is fixed: selective earnings-preview news and analyst Q&A priors by topic.

## Addendum: Kalshi-Style Phrase Expansion + Base Ablation Harness (2026-05-20)

**Goal:** Train on less business-generic target phrases (including Kalshi-style terms), add a reusable Kalshi earnings-market interaction class, and ship a repeatable base-train ablation harness that quantifies Brier impact across feature/data variants.

### Plan (Kalshi-Style Expansion)

- [x] Expand synthetic/base target phrase handling to include Kalshi-style terms and optional global phrase expansion from historical market contracts.
- [x] Add a dedicated Kalshi interactions class for earnings market discovery by company/event with pagination and robust filtering.
- [x] Add CLI command(s) for easier earnings market discovery using that class.
- [x] Add a base ablation harness command that builds variant datasets and runs optimized base-model training/evaluation for each variant.
- [x] Add focused unit/integration tests for new phrase expansion behavior, Kalshi interactions class, and ablation harness output shape.
- [x] Run targeted pytest suite + lint and record results.

### Review (to fill after implementation)

- [x] Phrase expansion behavior verified on synthetic dataset build (`synthetic-plus-wmt-everything-kalshi-terms.json`) with 27 WMT targets including `openai`, `omnichannel`, and `salmon`.
- [x] Kalshi earnings discovery class validated with mocked API responses (`tests/unit/test_kalshi_earnings.py`).
- [x] Ablation harness writes per-variant artifacts and sortable Brier summary (`artifacts/model1/ablations/base-2026-05-20/eval/base-ablation-summary.json`).
- [x] Recommendation updated on news/reliability in base training: keep optional by default until broader historical news coverage exists; market-phrase expansion is clearly worthwhile.

## Addendum: DefeatBeta Pre-Earnings Week Workflow for Walmart (2026-05-20)

**Goal:** Build a DefeatBeta-based historical news ingestion path (with yfinance fallback), then train/evaluate Walmart using week-before-earnings windows.

### Plan (DefeatBeta)

- [x] Add a dedicated DefeatBeta client path aligned with the Python library data model (`stock_news` parquet schema) and typed parse/runtime errors.
- [x] Add a CLI command that maps transcript fiscal periods to estimated call dates and fetches pre-earnings-week news per event.
- [x] Add yfinance fallback within that command for cases where DefeatBeta credentials or endpoint access are unavailable.
- [x] Add focused unit/integration tests for client behavior and CLI manifest generation.
- [x] Run the workflow on Walmart and train/evaluate a company model on the resulting dataset slice.

### Review (DefeatBeta)

- [x] New command added: `collect-defeatbeta-pre-earnings-week-articles`.
- [x] Tests passed: `test_defeatbeta.py` and CLI integration coverage in `test_cli_historical_training.py`.
- [x] User correction applied: DefeatBeta is treated as a Python-library-backed dataset workflow rather than only direct host calls.
- [x] Live Walmart run completed with library-backed ingestion and produced `100` pre-earnings-week news manifests (`data/manifests/defeatbeta-wmt-preweek.json`).
- [x] Performance fix: DefeatBeta collection now does one full-range pull per symbol and slices per-quarter windows locally (WMT runtime improved from ~`368s` to ~`21s` on the same command).
- [x] Trained/evaluated using merged SEC + DefeatBeta-preweek manifest set:
  - Dataset: `artifacts/model1/datasets/synthetic-wmt-defeatbeta-preweek.json` (243 rows)
  - Model: `artifacts/model1/models/model1-company-wmt-defeatbeta-preweek-ema.json`
  - Eval: `artifacts/model1/eval/model1-company-wmt-defeatbeta-preweek-ema-brier.json`
  - Holdout Brier: `0.093486` (worse than prior best `0.065403`; in-sample Brier improved to `0.047257`, indicating current preweek news may be adding variance rather than holdout signal)

## Webapp Implementation Plan (2026-05-20)

- [x] Task 1: Backend run-store skeleton (event/company cache layout)
- [x] Task 2: Job registry + idempotency + resource budgets
- [x] Task 3: Kalshi market service normalization for web API
- [x] Task 4: FastAPI routes + websocket job stream
- [x] Task 5: Cache-aware job execution with explicit decision cutoff
- [x] Task 6: React + Tailwind + shadcn scaffold
- [x] Task 7: Market UI, run selector, uploads, background rail, toasts
- [x] Task 8: End-to-end verification (tests, smoke flow, lint)

## Addendum: Mention Model Quality Upgrades (2026-05-20)

**Goal:** Improve mention-model robustness without changing web API contracts by adding pre-train gating, exact-vs-semantic signal splitting, hard-negative mining features, and stronger event-time leakage guardrails.

### Plan (Model Quality Upgrades)

- [x] Add pre-train gate checks to model training commands with enforce/fail behavior and diagnostics output.
- [x] Split exact vs semantic evidence into explicit feature channels and add alignment/gap features.
- [x] Add hard-negative mining features based on similar neighbor phrases mentioned in evidence.
- [x] Tighten event-time leakage guardrails in dataset/example construction logic.
- [x] Add/adjust unit and integration tests for new feature keys, guardrails, and pre-train gating behavior.
- [x] Run focused pytest suite and verify lint clean.

### Review (Model Quality Upgrades)

- [x] Added optional strict pre-train gating to `train-model1`, `train-model1-optimized`, and `train-model1-company` with JSON diagnostics output (`--pretrain-gate-report-out`) and hard fail mode (`--enforce-pretrain-gate`).
- [x] Added explicit exact-vs-semantic feature channels (`exact_signal_binary`, `semantic_signal_*`, `semantic_exact_gap`) while retaining existing feature keys for compatibility.
- [x] Added hard-negative mining features (`hard_negative_neighbor_*`) based on TF-IDF similarity among target phrases when similar neighbors are mentioned but the target is not.
- [x] Tightened event-time leakage guardrail by clamping transcript-derived call-time proxy to `[fiscal_period_end, fiscal_period_end + 120 days]`.
- [x] Applied lightweight `<entity>` handling for settlement labels by filtering analyst/question context in transcript labeling while keeping operator/company speech eligible.
- [x] Verification:
  - Focused tests passed: `python -m pytest tests/unit/test_labeling.py tests/unit/test_features.py tests/unit/test_real_training_data.py tests/integration/test_cli_historical_training.py -q` (`38 passed`).
  - IDE lints clean on changed files.
  - Full suite currently has two environment-dependent failures due missing `Earnings-Release-2026-Q1.pdf` in local workspace; unrelated to this change set.

## Addendum: ECE + News Coverage + Closed-Market Checks (2026-05-21)

**Goal:** Add calibration-error reporting, fix NVDA news coverage in training rows, and run additional closed-market tests beyond NVDA with explicit sufficiency checks.

### Plan

- [x] Add ECE output to evaluation artifacts and CLI summaries.
- [x] Repair NVDA news coverage by collecting historical pre-earnings-week news across transcript periods.
- [x] Rebuild/retrain NVDA and rerun closed-event evaluation with Brier + ECE.
- [x] Run additional closed-market evaluation for CAVA and document data sufficiency constraints.
- [x] Check discovery coverage for other symbols and report resource/rate-limit gaps.

### Data Hunt Review

- [x] Added `expected_calibration_error` to `EvaluationReport` (`src/kalorie/ml/evaluation.py`) and surfaced ECE in `evaluate-run` + `evaluate-model1` CLI outputs.
- [x] Verification passed:
  - `python -m pytest tests/unit/test_evaluation.py tests/integration/test_cli_metrics.py tests/integration/test_cli_historical_training.py -q` (`21 passed`).
- [x] NVDA news coverage fixed for training rows:
  - Before: `synthetic-plus-nvda-news-finetune-v2-20260521.json` -> `news_rows=0/429`.
  - After DefeatBeta preweek integration: `synthetic-plus-nvda-defeatbeta-finetune-v2-20260521.json` -> `news_rows=66/429` (`mean_news_ratio=0.147091`).
- [x] NVDA closed-event rerun (`KXEARNINGSMENTIONNVDA-26MAY20`) with news-covered model:
  - `model_company_brier_score=0.177935`, `model_company_ece=0.248408`
  - `kalshi_yes_ask_brier_score=0.258635`, `kalshi_yes_ask_ece=0.351765`
  - `kalshi_yes_mid_brier_score=0.250344`, `kalshi_yes_mid_ece=0.384706`
  - Artifact: `artifacts/model1/predictions/nvda-26may20-market-comparison-defeatbeta-v2-20260521.json`.
- [x] Additional closed-market check on CAVA (`KXEARNINGSMENTIONCAVA-26MAY19`):
  - `model_brier_score=0.614645`, `model_ece=0.729624`
  - `kalshi_yes_ask_brier_score=1.000000`, `kalshi_yes_mid_brier_score=0.250000`
  - Artifact: `artifacts/model1/predictions/cava-26may19-market-comparison-v2-20260521.json`.
- [x] Sufficiency findings:
  - CAVA retraining gate fails due sparse/one-class history (`event_count=3`, `positive_rate=0.0`).
  - Closed-market discovery for `WMT`, `AAPL`, and `NVDA` currently returns `market_count=0` in live pulls; some alternate symbol pulls hit Kalshi `429` rate limits.

## Addendum: Base Model Retrain with 5-Event Pack (2026-05-21)

**Goal:** Retrain base models after adding the newly cached five-event closed-market pack, then verify whether base Brier/ECE improve on that same pack.

- [x] Build merged base dataset (`existing_base + 5-event pack`) and verify row counts.
- [x] Train `train-model1` baseline artifact with pre-train gate report.
- [x] Train `train-model1-optimized` artifact + optimization report with pre-train gate report.
- [x] Evaluate base artifacts on `historical-examples-5event-pack.json` and summarize Brier/ECE deltas versus existing `model1-optimized.json`.
- [x] Record resulting artifact paths + conclusions in this section.

### Artifacts (5-Event In-Sample)

- Merged dataset: `artifacts/model1/datasets/base-plus-5event-pack-20260521.json` (`2417` rows = `2340` base + `77` event-pack rows).
- Merge stats: `artifacts/model1/datasets/base-plus-5event-pack-20260521-stats.json`.
- Baseline retrained model: `artifacts/model1/models/model1-base-plus-5event-pack-20260521.json`.
- Baseline pre-train gate report: `artifacts/model1/eval/model1-base-plus-5event-pack-pretrain-gate.json`.
- Optimized retrained model: `artifacts/model1/models/model1-optimized-plus-5event-pack-20260521.json`.
- Optimizer report: `artifacts/model1/eval/model1-optimized-plus-5event-pack-20260521-report.json`.
- Optimized pre-train gate report: `artifacts/model1/eval/model1-optimized-plus-5event-pack-pretrain-gate.json`.
- 5-event evals:
  - Legacy model: `artifacts/model1/eval/model1-optimized-legacy-on-5event-pack-20260521.json`
  - New baseline: `artifacts/model1/eval/model1-base-plus-5event-pack-on-5event-pack-20260521.json`
  - New optimized: `artifacts/model1/eval/model1-optimized-plus-5event-pack-on-5event-pack-20260521.json`
- Event-wise comparison: `artifacts/model1/eval/model1-5event-pack-per-event-comparison-20260521.json`.

### Results (5-Event In-Sample)

- On the 5-event pack (`77` contracts):
  - Legacy optimized (`model1-optimized.json`): Brier `0.399258`, ECE `0.373491`.
  - New baseline (`model1-base-plus-5event-pack-20260521.json`): Brier `0.123069`, ECE `0.065290`.
  - New optimized (`model1-optimized-plus-5event-pack-20260521.json`): Brier `0.150733`, ECE `0.095048`.
- Optimizer holdout on merged dataset: best holdout Brier `0.141670`, log loss `0.462932`; best params `C=0.3`, `min_company_rows=8`, `blend_weight=0.5`, `class_weight_balanced=False`, `target_indicator=True`, isotonic enabled.
- Legacy-vs-new behavior on original base dataset (`synthetic-phrase-presence-2kplus-with-templates.json`):
  - Legacy optimized: Brier `0.074579`.
  - New baseline: Brier `0.111785`.
  - New optimized: Brier `0.112946`.

### Important caveat

- Pre-train gate failed (not enforced for this run) due legacy rows in the old base dataset:
  - `leakage_violations=2280`
  - exact/semantic/hard-negative feature coverage each `~0.0319`
- Interpretation: the newly trained artifacts improve markedly on the newly added 5-event pack but degrade on the older base distribution, indicating distribution shift and stale-feature mismatch in the merged training corpus.

## Addendum: 2h Pre-Call Cutoff + Feature-Coverage Repair (2026-05-21)

**Goal:** Enforce `evidence_cutoff = call_start_proxy - 2h` and rebuild training rows with the current feature schema so pre-train gate metrics are meaningful.

- [x] Updated dataset builders to filter evidence documents at `call_time_proxy - 2h` and persist that fixed cutoff (instead of using max evidence timestamp).
- [x] Added regression test to lock this behavior: `test_build_synthetic_phrase_examples_uses_two_hour_pre_call_cutoff`.
- [x] Rebuilt base synthetic dataset with the new cutoff and current features:
  - `artifacts/model1/datasets/synthetic-phrase-presence-2kplus-with-templates-cutoff2h-20260521.json`
- [x] Merged repaired base rows with 5-event pack rows:
  - `artifacts/model1/datasets/base-cutoff2h-plus-5event-pack-20260521.json`
  - `artifacts/model1/datasets/base-cutoff2h-plus-5event-pack-20260521-stats.json`
- [x] Trained gate-clean models with enforcement enabled:
  - `artifacts/model1/models/model1-base-cutoff2h-plus-5event-pack-20260521.json`
  - `artifacts/model1/models/model1-optimized-cutoff2h-plus-5event-pack-20260521.json`
- [x] Verified gate passes on merged repaired dataset:
  - `sample_count=137`, `event_count=6`, `leakage_violations=0`
  - `exact/semantic/hard_negative coverage = 1.0`
  - report: `artifacts/model1/eval/model1-base-cutoff2h-plus-5event-pack-pretrain-gate.json`

### Results (2h cutoff run)

- Five-event evaluation (`historical-examples-5event-pack.json`, 77 rows):
  - Legacy optimized baseline: Brier `0.399258`
  - New base (2h cutoff, gate-clean): Brier `0.111456`
  - New optimized (2h cutoff, gate-clean): Brier `0.166489`
- Note: strict cutoff plus current manifest alignment reduced old synthetic base rows from `2340` to `60` (mostly one event), so this run is leakage-clean but data-sparse.

## Addendum: Event Scenario Model + Kalshi Benchmark Target (2026-05-21)

**Goal:** Restore a large leakage-clean synthetic phrase dataset, add event-level scenario features, and benchmark model probabilities against real Kalshi bid/ask snapshots captured at `call_start - 10 minutes` where real market data exists.

### Decisions

- [x] Most training rows do not need a corresponding Kalshi market. The model may create synthetic candidate phrase markets that ask whether eligible company-side transcript text contains the target word or phrase under `rules_earningsmentions.pdf`, including plurals and permitted variants.
- [x] Real Kalshi markets are primarily for real-world benchmark/effect checks, including NVDA 2026-05-20 and a small set of other closed earnings mention markets.
- [x] Evidence cutoff is `call_start - 10 minutes` for time-sensitive evidence.
- [x] Event call-time hierarchy: explicit event/call timestamp, then Kalshi close/snapshot target metadata, then transcript mtime proxy, then fiscal-period proxy; persist `call_time_source`.
- [x] Event identity: use Kalshi `event_ticker` when available, otherwise `company_symbol + fiscal_year + fiscal_quarter`.
- [x] Scenario catalogs may model broad call dynamics, including likely analyst questions and management answers, but settlement labels remain company-speaker-only.
- [x] Use existing environment credentials on a bounded best-effort basis; do not print secrets. If live calls are unavailable, skip live artifact generation and document the missing prerequisite.
- [x] If exact T-10 Kalshi quotes are unavailable locally, try to hydrate historical quotes if supported; otherwise use the nearest available snapshot before T-10 and record the time delta.
- [x] Generate a large synthetic research phrase set and add diagnostics so low-quality candidates can be filtered.
- [x] Add optional `xgboost` only as a benchmark dependency; keep logistic regression as production baseline unless event-grouped Brier/ECE clearly justify a later switch.

### Implementation Tracking

- [ ] Split `event_baseline` evidence from `time_sensitive` evidence and update cutoff/provenance tests.
- [ ] Rebuild event-aligned evidence joins with coverage diagnostics and `call_time_source` counts.
- [ ] Add rules-aware large synthetic candidate phrase generation and phrase provenance.
- [ ] Add event-level LLM scenario/Q&A catalog generation from pre-call materials only.
- [ ] Add scenario semantic features and wire matching catalog generation/loading into prediction jobs.
- [ ] Train logistic baseline and optional XGBoost benchmark on event-grouped holdouts.
- [ ] Benchmark on NVDA 2026-05-20 plus other available Kalshi market events with T-10 bid/ask/mid Brier and ECE.
- [ ] Report model-minus-Kalshi Brier/ECE deltas; target model Brier at least `0.02` below Kalshi and aim below the research-paper reference of about `0.14`.
- [ ] Document artifact paths, missing-data gaps, and whether synthetic-row validation agrees with real Kalshi benchmark rows.

### Results

- [x] Split `event_baseline` evidence from `time_sensitive` evidence and added document-role provenance on training examples.
- [x] Added explicit call-start metadata support on transcript records so future event remaps can avoid transcript-mtime leakage.
- [x] Added large evidence-derived synthetic phrase candidates to `build-synthetic-phrase-dataset` via `--include-evidence-phrase-candidates`.
- [x] Added event-level scenario catalog generation and optional scenario embedding features.
- [x] Added `evaluate-model1-kalshi-benchmark` for model-vs-Kalshi yes-bid/yes-ask/yes-mid Brier and ECE reports.
- [x] Added optional `xgboost` to the dev dependency group for benchmark-only work.
- [x] Unit verification: `python -m pytest tests/unit -q` passed (`143` tests).
- [x] Focused lint verification passed on touched non-CLI files; full `src/kalorie/app/cli.py` still has pre-existing line-length lint debt outside this change.
- [x] Local smoke rebuild:
  - Dataset: `artifacts/model1/datasets/smoke-synthetic-nvda-evidence-candidates-20260521.json`.
  - Gate report: `artifacts/model1/eval/smoke-model1-nvda-evidence-candidates-gate-20260521.json`.
  - Gate result: PASS, `sample_count=520`, `event_count=13`, `leakage_violations=0`, feature coverage `1.0`, positive rate `0.694231`.
  - Model: `artifacts/model1/models/smoke-model1-nvda-evidence-candidates-20260521.json`.
- [x] Cached 5-event Kalshi benchmark smoke:
  - Combined snapshots: `artifacts/model1/event-pack-20260521/preclose_snapshots-combined.json`.
  - Report: `artifacts/model1/eval/model1-base-cutoff2h-vs-kalshi-preclose-20260521.json`.
  - Table: `artifacts/model1/eval/model1-base-cutoff2h-vs-kalshi-preclose-20260521.table.md`.
  - Detail log: `artifacts/model1/eval/model1-base-cutoff2h-vs-kalshi-preclose-20260521.log`.
  - Existing base model Brier `0.111456` versus Kalshi mid Brier `0.197245` on the cached benchmark rows.

### Remaining Gaps

- [ ] Exact `call_start - 10 minutes` quote snapshots still need hydration for NVDA 2026-05-20 and other real Kalshi benchmark events; the cached 5-event pack uses pre-close snapshots and records snapshot times.
- [ ] Scenario catalogs were implemented and tested but not generated live in this run.
- [ ] XGBoost is available as an optional dependency but no benchmark report was run in this smoke pass.

## Addendum: Clean 5-Event Holdout Benchmark (2026-05-21)

**Goal:** Retrain without the five cached Kalshi benchmark events, then evaluate only on those held-out real Kalshi rows to remove the in-sample contamination from the earlier benchmark.

### Clean Holdout Plan

- [x] Identify benchmark event keys from `artifacts/model1/event-pack-20260521/historical-examples-5event-pack.json`.
- [x] Build a clean training artifact from gate-clean non-overlapping rows only.
- [x] Train a baseline clean model with the pre-train gate enforced.
- [x] Train an optimized clean model with the pre-train gate enforced.
- [x] Evaluate both models on the unchanged five-event Kalshi benchmark pack and regenerate `.json`, `.table.md`, and `.log` outputs.

### Clean Holdout Artifacts

- Clean training dataset: `artifacts/model1/datasets/clean-no-5event-benchmark-training-20260521.json`.
- Baseline clean model: `artifacts/model1/models/model1-clean-no-5event-benchmark-20260521.json`.
- Baseline gate report: `artifacts/model1/eval/model1-clean-no-5event-benchmark-pretrain-gate-20260521.json`.
- Baseline held-out report: `artifacts/model1/eval/model1-clean-no-5event-vs-kalshi-preclose-20260521.json`.
- Baseline held-out table: `artifacts/model1/eval/model1-clean-no-5event-vs-kalshi-preclose-20260521.table.md`.
- Baseline held-out log: `artifacts/model1/eval/model1-clean-no-5event-vs-kalshi-preclose-20260521.log`.
- Optimized clean model: `artifacts/model1/models/model1-clean-no-5event-benchmark-optimized-20260521.json`.
- Optimizer report: `artifacts/model1/eval/model1-clean-no-5event-benchmark-optimized-training-report-20260521.json`.
- Optimized held-out report: `artifacts/model1/eval/model1-clean-no-5event-optimized-vs-kalshi-preclose-20260521.json`.
- Optimized held-out table: `artifacts/model1/eval/model1-clean-no-5event-optimized-vs-kalshi-preclose-20260521.table.md`.
- Optimized held-out log: `artifacts/model1/eval/model1-clean-no-5event-optimized-vs-kalshi-preclose-20260521.log`.

### Clean Holdout Results

- Clean training set: `580` rows, `14` events, companies `ACN` and `NVDA`, zero overlap with the five benchmark event keys.
- Pre-train gate passed for both clean training runs: `sample_count=580`, `event_count=14`, `leakage_violations=0`, exact/semantic/hard-negative feature coverage `1.0`.
- Held-out five-event benchmark (`77` rows):
  - Baseline clean model: Brier `0.355938`, ECE `0.379973`.
  - Optimized clean model: Brier `0.370241`, ECE `0.414613`.
  - Kalshi mid: Brier `0.197245`, ECE `0.109351`.
- Interpretation: after removing the five benchmark events from training, the current synthetic-only model does not beat Kalshi on the real held-out events. The earlier `~0.11` Brier should be treated as an in-sample smoke result, not a clean estimate of edge.

## Addendum: Clean Benchmark Repair Plan (2026-05-21)

**Goal:** Improve clean held-out Kalshi mention-market performance by repairing the training distribution, adding failure diagnostics, and using conservative event-held-out model selection before trying higher-capacity models.

### Repair Plan

- [x] Add benchmark diagnostics that expose false positives/negatives, probability saturation, phrase categories, wide-spread quote artifacts, and per-row Brier contribution.
- [x] Train and benchmark a conservative global logistic run with `target_indicator=False`, isotonic calibration disabled, and company overrides effectively disabled unless enough real rows exist.
- [x] Rebuild a larger leakage-clean training dataset across benchmark-adjacent companies, prioritizing Kalshi-style contract phrases and strict `call_start - 10 minutes` evidence cutoffs.
- [ ] Merge pre-call news manifests into training builds so `evidence_news_doc_ratio` is non-zero and closer to benchmark inference rows.
- [ ] Add phrase prior and company-topic affinity features after the diagnostic baseline is stable.
- [x] Re-run the unchanged five-event holdout benchmark after each slice, comparing model Brier/ECE to Kalshi bid/ask/mid and recording artifact paths.
- [x] Defer XGBoost until there are enough real event-aligned rows for event-grouped validation; use it as a benchmark, not as the first fix.

### First Implementation Slice

- [x] Add tests for a benchmark diagnostics report generated by `evaluate-model1-kalshi-benchmark`.
- [x] Implement diagnostics output without changing model predictions.
- [x] Run the conservative clean model benchmark using existing data.
- [x] Use the diagnostics to choose the exact data backfill targets and feature additions.

### Repair Results

- Diagnostics now emit by default beside benchmark reports:
  - `artifacts/model1/eval/model1-clean-no-5event-vs-kalshi-preclose-20260521.diagnostics.md`.
  - The clean synthetic-only model had `25` false positives, `6` false negatives, mean model probability `0.818872`, and `54/77` rows at `p >= 0.90`.
- Repaired the training distribution by recollecting properly paired SEC EX-99 manifests for `AAPL`, `AMZN`, `BAC`, `GOOGL`, `MSFT`, and `NVDA`:
  - Merged manifest: `data/manifests/benchmark-repair-ex99-six-companies.json` (`73` rows).
  - Repaired dataset: `artifacts/model1/datasets/benchmark-company-sec-market-phrases-no-5event-20260521.json`.
  - Dataset size: `1151` rows, `41` historical events, zero overlap with the five benchmark event keys.
- Best repaired clean benchmark so far:
  - Model: `artifacts/model1/models/model1-benchmark-company-sec-market-phrases-optimized-conservative-20260521.json`.
  - Report: `artifacts/model1/eval/model1-benchmark-company-sec-market-phrases-optimized-conservative-vs-kalshi-preclose-20260521.json`.
  - Table: `artifacts/model1/eval/model1-benchmark-company-sec-market-phrases-optimized-conservative-vs-kalshi-preclose-20260521.table.md`.
  - Diagnostics: `artifacts/model1/eval/model1-benchmark-company-sec-market-phrases-optimized-conservative-vs-kalshi-preclose-20260521.diagnostics.md`.
  - Brier `0.181180` vs Kalshi mid Brier `0.197245`, an improvement of `0.016065`.
  - ECE `0.125856` vs Kalshi mid ECE `0.109351`.
- Remaining gap to the explicit target: model beats Kalshi mid, but by `0.016065`, short of the desired `0.02` Brier edge. The largest remaining errors are GOOGL/BAC codenames and macro false positives, which likely need pre-call news/scenario context rather than more logistic tuning.
- Added temperature calibration as a safer alternative to isotonic:
  - Model: `artifacts/model1/models/model1-benchmark-company-sec-market-phrases-temperature-optimized-20260521.json`.
  - Report: `artifacts/model1/eval/model1-benchmark-company-sec-market-phrases-temperature-optimized-vs-kalshi-preclose-20260521.json`.
  - Table: `artifacts/model1/eval/model1-benchmark-company-sec-market-phrases-temperature-optimized-vs-kalshi-preclose-20260521.table.md`.
  - Diagnostics: `artifacts/model1/eval/model1-benchmark-company-sec-market-phrases-temperature-optimized-vs-kalshi-preclose-20260521.diagnostics.md`.
  - Learned temperature: `1.5`, fit from training time-holdout, not from the benchmark.
  - Brier `0.173332` vs Kalshi mid Brier `0.197245`, an improvement of `0.023913`, clearing the `>0.02` target.
  - ECE `0.108561` vs Kalshi mid ECE `0.109351`.
- Review hardening after the first target-clearing result:
  - `_time_split_examples` now orders event groups by actual `evidence_cutoff` rather than fiscal-period tuple ordering.
  - Benchmark reports now include `skip_summary` and fail if no rows have matching bid/ask snapshots.
  - Chronological-split rerun: `artifacts/model1/eval/model1-benchmark-company-sec-market-phrases-temperature-optimized-chronosplit-vs-kalshi-preclose-20260521.json`.
  - Chronological-split rerun kept the same Brier `0.173332`, Kalshi mid Brier `0.197245`, and `skip_summary.evaluated_rows=77`, `skipped_rows=0`.
  - Caveat: the five-event pack is now an iterative validation benchmark because it guided repair/model-selection steps. The result is still clean of direct training-row overlap, but a fresh untouched closed-event pack is needed before treating this as a final estimate of live edge.

## Addendum: Fresh Two-Event Validation (2026-05-21)

**Goal:** Check whether the `0.173332` five-event result survives a fresh, non-tuned evaluation pack, and separate actual fixes from benchmark-guided model selection.

### Result Attribution

- [x] The `0.173332` model did not train on the five benchmark events and its temperature `1.5` was fit on the training time-holdout, not directly on benchmark labels.
- [x] The model family, data repair targets, conservative hyperparameter grid, and temperature-calibration choice were nevertheless selected after repeated looks at the same five-event benchmark.
- [x] Treat the five-event score as iterative validation, not as a pristine final out-of-sample estimate.

### Fresh Validation Build

- [x] Selected fresh finalized events outside the five-event pack: `KXEARNINGSMENTIONTGT-26MAY20` and `KXEARNINGSMENTIONLOW-26MAY20`.
- [x] Fetched Kalshi contracts:
  - `artifacts/model1/fresh-validation/tgt-26may20-contracts.json`.
  - `artifacts/model1/fresh-validation/low-26may20-contracts.json`.
- [x] Fetched/constructed available pre-call evidence:
  - Target has no document included because its SEC release filing time was after the strict `call_start - 10 minutes` cutoff.
  - Lowe's uses `artifacts/model1/fresh-validation/sec/LOW-2026-Q1-ab546139e636.txt`, available before the 9:00 ET call cutoff.
- [x] Built examples and non-leaky Kalshi snapshots using only candlesticks ending at or before `call_start - 10 minutes`:
  - Examples: `artifacts/model1/fresh-validation/tgt-low-26may20-fresh-examples.json`.
  - Snapshots: `artifacts/model1/fresh-validation/tgt-low-26may20-fresh-snapshots.json`.
  - Build summary: `artifacts/model1/fresh-validation/tgt-low-26may20-build-summary.json`.
- [x] Evaluated the frozen chronological-split model without changing hyperparameters:
  - Model: `artifacts/model1/models/model1-benchmark-company-sec-market-phrases-temperature-optimized-chronosplit-20260521.json`.
  - Report: `artifacts/model1/fresh-validation/tgt-low-26may20-frozen-model-vs-kalshi.json`.
  - Table: `artifacts/model1/fresh-validation/tgt-low-26may20-frozen-model-vs-kalshi.md`.
  - Log: `artifacts/model1/fresh-validation/tgt-low-26may20-frozen-model-vs-kalshi.log`.

### Fresh Validation Results

- [x] Combined fresh pack: `23` rows, model Brier `0.242701`, Kalshi mid Brier `0.262412`, model-minus-Kalshi Brier `-0.019711`, model ECE `0.181424`, Kalshi mid ECE `0.256304`.
- [x] Target event: `13` rows, model Brier `0.226624`, Kalshi mid Brier `0.308312`, model-minus-Kalshi Brier `-0.081688`.
- [x] Lowe's event: `10` rows, model Brier `0.263602`, Kalshi mid Brier `0.202742`, model-minus-Kalshi Brier `0.060860`.
- [x] Interpretation: fresh validation directionally supports real signal versus Kalshi on the combined two-event pack, but the edge is `0.019711`, just shy of the `0.02` target and highly event-sensitive.

## Addendum: Structured Benchmark Pack and Runner Plan (2026-05-21)

**Goal:** Replace ad-hoc benchmark assembly with a structured event-pack format and one benchmark runner that makes model family, data provenance, Kalshi snapshots, and split hygiene explicit.

**Architecture:** Add a small benchmark package under `src/kalorie/benchmarking/` for typed pack models, Kalshi quote hydration, pack validation, and benchmark execution. Keep current `HistoricalTrainingExample` and `evaluate-model1-kalshi-benchmark` behavior available, but route new work through a reproducible pack manifest. Use Kalshi public market-data endpoints directly with exact `event_ticker` requests, pagination where needed, and centralized 429 backoff.

**Tech Stack:** Python, Pydantic models, Typer CLI, existing `KalshiPublicClient`, existing model1 predictor, pytest, ruff.

### Design Decisions

- [ ] Official benchmark default is `model_family=global_base`, using one frozen model for all rows.
- [ ] Company-niched models are allowed only as separate `model_family=company_niched` runs, never silently mixed into the global benchmark.
- [ ] Later hierarchical models should be benchmarked as `model_family=hierarchical`, with explicit global/company/topic components in the report.
- [ ] Every event pack must record whether it is `blind`, `validation`, or `training`, and any post-error inspection converts `blind` to `validation`.
- [ ] Kalshi snapshots must use the last candle ending at or before `call_start - 10 minutes`; never use candles ending after the target time.
- [ ] Broad company/status searches are fallback only. Exact event ticker market discovery is preferred because Kalshi public docs support unauthenticated market data and pagination, and exact event lookup reduces rate-limit pressure.

### Task 1: Add Benchmark Pack Domain Models

**Files:**
- Create: `src/kalorie/benchmarking/packs.py`
- Test: `tests/unit/test_benchmark_packs.py`

- [x] **Step 1: Write failing tests for pack validation**

Cover:
- event and market rows must use timezone-aware timestamps,
- `snapshot.candle_end_ts <= snapshot_target_time`,
- market ids in examples and snapshots must match,
- pack split must be one of `blind`, `validation`, or `training`,
- model family must be recorded by benchmark reports.

- [x] **Step 2: Run red test**

Run: `python -m pytest tests/unit/test_benchmark_packs.py -q`

Expected: fail because `kalorie.benchmarking.packs` does not exist.

- [x] **Step 3: Implement Pydantic models**

Add:
- `BenchmarkSplit = Literal["blind", "validation", "training"]`
- `BenchmarkModelFamily = Literal["global_base", "company_niched", "hierarchical", "ensemble_research"]`
- `BenchmarkEvent`
- `BenchmarkMarket`
- `BenchmarkSnapshot`
- `BenchmarkEvidenceDocument`
- `BenchmarkPackManifest`
- `BenchmarkPack`
- `validate_benchmark_pack(pack: BenchmarkPack) -> None`

- [x] **Step 4: Run green test**

Run: `python -m pytest tests/unit/test_benchmark_packs.py -q`

Expected: all tests pass.

### Task 2: Centralize Kalshi Market Data Fetching and Backoff

**Files:**
- Modify: `src/kalorie/clients/kalshi.py`
- Test: `tests/unit/test_kalshi.py`

- [x] **Step 1: Write failing tests for 429 retry and exact event market fetch**

Cover:
- `get_event_mention_markets(event_ticker)` should use exact event endpoint first and fallback to `/markets?event_ticker=...` only on `404`.
- retryable `429` responses should back off and retry before failing.
- paginated `/markets` responses should continue until cursor is exhausted or limit is reached.

- [x] **Step 2: Run red test**

Run: `python -m pytest tests/unit/test_kalshi.py -q`

Expected: fail on missing retry/backoff behavior.

- [x] **Step 3: Implement minimal client request helper**

Add a private helper such as `_get_json(path, params=None)` that:
- sends unauthenticated public requests to `https://api.elections.kalshi.com/trade-api/v2`,
- retries `429` with bounded exponential backoff,
- preserves current error behavior for non-retryable HTTP errors,
- is used by `get_market`, `get_event_mention_markets`, `get_historical_markets`, `get_market_candlesticks`, and internal market pagination.

- [x] **Step 4: Run green test**

Run: `python -m pytest tests/unit/test_kalshi.py -q`

Expected: all tests pass.

### Task 3: Add T-10 Snapshot Hydration

**Files:**
- Create: `src/kalorie/benchmarking/kalshi_snapshots.py`
- Test: `tests/unit/test_benchmark_kalshi_snapshots.py`

- [x] **Step 1: Write failing tests for snapshot selection**

Cover:
- chooses the latest candle with `end_period_ts <= snapshot_target_time`,
- rejects or skips candles missing bid/ask,
- never chooses a candle after target time,
- emits `preclose_yes_bid`, `preclose_yes_ask`, `snapshot_target_time`, and `candle_end_ts`.

- [x] **Step 2: Run red test**

Run: `python -m pytest tests/unit/test_benchmark_kalshi_snapshots.py -q`

Expected: fail because snapshot hydration module does not exist.

- [x] **Step 3: Implement snapshot hydration helper**

Add:
- `hydrate_event_snapshots(client, event, markets, *, lookback_minutes=120) -> list[BenchmarkSnapshot]`
- support exact event series tickers but default to `KXEARNINGSMENTION` when the event-specific series is unreliable,
- choose a 1-minute interval by default for benchmark packs,
- return structured skipped-market diagnostics.

- [x] **Step 4: Run green test**

Run: `python -m pytest tests/unit/test_benchmark_kalshi_snapshots.py -q`

Expected: all tests pass.

### Task 4: Build Pack CLI

**Files:**
- Modify: `src/kalorie/app/cli.py`
- Test: `tests/integration/test_benchmark_pack_cli.py`

- [x] **Step 1: Write failing CLI tests**

Cover:
- `build-benchmark-pack` reads event config, contracts, evidence manifests, and examples, then writes a pack folder.
- output includes `manifest.json`, `events.json`, `markets.json`, `snapshots.json`, `evidence.json`, `examples.json`, and `pack.json`.
- command fails if any market lacks a valid non-leaky T-10 snapshot unless `--allow-missing-snapshots` is passed.

- [x] **Step 2: Run red test**

Run: `python -m pytest tests/integration/test_benchmark_pack_cli.py -q`

Expected: fail because command does not exist.

- [x] **Step 3: Implement command**

Add `build-benchmark-pack` with options:
- `--event-config`
- `--contracts`
- `--examples`
- `--evidence-manifests`
- `--snapshots` for prehydrated snapshots or `--hydrate-kalshi-snapshots`
- `--split blind|validation|training`
- `--out`

- [x] **Step 4: Run green test**

Run: `python -m pytest tests/integration/test_benchmark_pack_cli.py -q`

Expected: all tests pass.

### Task 5: Add Pack Benchmark Runner

**Files:**
- Create: `src/kalorie/benchmarking/runner.py`
- Modify: `src/kalorie/app/cli.py`
- Test: `tests/unit/test_benchmark_runner.py`, `tests/integration/test_benchmark_pack_cli.py`

- [x] **Step 1: Write failing tests for model-family explicit benchmark output**

Cover:
- `run-benchmark-pack` requires `--model-family`.
- report records `model_family`, `model_path`, `pack_path`, `pack_split`, `excluded_events`, `calibration`, and per-row prediction source.
- global-base runner uses the same model for every row.
- company-niched runner fails if no company model mapping is supplied.

- [x] **Step 2: Run red tests**

Run:
`python -m pytest tests/unit/test_benchmark_runner.py tests/integration/test_benchmark_pack_cli.py -q`

Expected: fail because runner does not exist.

- [x] **Step 3: Implement runner**

Add:
- `run_model1_pack_benchmark(pack, model, model_family, metadata) -> dict`
- `run-benchmark-pack` CLI that writes `.json`, `.table.md`, `.log`, and `.diagnostics.md`
- compatibility path that reuses `_kalshi_benchmark_metric_report`, `_kalshi_benchmark_table_markdown`, `_kalshi_benchmark_detail_log`, and diagnostics helpers.

- [x] **Step 4: Run green tests**

Run:
`python -m pytest tests/unit/test_benchmark_runner.py tests/integration/test_benchmark_pack_cli.py -q`

Expected: all tests pass.

### Task 6: Rebuild Current Validation Packs in the New Format

**Files:**
- Data outputs under: `artifacts/benchmarks/`
- Documentation update: this file.

- [ ] **Step 1: Build packs for existing validation events**

Create:
- `artifacts/benchmarks/five-event-validation-20260521`
- `artifacts/benchmarks/tgt-low-fresh-validation-20260521`
- optionally `artifacts/benchmarks/wmt-26may21-live-20260521` if WMT labels and exact T-10 quotes are complete.

- [ ] **Step 2: Run frozen global-base benchmark**

Run the frozen model:
`artifacts/model1/models/model1-benchmark-company-sec-market-phrases-temperature-optimized-chronosplit-20260521.json`

against each pack with `--model-family global_base`.

- [ ] **Step 3: Verify reports**

Check:
- row counts match old outputs,
- `skip_summary.skipped_rows == 0`,
- no snapshot candle ends after target,
- five-event metrics match the previous `0.173332` report within rounding,
- TGT/LOW metrics match `0.242701` model Brier and `0.262412` Kalshi mid Brier within rounding.

### Task 7: Add Company-Niched Comparison Only After the Base Runner Is Stable

**Files:**
- Create or modify as needed after Tasks 1-6 pass.

- [ ] **Step 1: Add explicit company-model mapping input**

Use a separate command option such as:
`--company-model-map artifacts/model1/models/company-model-map.json`

- [ ] **Step 2: Enforce minimum evidence**

Fail `model_family=company_niched` unless each company has either:
- a supplied company model, or
- an explicit fallback policy recorded in the report.

- [ ] **Step 3: Compare model families side-by-side**

Output one summary table comparing:
- `global_base`
- `company_niched`
- later `hierarchical`

Do not let the best family overwrite the official baseline without a separate blind-pack result.

### Verification Checklist

- [ ] `python -m pytest tests/unit/test_kalshi.py tests/unit/test_benchmark_packs.py tests/unit/test_benchmark_kalshi_snapshots.py tests/unit/test_benchmark_runner.py -q`
- [ ] `python -m pytest tests/integration/test_benchmark_pack_cli.py -q`
- [ ] `python -m ruff check src tests`
- [ ] Fresh pack build produces no skipped snapshots.
- [ ] Benchmark reports clearly identify `model_family`.
- [ ] No benchmark row uses a quote candle after `call_start - 10 minutes`.

## Addendum: Closed Kalshi Mention Data Hunt (2026-05-21)

**Goal:** Find and hydrate additional finalized Kalshi earnings-mention events outside the contaminated validation sets, producing leakage-safe data artifacts for future benchmark training/evaluation without tuning any model.

**Output Root:** `artifacts/model1/data-hunt-20260521/`

- [x] Inspect cached `KXEARNINGSMENTION*` event candidates and exclude the old five-event pack, fresh TGT/LOW validation, and partial/local NVDA/CAVA/WMT candidates unless needed as comparison context.
- [x] Fetch exact Kalshi event/market metadata with event ticker lookups first, falling back only to event-scoped `/markets` queries.
- [x] For each contract, select the latest 1-minute candle with `end_period_ts <= call_start - 10 minutes`; never use candles after the target time.
- [x] Hydrate or reuse transcript, SEC EX-99, and pre-call news evidence manifests with explicit provenance paths.
- [x] Write per-event `event.json`, `contracts.json`, `snapshots.json`, `evidence-manifests.json`, transcript artifact/manifest, and `README.md`.
- [x] Write `candidate-events-summary.json` with readiness flags, blockers, counts, and artifact paths.
- [x] Verify that artifact JSON is parseable, no selected candle is after the target time, and ready events have finalized outcomes, snapshots, transcript, and at least one pre-call evidence source.

### Review

- [x] Inspected 54 eligible closed candidate events from the cached prefix file via read-only exploration, then hydrated 6 high-value events: META, NFLX, JPM, UBER, TSLA, and SBUX.
- [x] Ready count: 6 of 6 hydrated events. Total ready contracts/snapshots: 100.
- [x] Candidate summary: `artifacts/model1/data-hunt-20260521/candidate-events-summary.json`.
- [x] Top recommended benchmark additions: `KXEARNINGSMENTIONTSLA-26APR22`, `KXEARNINGSMENTIONUBER-26MAY06`, `KXEARNINGSMENTIONMETA-26APR29`, `KXEARNINGSMENTIONNFLX-26APR16`, and `KXEARNINGSMENTIONSBUX-26APR28`.
- [x] Verification command: `python artifacts\model1\data-hunt-20260521\verify_data_hunt.py` -> `summary_rows=6 ready_rows=6 snapshots_checked=100 missing=0 violations=0`.
- [x] No model training, tuning, or evaluation was run in this data-hunt pass.

## Addendum: Kalshi Earnings Workflow Implementation (2026-05-21)

**Goal:** Implement reusable workflows for closed real Kalshi earnings event packs and broad historical synthetic Kalshi-style rows from the transcript corpus.

- [x] Added workflow config/artifact models and transcript inventory generation.
- [x] Added SEC evidence planning/collection with cached CIK support, hard request-budget checks, `.htm` exhibit preference, and manifest dedupe.
- [x] Added validated simple Kalshi-style phrase generation with deterministic mining, strict OpenAI JSON parsing, and local transcript matching.
- [x] Added historical synthetic row generation from transcripts, manifests, and phrase catalogs.
- [x] Added real Kalshi event-pack collection and real event-pack row conversion.
- [x] Added workflow CLI commands and event-pack artifact verification.

### Workflow Implementation Review

- [x] New workflow modules live under `src/kalorie/workflows/`.
- [x] New commands include `plan-historical-evidence-collection`, `collect-historical-evidence-pack`, `generate-transcript-kalshi-phrases`, `build-historical-synthetic-kalshi-rows`, `collect-kalshi-mention-event-pack`, `build-kalshi-event-pack-training-rows`, and `verify-kalshi-workflow-artifacts`.
- [x] Focused verification passed: `python -m pytest tests/unit/test_workflow_models.py tests/unit/test_workflow_phrase_generation.py tests/unit/test_workflow_evidence_collection.py tests/unit/test_workflow_kalshi_event_pack.py tests/unit/test_workflow_verification.py tests/integration/test_historical_synthetic_workflow_cli.py tests/integration/test_kalshi_event_pack_workflow_cli.py -q` -> `22 passed`.
- [x] Scoped lint passed: `python -m ruff check src/kalorie/workflows src/kalorie/ml/datasets.py tests/unit/test_workflow_models.py tests/unit/test_workflow_phrase_generation.py tests/unit/test_workflow_evidence_collection.py tests/unit/test_workflow_kalshi_event_pack.py tests/unit/test_workflow_verification.py tests/integration/test_historical_synthetic_workflow_cli.py tests/integration/test_kalshi_event_pack_workflow_cli.py`.
- [x] Code review gaps fixed: real event rows now use T-10 snapshot prices, transcript inventory includes explicit estimated call times, and event-pack verification checks transcript/evidence presence plus snapshot/contract parity.
- [x] Full pytest was attempted: `python -m pytest -q` -> `215 passed, 2 failed`; both failures require missing local fixture `Earnings-Release-2026-Q1.pdf`.

### Historical Workflow Run Review

- [x] Dry-run SEC plan wrote `artifacts/model1/workflows/historical-synthetic-20260521/sec-request-plan.json`: 88 projected requests across 51 companies.
- [x] Live SEC collection checkpointed after each request/fetch, but the SEC API rate-limited the run; final live output was 0 new manifests with summaries in `artifacts/model1/workflows/historical-synthetic-20260521/evidence-summary.json`.
- [x] Parallel phrase generation wrote `artifacts/model1/workflows/historical-synthetic-20260521/phrase-catalog.json`: 20,083 validated phrase rows, 0 phrase skips.
- [x] Historical row build used cached SEC manifests from `artifacts/model1/datasets/sec-corpus-plus-wmt-fy27-manifests-20260521.json` and wrote `artifacts/model1/datasets/historical-synthetic-kalshi-style-20260521.json`: 409 rows.
- [x] Row verification command: `python -c "... assert all(r['market_venue']=='synthetic' for r in rows) ..."` -> `409 rows verified synthetic`.

### Historical Synthetic Retrain Review

- [x] Input dataset: `artifacts/model1/datasets/historical-synthetic-kalshi-style-20260521.json` (`409` rows, `25` event keys, `2` companies, `294` positive rows, `115` negative rows).
- [x] Wrote enhanced `model1` artifact: `artifacts/model1/models/model1-current-historical-synthetic-kalshi-style-20260521.json`.
- [x] Wrote market-residual artifact: `artifacts/model1/models/market-residual-historical-synthetic-kalshi-style-20260521.pkl`.
- [x] Wrote split/in-sample summary: `artifacts/model1/eval/retrain-historical-synthetic-kalshi-style-20260521-summary.json`.
- [x] Wrote external benchmark summary: `artifacts/model1/eval/retrain-historical-synthetic-kalshi-style-20260521-external-benchmarks.json`.
- [x] Time-event holdout Brier: market prior `0.250000`, enhanced `model1` `0.171382`, market residual `0.256651`.
- [x] External benchmark Brier highlights: 5-event pack model1 `0.193626` vs Kalshi mid `0.197245`; TGT/LOW model1 `0.237034` vs Kalshi mid `0.262412`; residual underperformed on 5-event and TGT/LOW but remained strong on NVDA/CAVA.
- [x] Verification passed: `python -m pytest tests/unit/test_market_residual.py tests/unit/test_grouped_calibration.py tests/unit/test_benchmark_runner.py tests/unit/test_model1.py -q` -> `21 passed`.
- [x] Artifact reload check passed for the JSON model, residual pickle, split summary, and external benchmark report.

### Merged Historical Synthetic Plus Benchmark Retrain Review

- [x] Clarified row meaning: the 409-row historical synthetic dataset contains labeled `(company, quarter, target_phrase)` examples, not 409 words or 409 events.
- [x] Merged `artifacts/model1/datasets/benchmark-company-plus-clean-plus-wmt-20260521.json` with `artifacts/model1/datasets/historical-synthetic-kalshi-style-20260521.json`.
- [x] Wrote merged dataset: `artifacts/model1/datasets/benchmark-company-plus-historical-synthetic-20260521.json` (`2,978` rows after removing `10` duplicate `market_id`s).
- [x] Trained optimized merged model: `artifacts/model1/models/model1-historical-synthetic-plus-benchmark-optimized-20260521.json`.
- [x] Optimization report: `artifacts/model1/eval/model1-historical-synthetic-plus-benchmark-optimized-training-report-20260521.json`; time-event holdout Brier `0.169557`.
- [x] Pre-train gate report: `artifacts/model1/eval/model1-historical-synthetic-plus-benchmark-pretrain-gate-20260521.json`; gate passed with warning that news evidence ratio is zero for all rows.
- [x] 5-event benchmark report: `artifacts/model1/eval/model1-historical-synthetic-plus-benchmark-vs-5event-kalshi-20260521.json`; model Brier `0.222313` vs Kalshi mid `0.197245`.
- [x] TGT/LOW fresh benchmark report: `artifacts/model1/fresh-validation/tgt-low-26may20-historical-synthetic-plus-benchmark-vs-kalshi.json`; model Brier `0.273286` vs Kalshi mid `0.262412`.
- [x] NVDA fresh benchmark report: `artifacts/model1/fresh-validation/nvda-26may20-historical-synthetic-plus-benchmark-vs-kalshi.json`; model Brier `0.164865` vs Kalshi mid `0.272109`.
- [x] CAVA fresh benchmark report: `artifacts/model1/fresh-validation/cava-26may19-historical-synthetic-plus-benchmark-vs-kalshi.json`; model Brier `0.082048` vs Kalshi mid `0.250000`.
- [x] Artifact parse check passed for merged dataset, model artifact, training report, and all benchmark JSON outputs.

### DefeatBeta/YFinance News Evidence Review

- [x] Added leakage-safe historical news collector command: `collect-historical-news-manifests`.
- [x] TDD verification covered cutoff filtering, yfinance fallback, and CLI manifest/summary output.
- [x] Collected DefeatBeta/yfinance news from `artifacts/model1/datasets/historical-synthetic-kalshi-style-20260521.json`.
- [x] News manifest output: `artifacts/model1/workflows/historical-synthetic-20260521/news-manifests-defeatbeta-yfinance.json` (`99` manifests, all DefeatBeta, all WMT, zero leakage violations).
- [x] ACN remained uncovered: `23` ACN windows had no pre-cutoff DefeatBeta/yfinance articles; NewsData/Tiingo keys were not available in the loaded settings.
- [x] Merged SEC plus news manifests: `artifacts/model1/workflows/historical-synthetic-20260521/evidence-manifests-sec-plus-news.json` (`467` manifests).
- [x] Rebuilt news-enabled historical rows: `artifacts/model1/datasets/historical-synthetic-kalshi-style-sec-plus-news-20260521.json` (`409` rows, `26` rows with `evidence_news_doc_ratio > 0`).
- [x] Merged benchmark plus news-enabled historical rows: `artifacts/model1/datasets/benchmark-company-plus-historical-synthetic-sec-plus-news-20260521.json` (`2,978` rows, `26` rows with news features).
- [x] Trained news-enabled optimized model: `artifacts/model1/models/model1-historical-synthetic-sec-plus-news-optimized-20260521.json`; holdout Brier `0.168521`.
- [x] Benchmark deltas vs prior merged no-news model: 5-event `-0.006733`, TGT/LOW `-0.000301`, NVDA `+0.013715`, CAVA `+0.000596` Brier.
- [x] Verification passed: `python -m pytest tests/unit/test_workflow_evidence_collection.py tests/integration/test_historical_synthetic_workflow_cli.py -q` -> `11 passed`.
- [x] Scoped lint passed for `src/kalorie/workflows/evidence_collection.py` and the news workflow tests; CLI import/undefined-name check passed with `--select I,F`.

### Cached Event Dossier Implementation Plan

**Goal:** Generate and cache one pre-call event dossier per company-quarter so topics, likely Q&A, synthetic call snippets, and target phrase variants become reusable training features.

**Files:**

- Modify `src/kalorie/data_grepping/event_scenarios.py` to add target phrase variants and source digest metadata to `EventScenarioCatalog`.
- Add `src/kalorie/workflows/event_dossiers.py` for event grouping, source digesting, cache reuse, and catalog persistence.
- Modify `src/kalorie/workflows/historical_synthetic.py` so historical synthetic rows can consume event dossier scenario texts and phrase variants.
- Modify `src/kalorie/app/cli.py` with `generate-historical-event-dossiers` and `--event-dossiers` support for row building.
- Test with `tests/unit/test_event_scenarios.py`, `tests/unit/test_workflow_event_dossiers.py`, and `tests/integration/test_historical_synthetic_workflow_cli.py`.

**Execution Steps:**

- [x] Add failing tests for event dossier parsing, source digest stability, and cache reuse.
- [x] Implement the event dossier schema/generator/cache helpers.
- [x] Add failing integration tests proving cached dossiers add `scenario_*` and `template_*` features to rebuilt rows.
- [x] Wire CLI commands and historical row building options.
- [x] Generate dossiers for the current SEC+news historical manifests and phrase catalog.
- [x] Rebuild the historical dataset, merge into the benchmark training set, retrain, and rerun benchmarks.
- [x] Verify tests, lint, artifact parsing, and benchmark deltas.

### Cached Event Dossier Review

- [x] Added cached event dossier generation with stable source digests and per-event JSON reuse in `src/kalorie/workflows/event_dossiers.py`.
- [x] Extended event scenario catalogs with `source_digest`, `prompt_version`, and `target_phrase_variants`.
- [x] Added robust parsing for strict JSON, fenced JSON, and JSON objects embedded in LLM prose.
- [x] Added `generate-historical-event-dossiers`, `--event-dossiers`, `--record-concurrency`, and bounded dossier generation concurrency.
- [x] Generated `25` leakage-safe SEC+news event dossiers at `artifacts/model1/workflows/historical-synthetic-20260521/event-dossiers-sec-plus-news.json` after applying the call-start cutoff filter.
- [x] Rebuilt `409` historical synthetic rows with dossier-derived features at `artifacts/model1/datasets/historical-synthetic-kalshi-style-sec-plus-news-dossiers-20260521.json`; all `409` rows include `scenario_*` and `template_*` feature keys, with `352` rows having nonzero template variants.
- [x] Merged a new `2,978` row training set at `artifacts/model1/datasets/benchmark-company-plus-historical-synthetic-sec-plus-news-dossiers-20260521.json`.
- [x] Trained dossier-enabled optimized model at `artifacts/model1/models/model1-historical-synthetic-sec-plus-news-dossiers-optimized-20260521.json`; holdout Brier `0.173021` vs prior SEC+news `0.168521`.
- [x] External benchmark deltas vs prior SEC+news model: 5-event `+0.020196`, TGT/LOW `+0.034579`, NVDA `-0.011769`, CAVA `-0.025099` Brier.
- [x] Verification passed: `python -m pytest tests/unit/test_event_scenarios.py tests/unit/test_workflow_event_dossiers.py tests/unit/test_features.py tests/integration/test_historical_synthetic_workflow_cli.py -q` -> `27 passed`.
- [x] Scoped lint passed for the touched dossier, embedding cache, historical synthetic, and test files.
- [x] Current limitation: the external benchmark example files do not yet include event-specific dossier features, so the benchmark is still partially measuring feature-availability mismatch. The next clean evaluation should rebuild benchmark/fresh validation examples with their own pre-call dossiers.

### Dossier Feature Parity + Noise Control Design

**Goal:** Make dossier-derived features testable on the same surfaces where the model is benchmarked, while reducing noisy LLM variants that can dilute the signal.

**Design:**

- Add a reusable event-dossier feature context that maps `EventScenarioCatalog` rows into `scenario_texts_by_event` and event-scoped `template_phrases_by_event`.
- Filter generated phrase variants before feature extraction:
  - Drop single-character tokens, pronouns, filler words, and obvious question-function words such as `you`, `i`, `we`, `what`, `re`, and `s`.
  - Keep multiword business/domain variants and exact normalized target keys.
  - Preserve deterministic ordering and dedupe variants per target.
- Extend real Kalshi event-pack row building so benchmark rows can be rebuilt with the same dossier-derived feature keys used in training.
- Add CLI support to `build-kalshi-event-pack-training-rows` for `--event-dossiers`, mirroring historical synthetic row building.
- Rebuild the five-event pack examples with event dossiers, rerun dossier model benchmarks, and compare against both Kalshi mid and the previous no-dossier benchmark.

**Acceptance Criteria:**

- Unit tests prove variant filtering removes noisy pronoun/filler keys and keeps useful business variants.
- Integration tests prove real event-pack rows include nonzero `scenario_*` and `template_*` features when matching dossiers are provided.
- Existing historical synthetic dossier tests still pass.
- Corrected benchmark reports use dossier-enabled benchmark rows rather than missing-feature evaluation rows.

**Implementation Tasks:**

- [x] Add tests and implementation for filtered `phrase_variants_by_event`.
- [x] Add tests and implementation for event-pack dossier feature wiring.
- [x] Add CLI option and tests for `build-kalshi-event-pack-training-rows --event-dossiers`.
- [x] Generate/build dossier-enabled five-event benchmark rows.
- [x] Rerun benchmark reports and summarize Brier deltas.
- [x] Run final tests, scoped lint, artifact parse checks, and update this review section.

**Review Results:**

- Added feature parity for real Kalshi event-pack rows via `event_dossiers` and `embedding_provider` arguments in `build_real_event_pack_training_rows`.
- Added `--event-dossiers` support to `build-kalshi-event-pack-training-rows`, including compatibility with both normalized event-pack fixtures and the raw collected five-event artifact format.
- Filtered noisy LLM-generated target phrase variant keys before feature extraction.
- Generated five event-pack dossiers at `artifacts/model1/event-pack-20260521/event-dossiers.json`.
- Built corrected five-event examples at `artifacts/model1/event-pack-20260521/historical-examples-5event-pack-dossiers.json` (`77` rows; `77` with nonzero scenario features; `77` with nonzero template features).
- Verified the reusable CLI raw-pack path at `artifacts/model1/event-pack-20260521/historical-examples-5event-pack-dossiers-cli.json` (`77` rows; `0` skips; `77` with nonzero scenario features; `77` with nonzero template features).
- Rebuilt filtered dossier historical rows at `artifacts/model1/datasets/historical-synthetic-kalshi-style-sec-plus-news-dossiers-filtered-20260521.json` (`409` rows).
- Trained cleaned dossier model at `artifacts/model1/models/model1-historical-synthetic-sec-plus-news-dossiers-filtered-optimized-20260521.json`; holdout Brier `0.170938`, improving over the prior dossier holdout `0.173021`.
- Tested an additional stricter target-row filter, but rejected it after verification because it worsened holdout Brier to `0.177028` and five-event feature-parity Brier to `0.222390`.
- Corrected five-event benchmark:
  - Prior mismatched dossier eval: Brier `0.235776`.
  - Dossier eval with feature parity: Brier `0.220928`.
  - Filtered dossier model with feature parity: Brier `0.218049`.
  - Kalshi mid: Brier `0.197245`.
- Verification passed:
  - `python -m pytest tests/unit/test_workflow_event_dossiers.py tests/integration/test_kalshi_event_pack_workflow_cli.py tests/integration/test_historical_synthetic_workflow_cli.py -q` -> `18 passed`.
  - `python -m ruff check --select I,F ...` -> passed.
  - Generated artifact JSON parse checks passed.
  - IDE lints found no errors in touched source and test files.
