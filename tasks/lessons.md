# Lessons

## 2026-05-19: Do not duplicate binary Brier as MSE

When evaluating binary probability forecasts, Brier score is the mean squared error of the predicted probability against the binary outcome. Report Brier score as the primary metric and avoid presenting MSE as a separate model-quality metric unless explicitly needed for mathematical explanation.

## 2026-05-19: Anchor market snapshots to evidence availability

For earnings mention-market backtests, the market probability used for comparison should be captured at a decision time tied to evidence availability, such as five minutes after SEC supplemental material is filed. Do not use market close or post-resolution prices as the default backtest probability, because those can include call/transcript information and leak settlement outcome.

## 2026-05-19: Preserve original source artifacts separately from extracted text

When collecting SEC exhibits or other source documents, store the original fetched artifact (`.htm`, `.pdf`, etc.) as well as the extracted text used for modeling. Manifests should keep explicit paths for both so review/reprocessing never depends only on lossy text extraction.

## 2026-05-20: Do not require real Kalshi markets for phrase-presence training

When building transcript mention training data, separate the supervised learning target from the venue market catalog. We can train phrase-presence models from transcripts and pre-call materials using synthetic target phrases; Kalshi markets and odds are optional downstream comparison inputs for profit/backtest analysis, not prerequisites for building the core mention dataset.

## 2026-05-20: Keep base target phrases aligned with real market wording

Do not rely on only business-generic phrase lists (`revenue`, `margin`, etc.) for mention-market modeling. Always include Kalshi-style market terms and named entities (for example, product names, competitor names, and idiosyncratic phrases like `omnichannel`, `openai`, `salmon`) either via curated defaults or contract-derived phrase expansion.

## 2026-05-20: Verify historical news coverage before training claims

When adding a new news source, always measure effective coverage over training periods before attributing model behavior to feature relevance. Confirm provider entitlement (archive access), article date span, and `% training rows with nonzero news evidence` so "news had no effect" is not confused with "news was never present in the training rows."

## 2026-05-20: Confirm existing pipeline assumptions before asking architecture questions

When the user says a capability is already implemented in-repo (for example, market-specific fine-tuning logic), first inspect and anchor to that existing pipeline instead of asking model-stack selection questions that imply re-architecture.

## 2026-05-20: Verify provider modality before implementing connectors

Before wiring a new third-party source, confirm whether the provider is a hosted REST API, a local Python library, or a library-backed dataset abstraction. Build the connector against the actual modality first, then add transport fallbacks; this avoids spending cycles on unreachable endpoints that are not the intended integration surface.

## 2026-05-21: Preserve user-selected defaults unless explicitly changed

When the user confirms an option (for example, parallel job execution), carry that choice forward as the default and do not re-open it unless new technical constraints force a trade-off discussion.

## 2026-05-21: Persist shared historical caches for reruns

When designing run storage for iterative model workflows, keep a stable shared cache directory scoped to the natural evidence boundary (for example event/company scope) so reruns can reuse historical transcripts/news/filings instead of re-fetching every time.

## 2026-05-21: Scope caches and cutoffs to the event, not the contract

For earnings mention workflows, cache and provenance keys should be event/company scoped (shared across phrase contracts), and every run spec must include explicit leakage controls (decision cutoff), cache version signatures, and job idempotency semantics.

## 2026-05-21: Treat `<entity>` as company speech, not all transcript text

When implementing Kalshi mention settlement labels, do not assume the full transcript is eligible speech. `<entity>` should include company-side speakers (including operator) and exclude questioners in Q&A; if speaker metadata is partial, apply conservative analyst/question filtering and surface uncertainty instead of silently counting all mentions.

## 2026-05-20: External market API throttles should degrade gracefully

When listing markets from Kalshi, treat HTTP 429 as a normal transient condition and return best-effort partial results (or empty list) rather than raising a 500 to the UI. Paginated fetch loops must stop cleanly on rate-limit responses so the app remains usable under load.

## 2026-05-20: Verify payload content, not only HTTP success

For market discovery endpoints, a 200 response can still be functionally broken (`markets: []`). Always verify expected records (for example, a known active ticker like WMT) in live smoke checks before declaring the fetch path healthy.

## 2026-05-21: Use contract phrase fields, not question titles

For Kalshi mention markets, list endpoints can return a generic event question in `title` for every contract while the actionable phrase lives in `yes_sub_title` or `custom_strike`. UI phrase columns should bind to explicit phrase fields and avoid re-parsing the full question text.

## 2026-05-21: Use event_ticker getter for event pages

When the UI already knows the event scope, fetch contracts via one direct `GET /markets?event_ticker=...` request instead of scanning all mention series. Keep global discovery for the home index only, and use event-scoped getters for detail pages to avoid avoidable 20-30s cold loads.

## 2026-05-21: Prefer Kalshi v1 search for mention discovery

For fast mention-event discovery, `GET /v1/search/series` with `category=Mentions` returns event rows plus nested market contracts in one or two requests. Use this before brute-force series enumeration, and keep legacy `trade-api/v2` scans as fallback for resiliency.

## 2026-05-21: Enforce decision-time cutoff as call-start minus ten minutes

When generating training rows, `evidence_cutoff` must be anchored to decision time (`call_start - 10m`), not to the newest available evidence document timestamp. Filter evidence docs by that fixed cutoff and persist the cutoff directly into each example so leakage checks and model training operate on the intended pre-call information set.
