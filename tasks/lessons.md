# Lessons

## 2026-05-23: Verify market discovery against unfiltered historical coverage

When collecting Kalshi historical markets, do not assume a restrictive status filter returns the full finalized universe. Compare filtered discovery against an unfiltered discovery pass, then hydrate historical market detail for missing fields like `close_time`.

## 2026-05-23: Support both current and legacy candlestick shapes

Kalshi historical candlesticks may expose price values as `yes_bid.close` / `yes_ask.close`, while newer payloads may use `close_dollars`. Snapshot parsing must accept both shapes before declaring a market has no usable pre-close price.

## 2026-05-23: Use ticker-prefix search for prefix-defined datasets

For earnings mention markets, direct `GET /v1/search/series?query=KXEARNINGSMENTION` is broader than category-only `category=Mentions` discovery. When the target universe is defined by a ticker prefix, validate coverage with a prefix search before treating a category search as exhaustive.

Avoid adding ordering parameters unless they have been coverage-tested. In live checks, default prefix search returned a superset of explicitly start-time ordered prefix search.

## 2026-05-23: Retry transient transport failures during long market pulls

Long historical Kalshi pulls can fail from transient DNS or connection errors after many successful requests. Retry `httpx.TransportError` alongside HTTP 429 so a single temporary network failure does not discard a near-complete collection run.

## 2026-05-24: Verify artifact locations after user moves files

When referencing generated datasets, inspect the current artifact directory before claiming a file is missing. The canonical 3500-row Kalshi earnings mention corpus may be moved from a run-specific folder like `artifacts/full-v5/` into `artifacts/full/`.

## 2026-05-24: Execute after approval instead of re-asking

When the user resolves the one open design choice and then tells me to do the task, proceed with implementation and verification. Do not restate the design and wait for another approval unless a new blocker appears.

## 2026-05-24: Keep generated artifact directories intentionally small

For modeling runs, do not let broad artifact directories such as `artifacts/full/` accumulate derived experiments, reports, temporary predictions, skipped-market dumps, or stale intermediate files. Keep only canonical source datasets and explicitly useful final outputs; delete unnecessary generated files once their results are captured in `tasks/todo.md`, a report, or a reproducible command.

## 2026-05-25: Do not estimate OpenAI agentic web-search costs from final JSON size

For Responses API runs that use web search and frontier reasoning models, do not treat the saved final JSON packet size as the billed output. The API may bill for hidden reasoning, tool orchestration, retries, failed/rate-limited attempts, or other usage not reflected in the parsed artifact. Before scaling a paid collection, run a tiny metered sample, persist the raw response `usage`/billing metadata, and enforce explicit `max_paid_calls` / budget caps.

## 2026-05-26: Use absolute source paths in process launchers

When a launcher starts child Python processes, set `PYTHONPATH` to an absolute `src` path rather than a relative `src`. In Git Bash/MSYS on Windows, convert that path with `cygpath -w` before handing it to Windows Python. Child process working-directory and environment inheritance can differ from the parent shell's assumptions, and import failures should be caught by actually starting the launcher, not just parser checks.

## 2026-05-27: Keep model cards predictive-only

Do not put execution policy, minimum margin, ROI, trade selection, Kelly sizing, or other risk-tolerance assumptions inside saved model cards or model creation. Model cards should evaluate probability quality; trading overlays and expected-return distributions belong in separate risk preset artifacts/UI fields.
